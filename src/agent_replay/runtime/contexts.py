"""Recording and replay run contexts: one span per model step and per tool call.

Each context remembers, per span id, the step fields that are not OTel attributes (kind,
payload, tokens, latency, cost, cache status); the runner merges them with the exported
span dicts when it stores the run.

A tool that raises is a failed step: its span gets status ERROR and ``error.type``
(the exception class), the model gets ``{"error": "tool_error"}`` so the agent can go on, and
``tool_error`` remembers the first such class name so the runner stores the run as ``error``.
Tool payloads carry ``raised`` so replay reproduces that status from the source spans.

A ``RateLimitError`` that escapes the client (429 after all attempts) sets the model span's
``error.type`` to ``rate_limit``.

Replay applies tool faults (``runtime.faults``) to the stubbed result: the span gets the
fault's ``error.type`` and ``payload["fault"]`` holds the spec string. A malformed-output
fault reaches the agent as an ``ok`` result, so the agent's own validation is exercised.
Replaying a faulted run reproduces it: the recorded ``fault`` is carried forward and a
recorded malformed-output call again reaches the agent as ``ok``, so requests (and cache
keys) match the faulted run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Status, StatusCode, Tracer, format_span_id

from agent_replay.adapters.base import ToolResult
from agent_replay.codes import INJECTED_RATE_LIMIT, RATE_LIMIT
from agent_replay.llm import (
    InjectedRateLimitError,
    ModelRequest,
    ModelResponse,
    RateLimitError,
    make_tool_call_id,
    request_hash,
)
from agent_replay.llm.budget import BudgetTracker
from agent_replay.llm.provider import ModelProvider
from agent_replay.runtime.faults import (
    MALFORMED_TOOL_OUTPUT,
    TIMEOUT_ERROR_TYPE,
    TOOL_TIMEOUT,
    Fault,
    malformed_payload,
    tool_fault,
)
from agent_replay.runtime.stubs import UNRECORDED, UNRECORDED_TOOL_CALL, ToolStubs
from agent_replay.tracing import StepSpan, model_span, tool_span

Row = dict[str, Any]


@dataclass(frozen=True)
class ToolStep:
    """One executed or stubbed tool call.

    ``outcome`` is what the agent sees; ``result`` is stored in the span payload;
    ``error_type`` goes on the span (it differs from ``outcome.error_type`` for faults).
    """

    outcome: ToolResult
    result: Any
    latency_ms: int
    raised: bool = False
    error_type: str | None = None
    fault: str | None = None


def _step(outcome: ToolResult, result: Any, latency_ms: int, raised: bool) -> ToolStep:
    return ToolStep(outcome, result, latency_ms, raised, outcome.error_type)


def span_id_of(step: StepSpan) -> str:
    return format_span_id(step.span.get_span_context().span_id)


class BaseContext:
    """Shared model-call path and tool-call bookkeeping."""

    def __init__(
        self,
        tracer: Tracer,
        client: ModelProvider,
        *,
        tracker: BudgetTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tracer = tracer
        self._client = client
        self._tracker = tracker
        self._clock = clock
        self.steps: dict[str, Row] = {}
        self.diverged = False
        self.tool_error: str | None = None
        self._model_calls = 0
        self._pending_ids: list[str] = []
        self._tool_index = 0

    def _cache_status(self, response: ModelResponse) -> str:
        raise NotImplementedError

    def call_model(self, request: ModelRequest) -> ModelResponse:
        self._model_calls += 1
        self._pending_ids = []
        self._tool_index = 0
        key = request_hash(request)
        with model_span(self._tracer, request.model, provider=request.provider) as step:
            payload: Row = {
                "request": request.model_dump(mode="json"),
                "response": None,
                "request_hash": key,
                "cache_status": None,
                "latency_ms": 0,
            }
            row: Row = {"kind": "model", "payload": payload}
            self.steps[span_id_of(step)] = row
            try:
                response = self._client.generate(request)
            except RateLimitError as exc:
                injected = isinstance(exc, InjectedRateLimitError)
                step.set_error_type(INJECTED_RATE_LIMIT if injected else RATE_LIMIT)
                raise
            status = self._cache_status(response)
            step.set_usage(response.input_tokens, response.output_tokens)
            payload.update(
                response=response.model_dump(mode="json"),
                cache_status=status,
                latency_ms=response.latency_ms,
            )
            row.update(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                cache_status=status,
                cost_usd=self._tracker.cost_of(response, request.model)
                if self._tracker
                else 0.0,
            )
        self._pending_ids = [c.id for c in response.tool_calls]
        return response

    def _next_call_id(self) -> str:
        """Tool-call ids are taken in order from the last model response's tool_calls."""
        index = self._tool_index
        self._tool_index += 1
        if index < len(self._pending_ids):
            return self._pending_ids[index]
        return make_tool_call_id(self._model_calls, index)

    def _execute(self, name: str, args: dict[str, Any]) -> ToolStep:
        """Run or stub the tool."""
        raise NotImplementedError

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        call_id = self._next_call_id()
        with tool_span(self._tracer, name, call_id=call_id, arguments=args) as step:
            payload: Row = {
                "name": name,
                "call_id": call_id,
                "args": args,
                "result": None,
                "fault": None,
                "raised": False,
                "latency_ms": 0,
            }
            row: Row = {"kind": "tool", "payload": payload}
            self.steps[span_id_of(step)] = row
            done = self._execute(name, args)
            step.set_result(done.result)
            if done.error_type is not None:
                step.set_error_type(done.error_type)
                step.span.set_status(Status(StatusCode.ERROR, done.error_type))
            if done.raised and self.tool_error is None:
                self.tool_error = done.outcome.error_type
            payload.update(
                result=done.result,
                latency_ms=done.latency_ms,
                raised=done.raised,
                fault=done.fault,
            )
            row["latency_ms"] = done.latency_ms
        return done.outcome


def _as_tool_result(data: Any, error_type: str | None = None) -> ToolResult:
    if not isinstance(data, dict):
        data = {"error": "malformed_tool_output", "value": data}
        error_type = error_type or "malformed_tool_output"
    return ToolResult(
        ok=error_type is None,
        data=data,
        raw=json.dumps(data, sort_keys=True),
        error_type=error_type,
    )


class RecordingContext(BaseContext):
    """Live model (cache written, never read) and real tool functions."""

    def __init__(
        self,
        tracer: Tracer,
        client: ModelProvider,
        tools: Mapping[str, Callable[..., dict[str, Any]]],
        *,
        tracker: BudgetTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(tracer, client, tracker=tracker, clock=clock)
        self._tools = tools

    def _cache_status(self, response: ModelResponse) -> str:
        return "live"

    def _execute(self, name: str, args: dict[str, Any]) -> ToolStep:
        fn = self._tools.get(name)
        started = self._clock()
        raised = False
        if fn is None:
            outcome = _as_tool_result({"error": "unknown_tool"}, "unknown_tool")
        else:
            try:
                outcome = _as_tool_result(fn(**args))
            except Exception as exc:  # noqa: BLE001 - recorded as a failed step
                raised = True
                outcome = _as_tool_result({"error": "tool_error"}, type(exc).__name__)
        latency_ms = max(0, round((self._clock() - started) * 1000))
        return _step(outcome, outcome.data, latency_ms, raised)


class ReplayContext(BaseContext):
    """Cache-first model and tool results stubbed from the source run."""

    def __init__(
        self,
        tracer: Tracer,
        client: ModelProvider,
        stubs: ToolStubs,
        *,
        tracker: BudgetTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
        faults: Sequence[Fault] = (),
        tool_timeout_ms: int = 0,
    ) -> None:
        super().__init__(tracer, client, tracker=tracker, clock=clock)
        self._stubs = stubs
        self._faults = tuple(faults)
        self._tool_timeout_ms = tool_timeout_ms

    def _cache_status(self, response: ModelResponse) -> str:
        return "hit" if response.cache_status == "hit" else "miss"

    def _execute(self, name: str, args: dict[str, Any]) -> ToolStep:
        found = self._stubs.next_call(name, args)
        if found is not None and found.error_type == UNRECORDED_TOOL_CALL:
            # The source run had already diverged here; its stub is not a real result.
            self.diverged = True
        fault = tool_fault(self._faults, name)
        if fault is not None:
            # The recorded call is still consumed so later calls stay aligned.
            if found is None:
                self.diverged = True
            recorded = dict(UNRECORDED) if found is None else found.result
            latency_ms = 0 if found is None else found.latency_ms
            return self._faulted(fault, recorded, latency_ms)
        if found is None:
            self.diverged = True
            data = dict(UNRECORDED)
            return _step(_as_tool_result(data, UNRECORDED_TOOL_CALL), data, 0, False)
        # The recorded result and error type, unchanged, so the agent sends the model
        # exactly what it sent in the source run (same next request hash). A recorded
        # malformed-output fault was an ok result for the agent; keep it that way.
        agent_error = found.error_type
        if found.fault is not None and found.fault.startswith(
            f"{MALFORMED_TOOL_OUTPUT}:"
        ):
            agent_error = None
        return ToolStep(
            _as_tool_result(found.result, agent_error),
            found.result,
            found.latency_ms,
            found.raised,
            error_type=found.error_type,
            fault=found.fault,
        )

    def _faulted(self, fault: Fault, recorded: Any, latency_ms: int) -> ToolStep:
        if fault.kind == TOOL_TIMEOUT:
            data = {"error": TOOL_TIMEOUT}
            outcome = _as_tool_result(data, TIMEOUT_ERROR_TYPE)
            return ToolStep(
                outcome,
                data,
                self._tool_timeout_ms,
                error_type=TIMEOUT_ERROR_TYPE,
                fault=fault.spec,
            )
        if fault.kind == MALFORMED_TOOL_OUTPUT:
            data = malformed_payload(recorded)
            # An ok result: the agent must notice the payload breaks the output model.
            return ToolStep(
                _as_tool_result(data),
                data,
                latency_ms,
                error_type=MALFORMED_TOOL_OUTPUT,
                fault=fault.spec,
            )
        raise ValueError(f"not a tool fault: {fault.spec}")
