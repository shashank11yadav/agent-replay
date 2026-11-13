"""Run lifecycle: run an agent in a recording or replay context and store run + spans.

The run row and all of its spans are written in one transaction after the agent returns,
so a failed run never leaves spans without a run row. Status mapping:

- agent returns ok: ``ok`` (``diverged`` if a tool call had no recorded match);
- a tool raised: ``error`` with that exception class name, even if the agent went on;
- agent returns an error outcome or raises: ``error`` with the error class name;
- ``RateLimitError`` (429 after all attempts): ``error`` with ``rate_limit``, or
  ``injected_rate_limit`` when a ``rate_limit`` fault raised it;
- ``TransientError`` after all attempts (5xx, timeout): ``error`` with ``transient_error``;
- ``CacheMissNoLiveModel``: ``skipped``, the error text names the missing request hash;
- ``BudgetExceeded`` / ``QuotaExhausted``: ``error`` with ``budget_exceeded`` /
  ``quota_exhausted``; stored (and committed on a non-autocommit connection), then
  re-raised so a suite can stop.

This module is the single composition point for the run's model chain. The ``provider``
argument is the *live inner* provider only: pass ``llm.factory.build_live_provider`` output
(the registered provider, rate-limited when its spec says so; None for no live model). The
runner wraps it as LLMClient -> [FaultInjectingProvider] -> CacheFirstProvider ->
BudgetGuardProvider -> provider, sharing the process budget tracker. Faults are replay-only;
their specs are stored on the run. Never pass ``llm.factory.build_provider`` output: that
chain already has a cache and a budget guard, so spend would be recorded twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
from opentelemetry.trace import Status, StatusCode

from agent_replay import codes
from agent_replay.adapters.base import AgentAdapter, AgentOutcome
from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.db.cache_repo import CacheRepo
from agent_replay.demo import tools as demo_tools
from agent_replay.llm import (
    BudgetExceeded,
    CacheMissNoLiveModel,
    InjectedRateLimitError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)
from agent_replay.llm.budget import BudgetGuardProvider
from agent_replay.llm.cache_provider import CacheFirstProvider
from agent_replay.llm.client import LLMClient
from agent_replay.llm.factory import get_tracker, make_client
from agent_replay.llm.provider import ModelProvider
from agent_replay.runtime.contexts import (
    BaseContext,
    RecordingContext,
    ReplayContext,
    span_id_of,
)
from agent_replay.runtime.faults import Fault, FaultInjectingProvider
from agent_replay.runtime.stubs import ToolStubs
from agent_replay.taskspec import TaskSpec
from agent_replay.tracing import agent_span, new_tracer_with_provider

Row = dict[str, Any]
Mode = Literal["record", "replay"]

# Checked in order, so the injected subclass comes before RateLimitError.
_ERROR_CODES: dict[type[Exception], str] = {
    BudgetExceeded: codes.BUDGET_EXCEEDED,
    QuotaExhausted: codes.QUOTA_EXHAUSTED,
    InjectedRateLimitError: codes.INJECTED_RATE_LIMIT,
    RateLimitError: codes.RATE_LIMIT,
    # Still failing after the client's retries: the provider, not the agent, broke.
    TransientError: codes.TRANSIENT_ERROR,
}
# Agents that catch an exception report its class name; map the ones the gate reads.
_OUTCOME_CODES: dict[str, str] = {
    cls.__name__: code for cls, code in _ERROR_CODES.items()
}


def _client(
    conn: psycopg.Connection[Row],
    provider: ModelProvider | None,
    settings: Settings,
    mode: Mode,
    faults: Sequence[Fault] = (),
    recorded: Mapping[str, Row] | None = None,
) -> LLMClient:
    tracker = get_tracker(settings)
    live: ModelProvider | None = None
    if provider is not None:
        live = BudgetGuardProvider(provider, settings.run_budget_usd, tracker)
    # CacheRepo.put has explicit keywords; the ResponseCache protocol takes **kwargs.
    repo: Any = CacheRepo(conn)
    cache = CacheFirstProvider(
        repo,
        live,
        read=mode == "replay",
        write=True,
        overwrite=mode == "record",
        tracker=tracker,
        recorded=recorded,
    )
    inner: ModelProvider = cache
    if faults:
        # Above the cache: injected 429s never reach the cache, budget or live limiter.
        inner = FaultInjectingProvider(cache, faults)
    return make_client(inner, settings)


def recorded_responses(spans: Sequence[Row]) -> dict[str, Row]:
    """The responses a run received, by request hash, from its model spans."""
    found: dict[str, Row] = {}
    for span in spans:
        payload = span.get("payload") or {}
        if span.get("kind") != "model" or not isinstance(payload, dict):
            continue
        key, response = payload.get("request_hash"), payload.get("response")
        if isinstance(key, str) and isinstance(response, dict):
            found.setdefault(key, response)
    return found


def _started_at(span: Row) -> datetime | None:
    start_ns = span.get("start_ns")
    return None if start_ns is None else datetime.fromtimestamp(start_ns / 1e9, UTC)


def _span_rows(
    exported: list[Row],
    steps: dict[str, Row],
    root_span_id: str | None,
    task: TaskSpec | None = None,
) -> list[Row]:
    rows: list[Row] = []
    for span in sorted(exported, key=lambda s: s["seq"]):
        row: Row = {
            "seq": span["seq"],
            "trace_id": span["trace_id"],
            "span_id": span["span_id"],
            "parent_span_id": span["parent_span_id"],
            "name": span["name"],
            "status": "ok" if span["status"] == "unset" else span["status"],
            "error_type": span["error_type"],
            "attributes": span["attributes"],
            "started_at": _started_at(span),
            "kind": "agent" if span["span_id"] == root_span_id else "",
        }
        row.update(steps.get(span["span_id"], {}))
        rows.append(row)
    root = next((r for r in rows if r["span_id"] == root_span_id), None)
    if root is not None:
        # Virtual latency (ADR-004): the root is the sum of its steps.
        root["latency_ms"] = sum(r.get("latency_ms", 0) for r in rows if r is not root)
        if task is not None:
            # Snapshot so a replay without --tasks reuses the source task's limits.
            root["payload"] = {"task": task.model_dump(mode="json")}
    return rows


def _totals(rows: list[Row]) -> Row:
    steps = [r for r in rows if r.get("parent_span_id") is not None]
    models = [r for r in steps if r.get("kind") == "model"]
    return {
        "input_tokens": sum(r.get("input_tokens", 0) for r in steps),
        "output_tokens": sum(r.get("output_tokens", 0) for r in steps),
        "latency_ms": sum(r.get("latency_ms", 0) for r in steps),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in steps),
        "model_calls": len(models),
        "tool_calls": sum(1 for r in steps if r.get("kind") == "tool"),
        "cache_hits": sum(1 for r in models if r.get("cache_status") == "hit"),
        "cache_misses": sum(1 for r in models if r.get("cache_status") == "miss"),
    }


def _status(
    outcome: AgentOutcome | None,
    exc: Exception | None,
    diverged: bool,
    tool_error: str | None = None,
) -> tuple[str, str | None, str | None]:
    """(status, error_type, error) for the run row.

    Stop codes (``agent_replay.codes.STOP_CODES``) are stored in both ``error_type`` and
    ``error``: the gate reads them as the run's error.
    """
    if isinstance(exc, CacheMissNoLiveModel):
        return "skipped", "cache_miss_no_live_model", str(exc)
    if exc is not None:
        code = next(
            (c for cls, c in _ERROR_CODES.items() if isinstance(exc, cls)), None
        )
        if code is not None:
            return "error", code, code
        # Only the class name: exception text may carry provider details.
        return "error", type(exc).__name__, type(exc).__name__
    assert outcome is not None
    if outcome.status == "error":
        error = outcome.error or "agent_error"
        error = _OUTCOME_CODES.get(error, error)
        return "error", error, error
    if tool_error is not None:
        return "error", tool_error, tool_error
    return ("diverged" if diverged else "ok"), None, None


def _execute(
    *,
    conn: psycopg.Connection[Row],
    task: TaskSpec,
    agent: AgentAdapter,
    variant: str,
    mode: Mode,
    make_context: Callable[[Any], BaseContext],
    source_run_id: str | None,
    faults: Sequence[Fault] = (),
    suite_run_id: str | None = None,
) -> Any:
    started = datetime.now(UTC)
    provider, tracer, exporter = new_tracer_with_provider()
    ctx = make_context(tracer)
    outcome: AgentOutcome | None = None
    caught: Exception | None = None
    root_span_id: str | None = None
    try:
        with agent_span(tracer, agent.name) as root:
            root_span_id = span_id_of(root)
            outcome = agent.run(task, ctx)
            if outcome.status == "error":
                code = outcome.error or "agent_error"
                code = _OUTCOME_CODES.get(code, code)
                root.set_error_type(code)
                root.span.set_status(Status(StatusCode.ERROR, code))
    except Exception as exc:  # noqa: BLE001 - every agent failure is stored as run data
        caught = exc
    finally:
        provider.shutdown()

    rows = _span_rows(exporter.spans, ctx.steps, root_span_id, task)
    status, error_type, error = _status(outcome, caught, ctx.diverged, ctx.tool_error)
    run: Row = {
        "suite_run_id": suite_run_id,
        "source_run_id": source_run_id,
        "task_id": task.id,
        "variant": variant,
        "mode": mode,
        "status": status,
        "error_type": error_type,
        "error": error,
        "final_output": outcome.final_output if outcome is not None else None,
        "faults": [f.spec for f in faults],
        "started_at": started,
        "finished_at": datetime.now(UTC),
        **_totals(rows),
    }
    run_id = runs_repo.save_run(conn, run, rows)
    if isinstance(caught, BudgetExceeded | QuotaExhausted):
        if not conn.autocommit:
            # Otherwise the caller's pool context rolls the stored run back on re-raise.
            conn.commit()
        raise caught
    return run_id


def record_run(
    task: TaskSpec,
    agent: AgentAdapter,
    variant: str,
    *,
    conn: psycopg.Connection[Row],
    provider: ModelProvider,
    settings: Settings | None = None,
    tools: Mapping[str, Callable[..., dict[str, Any]]] | None = None,
    faults: Sequence[Fault] = (),
    suite_run_id: str | None = None,
) -> Any:
    """Record one run with the live model and real tools; returns the run id.

    ``provider`` is ``factory.build_live_provider(settings, "record")`` output (never
    ``build_provider`` output; see the module docstring). ``faults`` must be empty: fault
    injection is replay-only, so a non-empty list raises ValueError and stores nothing.
    """
    if faults:
        raise ValueError(
            "faults are only supported in replay mode; record runs the real model and tools"
        )
    s = settings or Settings.model_validate({})
    client = _client(conn, provider, s, "record")
    tracker = get_tracker(s)
    registry = demo_tools.TOOLS if tools is None else tools
    return _execute(
        conn=conn,
        task=task,
        agent=agent,
        variant=variant,
        mode="record",
        make_context=lambda tracer: RecordingContext(
            tracer, client, registry, tracker=tracker
        ),
        source_run_id=None,
        suite_run_id=suite_run_id,
    )


def replay_run(
    source_run_id: Any,
    *,
    conn: psycopg.Connection[Row],
    task: TaskSpec,
    agent: AgentAdapter,
    variant: str,
    provider: ModelProvider | None = None,
    settings: Settings | None = None,
    faults: Sequence[Fault] = (),
    suite_run_id: str | None = None,
) -> Any:
    """Replay a source run: tools stubbed from its spans, model responses served from its
    own recorded spans first, then the shared cache, then ``provider``.

    ``faults`` (from ``runtime.faults.parse_fault``) are injected into the tool stubs and
    the model chain; their specs are stored in ``runs.faults``.

    ``provider`` is ``factory.build_live_provider(settings, "replay")`` output, None when
    there is no live model (never ``build_provider`` output; see the module docstring).
    """
    source = str(source_run_id)
    if runs_repo.get_run(conn, source) is None:
        raise LookupError(f"source run {source} not found")
    spans = runs_repo.list_spans(conn, source)
    stubs = ToolStubs.from_spans(spans)
    s = settings or Settings.model_validate({})
    client = _client(conn, provider, s, "replay", faults, recorded_responses(spans))
    tracker = get_tracker(s)
    timeout_ms = round(s.tool_timeout_s * 1000)
    return _execute(
        conn=conn,
        task=task,
        agent=agent,
        variant=variant,
        mode="replay",
        make_context=lambda tracer: ReplayContext(
            tracer,
            client,
            stubs,
            tracker=tracker,
            faults=faults,
            tool_timeout_ms=timeout_ms,
        ),
        source_run_id=source,
        faults=faults,
        suite_run_id=suite_run_id,
    )
