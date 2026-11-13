"""Provider conformance kit: a reusable test base every model provider subclasses.

Usage (in ``tests/providers/test_<name>_conformance.py``; pytest puts this directory on
``sys.path``, so ``from conformance import ...`` works)::

    class TestMyProviderConformance(ProviderConformance):
        declared_errors = frozenset({RateLimitError, TransientError})
        undeclared_reason = "QuotaExhausted comes from the rate limiter"

        @pytest.fixture
        def backend(self, tmp_path) -> MyFakeBackend:
            return MyFakeBackend(tmp_path)

Fake-backend contract. The subclass supplies a ``backend`` fixture returning an object that
stands in for whatever the provider talks to (an SDK client, a subprocess, a script file).
It must offer these methods; nothing else of it is used by the kit:

- ``make_provider() -> ModelProvider``: a provider wired to this fake backend only (no
  network, no real CLI, no credentials). May be called once per test.
- ``script_text(text, *, input_tokens, output_tokens)``: the next backend call answers with
  this plain text reply (no tool calls) and reports these token counts.
- ``script_tool_calls(calls, *, step, input_tokens, output_tokens)``: the next backend call
  answers with these ``(name, args)`` tool calls and no text. ``step`` is the 1-based model
  turn being generated (1 + model messages in the request); a backend whose reply format
  carries the call ids itself (scripted) must emit ``make_tool_call_id(step, index)``.
- ``script_error(error_cls)``: the next backend call fails in the way that the provider must
  map to ``error_cls`` (one of RateLimitError, TransientError, PermanentModelError,
  QuotaExhausted). Only called for classes listed in ``declared_errors``.
- ``seen_request() -> SeenRequest``: what the backend received on the most recent call,
  reduced to the history's tool calls and tool results (see ``SeenRequest``).

Subclass class attributes:

- ``declared_errors``: the subset of the four error classes the backend can produce. The
  other ones are skipped with ``undeclared_reason``, which must then be a non-empty string,
  so a provider cannot silently leave a declared mapping untested.
- ``provider_name`` / ``model``: used in the ModelRequests the kit builds.

The base class is deliberately not named ``Test*``, so pytest only collects subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from agent_replay.llm import (
    Message,
    ModelProvider,
    ModelRequest,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ToolCall,
    ToolResult,
    ToolSchema,
    TransientError,
    make_tool_call_id,
)

ERROR_CLASSES: tuple[type[Exception], ...] = (
    RateLimitError,
    TransientError,
    PermanentModelError,
    QuotaExhausted,
)

SEARCH_TOOL = ToolSchema(
    name="search",
    description="Search the catalogue.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


@dataclass
class SeenRequest:
    """What a fake backend received, reduced to what the kit compares."""

    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tool_results: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ProviderConformance:
    """Base class; see the module docstring for the fake-backend contract."""

    declared_errors: ClassVar[frozenset[type[Exception]]] = frozenset()
    undeclared_reason: ClassVar[str] = ""
    provider_name: ClassVar[str] = "gemini"
    model: ClassVar[str] = "gemini-3.5-flash-lite"

    @pytest.fixture
    def backend(self) -> Any:  # pragma: no cover - subclasses override
        raise NotImplementedError("subclass must provide a `backend` fixture")

    def _request(self, messages: list[Message] | None = None) -> ModelRequest:
        return ModelRequest(
            provider=self.provider_name,
            model=self.model,
            system="You are a helper.",
            messages=messages or [Message(role="user", text="hi")],
            tools=[SEARCH_TOOL],
            temperature=0.0,
            seed=7,
        )

    def _provider(self, backend: Any) -> ModelProvider:
        return backend.make_provider()  # type: ignore[no-any-return]

    # ---- kit self-checks -------------------------------------------------------------

    def test_declared_errors_are_valid_and_undeclared_have_a_reason(self) -> None:
        assert set(self.declared_errors) <= set(ERROR_CLASSES)
        if set(self.declared_errors) != set(ERROR_CLASSES):
            assert self.undeclared_reason.strip(), (
                "undeclared error classes need a non-empty undeclared_reason"
            )

    # ---- replies ---------------------------------------------------------------------

    def test_plain_text_reply(self, backend: Any) -> None:
        backend.script_text("hello there", input_tokens=3, output_tokens=2)
        out = self._provider(backend).generate(self._request())
        assert out.text == "hello there"
        assert out.tool_calls == []

    @pytest.mark.parametrize("prior_model_turns", [0, 2])
    def test_two_tool_calls_with_ids_names_and_args(
        self, backend: Any, prior_model_turns: int
    ) -> None:
        messages = [Message(role="user", text="find things")]
        for i in range(prior_model_turns):
            call = ToolCall(id=make_tool_call_id(i + 1, 0), name="search", args={})
            messages.append(Message(role="model", tool_calls=[call]))
            messages.append(
                Message(
                    role="tool",
                    tool_results=[
                        ToolResult(call_id=call.id, name="search", response={"n": i})
                    ],
                )
            )
        step = 1 + prior_model_turns
        wanted = [("search", {"q": "alpha"}), ("search", {"q": "beta", "n": 2})]
        backend.script_tool_calls(wanted, step=step, input_tokens=5, output_tokens=4)
        out = self._provider(backend).generate(self._request(messages))
        assert [(c.name, c.args) for c in out.tool_calls] == wanted
        assert [c.id for c in out.tool_calls] == [
            make_tool_call_id(step, 0),
            make_tool_call_id(step, 1),
        ]

    def test_tool_result_round_trip_reaches_the_backend(self, backend: Any) -> None:
        call = ToolCall(id="call_1_0", name="search", args={"q": "x"})
        messages = [
            Message(role="user", text="find x"),
            Message(role="model", text="looking", tool_calls=[call]),
            Message(
                role="tool",
                tool_results=[
                    ToolResult(
                        call_id="call_1_0", name="search", response={"hits": [1, 2]}
                    )
                ],
            ),
        ]
        backend.script_text("found it", input_tokens=1, output_tokens=1)
        out = self._provider(backend).generate(self._request(messages))
        assert out.text == "found it"
        seen = backend.seen_request()
        assert ("search", {"q": "x"}) in seen.tool_calls
        assert ("search", {"hits": [1, 2]}) in seen.tool_results

    def test_usage_tokens_map_to_input_and_output(self, backend: Any) -> None:
        backend.script_text("ok", input_tokens=123, output_tokens=45)
        out = self._provider(backend).generate(self._request())
        assert (out.input_tokens, out.output_tokens) == (123, 45)

    # ---- errors ----------------------------------------------------------------------

    @pytest.mark.parametrize("error_cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_error_mapping(self, backend: Any, error_cls: type[Exception]) -> None:
        if error_cls not in self.declared_errors:
            pytest.skip(
                f"{error_cls.__name__} not declared: {self.undeclared_reason or '?'}"
            )
        backend.script_error(error_cls)
        with pytest.raises(error_cls) as info:
            self._provider(backend).generate(self._request())
        assert type(info.value) is error_cls
