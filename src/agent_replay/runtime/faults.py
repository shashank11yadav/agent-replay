"""Fault injection for replay: fault specs, tool-stub faults and a faulting provider.

Specs are ``<kind>:<target>``:

- ``tool_timeout:<tool|*>``: the stubbed call returns ``{"error": "tool_timeout"}`` with
  ``error.type`` ``timeout`` and latency equal to the configured tool timeout;
- ``malformed_tool_output:<tool|*>``: the stub returns a payload that fails the tool's output
  model (the agent sees an ``ok`` result and must validate it itself);
- ``rate_limit:<n>``: the first ``n`` model attempts of the run raise
  ``InjectedRateLimitError`` (a ``RateLimitError``). When ``n`` outlasts the client's retries
  the run ends as ``injected_rate_limit``, which is not a stop code, so a fault suite still
  replays every task.

``FaultInjectingProvider`` sits between the client and the cache, so injected 429s are retried
by the client but never reach the cache, the run budget guard or the live rate limiter.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_replay.codes import RATE_LIMIT
from agent_replay.llm import InjectedRateLimitError, ModelRequest, ModelResponse
from agent_replay.llm.provider import ModelProvider

TOOL_TIMEOUT = "tool_timeout"
MALFORMED_TOOL_OUTPUT = "malformed_tool_output"
FAULT_KINDS = (TOOL_TIMEOUT, MALFORMED_TOOL_OUTPUT, RATE_LIMIT)
TIMEOUT_ERROR_TYPE = "timeout"
ALL_TOOLS = "*"

_USAGE = (
    "valid fault specs: tool_timeout:<tool|*>, malformed_tool_output:<tool|*>, "
    "rate_limit:<n> (n >= 1)"
)


@dataclass(frozen=True)
class Fault:
    """One parsed fault spec."""

    kind: str
    target: str

    @property
    def spec(self) -> str:
        return f"{self.kind}:{self.target}"

    @property
    def count(self) -> int:
        """Number of injected 429s (0 for tool faults)."""
        return int(self.target) if self.kind == RATE_LIMIT else 0

    def matches_tool(self, name: str) -> bool:
        return self.kind != RATE_LIMIT and self.target in (ALL_TOOLS, name)


def parse_fault(spec: str) -> Fault:
    """Parse ``<kind>:<target>``; raises ValueError naming the valid kinds."""
    kind, sep, target = spec.partition(":")
    if not sep or kind not in FAULT_KINDS:
        raise ValueError(f"unknown fault spec {spec!r}; {_USAGE}")
    if kind == RATE_LIMIT:
        if not (target.isascii() and target.isdigit()) or int(target) < 1:
            raise ValueError(f"invalid rate_limit count in {spec!r}; {_USAGE}")
    elif not target or target != target.strip() or any(c.isspace() for c in target):
        raise ValueError(f"invalid tool name in {spec!r}; {_USAGE}")
    return Fault(kind, target)


def tool_fault(faults: Sequence[Fault], name: str) -> Fault | None:
    """The first tool fault that applies to this tool, or None."""
    return next((f for f in faults if f.matches_tool(name)), None)


def rate_limit_count(faults: Sequence[Fault]) -> int:
    """Total injected 429s across all rate_limit faults."""
    return sum(f.count for f in faults)


def malformed_payload(result: Any) -> dict[str, Any]:
    """A deterministic payload with none of the tool's fields: truncated recorded JSON."""
    text = json.dumps(result, sort_keys=True, default=str)
    return {MALFORMED_TOOL_OUTPUT: text[: len(text) // 2]}


class FaultInjectingProvider:
    """Raises ``InjectedRateLimitError`` for the first ``n`` calls, then delegates."""

    def __init__(self, inner: ModelProvider, faults: Sequence[Fault]) -> None:
        self._inner = inner
        self._remaining = rate_limit_count(faults)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._remaining > 0:
            self._remaining -= 1
            raise InjectedRateLimitError("injected rate_limit fault")
        return self._inner.generate(request)
