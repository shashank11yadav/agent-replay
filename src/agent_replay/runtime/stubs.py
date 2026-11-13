"""Tool stubs answered from a source run's recorded tool spans."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from agent_replay import semconv

UNRECORDED_TOOL_CALL = "unrecorded_tool_call"
UNRECORDED: dict[str, Any] = {"error": UNRECORDED_TOOL_CALL}


def canonical_args(args: dict[str, Any]) -> str:
    """Canonical JSON of tool arguments (sorted keys, no whitespace)."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_tool_span(span: dict[str, Any]) -> bool:
    if span.get("kind") == "tool":
        return True
    return str(span.get("name", "")).startswith(f"{semconv.OP_EXECUTE_TOOL} ")


@dataclass(frozen=True)
class RecordedCall:
    """One recorded tool call: its stored result, latency and outcome.

    ``fault`` is the fault spec that produced the recorded result (a faulted replay), if any.
    """

    result: Any
    latency_ms: int
    error_type: str | None = None
    raised: bool = False
    fault: str | None = None


class ToolStubs:
    """Matches calls on tool name plus canonical args; repeats are served in recorded order."""

    def __init__(self) -> None:
        self._recorded: dict[tuple[str, str], deque[RecordedCall]] = defaultdict(deque)

    @classmethod
    def from_spans(cls, spans: list[dict[str, Any]]) -> ToolStubs:
        """Build stubs from ``runs_repo.list_spans`` rows (seq order)."""
        stubs = cls()
        for span in spans:
            payload = span.get("payload") or {}
            if not _is_tool_span(span) or "name" not in payload:
                continue
            key = (payload["name"], canonical_args(payload.get("args") or {}))
            stubs._recorded[key].append(
                RecordedCall(
                    result=payload.get("result"),
                    latency_ms=int(span.get("latency_ms") or 0),
                    error_type=span.get("error_type"),
                    raised=bool(payload.get("raised")),
                    fault=payload.get("fault"),
                )
            )
        return stubs

    def next_call(self, name: str, args: dict[str, Any]) -> RecordedCall | None:
        """Next recorded call (result deep-copied) for this name and args, or None."""
        queue = self._recorded.get((name, canonical_args(args)))
        if not queue:
            return None
        recorded = queue.popleft()
        return RecordedCall(
            result=copy.deepcopy(recorded.result),
            latency_ms=recorded.latency_ms,
            error_type=recorded.error_type,
            raised=recorded.raised,
            fault=recorded.fault,
        )

    def lookup(self, name: str, args: dict[str, Any]) -> tuple[Any, int] | None:
        """Next recorded (result, latency_ms) for this call, or None if unrecorded."""
        found = self.next_call(name, args)
        return None if found is None else (found.result, found.latency_ms)

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Recorded result for this call, or ``{"error": "unrecorded_tool_call"}``."""
        found = self.lookup(name, args)
        return dict(UNRECORDED) if found is None else found[0]
