"""Pure deterministic checks over a run row and its span rows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_replay.semconv import GEN_AI_TOOL_NAME
from agent_replay.taskspec import TaskSpec

Row = dict[str, Any]


@dataclass(frozen=True)
class CheckResult:
    passed: bool | None
    detail: dict[str, Any] = field(default_factory=dict)


def expected_text(task: TaskSpec, run: Row, spans: list[Row]) -> CheckResult:
    output = run.get("final_output")
    if output is None:
        return CheckResult(
            False, {"missing": list(task.expected_text), "reason": "no output"}
        )
    haystack = str(output).lower()
    missing = [s for s in task.expected_text if s.lower() not in haystack]
    return CheckResult(not missing, {"missing": missing})


def must_call_tools(task: TaskSpec, run: Row, spans: list[Row]) -> CheckResult:
    called = {
        (s.get("attributes") or {}).get(GEN_AI_TOOL_NAME)
        for s in spans
        if s.get("kind") == "tool"
    }
    missing = [t for t in task.must_call_tools if t not in called]
    return CheckResult(not missing, {"missing": missing})


def max_steps(task: TaskSpec, run: Row, spans: list[Row]) -> CheckResult:
    calls = int(run.get("model_calls") or 0)
    return CheckResult(
        calls <= task.max_steps, {"model_calls": calls, "max_steps": task.max_steps}
    )


def completed(task: TaskSpec, run: Row, spans: list[Row]) -> CheckResult:
    status = run.get("status")
    return CheckResult(status == "ok", {"status": status})


CHECKS: dict[str, Callable[[TaskSpec, Row, list[Row]], CheckResult]] = {
    "expected_text": expected_text,
    "must_call_tools": must_call_tools,
    "max_steps": max_steps,
    "completed": completed,
}
