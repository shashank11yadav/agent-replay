"""Unit tests for recorded tool stubs (no database)."""

from __future__ import annotations

from typing import Any

from agent_replay.runtime.stubs import UNRECORDED, ToolStubs, canonical_args


def _tool(name: str, args: dict[str, Any], result: Any, latency: int = 0) -> Any:
    return {
        "name": f"execute_tool {name}",
        "kind": "tool",
        "latency_ms": latency,
        "payload": {"name": name, "args": args, "result": result, "fault": None},
    }


def test_canonical_args_ignore_key_order() -> None:
    assert canonical_args({"a": 1, "b": 2}) == canonical_args({"b": 2, "a": 1})


def test_lookup_returns_result_and_latency_in_recorded_order() -> None:
    stubs = ToolStubs.from_spans(
        [
            {"name": "invoke_agent demo-agent", "kind": "agent", "payload": {}},
            _tool("t", {"x": 1}, {"n": 0}, latency=5),
            {"name": "generate_content m", "kind": "model", "payload": {}},
            _tool("t", {"x": 1}, {"n": 1}, latency=7),
        ]
    )
    assert stubs.lookup("t", {"x": 1}) == ({"n": 0}, 5)
    assert stubs.lookup("t", {"x": 1}) == ({"n": 1}, 7)
    assert stubs.lookup("t", {"x": 1}) is None
    assert stubs.call("t", {"x": 1}) == UNRECORDED


def test_call_returns_a_copy_of_the_recorded_result() -> None:
    stubs = ToolStubs.from_spans(
        [_tool("t", {}, {"n": [1]}), _tool("t", {}, {"n": [1]})]
    )
    first = stubs.call("t", {})
    first["n"].append(2)
    assert stubs.call("t", {}) == {"n": [1]}


def test_next_call_keeps_recorded_error_type_and_raised_flag() -> None:
    span = _tool("t", {}, {"error": "tool_error"})
    span["error_type"] = "ValueError"
    span["payload"]["raised"] = True
    found = ToolStubs.from_spans([span]).next_call("t", {})
    assert found is not None
    assert (found.result, found.error_type, found.raised) == (
        {"error": "tool_error"},
        "ValueError",
        True,
    )
