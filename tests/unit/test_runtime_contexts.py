"""Unit tests for run contexts (no database)."""

from __future__ import annotations

from agent_replay.llm import ModelRequest, ModelResponse, ScriptedProvider, ToolCall
from agent_replay.runtime.contexts import RecordingContext, ReplayContext
from agent_replay.runtime.stubs import ToolStubs
from agent_replay.tracing import new_tracer

REQ = ModelRequest(provider="p", model="m", system="s", messages=[])


def _provider() -> ScriptedProvider:
    return ScriptedProvider(
        [ModelResponse(tool_calls=[ToolCall(id="x_1", name="t", args={})])]
    )


def test_call_ids_come_from_last_response_then_fall_back_to_step_index() -> None:
    tracer, _ = new_tracer()
    ctx = RecordingContext(tracer, _provider(), {"t": lambda: {"ok": 1}})
    ctx.call_model(REQ)
    ctx.call_tool("t", {})
    ctx.call_tool("t", {})
    ids = [s["payload"]["call_id"] for s in ctx.steps.values() if s["kind"] == "tool"]
    assert ids == ["x_1", "call_1_1"]


def test_recording_tool_exception_becomes_error_result() -> None:
    def boom() -> dict[str, int]:
        raise ValueError("bad")

    tracer, _ = new_tracer()
    ctx = RecordingContext(tracer, _provider(), {"t": boom})
    result = ctx.call_tool("t", {})
    assert not result.ok and result.error_type == "ValueError"
    assert result.data == {"error": "tool_error"}


def test_replay_unrecorded_call_marks_divergence() -> None:
    tracer, exporter = new_tracer()
    ctx = ReplayContext(tracer, _provider(), ToolStubs.from_spans([]))
    result = ctx.call_tool("t", {"a": 1})
    assert ctx.diverged
    assert result.error_type == "unrecorded_tool_call"
    assert exporter.spans[0]["status"] == "error"


def test_replaying_a_recorded_unrecorded_call_stays_diverged() -> None:
    source = {
        "kind": "tool",
        "name": "execute_tool t",
        "error_type": "unrecorded_tool_call",
        "latency_ms": 0,
        "payload": {
            "name": "t",
            "args": {"a": 1},
            "result": {"error": "unrecorded_tool_call"},
            "raised": False,
            "fault": None,
        },
    }
    tracer, _ = new_tracer()
    ctx = ReplayContext(tracer, _provider(), ToolStubs.from_spans([source]))
    result = ctx.call_tool("t", {"a": 1})
    assert result.error_type == "unrecorded_tool_call"
    assert ctx.diverged
