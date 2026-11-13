"""Tracing: pinned semconv constants, tracer setup and the collecting exporter."""

from __future__ import annotations

import json
import re

import pytest
from support import ROOT

from agent_replay import semconv, tracing

SRC = ROOT / "src" / "agent_replay"

MODEL = "gemini-3.5-flash-lite"


def _tracing():
    import agent_replay.tracing as t

    return t


def _semconv():
    import agent_replay.semconv as s

    return s


@pytest.fixture
def tr():
    t = _tracing()
    tracer, exporter = t.new_tracer()
    return t, tracer, exporter


def _by_name(exporter, name):
    found = [s for s in exporter.spans if s["name"] == name]
    assert len(found) == 1, [s["name"] for s in exporter.spans]
    return found[0]


def test_semconv_constants_values():
    s = _semconv()
    names = {v for v in vars(s).values() if isinstance(v, str)}
    for literal in [
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "error.type",
        "invoke_agent",
        "generate_content",
        "execute_tool",
    ]:
        assert literal in names, literal


def test_only_semconv_contains_gen_ai_literal():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if p.name != "semconv.py" and "gen_ai." in p.read_text()
    ]
    assert offenders == []
    assert "gen_ai." in (SRC / "semconv.py").read_text()


def test_new_tracer_local_provider_global_untouched():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    t = _tracing()
    before = trace.get_tracer_provider()
    tracer1, exp1 = t.new_tracer()
    _, exp2 = t.new_tracer()
    assert trace.get_tracer_provider() is before
    assert not isinstance(trace.get_tracer_provider(), TracerProvider)
    assert exp1 is not exp2
    with tracer1.start_as_current_span("only-one"):
        pass
    assert [s["name"] for s in exp1.spans] == ["only-one"]
    assert exp2.spans == []
    # synchronous processor: span is collected as soon as it ends
    from opentelemetry.sdk.trace.export import SpanExporter

    assert isinstance(exp1, SpanExporter)


def test_model_span_name_and_attributes(tr):
    t, tracer, exp = tr
    s = _semconv()
    with t.model_span(
        tracer, MODEL, provider="gemini", input_tokens=11, output_tokens=7
    ):
        pass
    d = _by_name(exp, f"generate_content {MODEL}")
    a = d["attributes"]
    assert (
        a[
            s.GEN_AI_OPERATION_NAME
            if hasattr(s, "GEN_AI_OPERATION_NAME")
            else "gen_ai.operation.name"
        ]
        == "generate_content"
    )
    assert a["gen_ai.provider.name"] == "gemini"
    assert a["gen_ai.request.model"] == MODEL
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert d["status"] != "error"


def test_tool_span_name_and_attributes(tr):
    t, tracer, exp = tr
    args = {"order_id": "A-1", "n": 2}
    result = {"status": "shipped", "items": [1, 2]}
    with t.tool_span(
        tracer, "get_order", call_id="call-9", arguments=args, result=result
    ):
        pass
    d = _by_name(exp, "execute_tool get_order")
    a = d["attributes"]
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert a["gen_ai.tool.name"] == "get_order"
    assert a["gen_ai.tool.call.id"] == "call-9"
    assert isinstance(a["gen_ai.tool.call.arguments"], str)
    assert isinstance(a["gen_ai.tool.call.result"], str)
    assert json.loads(a["gen_ai.tool.call.arguments"]) == args
    assert json.loads(a["gen_ai.tool.call.result"]) == result


def test_children_have_root_as_parent(tr):
    t, tracer, exp = tr
    with t.agent_span(tracer, "demo-agent"):
        with t.model_span(
            tracer, MODEL, provider="gemini", input_tokens=1, output_tokens=2
        ):
            pass
        with t.tool_span(
            tracer, "get_order", call_id="c1", arguments={}, result={"ok": 1}
        ):
            pass
    root = _by_name(exp, "invoke_agent demo-agent")
    model = _by_name(exp, f"generate_content {MODEL}")
    tool = _by_name(exp, "execute_tool get_order")
    assert root["parent_span_id"] is None
    assert root["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert model["parent_span_id"] == root["span_id"]
    assert tool["parent_span_id"] == root["span_id"]
    assert model["trace_id"] == tool["trace_id"] == root["trace_id"]


def test_exception_marks_error(tr):
    t, tracer, exp = tr
    with (
        pytest.raises(KeyError),
        t.agent_span(tracer, "demo-agent"),
        t.tool_span(tracer, "get_order", call_id="c1", arguments={}, result=None),
    ):
        raise KeyError("boom")
    tool = _by_name(exp, "execute_tool get_order")
    root = _by_name(exp, "invoke_agent demo-agent")
    for d in (tool, root):
        assert d["status"] == "error"
        assert d["error_type"] == "KeyError"
        assert d["attributes"]["error.type"] == "KeyError"


def test_ok_span_status_and_no_error_type(tr):
    _, tracer, exp = tr
    with tracer.start_as_current_span("plain"):
        pass
    d = exp.spans[0]
    assert d["status"] != "error"
    assert d["error_type"] is None


def test_collected_dict_shape_and_json_serialisable(tr):
    t, tracer, exp = tr
    with (
        t.agent_span(tracer, "demo-agent"),
        t.model_span(tracer, MODEL, provider="gemini", input_tokens=1, output_tokens=2),
    ):
        pass
    json.dumps(exp.spans)
    keys = {
        "trace_id",
        "span_id",
        "parent_span_id",
        "name",
        "start_ns",
        "end_ns",
        "status",
        "error_type",
        "attributes",
    }
    for d in exp.spans:
        assert keys <= set(d)
        assert re.fullmatch(r"[0-9a-f]{32}", d["trace_id"])
        assert re.fullmatch(r"[0-9a-f]{16}", d["span_id"])
        assert isinstance(d["start_ns"], int) and isinstance(d["end_ns"], int)
        assert d["end_ns"] >= d["start_ns"] > 0
        assert isinstance(d["attributes"], dict)
    root = _by_name(exp, "invoke_agent demo-agent")
    child = _by_name(exp, f"generate_content {MODEL}")
    assert root["parent_span_id"] is None
    assert re.fullmatch(r"[0-9a-f]{16}", child["parent_span_id"])


def test_export_returns_success_flush_and_shutdown(tr):
    from opentelemetry.sdk.trace.export import SpanExportResult

    _, tracer, exp = tr
    with tracer.start_as_current_span("x"):
        pass
    assert exp.export([]) is SpanExportResult.SUCCESS
    assert exp.force_flush() is True
    assert exp.shutdown() is None


def test_sequence_attributes_become_lists() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracer.start_as_current_span("s") as span:
        span.set_attribute("test.tags", ["stop"])
    attrs = exporter.spans[0]["attributes"]
    assert attrs["test.tags"] == ["stop"]
    json.dumps(exporter.spans)


def test_ok_span_status_is_unset_without_error_type() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracing.agent_span(tracer, "a"):
        pass
    span = exporter.spans[0]
    assert span["status"] == "unset"
    assert span["error_type"] is None
    assert semconv.ERROR_TYPE not in span["attributes"]


def test_error_type_uses_exception_class_name() -> None:
    tracer, exporter = tracing.new_tracer()
    with (
        pytest.raises(ValueError),
        tracing.model_span(tracer, "m", provider="p", input_tokens=0, output_tokens=0),
    ):
        raise ValueError("secret detail")
    span = exporter.spans[0]
    assert span["status"] == "error"
    assert span["error_type"] == "ValueError"


def test_clear_empties_collected_spans() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracer.start_as_current_span("s"):
        pass
    exporter.clear()
    assert exporter.spans == []


def test_new_tracer_registers_no_atexit_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import atexit

    calls: list[object] = []
    monkeypatch.setattr(atexit, "register", lambda *a, **k: calls.append(a))
    tracing.new_tracer()
    tracing.new_tracer()
    assert calls == []


def test_provider_can_be_shut_down_explicitly() -> None:
    provider, tracer, exporter = tracing.new_tracer_with_provider()
    with tracer.start_as_current_span("before"):
        pass
    provider.shutdown()
    with tracer.start_as_current_span("after"):
        pass
    assert [s["name"] for s in exporter.spans] == ["before"]


def test_existing_error_type_is_kept() -> None:
    tracer, exporter = tracing.new_tracer()
    with (
        pytest.raises(RuntimeError),
        tracing.model_span(tracer, "m", provider="p") as step,
    ):
        step.set_error_type("rate_limit")
        raise RuntimeError("429")
    span = exporter.spans[0]
    assert span["status"] == "error"
    assert span["error_type"] == "rate_limit"


def test_base_exception_marks_status_error() -> None:
    tracer, exporter = tracing.new_tracer()
    with pytest.raises(KeyboardInterrupt), tracing.agent_span(tracer, "a"):
        raise KeyboardInterrupt
    span = exporter.spans[0]
    assert span["status"] == "error"
    assert span["error_type"] == "KeyboardInterrupt"


def test_model_usage_set_after_call() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracing.model_span(tracer, "m", provider="p") as step:
        step.set_usage(input_tokens=3, output_tokens=4)
        step.set_attribute("test.tags", ["stop"])
    attrs = exporter.spans[0]["attributes"]
    assert attrs[semconv.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs[semconv.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs["test.tags"] == ["stop"]


def test_model_span_without_usage_omits_token_attributes() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracing.model_span(tracer, "m", provider="p"):
        pass
    attrs = exporter.spans[0]["attributes"]
    assert semconv.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert semconv.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs


def test_tool_result_set_after_call() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracing.tool_span(tracer, "t", call_id="c", arguments={}) as step:
        step.set_result({"ok": True})
    attrs = exporter.spans[0]["attributes"]
    assert json.loads(attrs[semconv.GEN_AI_TOOL_CALL_RESULT]) == {"ok": True}


def test_seq_is_start_order() -> None:
    tracer, exporter = tracing.new_tracer()
    with tracing.agent_span(tracer, "a"):
        with tracing.model_span(tracer, "m", provider="p"):
            pass
        with tracing.tool_span(tracer, "t", call_id="c", arguments={}):
            pass
    by_seq = sorted(exporter.spans, key=lambda s: s["seq"])
    assert [s["seq"] for s in by_seq] == [0, 1, 2]
    assert [s["name"] for s in by_seq] == [
        "invoke_agent a",
        "generate_content m",
        "execute_tool t",
    ]
    assert exporter.spans[-1]["name"] == "invoke_agent a"
