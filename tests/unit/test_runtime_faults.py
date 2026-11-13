"""Unit tests for fault specs, tool faults and the faulting provider (no database)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_replay.demo.tools import OUTPUT_MODELS
from agent_replay.llm import (
    ModelRequest,
    ModelResponse,
    RateLimitError,
    ScriptedProvider,
    ToolCall,
)
from agent_replay.llm.client import LLMClient
from agent_replay.runtime.contexts import ReplayContext
from agent_replay.runtime.faults import (
    FAULT_KINDS,
    FaultInjectingProvider,
    malformed_payload,
    parse_fault,
    rate_limit_count,
    tool_fault,
)
from agent_replay.runtime.stubs import ToolStubs
from agent_replay.tracing import new_tracer

REQ = ModelRequest(provider="p", model="m", system="s", messages=[])
ORDER = {
    "order_id": "A1001",
    "customer": "c",
    "status": "shipped",
    "items": ["x"],
    "total": 1.0,
    "currency": "USD",
}


def test_parse_fault_round_trips_spec() -> None:
    for spec in ("tool_timeout:get_order", "malformed_tool_output:*", "rate_limit:2"):
        assert parse_fault(spec).spec == spec


@pytest.mark.parametrize(
    "spec",
    [
        "explode:x",
        "tool_timeout",
        "tool_timeout:",
        "rate_limit:0",
        "rate_limit:-1",
        "rate_limit:+2",
        "rate_limit:abc",
        "",
        "tool_timeout: a",
    ],
)
def test_parse_fault_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError) as ei:
        parse_fault(spec)
    for kind in FAULT_KINDS:
        assert kind in str(ei.value)


def test_tool_fault_matches_name_or_star_first_wins() -> None:
    faults = [
        parse_fault("rate_limit:1"),
        parse_fault("tool_timeout:get_order"),
        parse_fault("malformed_tool_output:*"),
    ]
    assert tool_fault(faults, "get_order") == faults[1]
    assert tool_fault(faults, "search_products") == faults[2]
    assert tool_fault(faults[:2], "search_products") is None
    assert rate_limit_count(faults) == 1
    assert (
        rate_limit_count([parse_fault("rate_limit:2"), parse_fault("rate_limit:3")])
        == 5
    )


@pytest.mark.parametrize(
    "result", [ORDER, {"products": []}, None, "text", {"error": "x"}]
)
def test_malformed_payload_fails_every_output_model(result: Any) -> None:
    payload = malformed_payload(result)
    assert isinstance(payload, dict) and set(payload) != {"error"}
    for model in OUTPUT_MODELS.values():
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_faulting_provider_raises_first_n_then_delegates() -> None:
    inner = ScriptedProvider([ModelResponse(text="a"), ModelResponse(text="b")])
    p = FaultInjectingProvider(inner, [parse_fault("rate_limit:2")])
    for _ in range(2):
        with pytest.raises(RateLimitError):
            p.generate(REQ)
    assert inner.requests == []
    assert p.generate(REQ).text == "a"
    assert p.generate(REQ).text == "b"


def test_faulting_provider_under_client_retries_records_attempts() -> None:
    inner = ScriptedProvider([ModelResponse(text="a")])
    fp = FaultInjectingProvider(inner, [parse_fault("rate_limit:2")])
    client = LLMClient(fp, max_attempts=3, backoff_base_s=0.0, sleep=lambda _: None)
    assert client.generate(REQ).attempts == 3


def _replay_ctx(faults: list[str]) -> tuple[ReplayContext, Any]:
    spans = [
        {
            "name": "execute_tool get_order",
            "kind": "tool",
            "latency_ms": 12,
            "payload": {
                "name": "get_order",
                "args": {"order_id": "A1001"},
                "result": ORDER,
            },
        }
    ]
    tracer, exporter = new_tracer()
    provider = ScriptedProvider(
        [ModelResponse(tool_calls=[ToolCall(id="c1", name="get_order", args={})])]
    )
    ctx = ReplayContext(
        tracer,
        provider,
        ToolStubs.from_spans(spans),
        faults=[parse_fault(f) for f in faults],
        tool_timeout_ms=7000,
    )
    return ctx, exporter


def _tool_row(ctx: ReplayContext) -> Any:
    (row,) = [r for r in ctx.steps.values() if r["kind"] == "tool"]
    return row


def test_replay_tool_timeout_fault() -> None:
    ctx, exporter = _replay_ctx(["tool_timeout:get_order"])
    result = ctx.call_tool("get_order", {"order_id": "A1001"})
    assert not result.ok and result.data == {"error": "tool_timeout"}
    row = _tool_row(ctx)
    assert row["latency_ms"] == 7000
    assert row["payload"]["fault"] == "tool_timeout:get_order"
    assert row["payload"]["result"] == {"error": "tool_timeout"}
    assert exporter.spans[0]["error_type"] == "timeout"
    assert not ctx.diverged and ctx.tool_error is None


def test_replay_malformed_fault_returns_ok_but_invalid_payload() -> None:
    ctx, exporter = _replay_ctx(["malformed_tool_output:*"])
    result = ctx.call_tool("get_order", {"order_id": "A1001"})
    assert result.ok and result.error_type is None
    with pytest.raises(ValidationError):
        OUTPUT_MODELS["get_order"].model_validate(result.data)
    row = _tool_row(ctx)
    assert row["latency_ms"] == 12
    assert row["payload"]["fault"] == "malformed_tool_output:*"
    assert exporter.spans[0]["error_type"] == "malformed_tool_output"
    assert ctx.tool_error is None


def test_replay_without_matching_fault_is_unchanged() -> None:
    ctx, exporter = _replay_ctx(["tool_timeout:search_products"])
    result = ctx.call_tool("get_order", {"order_id": "A1001"})
    assert result.ok and result.data == ORDER
    row = _tool_row(ctx)
    assert row["payload"]["fault"] is None and row["latency_ms"] == 12
    assert exporter.spans[0]["error_type"] is None
