"""Recording and replay contexts, recorded tool stubs and the run lifecycle.

Run latency_ms is the sum of latency_ms over the non-root spans (ADR-004).
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from typing import Any

import pytest
from support import ROOT

from agent_replay import semconv
from agent_replay.adapters.base import AgentOutcome
from agent_replay.db import runs_repo
from agent_replay.llm import (
    BudgetExceeded,
    ModelRequest,
    ModelResponse,
    QuotaExhausted,
    ScriptedProvider,
    ToolCall,
    request_hash,
)
from agent_replay.taskspec import load_tasks

EXAMPLE_TASKS = ROOT / "src" / "agent_replay" / "demo" / "example" / "task.jsonl"


def _task() -> Any:
    return load_tasks(EXAMPLE_TASKS)[0]


def _agent(variant: str = "minimal") -> Any:
    from agent_replay.demo.agent import DemoAgent

    return DemoAgent(variant)


def _runner() -> Any:
    return importlib.import_module("agent_replay.runtime.runner")


def _resp1(sig: str | None = None) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="call_1_0",
                name="get_order",
                args={"order_id": "A1001"},
                signature=sig,
            )
        ],
        input_tokens=100,
        output_tokens=20,
        latency_ms=300,
    )


def _resp2() -> ModelResponse:
    return ModelResponse(
        text="Order A1001 has shipped.",
        input_tokens=150,
        output_tokens=30,
        latency_ms=450,
    )


def _provider(*responses: ModelResponse) -> ScriptedProvider:
    return ScriptedProvider(list(responses) or [_resp1(), _resp2()])


def _record(conn: Any, provider: Any = None, variant: str = "minimal") -> str:
    rid = _runner().record_run(
        _task(), _agent(variant), variant, conn=conn, provider=provider or _provider()
    )
    return str(rid)


def _replay(
    conn: Any,
    source: str,
    variant: str = "minimal",
    provider: Any = None,
    agent: Any = None,
) -> str:
    rid = _runner().replay_run(
        source,
        conn=conn,
        task=_task(),
        agent=agent or _agent(variant),
        variant=variant,
        provider=provider,
    )
    return str(rid)


def _spans(conn: Any, rid: str) -> list[dict[str, Any]]:
    return runs_repo.list_spans(conn, rid)


def _model_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if s["name"].startswith("generate_content")]


def _tool_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if s["name"].startswith("execute_tool")]


def _run(conn: Any, rid: str) -> dict[str, Any]:
    row = runs_repo.get_run(conn, rid)
    assert row is not None, f"no run row {rid}"
    return row


# --- record structure -------------------------------------------------------


def test_record_run_stores_one_record_run_with_root_model_and_tool_spans(
    conn: Any,
) -> None:
    rid = _record(conn)
    runs = runs_repo.list_runs(conn)
    assert len(runs) == 1
    run = runs[0]
    assert str(run["id"]) == rid
    assert run["mode"] == "record"
    assert run["status"] == "ok"
    assert run["task_id"] == "example-1"
    assert run["variant"] == "minimal"
    assert run["source_run_id"] is None
    assert run["final_output"] == "Order A1001 has shipped."

    spans = _spans(conn, rid)
    assert [s["seq"] for s in spans] == list(range(len(spans)))
    assert len(spans) == 4
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert len(roots) == 1
    assert roots[0]["seq"] == 0
    assert roots[0]["name"] == "invoke_agent demo-agent"
    models = _model_spans(spans)
    tools = _tool_spans(spans)
    assert len(models) == 2 and len(tools) == 1
    assert models[0]["name"].startswith("generate_content ")
    assert tools[0]["name"] == "execute_tool get_order"
    for s in models + tools:
        assert s["parent_span_id"] == roots[0]["span_id"]
    # step order: model, tool, model
    assert [s["name"].split()[0] for s in spans[1:]] == [
        "generate_content",
        "execute_tool",
        "generate_content",
    ]
    assert all(s["status"] in ("ok", "error") for s in spans), "unset must map to ok"


def test_record_model_and_tool_spans_carry_semconv_attributes(conn: Any) -> None:
    spans = _spans(conn, _record(conn))
    m = _model_spans(spans)[0]
    assert m["attributes"][semconv.GEN_AI_OPERATION_NAME] == "generate_content"
    assert m["attributes"][semconv.GEN_AI_REQUEST_MODEL]
    assert m["attributes"][semconv.GEN_AI_PROVIDER_NAME]
    assert m["attributes"][semconv.GEN_AI_USAGE_INPUT_TOKENS] == 100
    assert m["attributes"][semconv.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    t = _tool_spans(spans)[0]
    assert t["attributes"][semconv.GEN_AI_TOOL_NAME] == "get_order"
    assert t["attributes"][semconv.GEN_AI_TOOL_CALL_ID] == "call_1_0"
    assert json.loads(t["attributes"][semconv.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "order_id": "A1001"
    }
    result = json.loads(t["attributes"][semconv.GEN_AI_TOOL_CALL_RESULT])
    assert result["order_id"] == "A1001"


# --- model cache ------------------------------------------------------------


def test_record_writes_every_model_response_to_model_cache(conn: Any, raw: Any) -> None:
    rid = _record(conn)
    models = _model_spans(_spans(conn, rid))
    hashes = [m["payload"]["request_hash"] for m in models]
    assert len(set(hashes)) == 2
    rows = raw.execute(
        "SELECT request_hash, response, latency_ms FROM model_cache"
    ).fetchall()
    cached = {r[0]: r for r in rows}
    assert set(cached) == set(hashes)
    assert cached[hashes[0]][2] == 300
    assert cached[hashes[1]][2] == 450
    assert cached[hashes[1]][1]["text"] == "Order A1001 has shipped."
    assert cached[hashes[0]][1]["tool_calls"][0]["name"] == "get_order"


def test_record_rerecord_overwrites_cache_and_does_not_read_it(
    conn: Any, raw: Any
) -> None:
    first = _record(conn)
    live = _provider(
        _resp1().model_copy(update={"latency_ms": 999}),
        _resp2().model_copy(update={"latency_ms": 888}),
    )
    second = _record(conn, live)
    assert len(live.requests) == 2, (
        "record mode must call the live model, not the cache"
    )
    h = _model_spans(_spans(conn, second))[0]["payload"]["request_hash"]
    assert h == _model_spans(_spans(conn, first))[0]["payload"]["request_hash"]
    assert raw.execute(
        "SELECT latency_ms FROM model_cache WHERE request_hash = %s", (h,)
    ).fetchone() == (999,)


# --- totals -----------------------------------------------------------------


def test_run_totals_equal_span_sums(conn: Any) -> None:
    rid = _record(conn)
    run = _run(conn, rid)
    spans = _spans(conn, rid)
    steps = [s for s in spans if s["parent_span_id"] is not None]
    assert run["input_tokens"] == sum(s["input_tokens"] for s in steps) == 250
    assert run["output_tokens"] == sum(s["output_tokens"] for s in steps) == 50
    assert run["latency_ms"] == sum(s["latency_ms"] for s in steps)
    assert run["latency_ms"] >= 750
    assert run["model_calls"] == len(_model_spans(spans)) == 2
    assert run["tool_calls"] == len(_tool_spans(spans)) == 1
    models = _model_spans(spans)
    assert run["cache_hits"] == sum(1 for s in models if s["cache_status"] == "hit")
    assert run["cache_misses"] == sum(1 for s in models if s["cache_status"] == "miss")
    assert all(s["cache_status"] in ("miss", "live") for s in models)
    assert run["finished_at"] is not None


def test_run_totals_for_replay_equal_span_sums(conn: Any) -> None:
    src = _record(conn)
    rid = _replay(conn, src)
    run = _run(conn, rid)
    spans = _spans(conn, rid)
    steps = [s for s in spans if s["parent_span_id"] is not None]
    assert run["input_tokens"] == sum(s["input_tokens"] for s in steps)
    assert run["output_tokens"] == sum(s["output_tokens"] for s in steps)
    assert run["latency_ms"] == sum(s["latency_ms"] for s in steps)
    assert run["model_calls"] == 2 and run["tool_calls"] == 1
    assert run["cache_hits"] == 2 and run["cache_misses"] == 0


# --- payloads ---------------------------------------------------------------


def test_model_span_payload_holds_request_response_hash_status_latency(
    conn: Any,
) -> None:
    spans = _spans(conn, _record(conn, _provider(_resp1(sig="c2lnbmF0dXJl"), _resp2())))
    m = _model_spans(spans)[0]
    p = m["payload"]
    assert {"request", "response", "request_hash", "cache_status", "latency_ms"} <= set(
        p
    )
    assert p["latency_ms"] == 300 == m["latency_ms"]
    assert p["cache_status"] == m["cache_status"]
    assert re.fullmatch(r"[0-9a-f]{64}", p["request_hash"])
    assert p["request_hash"] == request_hash(ModelRequest.model_validate(p["request"]))
    assert p["request"]["messages"][0]["text"] == _task().prompt
    assert p["response"]["tool_calls"][0]["name"] == "get_order"
    assert p["response"]["tool_calls"][0]["signature"] == "c2lnbmF0dXJl"
    # second request carries the tool result
    p2 = _model_spans(spans)[1]["payload"]
    assert p2["request"]["messages"][-1]["role"] == "tool"
    assert p2["response"]["text"] == "Order A1001 has shipped."


def test_tool_span_payload_holds_name_call_id_args_result_fault(conn: Any) -> None:
    from agent_replay.demo import tools

    t = _tool_spans(_spans(conn, _record(conn)))[0]
    p = t["payload"]
    assert {"name", "call_id", "args", "result", "fault"} <= set(p)
    assert p["name"] == "get_order"
    assert p["call_id"] == "call_1_0"
    assert p["args"] == {"order_id": "A1001"}
    assert p["result"] == tools.get_order("A1001")
    assert p["fault"] is None


# --- ToolStubs --------------------------------------------------------------


def _stub_spans_with_repeats(conn: Any) -> list[dict[str, Any]]:
    two = ModelResponse(
        tool_calls=[
            ToolCall(id="call_1_0", name="get_order", args={"order_id": "A1001"}),
            ToolCall(id="call_1_1", name="get_order", args={"order_id": "A1001"}),
            ToolCall(id="call_1_2", name="get_order", args={"order_id": "A1002"}),
        ],
        input_tokens=10,
        output_tokens=5,
        latency_ms=10,
    )
    rid = _record(conn, _provider(two, _resp2()))
    spans = copy.deepcopy(_spans(conn, rid))
    tools = _tool_spans(spans)
    assert len(tools) == 3
    for i, t in enumerate(tools):
        marker = {"marker": f"result-{i}"}
        t["payload"]["result"] = marker
        t["attributes"][semconv.GEN_AI_TOOL_CALL_RESULT] = json.dumps(
            marker, sort_keys=True
        )
    return spans


def test_tool_stubs_answer_by_name_and_args_in_recorded_order(conn: Any) -> None:
    from agent_replay.runtime.stubs import ToolStubs

    stubs = ToolStubs.from_spans(_stub_spans_with_repeats(conn))
    assert stubs.call("get_order", {"order_id": "A1001"}) == {"marker": "result-0"}
    assert stubs.call("get_order", {"order_id": "A1001"}) == {"marker": "result-1"}
    # different args are matched by args, not by position
    assert stubs.call("get_order", {"order_id": "A1002"}) == {"marker": "result-2"}


def test_tool_stubs_canonical_args_ignore_key_order(conn: Any) -> None:
    from agent_replay.runtime.stubs import ToolStubs

    rid = _record(
        conn,
        _provider(
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1_0",
                        name="convert_currency",
                        args={
                            "amount": 5,
                            "from_currency": "USD",
                            "to_currency": "EUR",
                        },
                    )
                ],
                latency_ms=1,
            ),
            _resp2(),
        ),
    )
    stubs = ToolStubs.from_spans(_spans(conn, rid))
    out = stubs.call(
        "convert_currency", {"to_currency": "EUR", "from_currency": "USD", "amount": 5}
    )
    assert out != {"error": "unrecorded_tool_call"}
    assert out["converted"]


def test_tool_stubs_unrecorded_call_returns_error_dict(conn: Any) -> None:
    from agent_replay.runtime.stubs import ToolStubs

    stubs = ToolStubs.from_spans(_spans(conn, _record(conn)))
    assert stubs.call("get_order", {"order_id": "ZZZ"}) == {
        "error": "unrecorded_tool_call"
    }
    assert stubs.call("search_products", {"query": "x"}) == {
        "error": "unrecorded_tool_call"
    }


# --- divergence -------------------------------------------------------------


class _WanderAgent:
    """Replays the source's first model request, then calls a tool that was never recorded."""

    name = "demo-agent"

    def __init__(self, first_request: dict[str, Any], outcome: AgentOutcome) -> None:
        self._req = ModelRequest.model_validate(first_request)
        self._outcome = outcome
        self.tool_result: Any = None

    def run(self, task: Any, ctx: Any) -> AgentOutcome:
        ctx.call_model(self._req)
        self.tool_result = ctx.call_tool("get_order", {"order_id": "A9999"})
        return self._outcome


def _first_request(conn: Any, src: str) -> dict[str, Any]:
    return _model_spans(_spans(conn, src))[0]["payload"]["request"]


def test_unrecorded_tool_call_diverges_the_run(conn: Any) -> None:
    src = _record(conn)
    agent = _WanderAgent(
        _first_request(conn, src), AgentOutcome(status="ok", final_output="x")
    )
    rid = _replay(conn, src, agent=agent)
    assert agent.tool_result.data == {"error": "unrecorded_tool_call"} or (
        agent.tool_result.error_type == "unrecorded_tool_call"
    )
    run = _run(conn, rid)
    assert run["status"] == "diverged"
    assert run["mode"] == "replay"
    t = _tool_spans(_spans(conn, rid))
    assert len(t) == 1
    assert t[0]["error_type"] == "unrecorded_tool_call"
    assert t[0]["attributes"][semconv.ERROR_TYPE] == "unrecorded_tool_call"
    assert t[0]["payload"]["result"] == {"error": "unrecorded_tool_call"}
    assert t[0]["payload"]["args"] == {"order_id": "A9999"}


def test_later_error_overrides_diverged(conn: Any) -> None:
    src = _record(conn)
    agent = _WanderAgent(
        _first_request(conn, src), AgentOutcome(status="error", error="RuntimeError")
    )
    assert _run(conn, _replay(conn, src, agent=agent))["status"] == "error"


# --- replay -----------------------------------------------------------------


def test_replay_never_calls_real_tool_functions(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_replay.demo import tools

    calls: list[str] = []

    def wrap(name: str, fn: Any) -> Any:
        def spy(*a: Any, **kw: Any) -> Any:
            calls.append(name)
            return fn(*a, **kw)

        return spy

    for name, fn in list(tools.TOOLS.items()):
        spy = wrap(name, fn)
        monkeypatch.setitem(tools.TOOLS, name, spy)
        monkeypatch.setattr(tools, name, spy)

    src = _record(conn)
    assert calls == ["get_order"], "spy must observe the real tool during record"
    calls.clear()
    rid = _replay(conn, src)
    assert calls == []
    run = _run(conn, rid)
    assert run["status"] == "ok" and run["tool_calls"] == 1


def test_replay_of_unchanged_run_matches_source_without_live_provider(
    conn: Any,
) -> None:
    src = _record(conn)
    rid = _replay(conn, src)
    s, r = _run(conn, src), _run(conn, rid)
    assert r["id"] != s["id"]
    assert r["mode"] == "replay"
    assert str(r["source_run_id"]) == src
    assert r["status"] == "ok"
    assert r["variant"] == "minimal" and r["task_id"] == "example-1"
    spans = _spans(conn, rid)
    models = _model_spans(spans)
    assert len(models) == 2
    assert all(m["cache_status"] == "hit" for m in models)
    assert all(m["payload"]["cache_status"] == "hit" for m in models)
    for key in (
        "final_output",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "tool_calls",
    ):
        assert r[key] == s[key], key
    src_spans = _spans(conn, src)
    assert [m["latency_ms"] for m in models] == [
        m["latency_ms"] for m in _model_spans(src_spans)
    ]
    assert [m["payload"]["request_hash"] for m in models] == [
        m["payload"]["request_hash"] for m in _model_spans(src_spans)
    ]
    assert [t["payload"]["result"] for t in _tool_spans(spans)] == [
        t["payload"]["result"] for t in _tool_spans(src_spans)
    ]
    assert [s_["seq"] for s_ in spans] == list(range(len(spans)))
    # the source run is untouched
    assert _run(conn, src)["mode"] == "record"
    assert len(runs_repo.list_runs(conn)) == 2


def test_replay_with_changed_variant_and_no_live_provider_is_skipped(conn: Any) -> None:
    src = _record(conn)
    rid = _replay(conn, src, variant="structured")
    run = _run(conn, rid)
    assert run["status"] == "skipped"
    assert run["mode"] == "replay"
    assert str(run["source_run_id"]) == src
    assert run["variant"] == "structured"

    from agent_replay.config import Settings
    from agent_replay.demo.agent import load_prompt
    from agent_replay.demo.tools import TOOL_SCHEMAS
    from agent_replay.llm import Message

    s = Settings.model_validate({})
    missing = request_hash(
        ModelRequest(
            provider=s.provider,
            model=s.model,
            system=load_prompt("structured"),
            messages=[Message(role="user", text=_task().prompt)],
            tools=list(TOOL_SCHEMAS.values()),
            temperature=s.temperature,
            seed=s.seed,
        )
    )
    src_hashes = {m["payload"]["request_hash"] for m in _model_spans(_spans(conn, src))}
    assert missing not in src_hashes
    assert missing in (run["error"] or "")


def test_replay_with_changed_variant_and_live_provider_serves_changed_request_live(
    conn: Any,
) -> None:
    """With a live provider the changed request is served live and the run completes."""
    src = _record(conn)
    live = _provider(_resp2().model_copy(update={"latency_ms": 77}))
    rid = _replay(conn, src, variant="structured", provider=live)
    assert len(live.requests) == 1
    run = _run(conn, rid)
    assert run["status"] in ("ok", "diverged")
    assert run["final_output"] == "Order A1001 has shipped."


# --- lifecycle --------------------------------------------------------------


class _Boom:
    name = "demo-agent"

    def __init__(self, request: dict[str, Any] | None = None) -> None:
        self._req = request

    def run(self, task: Any, ctx: Any) -> AgentOutcome:
        if self._req is not None:
            ctx.call_model(ModelRequest.model_validate(self._req))
        raise RuntimeError("boom")


def test_unexpected_agent_exception_stores_error_run_with_class_name(conn: Any) -> None:
    rid = _runner().record_run(
        _task(), _Boom(), "minimal", conn=conn, provider=_provider()
    )
    run = _run(conn, str(rid))
    assert run["status"] == "error"
    assert run["mode"] == "record"
    assert "RuntimeError" in f"{run['error_type']} {run['error']}"
    assert (
        run["error_type"] == "RuntimeError"
        or run["error"] == "RuntimeError"
        or ("RuntimeError" in (run["error"] or ""))
    )


def test_error_after_a_model_step_keeps_spans_attached_to_the_run_row(
    conn: Any, raw: Any
) -> None:
    src = _record(conn)
    req = _first_request(conn, src)
    rid = str(
        _runner().record_run(
            _task(), _Boom(req), "minimal", conn=conn, provider=_provider()
        )
    )
    assert _run(conn, rid)["status"] == "error"
    spans = _spans(conn, rid)
    assert len(spans) >= 1 and [s["seq"] for s in spans] == list(range(len(spans)))
    assert raw.execute(
        "SELECT count(*) FROM spans WHERE run_id NOT IN (SELECT id FROM runs)"
    ).fetchone() == (0,)
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert len(roots) == 1 and roots[0]["status"] == "error"
    assert roots[0]["error_type"] == "RuntimeError"


def test_replay_with_unexpected_exception_is_stored_as_error(conn: Any) -> None:
    src = _record(conn)
    rid = _replay(conn, src, agent=_Boom())
    run = _run(conn, rid)
    assert run["status"] == "error" and run["mode"] == "replay"
    assert "RuntimeError" in f"{run['error_type']} {run['error']}"


class _Raises:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, request: Any) -> Any:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (BudgetExceeded(1.5, 1.5), "budget_exceeded"),
        (QuotaExhausted("rpd"), "quota_exhausted"),
    ],
)
def test_budget_and_quota_stop_conditions_are_stored_as_error_runs(
    conn: Any, exc: Exception, code: str
) -> None:
    try:
        _runner().record_run(
            _task(), _agent(), "minimal", conn=conn, provider=_Raises(exc)
        )
    except (BudgetExceeded, QuotaExhausted):
        pass  # the runner may re-raise after storing so a suite can stop
    runs = runs_repo.list_runs(conn)
    assert len(runs) == 1
    assert runs[0]["status"] == "error"
    assert code in f"{runs[0]['error_type']} {runs[0]['error']}"
