"""Runner unit tests against a throwaway Postgres database."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from support import ROOT, pg

from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.demo.agent import DemoAgent
from agent_replay.demo.tools import TOOLS
from agent_replay.llm import (
    BudgetExceeded,
    ModelResponse,
    RateLimitError,
    ScriptedProvider,
    ToolCall,
    TransientError,
)
from agent_replay.llm.factory import build_live_provider, get_tracker
from agent_replay.runtime import runner
from agent_replay.taskspec import load_tasks

EXAMPLE_TASKS = ROOT / "src" / "agent_replay" / "demo" / "example" / "task.jsonl"


def _task() -> Any:
    return load_tasks(EXAMPLE_TASKS)[0]


def _call(name: str, args: dict[str, Any], step: int = 1, i: int = 0) -> ToolCall:
    return ToolCall(id=f"call_{step}_{i}", name=name, args=args)


def _final(text: str = "done") -> ModelResponse:
    return ModelResponse(text=text, input_tokens=50, output_tokens=10, latency_ms=5)


def _tools_call(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        tool_calls=list(calls), input_tokens=100, output_tokens=20, latency_ms=5
    )


def _fresh_settings(**kw: Any) -> Settings:
    # Unique prices give this test its own process tracker.
    price = 1.0 + uuid.uuid4().int % 10_000 / 1e6
    return Settings.model_validate({"price_input_per_m": price, **kw})


def test_record_run_counts_each_live_call_once_with_build_live_provider(
    conn: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = [_tools_call(_call("get_order", {"order_id": "A1001"})), _final()]
    script = tmp_path / "script.json"
    script.write_text(json.dumps([r.model_dump(mode="json") for r in responses]))
    monkeypatch.setenv("AGENT_REPLAY_SCRIPT_PATH", str(script))
    s = _fresh_settings(provider="scripted")
    tracker = get_tracker(s)
    before = tracker.spent_usd
    live = build_live_provider(s, "record")
    assert isinstance(live, ScriptedProvider)
    runner.record_run(
        _task(),
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=live,
        settings=s,
    )
    expected = sum(tracker.cost_of(r) for r in responses)
    assert tracker.spent_usd - before == pytest.approx(expected)


def test_build_live_provider_replay_without_key_or_script_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_REPLAY_SCRIPT_PATH", raising=False)
    assert build_live_provider(Settings.model_validate({}), "replay") is None
    scripted = Settings.model_validate({"provider": "scripted"})
    assert build_live_provider(scripted, "replay") is None
    with pytest.raises(ValueError):
        build_live_provider(scripted, "record")


class _Raises:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, request: Any) -> Any:
        raise self.exc


def test_stop_condition_run_survives_on_non_autocommit_connection(db_url: str) -> None:
    from psycopg.rows import dict_row

    c = pg.connect(db_url, autocommit=False, row_factory=dict_row)
    try:
        c.execute("SELECT 1")  # an open transaction, as under a pool connection
        with pytest.raises(BudgetExceeded):
            runner.record_run(
                _task(),
                DemoAgent("minimal"),
                "minimal",
                conn=c,
                provider=_Raises(BudgetExceeded(1.5, 1.5)),
            )
        c.rollback()  # what pool.connection() does when the exception leaves it
    finally:
        c.close()
    with pg.connect(db_url, autocommit=True, row_factory=dict_row) as check:
        runs = runs_repo.list_runs(check)
        assert len(runs) == 1
        assert (runs[0]["status"], runs[0]["error"]) == ("error", "budget_exceeded")
        assert runs_repo.list_spans(check, str(runs[0]["id"]))


def test_rate_limit_after_retries_is_stored_as_rate_limit(conn: Any) -> None:
    s = Settings.model_validate({"max_attempts": 1})
    rid = runner.record_run(
        _task(),
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=_Raises(RateLimitError("429")),
        settings=s,
    )
    run = runs_repo.get_run(conn, str(rid))
    assert run is not None
    assert (run["status"], run["error_type"], run["error"]) == (
        "error",
        "rate_limit",
        "rate_limit",
    )
    spans = runs_repo.list_spans(conn, str(rid))
    model = [x for x in spans if x["name"].startswith("generate_content")]
    assert model[0]["error_type"] == "rate_limit"


def _boom(order_id: str) -> dict[str, Any]:
    raise ValueError("bad order")


def _not_a_dict(query: str) -> Any:
    return ["not", "a", "dict"]


def _failing_tools() -> dict[str, Any]:
    return {**TOOLS, "get_order": _boom, "search_products": _not_a_dict}


def _failing_script() -> ScriptedProvider:
    return ScriptedProvider(
        [
            _tools_call(
                _call("get_order", {"order_id": "A1001"}),
                _call("search_products", {"query": "x"}, i=1),
            ),
            _final("sorry"),
        ]
    )


def test_raising_tool_goes_back_to_model_and_marks_run_error(conn: Any) -> None:
    provider = _failing_script()
    rid = runner.record_run(
        _task(),
        DemoAgent("minimal"),
        "minimal",
        conn=conn,
        provider=provider,
        tools=_failing_tools(),
    )
    run = runs_repo.get_run(conn, str(rid))
    assert run is not None
    assert (run["status"], run["error_type"]) == ("error", "ValueError")
    assert run["final_output"] == "sorry"
    tool = [
        x
        for x in runs_repo.list_spans(conn, str(rid))
        if x["name"].startswith("execute_tool")
    ]
    assert (tool[0]["status"], tool[0]["error_type"]) == ("error", "ValueError")
    assert tool[0]["payload"]["raised"] is True
    assert tool[1]["payload"]["raised"] is False
    sent = provider.requests[1].messages[-1].tool_results
    assert sent[0].response == {"error": "tool_error"}


def test_replay_reproduces_recorded_tool_errors_and_hits_cache(conn: Any) -> None:
    src = runner.record_run(
        _task(),
        DemoAgent("minimal"),
        "minimal",
        conn=conn,
        provider=_failing_script(),
        tools=_failing_tools(),
    )
    rid = runner.replay_run(
        src, conn=conn, task=_task(), agent=DemoAgent("minimal"), variant="minimal"
    )
    source = runs_repo.get_run(conn, str(src))
    replay = runs_repo.get_run(conn, str(rid))
    assert source is not None and replay is not None
    assert replay["cache_misses"] == 0 and replay["cache_hits"] == 2
    for key in ("status", "error_type", "final_output", "tool_calls"):
        assert replay[key] == source[key], key

    def tools(run_id: Any) -> list[tuple[Any, ...]]:
        return [
            (x["status"], x["error_type"], x["payload"]["result"])
            for x in runs_repo.list_spans(conn, str(run_id))
            if x["name"].startswith("execute_tool")
        ]

    assert tools(rid) == tools(src)
    assert tools(src)[1][1] == "malformed_tool_output"


def test_cli_replay_without_tasks_uses_stored_task_snapshot(
    db_url: str,
    conn: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    from agent_replay.cli import run_cmd

    responses = [_tools_call(_call("get_order", {"order_id": "A1001"})), _final()]
    script = tmp_path / "script.json"
    script.write_text(json.dumps([r.model_dump(mode="json") for r in responses]))
    monkeypatch.setenv("AGENT_REPLAY_SCRIPT_PATH", str(script))
    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", db_url)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    monkeypatch.setenv("AGENT_REPLAY_MAX_STEPS", "6")
    s = _fresh_settings(provider="scripted", max_steps=6)
    task = _task()
    assert task.max_steps == 4
    source = runner.record_run(
        task,
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=build_live_provider(s, "record"),  # type: ignore[arg-type]
        settings=s,
    )
    args = argparse.Namespace(
        run=str(source), variant=None, tasks=None, cache_only=True
    )
    assert run_cmd._replay(args) == 0
    new_id = capsys.readouterr().out.strip()
    src_run = runs_repo.get_run(conn, str(source))
    new_run = runs_repo.get_run(conn, new_id)
    assert src_run is not None and new_run is not None
    assert new_run["status"] == src_run["status"]
    assert new_run["cache_misses"] == 0


@pytest.mark.parametrize("spec", ["malformed_tool_output:*", "tool_timeout:get_order"])
def test_replay_of_a_faulted_replay_is_deterministic(conn: Any, spec: str) -> None:
    from agent_replay.runtime.faults import parse_fault

    calls = [
        _call("get_order", {"order_id": "A1001"}, i=0),
        _call("search_products", {"query": "mouse"}, i=1),
    ]
    src = runner.record_run(
        _task(),
        DemoAgent("minimal"),
        "minimal",
        conn=conn,
        provider=ScriptedProvider([_tools_call(*calls), _final()]),
    )
    live = ScriptedProvider([_final()])
    faulted = runner.replay_run(
        src,
        conn=conn,
        task=_task(),
        agent=DemoAgent("minimal"),
        variant="minimal",
        provider=live,
        faults=[parse_fault(spec)],
    )
    assert len(live.requests) == 1  # the fault changed the second request
    again = runner.replay_run(  # no faults, no live model (cache only)
        faulted, conn=conn, task=_task(), agent=DemoAgent("minimal"), variant="minimal"
    )
    first = runs_repo.get_run(conn, str(faulted))
    second = runs_repo.get_run(conn, str(again))
    assert first is not None and second is not None
    assert second["status"] == first["status"] != "skipped"
    assert second["cache_hits"] == 2 and second["cache_misses"] == 0
    assert second["final_output"] == first["final_output"]

    def tools(run_id: Any) -> list[tuple[Any, ...]]:
        return [
            (x["error_type"], x["latency_ms"], x["payload"]["result"])
            for x in runs_repo.list_spans(conn, str(run_id))
            if x["name"].startswith("execute_tool")
        ]

    assert tools(again) == tools(faulted)


def test_transient_error_after_retries_is_stored_as_a_stop_code(conn: Any) -> None:
    s = Settings.model_validate({"max_attempts": 1})
    rid = runner.record_run(
        _task(),
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=_Raises(TransientError("503")),
        settings=s,
    )
    run = runs_repo.get_run(conn, str(rid))
    assert run is not None
    assert (run["status"], run["error_type"], run["error"]) == (
        "error",
        "transient_error",
        "transient_error",
    )


def test_replay_serves_the_source_runs_own_responses_over_a_newer_cache_entry(
    conn: Any,
) -> None:
    s = _fresh_settings(provider="scripted")
    shipped = _final("Order A1001 has shipped.")
    first = runner.record_run(
        _task(),
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=ScriptedProvider(
            [_tools_call(_call("get_order", {"order_id": "A1001"})), shipped]
        ),
        settings=s,
    )
    # A later recording sends the same first request and replaces its cache entry.
    runner.record_run(
        _task(),
        DemoAgent("minimal", s),
        "minimal",
        conn=conn,
        provider=ScriptedProvider([_final("I could not find that order.")]),
        settings=s,
    )
    replayed = runner.replay_run(
        first,
        conn=conn,
        task=_task(),
        agent=DemoAgent("minimal", s),
        variant="minimal",
        settings=s,
    )
    run = runs_repo.get_run(conn, str(replayed))
    assert run is not None
    assert (run["status"], run["final_output"]) == ("ok", "Order A1001 has shipped.")
    assert (run["tool_calls"], run["cache_hits"], run["cache_misses"]) == (1, 2, 0)
