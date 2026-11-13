"""Fault injection in replay: tool timeout, malformed tool output and 429s.

Only the scripted provider is used: no live calls.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from support import scrubbed_env

from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.demo.agent import DemoAgent
from agent_replay.demo.tools import OUTPUT_MODELS
from agent_replay.llm import ModelResponse, ScriptedProvider, ToolCall
from agent_replay.llm.factory import get_tracker
from agent_replay.llm.ratelimit import RateLimitedProvider, RateLimiter
from agent_replay.runtime import runner
from agent_replay.taskspec import load_tasks

SKEL = Path(str(resources.files("agent_replay") / "demo" / "example"))
TASKS = SKEL / "task.jsonl"
SETTINGS = Settings(backoff_base_s=0.0, tool_timeout_s=7.0, max_attempts=3)
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _faults() -> Any:
    from agent_replay.runtime import faults

    return faults


def pf(spec: str) -> Any:
    return _faults().parse_fault(spec)


def _task() -> Any:
    return load_tasks(TASKS)[0]


def _agent() -> Any:
    return DemoAgent("minimal", SETTINGS)


def _resp1() -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(id="call_1_0", name="get_order", args={"order_id": "A1001"}),
            ToolCall(id="call_1_1", name="search_products", args={"query": "mouse"}),
        ],
        input_tokens=100,
        output_tokens=20,
        latency_ms=300,
    )


def _resp2() -> ModelResponse:
    return ModelResponse(
        text="Done.", input_tokens=150, output_tokens=30, latency_ms=450
    )


def _record(conn: Any) -> str:
    return str(
        runner.record_run(
            _task(),
            _agent(),
            "minimal",
            conn=conn,
            provider=ScriptedProvider([_resp1(), _resp2()]),
            settings=SETTINGS,
        )
    )


def _replay(conn: Any, source: str, specs: list[str], provider: Any = None) -> str:
    return str(
        runner.replay_run(
            source,
            conn=conn,
            task=_task(),
            agent=_agent(),
            variant="minimal",
            provider=provider,
            settings=SETTINGS,
            faults=[pf(s) for s in specs],
        )
    )


def _spans(conn: Any, rid: str, prefix: str) -> list[dict[str, Any]]:
    return [s for s in runs_repo.list_spans(conn, rid) if s["name"].startswith(prefix)]


def _tool(conn: Any, rid: str, name: str) -> dict[str, Any]:
    (s,) = _spans(conn, rid, f"execute_tool {name}")
    return s


def _attempts(span: dict[str, Any]) -> int:
    p = span["payload"]
    return p["response"]["attempts"] if p.get("response") else p["attempts"]


def _tool_response(req: Any, name: str) -> Any:
    last = req.messages[-1]
    (r,) = [t for t in last.tool_results if t.name == name]
    return r.response


# --- parse_fault ------------------------------------------------------------


def test_parse_fault_valid_specs_return_fault_objects() -> None:
    a, b, c = (
        pf("tool_timeout:get_order"),
        pf("malformed_tool_output:*"),
        pf("rate_limit:2"),
    )
    assert a is not None and b is not None and c is not None
    assert not isinstance(a, str | dict | tuple)
    assert len({repr(a), repr(b), repr(c)}) == 3


def test_parse_fault_unknown_kind_lists_valid_kinds() -> None:
    with pytest.raises(ValueError) as ei:
        pf("explode:x")
    msg = str(ei.value)
    for kind in ("tool_timeout", "malformed_tool_output", "rate_limit"):
        assert kind in msg


def test_parse_fault_bad_rate_limit_count_raises() -> None:
    with pytest.raises(ValueError):
        pf("rate_limit:abc")


# --- tool_timeout -----------------------------------------------------------


def test_tool_timeout_stub_span_and_other_tools_unaffected(conn: Any) -> None:
    src = _record(conn)
    recorded_search = _tool(conn, src, "search_products")
    spec = "tool_timeout:get_order"
    live = ScriptedProvider(
        [_resp2()]
    )  # request 2 differs from the recording: cache miss
    new = _replay(conn, src, [spec], provider=live)

    t = _tool(conn, new, "get_order")
    assert t["payload"]["result"] == {"error": "tool_timeout"}
    assert t["error_type"] == "timeout"
    assert t["latency_ms"] == 7000
    assert spec in json.dumps(t["payload"]["fault"])

    other = _tool(conn, new, "search_products")
    assert other["payload"]["result"] == recorded_search["payload"]["result"]
    assert other["error_type"] is None
    assert other["latency_ms"] == recorded_search["latency_ms"]
    assert not other["payload"].get("fault")

    # the model is told about the timeout
    assert _tool_response(live.requests[0], "get_order") == {"error": "tool_timeout"}


# --- malformed_tool_output --------------------------------------------------


def test_malformed_tool_output_every_tool_fails_schema_and_run_does_not_raise(
    conn: Any,
) -> None:
    src = _record(conn)
    live = ScriptedProvider([_resp2()])
    new = _replay(
        conn, src, ["malformed_tool_output:*"], provider=live
    )  # must not raise

    for name in ("get_order", "search_products"):
        t = _tool(conn, new, name)
        assert t["error_type"] == "malformed_tool_output"
        result = t["payload"]["result"]
        with pytest.raises(ValidationError):
            OUTPUT_MODELS[name].model_validate(result)
        assert "malformed_tool_output:*" in json.dumps(t["payload"]["fault"])
        assert _tool_response(live.requests[0], name) == {
            "error": "malformed_tool_output"
        }
    run = runs_repo.get_run(conn, new)
    assert run is not None and run["final_output"] == "Done."


# --- rate_limit -------------------------------------------------------------


def test_rate_limit_2_with_3_attempts_completes_with_attempts_3(conn: Any) -> None:
    src = _record(conn)
    new = _replay(conn, src, ["rate_limit:2"])  # no live provider: all cached
    run = runs_repo.get_run(conn, new)
    assert run["status"] == "ok"
    models = _spans(conn, new, "generate_content")
    assert len(models) == 2
    assert _attempts(models[0]) == 3
    assert _attempts(models[1]) == 1


def test_rate_limit_5_with_3_attempts_fails_first_step(conn: Any) -> None:
    src = _record(conn)
    new = _replay(conn, src, ["rate_limit:5"])
    run = runs_repo.get_run(conn, new)
    assert run["status"] == "error"
    # Not the provider's stop code: an injected exhaustion must not stop a fault suite.
    assert run["error_type"] == run["error"] == "injected_rate_limit"
    models = _spans(conn, new, "generate_content")
    assert len(models) == 1
    assert models[0]["error_type"] == "injected_rate_limit"
    assert not _spans(conn, new, "execute_tool")


def test_rate_limit_uses_no_budget_rpm_or_live_call(conn: Any) -> None:
    src = _record(conn)
    spent = get_tracker(SETTINGS).spent_usd
    limiter = RateLimiter(1, 1)  # a single acquire would exhaust it
    live = ScriptedProvider([])  # any live call raises ScriptExhausted
    zero_budget = Settings(
        backoff_base_s=0.0, tool_timeout_s=7.0, max_attempts=3, run_budget_usd=0.0
    )
    new = str(
        runner.replay_run(
            src,
            conn=conn,
            task=_task(),
            agent=DemoAgent("minimal", SETTINGS),
            variant="minimal",
            provider=RateLimitedProvider(live, limiter),
            settings=zero_budget,
            faults=[pf("rate_limit:2")],
        )
    )
    run = runs_repo.get_run(conn, new)
    assert run["status"] == "ok"  # a zero budget would refuse any live call
    assert live.requests == []
    assert get_tracker(SETTINGS).spent_usd == spent
    limiter.acquire()  # still has its one request and one daily slot


# --- storage ----------------------------------------------------------------


def test_fault_specs_stored_on_run_row(conn: Any) -> None:
    src = _record(conn)
    specs = ["rate_limit:1", "tool_timeout:get_order"]
    new = _replay(conn, src, specs, provider=ScriptedProvider([_resp2()]))
    run = runs_repo.get_run(conn, new)
    stored = json.dumps(run["faults"])
    for s in specs:
        assert s in stored
    assert runs_repo.get_run(conn, src)["faults"] == []


def test_record_run_rejects_faults(conn: Any) -> None:
    with pytest.raises(ValueError) as ei:
        runner.record_run(
            _task(),
            _agent(),
            "minimal",
            conn=conn,
            provider=ScriptedProvider([_resp1(), _resp2()]),
            settings=SETTINGS,
            faults=[pf("rate_limit:1")],
        )
    assert "fault" in str(ei.value).lower()
    assert runs_repo.list_runs(conn) == []


# --- CLI --------------------------------------------------------------------


def cli(db_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env.update(
        AGENT_REPLAY_DATABASE_URL=db_url,
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_PROVIDER="scripted",
        AGENT_REPLAY_SCRIPT_PATH=str(SKEL / "script.json"),
        AGENT_REPLAY_BACKOFF_BASE_S="0",
    )
    code = "import sys; from agent_replay.cli import main; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _cli_record(db_url: str) -> str:
    p = cli(
        db_url,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "structured",
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    return p.stdout.split()[-1]


def test_cli_replay_accepts_repeated_fault_flags(db_url: str, raw: Any) -> None:
    rid = _cli_record(db_url)
    p = cli(
        db_url,
        "replay",
        "--run",
        rid,
        "--cache-only",
        "--fault",
        "tool_timeout:get_order",
        "--fault",
        "rate_limit:1",
    )
    # The timed-out tool changes the next request, so --cache-only misses: the run is
    # stored as skipped and the command exits 1.
    assert p.returncode == 1, (p.stdout, p.stderr)
    assert "run status skipped" in p.stderr
    new = p.stdout.split()[-1]
    assert _UUID.match(new)
    (faults,) = raw.execute("SELECT faults FROM runs WHERE id=%s", (new,)).fetchone()
    assert "tool_timeout:get_order" in json.dumps(faults)
    assert "rate_limit:1" in json.dumps(faults)


def test_cli_replay_invalid_fault_exits_nonzero_and_lists_kinds(
    db_url: str, raw: Any
) -> None:
    rid = _cli_record(db_url)
    p = cli(db_url, "replay", "--run", rid, "--cache-only", "--fault", "explode:x")
    assert p.returncode != 0
    for kind in ("tool_timeout", "malformed_tool_output", "rate_limit"):
        assert kind in p.stderr
    (n,) = raw.execute("SELECT count(*) FROM runs WHERE mode='replay'").fetchone()
    assert n == 0


def test_cli_record_with_fault_is_rejected(db_url: str, raw: Any) -> None:
    p = cli(
        db_url,
        "record",
        "--task",
        "example-1",
        "--tasks",
        str(TASKS),
        "--variant",
        "structured",
        "--fault",
        "rate_limit:1",
    )
    assert p.returncode == 2  # faults are replay-only, so record has no --fault flag
    assert "unrecognized arguments: --fault" in p.stderr
    (n,) = raw.execute("SELECT count(*) FROM runs").fetchone()
    assert n == 0
