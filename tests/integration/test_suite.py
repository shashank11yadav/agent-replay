"""Suite runner: resumable record, replay and score, through the CLI.

Failures are simulated in the CLI subprocess by patching ScriptedProvider.generate (quota
on the Nth call, call counting) and by AGENT_REPLAY_RUN_BUDGET_USD with inflated prices.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from support import scrubbed_env

from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.llm import QuotaExhausted
from agent_replay.suite import runner as suite_runner
from agent_replay.suite.runner import (
    file_sha256,
    pending_tasks,
    plan_replay,
    record_suite,
    score_suite,
    select_subset,
)
from agent_replay.taskspec import TaskSpec

N = 5
TASK_IDS = [f"s-{i}" for i in range(1, N + 1)]
PRELUDE = r"""
import os, sys
from agent_replay.llm.fake import ScriptedProvider
from agent_replay.llm.types import QuotaExhausted
_orig = ScriptedProvider.generate
_n = [0]
def _gen(self, req):
    _n[0] += 1
    p = os.environ.get('TEST_CALL_COUNT_FILE')
    if p:
        open(p, 'a').write('x\n')
    q = os.environ.get('TEST_QUOTA_AT')
    if q and _n[0] >= int(q):
        raise QuotaExhausted('quota')
    return _orig(self, req)
ScriptedProvider.generate = _gen
from agent_replay.cli import main
sys.exit(main())
"""


def write_tasks(tmp: Path) -> Path:
    lines = [
        json.dumps(
            {
                "id": tid,
                "prompt": f"What is the status of order A1001 (case {i})?",
                "category": "lookup",
                "expected_text": ["shipped"],
                "must_call_tools": ["get_order"],
                "max_steps": 4,
                "rubric": f"States that order A1001 has shipped (case {i}).",
            }
        )
        for i, tid in enumerate(TASK_IDS, start=1)
    ]
    p = tmp / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def write_agent_script(tmp: Path) -> Path:
    pair = [
        {
            "tool_calls": [
                {"id": "x", "name": "get_order", "args": {"order_id": "A1001"}}
            ]
        },
        {
            "text": "Order A1001 has shipped.",
            "input_tokens": 1_000_000,
            "output_tokens": 0,
        },
    ]
    p = tmp / "agent_script.json"
    p.write_text(json.dumps(pair * N))
    return p


def write_judge_script(tmp: Path) -> Path:
    entry = {
        "text": '{"pass": true, "reason": "ok"}',
        "input_tokens": 1_000_000,
        "output_tokens": 0,
    }
    p = tmp / "judge_script.json"
    p.write_text(json.dumps([entry] * (2 * N)))
    return p


def cli(
    db_url: str,
    script: Path,
    *args: str,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env.update(
        AGENT_REPLAY_DATABASE_URL=db_url,
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_PROVIDER="scripted",
        AGENT_REPLAY_SCRIPT_PATH=str(script),
        AGENT_REPLAY_BACKOFF_BASE_S="0",
        AGENT_REPLAY_RUN_BUDGET_USD="100",
    )
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", PRELUDE, *args],
        env=env,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def suite_row(conn: Any, sid: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM suite_runs WHERE id = %s", (sid,)).fetchone()
    assert row is not None, f"no suite_runs row {sid}"
    return dict(row)


def runs_of(conn: Any, sid: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE suite_run_id = %s ORDER BY started_at, id", (sid,)
    ).fetchall()
    return [dict(r) for r in rows]


def count_calls(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


class Env:
    def __init__(self, db_url: str, tmp: Path) -> None:
        self.db_url = db_url
        self.tmp = tmp
        self.tasks = write_tasks(tmp)
        self.script = write_agent_script(tmp)
        self.judge_script = write_judge_script(tmp)

    def run(self, *args: str, **kw: Any) -> subprocess.CompletedProcess[str]:
        script = kw.pop("script", self.script)
        return cli(self.db_url, script, *args, cwd=self.tmp, **kw)

    def record(self, **kw: Any) -> subprocess.CompletedProcess[str]:
        # A relative path, so later commands may fall back to the stored one.
        return self.run(
            "suite",
            "record",
            "--tasks",
            self.tasks.name,
            "--variant",
            "structured",
            "--label",
            "base",
            **kw,
        )

    def record_ok(self) -> str:
        p = self.record()
        assert p.returncode == 0, (p.stdout, p.stderr)
        return last_line(p)

    def replay(
        self, source: str, *extra: str, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        variant = kw.pop("variant", "structured")
        return self.run(
            "suite", "replay", "--source", source, "--variant", variant, *extra, **kw
        )


@pytest.fixture
def env(db_url: str, tmp_path: Path) -> Env:
    return Env(db_url, tmp_path)


STOP_BUDGET = {
    "AGENT_REPLAY_RUN_BUDGET_USD": "3.0",
    "AGENT_REPLAY_PRICE_INPUT_PER_M": "1.0",
}
STOP_QUOTA = {"TEST_QUOTA_AT": "7"}  # call 7 is the first model call of the 4th task


# --- record -------------------------------------------------------------------


def test_suite_record_creates_one_suite_row_and_one_run_per_task(
    env: Env, conn: Any
) -> None:
    p = env.record()
    assert p.returncode == 0, (p.stdout, p.stderr)
    sid = last_line(p)
    rows = conn.execute("SELECT id FROM suite_runs").fetchall()
    assert len(rows) == 1 and str(rows[0]["id"]) == sid
    s = suite_row(conn, sid)
    assert (
        s["label"] == "base" and s["variant"] == "structured" and s["mode"] == "record"
    )
    assert s["status"] == "complete"
    assert s["task_count"] == N
    assert s["tasks_path"] == env.tasks.name
    assert s["tasks_sha256"] == hashlib.sha256(env.tasks.read_bytes()).hexdigest()
    runs = runs_of(conn, sid)
    assert sorted(r["task_id"] for r in runs) == sorted(TASK_IDS)
    assert all(r["mode"] == "record" and r["status"] == "ok" for r in runs)
    assert all(r["variant"] == "structured" for r in runs)


# --- stop + resume ------------------------------------------------------------


def _stopped(env: Env, conn: Any, extra_env: dict[str, str]) -> tuple[str, str]:
    p = env.record(env_extra=extra_env)
    assert p.returncode != 0, (p.stdout, p.stderr)
    out = p.stdout + p.stderr
    sid = next(
        str(r["id"]) for r in conn.execute("SELECT id FROM suite_runs").fetchall()
    )
    return sid, out


def _check_stop(env: Env, conn: Any, sid: str, out: str) -> None:
    runs = runs_of(conn, sid)
    ok = {r["task_id"] for r in runs if r["status"] == "ok"}
    assert ok == {"s-1", "s-2", "s-3"}, [(r["task_id"], r["status"]) for r in runs]
    assert not any(r["task_id"] == "s-5" for r in runs)
    assert suite_row(conn, sid)["status"] == "stopped"
    remain = [ln for ln in out.splitlines() if "remain" in ln.lower()]
    assert any(re.search(r"\b2\b", ln) for ln in remain), out
    cmds = [ln for ln in out.splitlines() if f"--resume {sid}" in ln]
    assert any("suite record" in ln for ln in cmds), out


def test_record_stops_on_budget_exceeded_keeps_runs_and_prints_resume(
    env: Env, conn: Any
) -> None:
    sid, out = _stopped(env, conn, STOP_BUDGET)
    _check_stop(env, conn, sid, out)


def test_record_stops_on_quota_exhausted_keeps_runs_and_prints_resume(
    env: Env, conn: Any
) -> None:
    sid, out = _stopped(env, conn, STOP_QUOTA)
    _check_stop(env, conn, sid, out)


def _printed_resume_args(out: str, sid: str) -> list[str]:
    line = next(
        ln
        for ln in out.splitlines()
        if f"--resume {sid}" in ln and "suite record" in ln
    )
    toks = shlex.split(line.strip().lstrip("$ "))
    return toks[toks.index("suite") :]


def test_resume_skips_recorded_tasks_and_records_the_rest(
    env: Env, conn: Any, tmp_path: Path
) -> None:
    sid, out = _stopped(env, conn, STOP_QUOTA)
    before = {r["task_id"]: r["id"] for r in runs_of(conn, sid) if r["status"] == "ok"}
    counter = tmp_path / "calls.txt"
    p = env.run(
        *_printed_resume_args(out, sid),
        env_extra={"TEST_CALL_COUNT_FILE": str(counter)},
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert last_line(p) == sid
    runs = runs_of(conn, sid)
    for tid, rid in before.items():
        mine = [r for r in runs if r["task_id"] == tid]
        assert [r["id"] for r in mine] == [rid], f"{tid} was re-recorded"
    for tid in TASK_IDS:
        assert any(r["task_id"] == tid and r["status"] == "ok" for r in runs), tid
    assert count_calls(counter) == 4, (
        "only s-4 and s-5 (2 model calls each) may be recorded"
    )
    assert suite_row(conn, sid)["status"] == "complete"
    assert len(conn.execute("SELECT 1 FROM suite_runs").fetchall()) == 1


def test_resume_of_a_complete_suite_records_nothing(
    env: Env, conn: Any, tmp_path: Path
) -> None:
    sid = env.record_ok()
    counter = tmp_path / "calls.txt"
    p = env.run(
        "suite",
        "record",
        "--resume",
        sid,
        env_extra={"TEST_CALL_COUNT_FILE": str(counter)},
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert count_calls(counter) == 0
    assert len(runs_of(conn, sid)) == N


def test_resumed_record_starts_with_a_fresh_run_budget(env: Env, conn: Any) -> None:
    # Budget 3.0 USD = three tasks per process. Five tasks only finish if the resume
    # process starts with spent_usd 0.
    sid, _ = _stopped(env, conn, STOP_BUDGET)
    p = env.run("suite", "record", "--resume", sid, env_extra=STOP_BUDGET)
    assert p.returncode == 0, (p.stdout, p.stderr)
    runs = runs_of(conn, sid)
    for tid in TASK_IDS:
        assert any(r["task_id"] == tid and r["status"] == "ok" for r in runs), tid
    assert suite_row(conn, sid)["status"] == "complete"


# --- replay -------------------------------------------------------------------


def test_suite_replay_creates_replay_suite_with_one_run_per_source_run(
    env: Env, conn: Any
) -> None:
    src = env.record_ok()
    p = env.replay(src)
    assert p.returncode == 0, (p.stdout, p.stderr)
    new = last_line(p)
    assert new != src
    s = suite_row(conn, new)
    assert s["mode"] == "replay" and str(s["source_suite_run_id"]) == src
    assert s["variant"] == "structured" and s["status"] == "complete"
    assert s["task_count"] == N
    src_runs = {r["task_id"]: r for r in runs_of(conn, src)}
    runs = runs_of(conn, new)
    assert sorted(r["task_id"] for r in runs) == sorted(TASK_IDS)
    for r in runs:
        assert r["mode"] == "replay"
        assert r["source_run_id"] == src_runs[r["task_id"]]["id"]
        assert r["suite_run_id"] is not None


def test_record_then_replay_gives_identical_final_outputs(env: Env, conn: Any) -> None:
    src = env.record_ok()
    new = last_line(env.replay(src))
    a = {r["task_id"]: r["final_output"] for r in runs_of(conn, src)}
    b = {r["task_id"]: r["final_output"] for r in runs_of(conn, new)}
    assert set(a) == set(b) == set(TASK_IDS)
    for tid in TASK_IDS:
        assert a[tid] and "shipped" in a[tid]
        assert a[tid] == b[tid], tid
    assert all(r["status"] == "ok" for r in runs_of(conn, new))


def test_suite_replay_stores_repeated_faults_on_the_suite_run(
    env: Env, conn: Any
) -> None:
    src = env.record_ok()
    p = env.replay(src, "--fault", "tool_timeout:get_order", "--fault", "rate_limit:1")
    assert p.returncode == 0, (p.stdout, p.stderr)
    new = last_line(p)
    blob = json.dumps(suite_row(conn, new)["faults"])
    assert "tool_timeout:get_order" in blob and "rate_limit:1" in blob
    assert len(runs_of(conn, new)) == N


def test_replay_stops_on_budget_exceeded(env: Env, conn: Any) -> None:
    src = env.record_ok()
    # minimal variant changes every request -> cache misses -> live scripted calls.
    p = env.replay(src, variant="minimal", env_extra=STOP_BUDGET)
    assert p.returncode != 0, (p.stdout, p.stderr)
    _check_replay_stop(conn, src, "budget_exceeded")


def test_replay_stops_on_quota_exhausted(env: Env, conn: Any) -> None:
    src = env.record_ok()
    p = env.replay(src, variant="minimal", env_extra=STOP_QUOTA)
    assert p.returncode != 0, (p.stdout, p.stderr)
    _check_replay_stop(conn, src, "quota_exhausted")


def test_injected_rate_limit_past_all_attempts_does_not_stop_the_suite(
    env: Env, conn: Any
) -> None:
    # The 429s come from the fault, not the provider, so every task still gets its run.
    src = env.record_ok()
    p = env.replay(
        src, "--fault", "rate_limit:5", env_extra={"AGENT_REPLAY_MAX_ATTEMPTS": "3"}
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    new = _new_suite(conn, src)
    runs = runs_of(conn, new)
    assert len(runs) == N
    assert {(r["status"], r["error"]) for r in runs} == {
        ("error", "injected_rate_limit")
    }
    assert suite_row(conn, new)["status"] == "complete"


def _new_suite(conn: Any, src: str) -> str:
    rows = conn.execute(
        "SELECT id FROM suite_runs WHERE source_suite_run_id = %s", (src,)
    ).fetchall()
    assert len(rows) == 1
    return str(rows[0]["id"])


def _check_replay_stop(conn: Any, src: str, code: str) -> None:
    new = _new_suite(conn, src)
    runs = runs_of(conn, new)
    assert [r["status"] for r in runs] == ["ok", "ok", "ok", "error"], [
        (r["task_id"], r["status"], r["error"]) for r in runs
    ]
    assert runs[-1]["error"] == code
    assert suite_row(conn, new)["status"] == "stopped"
    assert len(runs_of(conn, src)) == N, "source suite untouched"


def test_replay_tasks_subset_replays_only_those_ids(env: Env, conn: Any) -> None:
    src = env.record_ok()
    p = env.replay(src, "--tasks-subset", "s-2,s-4")
    assert p.returncode == 0, (p.stdout, p.stderr)
    new = last_line(p)
    assert sorted(r["task_id"] for r in runs_of(conn, new)) == ["s-2", "s-4"]
    assert all(r["status"] == "ok" for r in runs_of(conn, new))


# --- score --------------------------------------------------------------------


def scores_of(conn: Any, run_id: Any) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM scores WHERE run_id = %s", (run_id,)
        ).fetchall()
    ]


def test_suite_score_no_judge_scores_every_run_deterministically(
    env: Env, conn: Any
) -> None:
    sid = env.record_ok()
    conn.execute("DELETE FROM scores")
    p = env.run("suite", "score", "--suite-run", sid, "--no-judge")
    assert p.returncode == 0, (p.stdout, p.stderr)
    for r in runs_of(conn, sid):
        rows = scores_of(conn, r["id"])
        assert rows, r["task_id"]
        assert not any(s["check_name"] == "judge" for s in rows)
        assert all(s["passed"] is True for s in rows), rows


def test_suite_score_with_judge_writes_a_judge_row_per_run(env: Env, conn: Any) -> None:
    sid = env.record_ok()
    p = env.run("suite", "score", "--suite-run", sid, script=env.judge_script)
    assert p.returncode == 0, (p.stdout, p.stderr)
    for r in runs_of(conn, sid):
        judge = [s for s in scores_of(conn, r["id"]) if s["check_name"] == "judge"]
        assert len(judge) == 1 and judge[0]["passed"] is True, r["task_id"]


def test_suite_score_judge_budget_exceeded_still_scores_remaining_runs(
    env: Env, conn: Any, tmp_path: Path
) -> None:
    sid = env.record_ok()
    runs = {r["task_id"]: r for r in runs_of(conn, sid)}
    price = {"AGENT_REPLAY_PRICE_INPUT_PER_M": "1.0"}
    # Pre-judge s-1 and s-2 so their judge responses are cached.
    for tid in ("s-1", "s-2"):
        p = env.run(
            "score",
            "--run",
            str(runs[tid]["id"]),
            "--tasks",
            str(env.tasks),
            script=env.judge_script,
            env_extra=price,
        )
        assert p.returncode == 0, (p.stdout, p.stderr)
    conn.execute("DELETE FROM scores")
    counter = tmp_path / "judge_calls.txt"
    # Budget 1.5 USD at 1 USD per judge call: one live call fits, the next is refused.
    p = env.run(
        "suite",
        "score",
        "--suite-run",
        sid,
        script=env.judge_script,
        env_extra={
            **price,
            "AGENT_REPLAY_RUN_BUDGET_USD": "1.5",
            "TEST_CALL_COUNT_FILE": str(counter),
        },
    )
    assert p.returncode != 0, (p.stdout, p.stderr)
    verdicts: dict[str, Any] = {}
    for tid, r in runs.items():
        judge = [s for s in scores_of(conn, r["id"]) if s["check_name"] == "judge"]
        assert len(judge) == 1, f"{tid} has no judge row: run was not scored"
        verdicts[tid] = judge[0]
    assert verdicts["s-1"]["passed"] is True and verdicts["s-2"]["passed"] is True
    exceeded = [
        t
        for t, j in verdicts.items()
        if j["detail"].get("verdict") == "budget_exceeded"
    ]
    assert len(exceeded) == 2, {
        t: (j["passed"], j["detail"]) for t, j in verdicts.items()
    }
    assert "s-1" not in exceeded and "s-2" not in exceeded
    assert count_calls(counter) == 1, (
        "no live judge call after the budget guard tripped"
    )


def _task(tid: str) -> TaskSpec:
    return TaskSpec(
        id=tid,
        prompt="p",
        category="c",
        expected_text=[],
        must_call_tools=[],
        max_steps=3,
        rubric="r",
    )


def _run(tid: str, status: str = "ok", error_type: str | None = None) -> dict[str, Any]:
    return {
        "id": f"{tid}-{status}",
        "task_id": tid,
        "status": status,
        "error_type": error_type,
    }


TASKS = [_task("a"), _task("b"), _task("c")]


def test_file_sha256_hashes_the_bytes(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_bytes(b"abc\n")
    assert file_sha256(f) == hashlib.sha256(b"abc\n").hexdigest()


def test_pending_skips_finished_tasks_and_keeps_file_order() -> None:
    runs = [_run("a"), _run("c", "error", "ToolError")]
    assert [t.id for t in pending_tasks(TASKS, runs)] == ["b"]


@pytest.mark.parametrize("code", ["budget_exceeded", "quota_exhausted", "rate_limit"])
def test_pending_includes_tasks_whose_run_was_stopped(code: str) -> None:
    runs = [_run("a"), _run("b", "error", code)]
    assert [t.id for t in pending_tasks(TASKS, runs)] == ["b", "c"]


def test_plan_replay_prefers_latest_ok_run_and_skips_missing_tasks() -> None:
    runs = [
        _run("a", "error", "quota_exhausted"),
        _run("a"),
        _run("c", "error", "ToolError"),
    ]
    plan = plan_replay(TASKS, runs)
    assert [(t.id, r["id"]) for t, r in plan] == [("a", "a-ok"), ("c", "c-error")]


def test_select_subset_filters_and_rejects_unknown_ids() -> None:
    assert [t.id for t in select_subset(TASKS, ["c", "a"])] == ["a", "c"]
    with pytest.raises(ValueError, match="zz"):
        select_subset(TASKS, ["a", "zz"])


# --- DB-backed: stop, resume, score ----------------------------------------------------


def _fake_record(stop_at: dict[str, str]) -> Any:
    """record_run stand-in: stores a run; a task in ``stop_at`` stops as that code."""

    def fake(task: TaskSpec, agent: Any, variant: str, **kw: Any) -> Any:
        code = stop_at.get(task.id)
        run_id = runs_repo.insert_run(
            kw["conn"],
            {
                "suite_run_id": kw["suite_run_id"],
                "task_id": task.id,
                "variant": variant,
                "mode": "record",
                "status": "error" if code else "ok",
                "error_type": code,
                "error": code,
                "final_output": None if code else "shipped",
            },
        )
        if code == "quota_exhausted":
            raise QuotaExhausted("quota")
        return run_id

    return fake


def _record(conn: Any, sid: str, stop_at: dict[str, str]) -> Any:
    return record_suite(
        conn,
        sid,
        TASKS,
        variant="v",
        provider=None,  # type: ignore[arg-type]
        settings=Settings.model_validate({}),
        make_agent=lambda v, s: None,  # type: ignore[arg-type,return-value]
    )


def test_resume_leaves_one_run_per_task_and_score_scores_each(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = str(runs_repo.create_suite_run(conn, variant="v", task_count=3))
    monkeypatch.setattr(
        suite_runner, "record_run", _fake_record({"b": "quota_exhausted"})
    )
    first = _record(conn, sid, {})
    assert first.stop_code == "quota_exhausted" and first.remaining == 2
    assert len(runs_repo.list_suite_run_rows(conn, sid)) == 2

    monkeypatch.setattr(suite_runner, "record_run", _fake_record({}))
    second = _record(conn, sid, {})
    assert not second.stopped
    rows = runs_repo.list_suite_run_rows(conn, sid)
    assert sorted(r["task_id"] for r in rows) == ["a", "b", "c"]
    assert all(r["status"] == "ok" for r in rows)
    assert runs_repo.get_suite_run(conn, sid)["status"] == "complete"  # type: ignore[index]
    assert score_suite(conn, sid, TASKS, None).scored == 3


def test_record_stops_on_a_rate_limit_run_without_an_exception(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = str(runs_repo.create_suite_run(conn, variant="v", task_count=3))
    monkeypatch.setattr(suite_runner, "record_run", _fake_record({"b": "rate_limit"}))
    out = _record(conn, sid, {})
    assert out.stop_code == "rate_limit" and out.remaining == 2
    assert [r["task_id"] for r in runs_repo.list_suite_run_rows(conn, sid)] == [
        "a",
        "b",
    ]
    assert runs_repo.get_suite_run(conn, sid)["status"] == "stopped"  # type: ignore[index]


def test_resume_after_a_replay_pointed_at_the_stopped_run(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = str(runs_repo.create_suite_run(conn, variant="v", task_count=3))
    monkeypatch.setattr(
        suite_runner, "record_run", _fake_record({"b": "quota_exhausted"})
    )
    _record(conn, sid, {})
    runs = runs_repo.list_suite_run_rows(conn, sid)
    stopped = next(r for r in runs if r["task_id"] == "b")
    assert [t.id for t, _ in plan_replay(TASKS, runs)] == ["a"]
    other = str(runs_repo.create_suite_run(conn, variant="v", mode="replay"))
    runs_repo.insert_run(
        conn,
        {
            "suite_run_id": other,
            "source_run_id": str(stopped["id"]),
            "task_id": "b",
            "mode": "replay",
        },
    )
    monkeypatch.setattr(suite_runner, "record_run", _fake_record({}))
    assert not _record(conn, sid, {}).stopped
    rows = runs_repo.list_suite_run_rows(conn, sid)
    assert sorted(r["task_id"] for r in rows) == ["a", "b", "c"]


def test_an_unexpected_error_mid_record_marks_the_suite_stopped(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = str(runs_repo.create_suite_run(conn, variant="v", task_count=3))
    fake = _fake_record({})

    def breaks_on_b(task: TaskSpec, *args: Any, **kw: Any) -> Any:
        if task.id == "b":
            raise RuntimeError("database went away")
        return fake(task, *args, **kw)

    monkeypatch.setattr(suite_runner, "record_run", breaks_on_b)
    with pytest.raises(RuntimeError):
        _record(conn, sid, {})
    assert runs_repo.get_suite_run(conn, sid)["status"] == "stopped"  # type: ignore[index]


def test_suite_replay_with_a_removed_variant_is_a_clean_error(
    env: Env, conn: Any
) -> None:
    sid = env.record_ok()
    conn.execute("UPDATE suite_runs SET variant = 'old_prompt' WHERE id = %s", (sid,))
    p = env.run("suite", "replay", "--source", sid)
    assert p.returncode == 1
    assert "unknown prompt variant 'old_prompt'" in p.stderr
    assert "Traceback" not in p.stderr
    assert [r["mode"] for r in conn.execute("SELECT mode FROM suite_runs")] == [
        "record"
    ]


def test_suite_resume_with_a_removed_variant_leaves_the_suite_alone(
    env: Env, conn: Any
) -> None:
    sid = env.record_ok()
    conn.execute(
        "UPDATE suite_runs SET variant = 'old_prompt', status = 'stopped' "
        "WHERE id = %s",
        (sid,),
    )
    p = env.run("suite", "record", "--resume", sid)
    assert p.returncode == 1
    assert "unknown prompt variant" in p.stderr and "Traceback" not in p.stderr
    assert suite_row(conn, sid)["status"] == "stopped"


@pytest.mark.parametrize("stored", ["/etc/passwd", "../outside/tasks.jsonl"])
def test_a_stored_tasks_path_outside_the_working_directory_is_refused(
    env: Env, conn: Any, stored: str
) -> None:
    sid = env.record_ok()
    conn.execute("UPDATE suite_runs SET tasks_path = %s WHERE id = %s", (stored, sid))
    p = env.run("suite", "score", "--suite-run", sid, "--no-judge")
    assert p.returncode == 1
    assert "pass --tasks" in p.stderr
    assert "root:" not in p.stderr
    ok = env.run(
        "suite", "score", "--suite-run", sid, "--no-judge", "--tasks", str(env.tasks)
    )
    assert ok.returncode == 0, (ok.stdout, ok.stderr)
