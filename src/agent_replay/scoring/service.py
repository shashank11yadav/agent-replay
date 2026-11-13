"""Score a stored run: run the deterministic checks and upsert one score row each."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg

from agent_replay.db import runs_repo
from agent_replay.scoring.checks import CHECKS, CheckResult
from agent_replay.taskspec import TaskSpec

Row = dict[str, Any]
# Hook for the LLM judge (scoring.judge.make_judge): called with (task, run, spans);
# its result is stored as kind 'judge'.
Judge = Callable[[TaskSpec, Row, list[Row]], CheckResult]


def score_run(
    conn: psycopg.Connection[Row],
    run_id: str,
    task: TaskSpec,
    judge: Judge | None = None,
) -> dict[str, Any]:
    run = runs_repo.get_run(conn, run_id)
    if run is None:
        raise KeyError(f"run not found: {run_id}")
    spans = runs_repo.list_spans(conn, run_id)
    results = {name: check(task, run, spans) for name, check in CHECKS.items()}
    verdict = judge(task, run, spans) if judge is not None else None
    with conn.transaction():
        for name, res in results.items():
            runs_repo.upsert_score(conn, run_id, name, res.passed, res.detail)
        if verdict is not None:
            runs_repo.upsert_score(
                conn, run_id, "judge", verdict.passed, verdict.detail, kind="judge"
            )
        else:
            runs_repo.delete_judge_score(conn, run_id)
    checks: dict[str, Any] = {
        name: {"passed": res.passed, "detail": res.detail}
        for name, res in results.items()
    }
    summary: dict[str, Any] = {
        "checks": checks,
        "deterministic_pass": all(c["passed"] is True for c in checks.values()),
    }
    summary["task_pass"] = bool(
        summary["deterministic_pass"] and verdict is not None and verdict.passed is True
    )
    if verdict is not None:
        summary["judge"] = {"passed": verdict.passed, "detail": verdict.detail}
    return summary
