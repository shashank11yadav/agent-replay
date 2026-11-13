"""Run a whole task file: record (resumable), replay and score every task.

A suite run is one ``suite_runs`` row plus one run per task. Stop codes
(``agent_replay.codes.STOP_CODES``) end the suite: the run that hit them is already stored
(by ``runtime.runner``) and the suite status becomes ``stopped``.
Callers own the connection and commit: every function here commits after each unit of work
so a later stop never loses stored runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

from agent_replay.adapters.base import AgentAdapter
from agent_replay.codes import (
    BUDGET_EXCEEDED,
    JUDGE_STOP_CODES,
    QUOTA_EXHAUSTED,
    STOP_CODES,
)
from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.llm import BudgetExceeded, QuotaExhausted
from agent_replay.llm.provider import ModelProvider
from agent_replay.runtime.faults import Fault
from agent_replay.runtime.runner import record_run, replay_run
from agent_replay.scoring.service import Judge, score_run
from agent_replay.taskspec import TaskSpec

Row = dict[str, Any]


@dataclass
class SuiteOutcome:
    suite_run_id: str
    stop_code: str | None = None
    remaining: int = 0

    @property
    def stopped(self) -> bool:
        return self.stop_code is not None


@dataclass
class ScoreOutcome:
    scored: int = 0
    judge_stops: list[str] = field(default_factory=list)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finished(run: Row) -> bool:
    return run.get("error_type") not in STOP_CODES


def pending_tasks(tasks: Sequence[TaskSpec], runs: Sequence[Row]) -> list[TaskSpec]:
    """Tasks (file order) with no finished run; a run cut short by a stop code is not."""
    done = {r["task_id"] for r in runs if _finished(r)}
    return [t for t in tasks if t.id not in done]


def plan_replay(
    tasks: Sequence[TaskSpec], source_runs: Sequence[Row]
) -> list[tuple[TaskSpec, Row]]:
    """(task, source run) pairs in file order: the latest ok run, else the latest run.

    Runs cut short by a stop code are never replayed.
    """
    plan: list[tuple[TaskSpec, Row]] = []
    for task in tasks:
        mine = [r for r in source_runs if r["task_id"] == task.id and _finished(r)]
        ok = [r for r in mine if r["status"] == "ok"]
        chosen = ok[-1] if ok else (mine[-1] if mine else None)
        if chosen is not None:
            plan.append((task, chosen))
    return plan


def select_subset(tasks: Sequence[TaskSpec], ids: Sequence[str]) -> list[TaskSpec]:
    known = {t.id for t in tasks}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown task ids: {', '.join(unknown)}")
    wanted = set(ids)
    return [t for t in tasks if t.id in wanted]


def _stop_code(exc: BaseException) -> str:
    return BUDGET_EXCEEDED if isinstance(exc, BudgetExceeded) else QUOTA_EXHAUSTED


def _stopped_by_run(conn: psycopg.Connection[Row], run_id: Any) -> str | None:
    run = runs_repo.get_run(conn, str(run_id))
    code = run.get("error_type") if run is not None else None
    return code if code in STOP_CODES else None


def _finish(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    stop_code: str | None,
    remaining: int,
) -> SuiteOutcome:
    runs_repo.set_suite_status(
        conn, suite_run_id, "stopped" if stop_code else "complete"
    )
    conn.commit()
    return SuiteOutcome(suite_run_id, stop_code, remaining)


def _stop_on_failure(conn: psycopg.Connection[Row], suite_run_id: str) -> None:
    """Mark the suite stopped after an unexpected error, so it is not left ``running``."""
    try:
        conn.rollback()
        runs_repo.set_suite_status(conn, suite_run_id, "stopped")
        conn.commit()
    except psycopg.Error:
        pass  # the original error is the one worth reporting


def discard_stopped_runs(conn: psycopg.Connection[Row], suite_run_id: str) -> int:
    """Detach the suite's runs cut short by a stop code (the rows are kept, outside the suite).

    A resume re-records those tasks; keeping the stale run would give a task two runs.
    Detaching, not deleting, keeps replay runs that point at them valid.
    Returns how many were detached.
    """
    stale = [
        r for r in runs_repo.list_suite_run_rows(conn, suite_run_id) if not _finished(r)
    ]
    for run in stale:
        runs_repo.detach_run(conn, str(run["id"]))
    conn.commit()
    return len(stale)


def record_suite(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    tasks: Sequence[TaskSpec],
    *,
    variant: str,
    provider: ModelProvider,
    settings: Settings,
    make_agent: Callable[[str, Settings], AgentAdapter],
) -> SuiteOutcome:
    """Record every task without a finished run in the suite, stopping on a stop code.

    Runs left by an earlier stop are detached first, so a suite keeps one run per task.
    Any other exception marks the suite ``stopped`` (committed) and is re-raised.
    """
    try:
        return _record_loop(
            conn,
            suite_run_id,
            tasks,
            variant=variant,
            provider=provider,
            settings=settings,
            make_agent=make_agent,
        )
    except BaseException:
        _stop_on_failure(conn, suite_run_id)
        raise


def _record_loop(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    tasks: Sequence[TaskSpec],
    *,
    variant: str,
    provider: ModelProvider,
    settings: Settings,
    make_agent: Callable[[str, Settings], AgentAdapter],
) -> SuiteOutcome:
    discard_stopped_runs(conn, suite_run_id)
    todo = pending_tasks(tasks, runs_repo.list_suite_run_rows(conn, suite_run_id))
    for index, task in enumerate(todo):
        stop: str | None = None
        try:
            run_id = record_run(
                task,
                make_agent(variant, settings),
                variant,
                conn=conn,
                provider=provider,
                settings=settings,
                suite_run_id=suite_run_id,
            )
            conn.commit()
            stop = _stopped_by_run(conn, run_id)
        except (BudgetExceeded, QuotaExhausted) as exc:
            stop = _stop_code(exc)
        if stop is not None:
            return _finish(conn, suite_run_id, stop, len(todo) - index)
    return _finish(conn, suite_run_id, None, 0)


def replay_suite(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    plan: Sequence[tuple[TaskSpec, Row]],
    *,
    variant: str,
    provider: ModelProvider | None,
    settings: Settings,
    faults: Sequence[Fault],
    make_agent: Callable[[str, Settings], AgentAdapter],
) -> SuiteOutcome:
    """Replay each planned source run into the suite, stopping on a stop code.

    Any other exception marks the suite ``stopped`` (committed) and is re-raised.
    """
    try:
        return _replay_loop(
            conn,
            suite_run_id,
            plan,
            variant=variant,
            provider=provider,
            settings=settings,
            faults=faults,
            make_agent=make_agent,
        )
    except BaseException:
        _stop_on_failure(conn, suite_run_id)
        raise


def _replay_loop(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    plan: Sequence[tuple[TaskSpec, Row]],
    *,
    variant: str,
    provider: ModelProvider | None,
    settings: Settings,
    faults: Sequence[Fault],
    make_agent: Callable[[str, Settings], AgentAdapter],
) -> SuiteOutcome:
    for index, (task, source) in enumerate(plan):
        stop: str | None = None
        try:
            run_id = replay_run(
                source["id"],
                conn=conn,
                task=task,
                agent=make_agent(variant, settings),
                variant=variant,
                provider=provider,
                settings=settings,
                faults=faults,
                suite_run_id=suite_run_id,
            )
            conn.commit()
            stop = _stopped_by_run(conn, run_id)
        except (BudgetExceeded, QuotaExhausted) as exc:
            stop = _stop_code(exc)
        if stop is not None:
            return _finish(conn, suite_run_id, stop, len(plan) - index)
    return _finish(conn, suite_run_id, None, 0)


def score_suite(
    conn: psycopg.Connection[Row],
    suite_run_id: str,
    tasks: Sequence[TaskSpec],
    judge: Judge | None,
) -> ScoreOutcome:
    """Score every run of the suite; a judge stop never ends the loop (the rest still score)."""
    by_id = {t.id: t for t in tasks}
    runs = runs_repo.list_suite_run_rows(conn, suite_run_id)
    missing = sorted({r["task_id"] for r in runs if r["task_id"] not in by_id})
    if missing:
        raise KeyError(f"tasks not in the task file: {', '.join(missing)}")
    outcome = ScoreOutcome()
    for run in runs:
        summary = score_run(conn, str(run["id"]), by_id[run["task_id"]], judge=judge)
        conn.commit()
        outcome.scored += 1
        verdict = ((summary.get("judge") or {}).get("detail") or {}).get("verdict")
        if verdict in JUDGE_STOP_CODES:
            outcome.judge_stops.append(verdict)
    return outcome
