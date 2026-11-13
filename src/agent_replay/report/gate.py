"""Regression gate: compare a candidate suite run with a baseline suite run.

Verdict precedence is ERROR > SKIPPED > FAIL > PASS. A delta exactly equal to a threshold
passes. Zero baselines never divide: a 0 -> 0 change is 0%, a rise from 0 is a FAIL
(any increase over a zero baseline exceeds every finite percentage). A missing p95 (a
suite without runs) is reported as N/A and cannot fail the gate on its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg
from pydantic import BaseModel, Field

from agent_replay import jsonio
from agent_replay.codes import (
    JUDGE_ERROR,
    JUDGE_SKIPPED,
    JUDGE_STOP_CODES,
    STOP_CODES,
    UNPARSEABLE_VERDICT,
)
from agent_replay.db import runs_repo
from agent_replay.report.metrics import SuiteMetrics, compute_metrics
from agent_replay.scoring.task_pass import task_passed

Row = dict[str, Any]

_EPS_DIGITS = 9


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class Thresholds(BaseModel):
    max_pass_rate_drop_pp: float = Field(ge=0)
    max_token_increase_pct: float = Field(ge=0)
    max_p95_increase_pct: float = Field(ge=0)


class SuiteNotFoundError(LookupError):
    """A suite run id does not exist."""


class MetricCheck(BaseModel):
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None  # percentage points (pass rate) or percent (others)
    unit: str
    threshold: float
    status: str  # PASS, FAIL or N/A
    note: str = ""


class GateResult(BaseModel):
    verdict: Verdict
    baseline_id: str
    candidate_id: str
    baseline: SuiteMetrics
    candidate: SuiteMetrics
    checks: list[MetricCheck]
    error_causes: dict[str, int]  # stop code or judge_error -> number of candidate rows
    missing_runs: int  # baseline tasks with no candidate run
    # Candidate task ids that are not in the baseline, and ones with more than one run.
    unexpected_tasks: list[str] = Field(default_factory=list)
    duplicate_tasks: list[str] = Field(default_factory=list)
    unscored_runs: int = 0  # candidate runs without deterministic and judge scores
    baseline_problem: str = ""  # why the baseline cannot be gated against
    skipped_requests: int
    flipped_to_fail: list[str]
    flipped_to_pass: list[str]


def load_thresholds(path: str | Path) -> Thresholds:
    """Read the gate thresholds; ValueError when missing, negative or not JSON."""
    data = jsonio.read_json(path)
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004
            "thresholds file must hold a JSON object"
        )
    return Thresholds.model_validate(data)


def _pct_change(base: float, cand: float) -> tuple[float | None, bool]:
    """(percent change, undefined). A zero baseline has no percentage."""
    if base == 0:
        return (0.0, False) if cand == 0 else (None, True)
    return round((cand - base) / base * 100, _EPS_DIGITS), False


def _relative_check(
    metric: str, base: float | None, cand: float | None, limit: float
) -> MetricCheck:
    def make(delta: float | None, status: str, note: str = "") -> MetricCheck:
        return MetricCheck(
            metric=metric,
            baseline=base,
            candidate=cand,
            delta=delta,
            unit="%",
            threshold=limit,
            status=status,
            note=note,
        )

    if base is None or cand is None:
        return make(None, "N/A", "no runs to compare")
    delta, undefined = _pct_change(base, cand)
    if undefined or delta is None:
        return make(None, "FAIL", "baseline is 0")
    return make(delta, "FAIL" if delta > limit else "PASS")


def _pass_rate_check(base: float, cand: float, limit: float) -> MetricCheck:
    delta = round((cand - base) * 100, _EPS_DIGITS)
    return MetricCheck(
        metric="Pass rate",
        baseline=base,
        candidate=cand,
        delta=delta,
        unit="pp",
        threshold=limit,
        status="FAIL" if -delta > limit else "PASS",
    )


def _per_task_pass(runs: Sequence[Row], scores: Sequence[Row]) -> dict[str, bool]:
    """Pass or fail per task.

    A task with several runs has no single answer and is left out; the gate already
    reports it as a duplicate.
    """
    by_run: dict[str, list[Row]] = {}
    for s in scores:
        by_run.setdefault(str(s["run_id"]), []).append(s)
    counts = Counter(str(r["task_id"]) for r in runs)
    return {
        str(r["task_id"]): task_passed(by_run.get(str(r["id"]), [])) is True
        for r in runs
        if counts[str(r["task_id"])] == 1
    }


def _baseline_problems(runs: Sequence[Row], metrics: SuiteMetrics) -> str:
    """Why the baseline cannot be gated against, or "" when it can."""
    if not runs:
        return "the baseline suite has no runs"
    problems = []
    counts = Counter(str(r["task_id"]) for r in runs)
    duplicates = sorted(t for t, n in counts.items() if n > 1)
    if duplicates:
        problems.append(
            "the baseline suite has more than one run for: " + ", ".join(duplicates)
        )
    skipped = sum(1 for r in runs if r["status"] == "skipped")
    if skipped:
        problems.append(f"the baseline suite has {skipped} skipped run(s)")
    stopped = sorted(
        {r["error_type"] for r in runs if r.get("error_type") in STOP_CODES}
    )
    if stopped:
        problems.append(
            "the baseline suite has run(s) stopped by a provider or budget error "
            f"({', '.join(stopped)})"
        )
    if metrics.scored_count < len(runs):
        problems.append("the baseline suite is not fully scored")
    return "; ".join(problems)


def _verdict_of(detail: object) -> object:
    return detail.get("verdict") if isinstance(detail, dict) else None


def _judge_failed(detail: object) -> bool:
    """A judge_error from the provider (rate limit, timeout, ...), not from the reply."""
    return isinstance(detail, dict) and detail.get("error") != UNPARSEABLE_VERDICT


def evaluate(
    baseline_id: str,
    candidate_id: str,
    base_runs: Sequence[Row],
    base_scores: Sequence[Row],
    cand_runs: Sequence[Row],
    cand_scores: Sequence[Row],
    thresholds: Thresholds,
) -> GateResult:
    """Pure gate logic over already-loaded rows."""
    base = compute_metrics(base_runs, base_scores)
    cand = compute_metrics(cand_runs, cand_scores)

    causes: dict[str, int] = {}
    for r in cand_runs:
        code = r.get("error_type") or r.get("error")
        if code in STOP_CODES:
            causes[code] = causes.get(code, 0) + 1
    for s in cand_scores:
        if s["kind"] != "judge":
            continue
        code = _verdict_of(s.get("detail"))
        if isinstance(code, str) and code in JUDGE_STOP_CODES:
            causes[code] = causes.get(code, 0) + 1
        elif code == JUDGE_ERROR and _judge_failed(s.get("detail")):
            causes[JUDGE_ERROR] = causes.get(JUDGE_ERROR, 0) + 1

    base_ids = {str(r["task_id"]) for r in base_runs}
    # A baseline run cut short by a stop code is never replayed, so its task is a baseline
    # problem rather than a missing candidate run.
    replayable = {
        str(r["task_id"]) for r in base_runs if r.get("error_type") not in STOP_CODES
    }
    cand_counts = Counter(str(r["task_id"]) for r in cand_runs)
    missing = len(replayable - cand_counts.keys())
    unexpected = sorted(cand_counts.keys() - base_ids)
    duplicates = sorted(t for t, n in cand_counts.items() if n > 1)
    unscored = len(cand_runs) - cand.scored_count
    baseline_problem = _baseline_problems(base_runs, base)

    changed = {str(r["id"]) for r in cand_runs if r["status"] == "skipped"}
    for s in cand_scores:
        if s["kind"] == "judge" and _verdict_of(s.get("detail")) == JUDGE_SKIPPED:
            changed.add(str(s["run_id"]))

    checks = [
        _pass_rate_check(
            base.pass_rate, cand.pass_rate, thresholds.max_pass_rate_drop_pp
        ),
        _relative_check(
            "Mean total tokens",
            base.mean_total_tokens,
            cand.mean_total_tokens,
            thresholds.max_token_increase_pct,
        ),
        _relative_check(
            "p95 latency (ms)",
            base.p95_latency_ms,
            cand.p95_latency_ms,
            thresholds.max_p95_increase_pct,
        ),
    ]

    if causes or missing or unexpected or duplicates or unscored or baseline_problem:
        verdict = Verdict.ERROR
    elif changed:
        verdict = Verdict.SKIPPED
    elif any(c.status == "FAIL" for c in checks):
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.PASS

    was = _per_task_pass(base_runs, base_scores)
    now = _per_task_pass(cand_runs, cand_scores)
    both = sorted(was.keys() & now.keys())
    return GateResult(
        verdict=verdict,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        baseline=base,
        candidate=cand,
        checks=checks,
        error_causes=causes,
        missing_runs=missing,
        unexpected_tasks=unexpected,
        duplicate_tasks=duplicates,
        unscored_runs=unscored,
        baseline_problem=baseline_problem,
        skipped_requests=len(changed),
        flipped_to_fail=[t for t in both if was[t] and not now[t]],
        flipped_to_pass=[t for t in both if not was[t] and now[t]],
    )


def _load(
    conn: psycopg.Connection[Row], suite_run_id: str
) -> tuple[list[Row], list[Row]]:
    if runs_repo.get_suite_run(conn, suite_run_id) is None:
        raise SuiteNotFoundError(f"suite run not found: {suite_run_id}")
    return (
        runs_repo.list_suite_run_rows(conn, suite_run_id),
        runs_repo.list_suite_score_rows(conn, suite_run_id),
    )


def gate(
    conn: psycopg.Connection[Row],
    baseline_id: str,
    candidate_id: str,
    thresholds: Thresholds,
) -> GateResult:
    """Compare two suite runs; SuiteNotFoundError for an unknown id."""
    base_runs, base_scores = _load(conn, baseline_id)
    cand_runs, cand_scores = _load(conn, candidate_id)
    return evaluate(
        baseline_id,
        candidate_id,
        base_runs,
        base_scores,
        cand_runs,
        cand_scores,
        thresholds,
    )
