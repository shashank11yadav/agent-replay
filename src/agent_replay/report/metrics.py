"""Suite metrics: pass rates, token means and p95 latency for one suite run."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

import psycopg
from pydantic import BaseModel

from agent_replay.db import runs_repo
from agent_replay.scoring.task_pass import task_passed

Row = dict[str, Any]

# Footnote for every table that shows these metrics.
METRICS_NOTE = "Token means and p95 latency include error and skipped runs."


class SuiteMetrics(BaseModel):
    """Per-suite metrics. Token means and p95 latency include error and skipped runs."""

    task_count: int
    scored_count: int  # runs with both deterministic and judge rows
    pass_rate: float
    deterministic_pass_rate: float
    judge_pass_rate: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_total_tokens: float
    total_tokens: int
    p95_latency_ms: float | None
    mean_model_calls: float
    error_count: int
    diverged_count: int
    skipped_count: int


def p95(values: Sequence[float]) -> float | None:
    """95th percentile (inclusive method); None for no values, the value itself for one."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def compute_metrics(runs: Sequence[Row], scores: Sequence[Row]) -> SuiteMetrics:
    """Metrics from run rows and score rows.

    A task passes by ``task_passed``; a run is scored only when it has both deterministic
    and judge rows, and an unscored run counts as not passed.
    """
    n = len(runs)
    by_run: dict[str, list[Row]] = {str(r["id"]): [] for r in runs}
    for s in scores:
        rows = by_run.get(str(s["run_id"]))
        if rows is not None:
            rows.append(s)
    det = judge = both = scored = 0
    for rows in by_run.values():
        checks = [s for s in rows if s["kind"] != "judge"]
        judges = [s for s in rows if s["kind"] == "judge"]
        det += bool(checks) and all(s["passed"] is True for s in checks)
        judge += bool(judges) and all(s["passed"] is True for s in judges)
        passed = task_passed(rows)
        scored += passed is not None
        both += passed is True
    in_tok = sum(r["input_tokens"] for r in runs)
    out_tok = sum(r["output_tokens"] for r in runs)

    def count(status: str) -> int:
        return sum(1 for r in runs if r["status"] == status)

    return SuiteMetrics(
        task_count=n,
        scored_count=scored,
        pass_rate=_rate(both, n),
        deterministic_pass_rate=_rate(det, n),
        judge_pass_rate=_rate(judge, n),
        mean_input_tokens=_rate(in_tok, n),
        mean_output_tokens=_rate(out_tok, n),
        mean_total_tokens=_rate(in_tok + out_tok, n),
        total_tokens=in_tok + out_tok,
        p95_latency_ms=p95([r["latency_ms"] for r in runs]),
        mean_model_calls=_rate(sum(r["model_calls"] for r in runs), n),
        error_count=count("error"),
        diverged_count=count("diverged"),
        skipped_count=count("skipped"),
    )


def suite_metrics(conn: psycopg.Connection[Row], suite_run_id: str) -> SuiteMetrics:
    runs = runs_repo.list_suite_run_rows(conn, suite_run_id)
    scores = runs_repo.list_suite_score_rows(conn, suite_run_id)
    return compute_metrics(runs, scores)
