"""Multi-suite comparison table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg

from agent_replay.db import runs_repo
from agent_replay.report.gate import SuiteNotFoundError
from agent_replay.report.markdown import escape, table
from agent_replay.report.metrics import METRICS_NOTE, SuiteMetrics, suite_metrics

Row = dict[str, Any]

HEADER = [
    "Suite run",
    "Variant",
    "Pass rate",
    "Mean total tokens",
    "p95 latency (ms)",
    "Errors",
    "Diverged",
    "Skipped",
]


def _cells(suite_id: str, variant: str, m: SuiteMetrics) -> list[str]:
    p95 = "n/a" if m.p95_latency_ms is None else f"{m.p95_latency_ms:.0f}"
    return [
        escape(suite_id),
        escape(variant),
        f"{m.pass_rate * 100:.1f}%",
        f"{m.mean_total_tokens:.1f}",
        p95,
        str(m.error_count),
        str(m.diverged_count),
        str(m.skipped_count),
    ]


def compare(conn: psycopg.Connection[Row], suite_run_ids: Sequence[str]) -> str:
    """One table row per suite run; SuiteNotFoundError for an unknown id."""
    body: list[list[str]] = []
    for sid in suite_run_ids:
        suite = runs_repo.get_suite_run(conn, sid)
        if suite is None:
            raise SuiteNotFoundError(f"suite run not found: {sid}")
        body.append(_cells(sid, suite["variant"], suite_metrics(conn, sid)))
    lines = ["## Suite comparison", "", *table(HEADER, body), ""]
    lines += [METRICS_NOTE, ""]
    return "\n".join(lines)
