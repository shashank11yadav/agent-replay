"""Markdown rendering for the gate report. All dynamic text goes through ``escape``."""

from __future__ import annotations

import re

from agent_replay import codes
from agent_replay.report.gate import GateResult, MetricCheck, Verdict
from agent_replay.report.metrics import METRICS_NOTE

_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~&])")


def escape(text: object) -> str:
    """Make text safe inside a table cell or list item (no markup, no line breaks)."""
    flat = " ".join(str(text).split())
    return _SPECIAL.sub(r"\\\1", flat).replace("@", "&#64;")


def table(header: list[str], body: list[list[str]]) -> list[str]:
    """Rows are already-escaped cells."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return lines


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _check_cells(c: MetricCheck) -> list[str]:
    if c.unit == "pp":
        base = f"{_num((c.baseline or 0.0) * 100)}%"
        cand = f"{_num((c.candidate or 0.0) * 100)}%"
        threshold = f"drop at most {c.threshold:g} pp"
    else:
        base, cand = _num(c.baseline), _num(c.candidate)
        threshold = f"rise at most {c.threshold:g}%"
    if c.delta is None:
        delta = f"n/a ({c.note})" if c.note else "n/a"
    else:
        delta = f"{c.delta:+.1f} {c.unit}"
    return [
        escape(c.metric),
        escape(base),
        escape(cand),
        escape(delta),
        escape(threshold),
        c.status,
    ]


_CAUSE_TEXT = {
    codes.RATE_LIMIT: (
        "Rate limit",
        (
            "the provider kept answering 429 after all attempts. "
            "Re-run the job after the rate limit window has passed."
        ),
    ),
    codes.QUOTA_EXHAUSTED: (
        "Request quota",
        "the provider request quota ran out. Re-run the job once the quota resets.",
    ),
    codes.BUDGET_EXCEEDED: (
        "Run budget",
        "the run budget was used up. Raise AGENT_REPLAY_RUN_BUDGET_USD or re-run the job.",
    ),
    codes.TRANSIENT_ERROR: (
        "Provider error",
        (
            "the provider kept failing (server error or timeout) after all attempts. "
            "Re-run the job; this is not a regression in the agent."
        ),
    ),
    codes.JUDGE_ERROR: (
        "Judge error",
        (
            "the judge model call failed after all attempts, so the run has no verdict. "
            "Re-run the job; this is not a regression in the agent."
        ),
    ),
}


def _error_lines(r: GateResult) -> list[str]:
    lines = [
        "The gate cannot compare these suites, so no pass/fail result is given.",
        "",
    ]
    for code, count in sorted(r.error_causes.items()):
        title, text = _CAUSE_TEXT[code]
        lines.append(f"- {title}: {count} candidate row(s) stopped because {text}")
    if r.baseline_problem:
        lines.append(
            f"- Baseline: {escape(r.baseline_problem)}. The fault is in the baseline, "
            "not in this change. Re-record the baseline, or score it "
            f"(`agent-replay suite score --suite-run {r.baseline_id}`) if it is only "
            "missing scores, and export it again."
        )
    if r.unscored_runs:
        lines.append(
            f"- Candidate not fully scored: {r.unscored_runs} run(s) lack deterministic "
            "or judge scores. "
            f"Run `agent-replay suite score --suite-run {r.candidate_id}` (with the "
            "judge) and re-run the gate."
        )
    if r.missing_runs:
        lines.append(
            f"- Missing runs: {r.missing_runs} baseline task(s) have no candidate run. "
            "Re-run the job so every task is replayed."
        )
    if r.unexpected_tasks:
        lines.append(
            "- Unexpected tasks: the candidate has runs for tasks the baseline does not "
            "have: " + ", ".join(escape(t) for t in r.unexpected_tasks) + "."
        )
    if r.duplicate_tasks:
        lines.append(
            "- Duplicate tasks: the candidate has more than one run for: "
            + ", ".join(escape(t) for t in r.duplicate_tasks)
            + "."
        )
    return lines


def _flip_lines(title: str, tasks: list[str]) -> list[str]:
    lines = [f"**{title}** ({len(tasks)})", ""]
    lines += [f"- {escape(t)}" for t in tasks] or ["- none"]
    return lines


def render_report(r: GateResult) -> str:
    """The gate report; for SKIPPED the first line is the notice."""
    out: list[str] = []
    if r.verdict is Verdict.SKIPPED:
        out += [
            (
                f"Notice: {r.skipped_requests} model request(s) changed and could not "
                "be replayed without a live model provider; a maintainer must run the "
                "gate locally with one, or the PR must carry its recordings in "
                "evals/candidate/."
            ),
            "",
        ]
    out += [f"## Regression gate: {r.verdict.value}", ""]
    if r.verdict is Verdict.ERROR:
        out += _error_lines(r) + [""]
    out += table(
        ["Metric", "Baseline", "Candidate", "Delta", "Threshold", "Status"],
        [_check_cells(c) for c in r.checks],
    )
    out += [
        "",
        METRICS_NOTE,
        "",
        *_flip_lines("Flipped pass to fail", r.flipped_to_fail),
        "",
        *_flip_lines("Flipped fail to pass", r.flipped_to_pass),
        "",
    ]
    return "\n".join(out)
