"""Unit tests for the pure metric math (no database)."""

from __future__ import annotations

import json

import pytest

from agent_replay.report.metrics import compute_metrics, p95


def _run(
    i: str, status: str = "ok", tin: int = 1, tout: int = 1, lat: int = 10
) -> dict:
    return {
        "id": i,
        "status": status,
        "input_tokens": tin,
        "output_tokens": tout,
        "latency_ms": lat,
        "model_calls": 1,
    }


def _s(rid: str, passed: bool | None, kind: str = "deterministic") -> dict:
    return {
        "run_id": rid,
        "check_name": kind + str(passed),
        "kind": kind,
        "passed": passed,
    }


def test_p95_edges() -> None:
    assert p95([]) is None
    assert p95([7]) == 7
    assert p95([1, 2]) == pytest.approx(1.95)


def test_empty_suite_has_no_division_by_zero() -> None:
    m = compute_metrics([], [])
    assert m.task_count == 0
    assert m.pass_rate == 0.0
    assert m.p95_latency_ms is None
    json.dumps(m.model_dump())


def test_pass_needs_deterministic_and_judge() -> None:
    runs = [_run("a"), _run("b"), _run("c")]
    scores = [
        _s("a", True),
        _s("a", True, "judge"),
        _s("b", False),
        _s("b", True, "judge"),
        _s("c", True),
        _s("c", None, "judge"),
    ]
    m = compute_metrics(runs, scores)
    assert m.pass_rate == pytest.approx(1 / 3)
    assert m.deterministic_pass_rate == pytest.approx(2 / 3)
    assert m.judge_pass_rate == pytest.approx(2 / 3)


def test_token_totals_and_status_counts() -> None:
    m = compute_metrics([_run("a", "error", 10, 5), _run("b", "skipped", 20, 5)], [])
    assert m.total_tokens == 40
    assert m.mean_total_tokens == 20
    assert (m.error_count, m.skipped_count, m.diverged_count) == (1, 1, 0)


def test_unscored_suite_is_not_healthy() -> None:
    m = compute_metrics([_run("a"), _run("b")], [])
    assert m.scored_count == 0
    assert m.deterministic_pass_rate == 0.0
    assert m.pass_rate == 0.0


def test_judge_only_run_is_not_deterministically_passed() -> None:
    m = compute_metrics([_run("a")], [_s("a", True, "judge")])
    assert m.scored_count == 0
    assert m.judge_pass_rate == 1.0
    assert m.deterministic_pass_rate == 0.0
    assert m.pass_rate == 0.0


def test_run_scored_without_a_judge_is_not_scored() -> None:
    m = compute_metrics([_run("a")], [_s("a", True)])
    assert m.scored_count == 0
    assert m.deterministic_pass_rate == 1.0
    assert m.pass_rate == 0.0
