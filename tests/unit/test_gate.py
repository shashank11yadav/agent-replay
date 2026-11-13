"""Unit tests for gate verdict precedence and threshold boundaries (no database)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_replay.report.gate import Thresholds, Verdict, evaluate, load_thresholds
from agent_replay.report.markdown import escape, render_report

TH = Thresholds(
    max_pass_rate_drop_pp=5, max_token_increase_pct=15, max_p95_increase_pct=20
)


def _run(i: int, **kw: Any) -> dict[str, Any]:
    base = {
        "id": f"r{i}",
        "task_id": f"t{i}",
        "status": "ok",
        "error_type": None,
        "error": None,
        "input_tokens": 60,
        "output_tokens": 40,
        "latency_ms": 1000,
        "model_calls": 1,
    }
    return {**base, **kw}


def _scores(runs: list[dict[str, Any]], bad: set[str] = frozenset()) -> list[Any]:  # type: ignore[assignment]
    out: list[dict[str, Any]] = []
    for r in runs:
        ok = r["id"] not in bad
        out.append(
            {"run_id": r["id"], "kind": "deterministic", "passed": ok, "detail": {}}
        )
        out.append(
            {
                "run_id": r["id"],
                "kind": "judge",
                "passed": ok,
                "detail": {"verdict": "x"},
            }
        )
    return out


def _gate(base: list[dict[str, Any]], cand: list[dict[str, Any]], **kw: Any) -> Any:
    return evaluate(
        "b", "c", base, _scores(base), cand, kw.get("scores") or _scores(cand), TH
    )


def test_equal_is_pass() -> None:
    runs = [_run(i) for i in range(4)]
    assert _gate(runs, runs).verdict is Verdict.PASS


def test_zero_baseline_tokens() -> None:
    zero = [_run(0, input_tokens=0, output_tokens=0)]
    assert _gate(zero, zero).verdict is Verdict.PASS
    assert _gate(zero, [_run(0)]).verdict is Verdict.FAIL


def test_none_p95_is_not_a_failure() -> None:
    r = evaluate("b", "c", [], [], [], [], TH)
    assert r.checks[2].status == "N/A"


def test_pass_rate_boundary() -> None:
    base = [_run(i) for i in range(20)]
    one_down = _scores(base, {"r0"})  # exactly 5pp
    two_down = _scores(base, {"r0", "r1"})
    assert (
        evaluate("b", "c", base, _scores(base), base, one_down, TH).verdict
        is Verdict.PASS
    )
    assert (
        evaluate("b", "c", base, _scores(base), base, two_down, TH).verdict
        is Verdict.FAIL
    )


def test_precedence() -> None:
    base = [_run(i) for i in range(3)]
    cand = [_run(0, status="skipped"), _run(1, latency_ms=9999), _run(2)]
    assert _gate(base, cand).verdict is Verdict.SKIPPED
    cand[1]["error_type"] = "budget_exceeded"
    assert _gate(base, cand).verdict is Verdict.ERROR
    assert _gate(base, cand[:2]).verdict is Verdict.ERROR


def test_judge_skipped_gates_skipped_and_counts_once() -> None:
    base = [_run(0)]
    cand = [_run(0, status="skipped")]
    scores = _scores(cand)[:1] + [
        {
            "run_id": "r0",
            "kind": "judge",
            "passed": None,
            "detail": {"verdict": "judge_skipped"},
        }
    ]
    r = evaluate("b", "c", base, _scores(base), cand, scores, TH)
    assert r.verdict is Verdict.SKIPPED
    assert r.skipped_requests == 1


def test_escape_blocks_table_breakout() -> None:
    assert "|" not in escape("a|b").replace("\\|", "")
    assert "\n" not in escape("a\nb")
    assert escape("<b>") == "\\<b\\>"


def test_report_escapes_task_ids() -> None:
    base = [_run(0, task_id="x|y")]
    cand = [_run(0, task_id="x|y")]
    r = evaluate("b", "c", base, _scores(base), cand, _scores(cand, {"r0"}), TH)
    assert "x\\|y" in render_report(r)


def test_load_thresholds_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps([1]))
    with pytest.raises(ValueError):
        load_thresholds(p)


def test_load_thresholds_rejects_nan(tmp_path: Path) -> None:
    # NaN compares false against everything, so a NaN limit would never fail a check.
    p = tmp_path / "t.json"
    p.write_text('{"max_token_increase_pct": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_thresholds(p)


def test_empty_baseline_is_error() -> None:
    r = evaluate("b", "c", [], [], [], [], TH)
    assert r.verdict is Verdict.ERROR
    assert "Baseline" in render_report(r)


def test_unscored_candidate_is_error() -> None:
    base = [_run(i) for i in range(2)]
    cand = [_run(i) for i in range(2)]
    r = evaluate("b", "c", base, _scores(base), cand, _scores(cand[:1]), TH)
    assert r.verdict is Verdict.ERROR
    assert "suite score --suite-run c" in render_report(r)


def test_unscored_baseline_is_error() -> None:
    base = [_run(i) for i in range(2)]
    r = evaluate("b", "c", base, _scores(base[:1]), base, _scores(base), TH)
    assert r.verdict is Verdict.ERROR
    assert r.baseline_problem


def test_judge_skipped_on_ok_run_is_skipped() -> None:
    base = [_run(0)]
    cand = [_run(0)]
    scores = _scores(cand)[:1] + [
        {
            "run_id": "r0",
            "kind": "judge",
            "passed": None,
            "detail": {"verdict": "judge_skipped"},
        }
    ]
    r = evaluate("b", "c", base, _scores(base), cand, scores, TH)
    assert r.verdict is Verdict.SKIPPED
    assert r.skipped_requests == 1


def test_escape_neutralises_mentions() -> None:
    assert "@" not in escape("@octocat")


def _judge(run_id: str, verdict: str, error: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": "judge",
        "passed": None,
        "detail": {"verdict": verdict, "error": error},
    }


def test_candidate_with_other_tasks_but_same_count_is_error() -> None:
    base = [_run(0, task_id="a"), _run(1, task_id="b", input_tokens=10000)]
    cand = [_run(0, task_id="a"), _run(1, task_id="c")]
    r = evaluate("b", "c", base, _scores(base, {"r1"}), cand, _scores(cand), TH)
    assert r.verdict is Verdict.ERROR
    assert r.missing_runs == 1
    assert r.unexpected_tasks == ["c"]
    assert "Unexpected tasks" in render_report(r)


def test_duplicate_candidate_task_is_error() -> None:
    base = [_run(0, task_id="a"), _run(1, task_id="b")]
    cand = [_run(0, task_id="a"), _run(1, task_id="b"), _run(2, task_id="b")]
    r = _gate(base, cand)
    assert r.verdict is Verdict.ERROR
    assert r.duplicate_tasks == ["b"]
    assert r.missing_runs == 0
    assert "Duplicate tasks" in render_report(r)


def test_judge_provider_error_is_error_not_fail() -> None:
    base = [_run(i) for i in range(2)]
    scores = _scores(base)[:3] + [_judge("r1", "judge_error", "RateLimitError")]
    r = evaluate("b", "c", base, _scores(base), base, scores, TH)
    assert r.verdict is Verdict.ERROR
    assert r.error_causes == {"judge_error": 1}
    assert "Judge error" in render_report(r)


def test_unparseable_judge_reply_is_a_plain_fail() -> None:
    base = [_run(i) for i in range(2)]
    scores = _scores(base)[:3] + [_judge("r1", "judge_error", "unparseable verdict")]
    r = evaluate("b", "c", base, _scores(base), base, scores, TH)
    assert r.verdict is Verdict.FAIL
    assert r.error_causes == {}


def test_transient_provider_failure_is_error_not_fail() -> None:
    base = [_run(i) for i in range(2)]
    cand = [_run(0), _run(1, status="error", error_type="transient_error")]
    r = evaluate("b", "c", base, _scores(base), cand, _scores(cand, {"r1"}), TH)
    assert r.verdict is Verdict.ERROR
    assert r.error_causes == {"transient_error": 1}
    assert "Provider error" in render_report(r)


def test_suites_scored_without_judge_are_error() -> None:
    base = [_run(i) for i in range(2)]
    cand = [_run(i, status="error") for i in range(2)]

    def deterministic(runs: list[dict[str, Any]], ok: bool) -> list[dict[str, Any]]:
        return [
            {"run_id": r["id"], "kind": "deterministic", "passed": ok, "detail": {}}
            for r in runs
        ]

    r = evaluate(
        "b", "c", base, deterministic(base, True), cand, deterministic(cand, False), TH
    )
    assert r.verdict is Verdict.ERROR
    assert r.unscored_runs == 2
    assert r.baseline_problem == "the baseline suite is not fully scored"


def test_duplicate_baseline_task_is_a_baseline_error() -> None:
    base = [_run(0, task_id="a"), _run(1, task_id="b"), _run(2, task_id="b")]
    cand = [_run(0, task_id="a"), _run(1, task_id="b")]
    r = _gate(base, cand)
    assert r.verdict is Verdict.ERROR
    assert r.baseline_problem == "the baseline suite has more than one run for: b"
    report = render_report(r)
    assert "- Baseline:" in report
    assert "Missing runs" not in report


def test_skipped_baseline_run_is_a_baseline_error() -> None:
    base = [_run(0), _run(1, status="skipped")]
    r = _gate(base, [_run(0), _run(1)])
    assert r.verdict is Verdict.ERROR
    assert r.baseline_problem == "the baseline suite has 1 skipped run(s)"
    assert "Re-record the baseline" in render_report(r)


def test_stopped_baseline_run_blames_the_baseline() -> None:
    # A run cut short by a stop code is never replayed, so the candidate has no run for it.
    # That gap is the baseline's fault, not a missing candidate run.
    base = [_run(0), _run(1, status="error", error_type="rate_limit")]
    cand = [_run(0)]
    r = _gate(base, cand)
    assert r.verdict is Verdict.ERROR
    assert r.baseline_problem == (
        "the baseline suite has run(s) stopped by a provider or budget error (rate_limit)"
    )
    assert r.missing_runs == 0
    report = render_report(r)
    assert "Missing runs" not in report
    assert "re-run the job" not in report.lower()


def test_several_baseline_problems_are_all_named() -> None:
    base = [_run(0, status="skipped"), _run(1), _run(2, task_id="t1")]
    r = evaluate("b", "c", base, _scores(base[:2]), base, _scores(base), TH)
    assert r.baseline_problem == (
        "the baseline suite has more than one run for: t1; "
        "the baseline suite has 1 skipped run(s); "
        "the baseline suite is not fully scored"
    )


def test_duplicate_task_is_left_out_of_flips() -> None:
    base = [_run(0, task_id="a")]
    cand = [_run(0, task_id="a"), _run(1, task_id="a")]
    r = evaluate("b", "c", base, _scores(base), cand, _scores(cand, {"r1"}), TH)
    assert r.verdict is Verdict.ERROR
    assert r.flipped_to_fail == []
