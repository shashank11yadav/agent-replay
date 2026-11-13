"""Suite metrics from the database: pass rates, tokens and p95 latency.

judge_pass_rate counts null verdicts as not passed.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

import pytest

from agent_replay.db import runs_repo


def _metrics_mod() -> Any:
    from agent_replay.report import metrics

    return metrics


def _expected_p95(data: list[float]) -> float:
    return statistics.quantiles(data, n=100, method="inclusive")[94]


@pytest.mark.parametrize(
    "data",
    [
        [1, 2],
        [5, 1, 3],
        [10, 20, 30, 40],
        list(range(1, 31)),
        [1000, 5, 77, 300, 300, 12, 9000, 42, 7, 8, 9],
    ],
)
def test_p95_matches_statistics_quantiles(data: list[int]) -> None:
    assert _metrics_mod().p95(data) == pytest.approx(_expected_p95(data))


_STATUSES = ["ok", "ok", "diverged", "error"]
_IN = [100, 200, 300, 400]
_OUT = [50, 70, 30, 50]
_LAT = [1000, 2000, 3000, 4000]
_CALLS = [2, 3, 4, 1]
# (deterministic check passed flags, judge passed, judge verdict detail)
_SCORES: list[tuple[list[bool], bool | None, str | None]] = [
    ([True, True], True, "pass"),
    ([True, True], False, "fail"),
    ([True, False], None, "judge_error"),
    ([True, True], None, "judge_skipped"),
]


def _add_run(
    conn: Any,
    suite: Any,
    i: int,
    *,
    status: str,
    tokens: tuple[int, int],
    latency: int,
    calls: int,
    det: list[bool],
    judge: bool | None,
    verdict: str | None,
    with_judge_row: bool = True,
) -> Any:
    run_id = runs_repo.save_run(
        conn,
        {
            "suite_run_id": suite,
            "task_id": f"task-{i}",
            "variant": "v",
            "mode": "record",
            "status": status,
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "latency_ms": latency,
            "model_calls": calls,
            "started_at": f"2026-01-01T00:00:{i:02d}+00:00",
        },
        [],
    )
    for j, ok in enumerate(det):
        runs_repo.upsert_score(conn, run_id, f"check_{j}", ok, {})
    if with_judge_row:
        runs_repo.upsert_score(
            conn, run_id, "judge", judge, {"verdict": verdict}, kind="judge"
        )
    return run_id


def _suite4(conn: Any) -> Any:
    suite = runs_repo.create_suite_run(conn, task_count=4)
    for i in range(4):
        det, judge, verdict = _SCORES[i]
        _add_run(
            conn,
            suite,
            i,
            status=_STATUSES[i],
            tokens=(_IN[i], _OUT[i]),
            latency=_LAT[i],
            calls=_CALLS[i],
            det=det,
            judge=judge,
            verdict=verdict,
        )
    return suite


def test_suite_metrics_returns_all_fields(conn: Any) -> None:
    m = _metrics_mod().suite_metrics(conn, str(_suite4(conn)))
    dumped = m.model_dump()
    for key in (
        "task_count",
        "pass_rate",
        "deterministic_pass_rate",
        "judge_pass_rate",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "total_tokens",
        "p95_latency_ms",
        "mean_model_calls",
        "error_count",
        "diverged_count",
        "skipped_count",
    ):
        assert key in dumped, key


def test_hand_built_suite_matches_hand_computed(conn: Any) -> None:
    m = _metrics_mod().suite_metrics(conn, str(_suite4(conn)))
    assert m.task_count == 4
    assert m.pass_rate == pytest.approx(0.25)  # only task 0
    assert m.deterministic_pass_rate == pytest.approx(0.75)  # tasks 0, 1, 3
    assert m.judge_pass_rate == pytest.approx(0.25)  # only task 0 judged pass
    assert m.mean_input_tokens == pytest.approx(250)
    assert m.mean_output_tokens == pytest.approx(50)
    assert m.mean_total_tokens == pytest.approx(300)
    assert m.total_tokens == 1200
    assert m.p95_latency_ms == pytest.approx(_expected_p95(_LAT))
    assert m.mean_model_calls == pytest.approx(2.5)
    assert m.error_count == 1
    assert m.diverged_count == 1
    assert m.skipped_count == 0


def test_skipped_runs_are_counted(conn: Any) -> None:
    suite = runs_repo.create_suite_run(conn, task_count=2)
    for i, status in enumerate(["skipped", "ok"]):
        _add_run(
            conn, suite, i, status=status, tokens=(1, 1), latency=10, calls=1,
            det=[True], judge=True, verdict="pass",
        )  # fmt: skip
    m = _metrics_mod().suite_metrics(conn, str(suite))
    assert (m.skipped_count, m.error_count, m.diverged_count) == (1, 0, 0)


def test_pass_requires_every_check_and_judge(conn: Any) -> None:
    suite = runs_repo.create_suite_run(conn, task_count=5)
    cases: list[tuple[list[bool], bool | None, str | None, bool]] = [
        ([True, True], True, "pass", True),
        ([True, False], True, "pass", False),  # a failed deterministic check
        ([True, True], False, "fail", False),
        ([True, True], None, "judge_error", False),
        ([True, True], None, "judge_skipped", False),
    ]
    for i, (det, judge, verdict, _) in enumerate(cases):
        _add_run(
            conn, suite, i, status="ok", tokens=(1, 1), latency=10, calls=1,
            det=det, judge=judge, verdict=verdict,
        )  # fmt: skip
    m = _metrics_mod().suite_metrics(conn, str(suite))
    assert m.task_count == 5
    assert m.pass_rate == pytest.approx(1 / 5)
    assert m.deterministic_pass_rate == pytest.approx(4 / 5)


def test_run_without_judge_row_is_not_passed(conn: Any) -> None:
    suite = runs_repo.create_suite_run(conn, task_count=2)
    _add_run(
        conn, suite, 0, status="ok", tokens=(1, 1), latency=10, calls=1,
        det=[True], judge=True, verdict="pass",
    )  # fmt: skip
    _add_run(
        conn, suite, 1, status="ok", tokens=(1, 1), latency=10, calls=1,
        det=[True], judge=None, verdict=None, with_judge_row=False,
    )  # fmt: skip
    m = _metrics_mod().suite_metrics(conn, str(suite))
    assert m.pass_rate == pytest.approx(0.5)


def test_metrics_ignore_other_suites(conn: Any) -> None:
    other = runs_repo.create_suite_run(conn, task_count=1)
    _add_run(
        conn, other, 9, status="error", tokens=(9999, 9999), latency=99999, calls=50,
        det=[False], judge=False, verdict="fail",
    )  # fmt: skip
    m = _metrics_mod().suite_metrics(conn, str(_suite4(conn)))
    assert m.task_count == 4
    assert m.total_tokens == 1200
    assert m.error_count == 1


def test_single_run_and_empty_suite_latency(conn: Any) -> None:
    mod = _metrics_mod()
    one = runs_repo.create_suite_run(conn, task_count=1)
    _add_run(
        conn, one, 0, status="ok", tokens=(5, 5), latency=777, calls=1,
        det=[True], judge=True, verdict="pass",
    )  # fmt: skip
    assert mod.suite_metrics(conn, str(one)).p95_latency_ms == 777
    empty = runs_repo.create_suite_run(conn)
    m = mod.suite_metrics(conn, str(empty))
    assert m.task_count == 0
    assert m.p95_latency_ms is None
    json.dumps(m.model_dump())


def test_metrics_is_pydantic_and_json_serialisable(conn: Any) -> None:
    from pydantic import BaseModel

    m = _metrics_mod().suite_metrics(conn, str(_suite4(conn)))
    assert isinstance(m, BaseModel)
    dumped = m.model_dump()
    assert json.loads(json.dumps(dumped)) == dumped
