"""Deterministic checks and the scoring service against a real database."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

CHECKS = ["expected_text", "must_call_tools", "max_steps", "completed"]


def checks() -> Any:
    from agent_replay.scoring import checks as c

    return c


def make_task(**over: Any) -> Any:
    from agent_replay.taskspec import TaskSpec

    base: dict[str, Any] = {
        "id": "t1",
        "prompt": "p",
        "category": "lookup",
        "expected_text": ["Shipped", "42.50"],
        "must_call_tools": ["get_order"],
        "max_steps": 3,
        "rubric": "r",
    }
    base.update(over)
    return TaskSpec(**base)


def tool_span(seq: int, tool: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "tool",
        "name": f"execute_tool {tool}",
        "attributes": {"gen_ai.tool.name": tool},
    }


def model_span(seq: int) -> dict[str, Any]:
    return {"seq": seq, "kind": "model", "name": "generate_content m", "attributes": {}}


def run_row(**over: Any) -> dict[str, Any]:
    r: dict[str, Any] = {
        "final_output": "Order is SHIPPED, total 42.50 EUR",
        "status": "ok",
        "model_calls": 2,
    }
    r.update(over)
    return r


SPANS = [model_span(1), tool_span(2, "get_order"), model_span(3)]


def missing(res: Any) -> list[str]:
    return sorted(res.detail["missing"])


# ---- pure checks -------------------------------------------------------------------


def test_expected_text_passes_case_insensitively() -> None:
    res = checks().expected_text(make_task(), run_row(), SPANS)
    assert res.passed is True


def test_expected_text_fails_listing_missing() -> None:
    res = checks().expected_text(
        make_task(expected_text=["shipped", "refund", "99.99"]), run_row(), SPANS
    )
    assert res.passed is False
    assert missing(res) == ["99.99", "refund"]


def test_expected_text_partial_match_still_fails() -> None:
    res = checks().expected_text(
        make_task(expected_text=["shipped", "nope"]), run_row(), SPANS
    )
    assert res.passed is False
    assert missing(res) == ["nope"]


def test_expected_text_fails_when_final_output_none() -> None:
    res = checks().expected_text(make_task(), run_row(final_output=None), SPANS)
    assert res.passed is False


def test_must_call_tools_passes_when_all_called() -> None:
    task = make_task(must_call_tools=["get_order", "convert_currency"])
    spans = [*SPANS, tool_span(4, "convert_currency")]
    assert checks().must_call_tools(task, run_row(), spans).passed is True


def test_must_call_tools_fails_listing_missing() -> None:
    task = make_task(
        must_call_tools=["get_order", "convert_currency", "search_products"]
    )
    res = checks().must_call_tools(task, run_row(), SPANS)
    assert res.passed is False
    assert missing(res) == ["convert_currency", "search_products"]


def test_must_call_tools_ignores_model_spans_and_names() -> None:
    # a model span whose name mentions the tool must not count as a tool call
    spans = [{"seq": 1, "kind": "model", "name": "get_order", "attributes": {}}]
    res = checks().must_call_tools(make_task(), run_row(), spans)
    assert res.passed is False
    assert missing(res) == ["get_order"]


def test_must_call_tools_empty_requirement_passes() -> None:
    assert checks().must_call_tools(make_task(must_call_tools=[]), run_row(), []).passed


@pytest.mark.parametrize(
    ("calls", "limit", "expected"),
    [(2, 3, True), (3, 3, True), (4, 3, False), (0, 1, True)],
)
def test_max_steps_boundary(calls: int, limit: int, expected: bool) -> None:
    res = checks().max_steps(
        make_task(max_steps=limit), run_row(model_calls=calls), SPANS
    )
    assert res.passed is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [("ok", True), ("error", False), ("diverged", False), ("skipped", False)],
)
def test_completed_only_for_ok(status: str, expected: bool) -> None:
    assert (
        checks().completed(make_task(), run_row(status=status), SPANS).passed
        is expected
    )


# ---- service -----------------------------------------------------------------------


def store_run(conn: Any, spans: list[dict[str, Any]] | None = None, **over: Any) -> str:
    from agent_replay.db import runs_repo

    run = {
        "id": str(uuid.uuid4()),
        "task_id": "t1",
        "variant": "v",
        "mode": "record",
        "status": "ok",
        "final_output": "Order is SHIPPED, total 42.50 EUR",
        "model_calls": 2,
        "tool_calls": 1,
    }
    run.update(over)
    full = []
    for s in SPANS if spans is None else spans:
        full.append({**s, "trace_id": "tr", "span_id": f"s{s['seq']}"})
    runs_repo.save_run(conn, run, full)
    return str(run["id"])


def rows(raw: Any, run_id: str) -> dict[str, tuple[Any, ...]]:
    cur = raw.execute(
        "SELECT check_name, kind, passed, detail FROM scores WHERE run_id = %s",
        (run_id,),
    )
    return {r[0]: r for r in cur.fetchall()}


def test_score_run_upserts_one_row_per_check_and_summarises(
    conn: Any, raw: Any
) -> None:
    from agent_replay.scoring.service import score_run

    run_id = store_run(conn)
    summary = score_run(conn, run_id, make_task(), judge=None)
    stored = rows(raw, run_id)
    assert sorted(stored) == sorted(CHECKS)
    assert all(r[1] == "deterministic" and r[2] is True for r in stored.values())
    assert summary["deterministic_pass"] is True
    for name in CHECKS:
        assert summary["checks"][name]["passed"] is True


def test_score_run_overall_false_and_failures_stored(conn: Any, raw: Any) -> None:
    from agent_replay.scoring.service import score_run

    run_id = store_run(conn, status="error", final_output="nothing useful")
    summary = score_run(conn, run_id, make_task(), judge=None)
    stored = rows(raw, run_id)
    assert summary["deterministic_pass"] is False
    assert summary["checks"]["completed"]["passed"] is False
    assert summary["checks"]["expected_text"]["passed"] is False
    assert summary["checks"]["must_call_tools"]["passed"] is True
    assert summary["checks"]["max_steps"]["passed"] is True
    assert stored["completed"][2] is False
    assert stored["expected_text"][2] is False
    assert stored["must_call_tools"][2] is True
    assert sorted(stored["expected_text"][3]["missing"]) == ["42.50", "Shipped"]


def test_score_run_twice_leaves_one_row_per_check(conn: Any, raw: Any) -> None:
    from agent_replay.scoring.service import score_run

    run_id = store_run(conn)
    task = make_task()
    score_run(conn, run_id, task, judge=None)
    score_run(conn, run_id, task, judge=None)
    n = raw.execute(
        "SELECT count(*) FROM scores WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    assert n == len(CHECKS)


def test_rescoring_with_changed_task_updates_rows(conn: Any, raw: Any) -> None:
    from agent_replay.scoring.service import score_run

    run_id = store_run(conn)
    score_run(conn, run_id, make_task(), judge=None)
    summary = score_run(conn, run_id, make_task(max_steps=1), judge=None)
    assert summary["deterministic_pass"] is False
    stored = rows(raw, run_id)
    assert stored["max_steps"][2] is False
    assert len(stored) == len(CHECKS)


def test_score_run_reads_tool_spans_from_db(conn: Any, raw: Any) -> None:
    from agent_replay.scoring.service import score_run

    run_id = store_run(conn, spans=[model_span(1)], tool_calls=0)
    summary = score_run(conn, run_id, make_task(), judge=None)
    assert summary["checks"]["must_call_tools"]["passed"] is False
    assert rows(raw, run_id)["must_call_tools"][2] is False
