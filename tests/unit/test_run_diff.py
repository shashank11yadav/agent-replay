"""Run diff: step alignment, first divergence and run-level deltas.

A loaded run is the run row plus its ``spans`` and, optionally, ``scores``. A step counts
as changed when kind, name or payload differ; bookkeeping columns such as span_id and
started_at are ignored.
"""

from __future__ import annotations

import json
from typing import Any

from agent_replay.diff import diff_runs


def span(
    seq: int, kind: str = "model", name: str = "m", payload: Any = None, **kw: Any
) -> dict:
    base = {
        "seq": seq,
        "kind": kind,
        "name": name,
        "payload": payload if payload is not None else {"response": f"r{seq}"},
        "span_id": f"sp-{seq}",
        "trace_id": "tr",
        "started_at": "2026-01-01T00:00:00Z",
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 100,
        "cost_usd": 0.0,
    }
    base.update(kw)
    return base


def run(spans: list[dict], passed: bool = True, **kw: Any) -> dict:
    base = {
        "id": "run-x",
        "input_tokens": sum(s["input_tokens"] for s in spans),
        "output_tokens": sum(s["output_tokens"] for s in spans),
        "latency_ms": sum(s["latency_ms"] for s in spans),
        "model_calls": sum(1 for s in spans if s["kind"] == "model"),
        "tool_calls": sum(1 for s in spans if s["kind"] == "tool"),
        "spans": spans,
        "scores": [
            {"check_name": "c", "kind": "deterministic", "passed": passed},
            {"check_name": "judge", "kind": "judge", "passed": True},
        ],
    }
    base.update(kw)
    return base


def five() -> list[dict]:
    return [span(i, kind="tool" if i % 2 else "model") for i in range(1, 6)]


def entries(d: Any) -> dict[int, Any]:
    return {e.seq: e for e in d.steps}


def test_identical_runs_have_no_divergence_and_no_change():
    d = diff_runs(run(five()), run(five(), id="run-y"))
    assert d.first_divergence is None
    assert [e.seq for e in d.steps] == [1, 2, 3, 4, 5]
    assert all(e.changed is False and e.status == "same" for e in d.steps)
    for key in (
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "model_calls",
        "tool_calls",
    ):
        assert d.deltas[key].delta == 0, key


def test_span_ids_and_timestamps_are_ignored():
    right = five()
    for s in right:
        s["span_id"] = "other"
        s["trace_id"] = "other"
        s["started_at"] = "2030-01-01T00:00:00Z"
    d = diff_runs(run(five()), run(right))
    assert d.first_divergence is None
    assert not any(e.changed for e in d.steps)


def test_first_divergence_is_seq_3():
    right = five()
    right[2] = span(3, kind="tool", payload={"response": "DIFFERENT"})
    right[4] = span(5, kind="tool", payload={"response": "also different"})
    d = diff_runs(run(five()), run(right))
    assert d.first_divergence == 3
    e = entries(d)
    assert [e[i].changed for i in range(1, 6)] == [False, False, True, False, True]


def test_divergence_by_kind_and_by_tool_args():
    right = five()
    right[1] = span(2, kind="tool")  # seq 2 was model
    assert diff_runs(run(five()), run(right)).first_divergence == 2
    left = [span(1, kind="tool", payload={"args": {"q": "a"}, "result": 1})]
    other = [span(1, kind="tool", payload={"args": {"q": "b"}, "result": 1})]
    assert diff_runs(run(left), run(other)).first_divergence == 1


def test_token_and_latency_deltas_per_step_and_run_level():
    right = five()
    right[1] = span(2, input_tokens=30, output_tokens=2, latency_ms=250)
    d = diff_runs(run(five()), run(right))
    e = entries(d)[2]
    assert (e.input_tokens_delta, e.output_tokens_delta, e.latency_ms_delta) == (
        20,
        -3,
        150,
    )
    assert d.deltas["input_tokens"].delta == 20
    assert d.deltas["output_tokens"].delta == -3
    assert d.deltas["latency_ms"].delta == 150
    assert d.deltas["input_tokens"].left == 50 and d.deltas["input_tokens"].right == 70


def test_model_and_tool_call_deltas():
    left = run(five())
    right = run(
        five(), model_calls=left["model_calls"] + 2, tool_calls=left["tool_calls"] - 1
    )
    d = diff_runs(left, right)
    assert d.deltas["model_calls"].delta == 2
    assert d.deltas["tool_calls"].delta == -1


def test_task_pass_delta_reports_both_sides():
    d = diff_runs(run(five(), passed=True), run(five(), passed=False))
    assert d.deltas["task_pass"].left is True
    assert d.deltas["task_pass"].right is False


def test_right_with_two_extra_steps_lists_them_as_added():
    right = five() + [span(6), span(7)]
    d = diff_runs(run(five()), run(right))
    e = entries(d)
    assert [e[i].status for i in range(1, 6)] == ["same"] * 5
    assert e[6].status == "added" and e[7].status == "added"
    assert len(d.steps) == 7
    assert d.first_divergence is None or d.first_divergence == 6


def test_extra_left_steps_are_removed():
    left = five() + [span(6)]
    d = diff_runs(run(left), run(five()))
    e = entries(d)
    assert e[6].status == "removed"
    assert len(d.steps) == 6


def test_dump_is_json_serialisable():
    right = five()
    right[2] = span(3, payload={"response": "x"})
    d = diff_runs(run(five()), run(right + [span(6)]))
    dumped = d.model_dump()
    assert json.loads(json.dumps(dumped))["first_divergence"] == 3
