from __future__ import annotations

from typing import Any

import pytest

from agent_replay.diff import diff_runs


def model_span(
    seq: int, status: str, text: str = "hi", latency: int = 5
) -> dict[str, Any]:
    call = {
        "id": "c1",
        "name": "search",
        "args": {"q": "a"},
        "signature": f"sig-{status}",
    }
    return {
        "seq": seq,
        "kind": "model",
        "name": "m",
        "payload": {
            "request": {"model": "m"},
            "request_hash": "h1",
            "cache_status": status,
            "latency_ms": latency,
            "response": {
                "text": text,
                "tool_calls": [call],
                "input_tokens": 1,
                "output_tokens": 2,
                "latency_ms": latency,
                "cache_status": status,
                "attempts": 1 if status == "live" else 2,
            },
        },
    }


def tool_span(seq: int, q: str = "a", result: Any = None) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "tool",
        "name": "search",
        "payload": {
            "name": "search",
            "call_id": f"id{seq}",
            "args": {"q": q},
            "result": result or {"ok": 1},
            "fault": None,
            "raised": False,
            "latency_ms": seq,
        },
    }


def run(
    spans: list[dict[str, Any]], scores: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"spans": spans, "scores": scores or []}


def test_record_vs_identical_replay_has_no_change():
    d = diff_runs(
        run([model_span(1, "live", latency=9), tool_span(2)]),
        run([model_span(1, "hit", latency=0), tool_span(2)]),
    )
    assert d.first_divergence is None
    assert not any(s.changed for s in d.steps)


def test_different_response_text_is_changed():
    d = diff_runs(run([model_span(1, "live")]), run([model_span(1, "hit", text="bye")]))
    assert d.first_divergence == 1


def test_different_tool_args_or_result_is_changed():
    assert (
        diff_runs(run([tool_span(1)]), run([tool_span(1, q="b")])).first_divergence == 1
    )
    assert (
        diff_runs(run([tool_span(1)]), run([tool_span(1, result={"ok": 2})]))
        .steps[0]
        .changed
    )


def score(kind: str, passed: bool | None, name: str = "c") -> dict[str, Any]:
    return {"check_name": name, "kind": kind, "passed": passed}


def test_task_pass_is_unknown_without_a_judge_row():
    dets = [score("deterministic", True)]
    ok = diff_runs(run([], dets + [score("judge", True)]), run([], dets))
    assert ok.deltas["task_pass"].left is True
    assert ok.deltas["task_pass"].right is None  # scored with --no-judge


def test_task_pass_null_judge_and_failed_check():
    dets = [score("deterministic", True)]
    d = diff_runs(
        run([], dets + [score("judge", None)]),
        run([], [score("deterministic", False), score("judge", True)]),
    )
    assert d.deltas["task_pass"].left is False
    assert d.deltas["task_pass"].right is False


def test_no_scores_is_none():
    assert diff_runs(run([]), run([])).deltas["task_pass"].left is None


def test_duplicate_seq_raises():
    with pytest.raises(ValueError, match="duplicate"):
        diff_runs(run([tool_span(1), tool_span(1)]), run([]))
