"""Unit tests for fixtures: stable serialisation, reference collection, verify, validation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agent_replay import fixtures, jsonio
from agent_replay.fixtures import (
    FixtureError,
    RunRow,
    SpanRow,
    SuiteRow,
    collect_hashes,
    dumps,
    load_bundle,
)

SUITE = "11111111-1111-4111-8111-111111111111"
RUN = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def suite_row() -> dict[str, Any]:
    return {
        "id": SUITE, "label": "", "variant": "v", "mode": "record", "status": "complete",
        "tasks_path": None, "tasks_sha256": "ab", "task_count": 1,
        "source_suite_run_id": None, "faults": [], "metrics": None,
        "created_at": NOW, "finished_at": None,
    }  # fmt: skip


def run_row(**kw: Any) -> dict[str, Any]:
    base = {
        "id": RUN, "suite_run_id": SUITE, "source_run_id": None, "task_id": "t",
        "variant": "v", "mode": "record", "status": "ok", "error_type": None,
        "error": None, "final_output": "x", "faults": [], "input_tokens": 0,
        "output_tokens": 0, "latency_ms": 0, "model_calls": 0, "tool_calls": 0,
        "cache_hits": 0, "cache_misses": 0, "cost_usd": 0.0, "started_at": NOW,
        "finished_at": None,
    }  # fmt: skip
    return {**base, **kw}


def span(kind: str, payload: dict[str, Any], seq: int = 0) -> SpanRow:
    return SpanRow.model_validate(
        {
            "id": seq + 1, "run_id": RUN, "seq": seq, "trace_id": "", "span_id": "",
            "parent_span_id": None, "kind": kind, "name": "", "status": "ok",
            "error_type": None, "attributes": {}, "payload": payload,
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "cost_usd": 0.0,
            "cache_status": None, "started_at": NOW,
        }
    )  # fmt: skip


def test_dumps_is_sorted_compact_and_unicode() -> None:
    assert (
        dumps({"b": 1, "a": {"d": "é", "c": [1, 2]}})
        == '{"a":{"c":[1,2],"d":"é"},"b":1}'
    )


def test_dumps_rejects_nan() -> None:
    with pytest.raises(ValueError):
        dumps({"x": float("nan")})


def test_datetimes_are_normalised_to_utc() -> None:
    local = datetime(2026, 11, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    row = SuiteRow.model_validate({**suite_row(), "created_at": local})
    assert row.model_dump(mode="json")["created_at"] == "2026-11-01T12:00:00Z"


def test_naive_datetime_and_bad_uuid_are_rejected() -> None:
    with pytest.raises(ValueError):
        SuiteRow.model_validate({**suite_row(), "created_at": datetime(2026, 1, 1)})
    with pytest.raises(ValueError):
        RunRow.model_validate(run_row(id="../../etc/passwd"))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        RunRow.model_validate(run_row(surprise=1))


def test_collect_hashes_skips_unanswered_and_non_model_spans() -> None:
    spans = [
        span("model", {"request_hash": "h1", "response": {"text": "a"}}, 0),
        span("model", {"request_hash": "h2", "response": None}, 1),
        span("tool", {"request_hash": "h3", "response": {}}, 2),
    ]
    assert collect_hashes(spans) == {RUN: ["h1"]}


def _score(kind: str) -> dict[str, Any]:
    name = "judge" if kind == "judge" else "output_contains"
    return {
        "run_id": RUN, "check_name": name, "kind": kind, "passed": True,
        "detail": {}, "created_at": NOW,
    }  # fmt: skip


SCORED = [_score("deterministic"), _score("judge")]


def _bundle(
    spans: list[SpanRow],
    cache: list[str],
    judge: dict[str, str],
    scores: list[dict[str, Any]] = SCORED,
) -> Any:
    entry = {
        "provider": "", "model": "", "request": {}, "response": {}, "input_tokens": 0,
        "output_tokens": 0, "latency_ms": 0, "cost_usd": 0.0, "created_at": NOW,
    }  # fmt: skip
    manifest = {
        "format_version": 1, "suite_run_id": SUITE, "variant": "v", "task_count": 1,
        "run_count": 1, "tasks_sha256": None, "suite_run": suite_row(),
        "judge_requests": judge,
    }  # fmt: skip
    return fixtures.Bundle.model_validate(
        {
            "manifest": manifest,
            "runs": [run_row()],
            "spans": [s.model_dump() for s in spans],
            "scores": scores,
            "cache": [{**entry, "request_hash": h} for h in cache],
        }
    )


def test_verify_complete_bundle_has_no_gaps() -> None:
    spans = [
        span("model", {"request_hash": "h1", "response": {}}, 0),
        span("tool", {"name": "get_order", "result": {"ok": 1}}, 1),
    ]
    assert fixtures.verify_bundle(_bundle(spans, ["h1", "j1"], {RUN: "j1"})) == {}


def test_verify_lists_missing_model_judge_and_tool_gaps_by_run() -> None:
    spans = [
        span("model", {"request_hash": "h1", "response": {}}, 0),
        span("tool", {"name": "get_order", "result": None}, 1),
        span("tool", {"name": "lookup", "result": None, "raised": True}, 2),
    ]
    gaps = fixtures.verify_bundle(_bundle(spans, [], {RUN: "j1"}))
    assert list(gaps) == [RUN]
    assert sorted(gaps[RUN]) == [
        "missing judge cache entry j1",
        "missing model cache entry h1",
        "missing tool result for get_order",
    ]


def test_load_bundle_rejects_malformed_row_with_clean_error(tmp_path: Path) -> None:
    good = _bundle([], [], {})
    write_bundle(tmp_path, good)
    for name in ("runs", "spans", "scores", "cache"):
        (tmp_path / f"{name}.jsonl").write_text("")
    (tmp_path / "runs.jsonl").write_text('{"id": "x", "evil": "__import__(1)"}\n')
    with pytest.raises(FixtureError, match=r"runs\.jsonl:1: invalid row"):
        load_bundle(tmp_path)


def test_load_bundle_rejects_span_for_unknown_run(tmp_path: Path) -> None:
    good = _bundle([], [], {})
    (tmp_path / "manifest.json").write_text(
        json.dumps(good.manifest.model_dump(mode="json"))
    )
    stray = span("model", {}).model_dump(mode="json") | {"run_id": SUITE}
    (tmp_path / "runs.jsonl").write_text(
        dumps(good.runs[0].model_dump(mode="json")) + "\n"
    )
    (tmp_path / "spans.jsonl").write_text(dumps(stray) + "\n")
    (tmp_path / "scores.jsonl").write_text("")
    (tmp_path / "cache.jsonl").write_text("")
    with pytest.raises(FixtureError, match="span refers to a run"):
        load_bundle(tmp_path)


def test_clean_strips_control_characters_and_defuses_workflow_commands() -> None:
    assert fixtures.clean("a\nb\x1b[31m::error::x") == "ab[31m: :error: :x"


def test_judge_needs_entry_only_for_model_responses() -> None:
    base = {"run_id": RUN, "check_name": "judge", "kind": "judge", "created_at": NOW}

    def score(passed: bool | None, detail: dict[str, Any]) -> fixtures.ScoreRow:
        return fixtures.ScoreRow.model_validate(
            {**base, "passed": passed, "detail": detail}
        )

    assert fixtures.judge_needs_entry(score(True, {}))
    assert fixtures.judge_needs_entry(
        score(None, {"verdict": "judge_error", "error": "unparseable verdict"})
    )
    assert not fixtures.judge_needs_entry(score(None, {"verdict": "judge_skipped"}))
    assert not fixtures.judge_needs_entry(
        score(None, {"verdict": "judge_error", "error": "TransientError"})
    )


def test_span_id_must_be_positive_and_bounded() -> None:
    for bad in (0, -1, 2**53):
        with pytest.raises(ValueError):
            SpanRow.model_validate({**span("model", {}).model_dump(), "id": bad})


def test_symlinked_and_oversized_files_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "real.jsonl"
    target.write_text("")
    link = tmp_path / "runs.jsonl"
    link.symlink_to(target)
    with pytest.raises(FixtureError, match="symlink"):
        fixtures._read_text(link)
    monkeypatch.setattr(fixtures, "MAX_FILE_BYTES", 3)
    big = tmp_path / "big.jsonl"
    big.write_text("abcdef")
    with pytest.raises(FixtureError, match="larger than"):
        fixtures._read_text(big)


def test_verify_flags_skipped_and_stopped_runs() -> None:
    bundle = _bundle([], [], {})
    bundle.runs[0] = RunRow.model_validate(
        run_row(status="skipped", error_type="rate_limit")
    )
    gaps = fixtures.verify_bundle(bundle)
    assert gaps[RUN] == [
        "incomplete baseline: run skipped",
        "incomplete baseline: run stopped (rate_limit)",
    ]


def test_non_finite_numbers_are_rejected() -> None:
    for text in ('{"payload": {"x": NaN}}', "[Infinity]", "-Infinity"):
        with pytest.raises(ValueError):
            jsonio.loads(text)
    with pytest.raises(ValueError):
        RunRow.model_validate(run_row(cost_usd=float("inf")))


def write_bundle(directory: Path, bundle: Any) -> None:
    manifest = bundle.manifest.model_dump(mode="json")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for name, rows in (
        ("runs", bundle.runs),
        ("spans", bundle.spans),
        ("scores", bundle.scores),
        ("cache", bundle.cache),
    ):
        text = "".join(dumps(r.model_dump(mode="json")) + "\n" for r in rows)
        (directory / f"{name}.jsonl").write_text(text)


def test_verify_flags_a_cache_entry_that_disagrees_with_its_span() -> None:
    recorded = {
        "text": None,
        "tool_calls": [{"id": "c", "name": "get_order", "args": {}}],
    }
    bundle = _bundle(
        [span("model", {"request_hash": "h1", "response": recorded})], ["h1"], {}
    )
    bundle.cache[0].response = {
        "text": "I could not find that order.",
        "tool_calls": [],
    }
    assert fixtures.verify_bundle(bundle) == {
        RUN: ["cache entry h1 disagrees with span 0"]
    }


def test_verify_ignores_bookkeeping_differences_between_cache_and_span() -> None:
    call = {"id": "c", "name": "get_order", "args": {"order_id": "A1"}}
    recorded = {"text": "x", "tool_calls": [call], "cache_status": "hit", "attempts": 2}
    bundle = _bundle(
        [span("model", {"request_hash": "h1", "response": recorded})], ["h1"], {}
    )
    bundle.cache[0].response = {
        "text": "x", "tool_calls": [{**call, "signature": "s"}], "cache_status": "live",
    }  # fmt: skip
    assert fixtures.verify_bundle(bundle) == {}


@pytest.mark.parametrize(
    ("scores", "gap"),
    [
        ([], "incomplete baseline: run has no deterministic scores"),
        ([_score("judge")], "incomplete baseline: run has no deterministic scores"),
        ([_score("deterministic")], "incomplete baseline: run has no judge score"),
    ],
)
def test_verify_flags_runs_without_deterministic_or_judge_scores(
    scores: list[dict[str, Any]], gap: str
) -> None:
    assert gap in fixtures.verify_bundle(_bundle([], [], {}, scores))[RUN]


def test_load_bundle_rejects_a_run_count_that_does_not_match(tmp_path: Path) -> None:
    bundle = _bundle([], [], {})
    bundle.manifest.run_count = 2
    write_bundle(tmp_path, bundle)
    with pytest.raises(FixtureError, match="run_count is 2 but runs.jsonl has 1"):
        load_bundle(tmp_path)


def test_load_bundle_rejects_a_task_count_that_does_not_match(tmp_path: Path) -> None:
    bundle = _bundle([], [], {})
    bundle.manifest.task_count = 2
    write_bundle(tmp_path, bundle)
    with pytest.raises(FixtureError, match="task_count"):
        load_bundle(tmp_path)


def test_a_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "runs.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(FixtureError, match="not a regular file"):
        fixtures._read_text(fifo)


def test_export_write_replaces_a_symlink_instead_of_following_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text("keep")
    link = tmp_path / "manifest.json"
    link.symlink_to(target)
    fixtures._write_replacing(link, b"{}\n")
    assert target.read_text() == "keep"
    assert not link.is_symlink()
    assert link.read_text() == "{}\n"
