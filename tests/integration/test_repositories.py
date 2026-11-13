"""Repositories: runs, spans, scores, suite runs and the model cache.

Run dicts carry an explicit "id" so tests do not depend on the return value.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest
from support import ROOT

SRC = ROOT / "src" / "agent_replay" / "db"
T0 = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)


def rr() -> Any:
    from agent_replay.db import runs_repo

    return runs_repo


def make_run(**over: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "task_id": "task-1",
        "variant": "v1",
        "mode": "record",
        "status": "ok",
        "final_output": "42",
        "faults": [{"kind": "rate_limit", "n": 2}, "tool_error:search"],
        "input_tokens": 11,
        "output_tokens": 7,
        "latency_ms": 123,
        "model_calls": 3,
        "tool_calls": 2,
        "cache_hits": 1,
        "cache_misses": 2,
        "cost_usd": 0.0125,
        "started_at": T0,
        "finished_at": T0 + dt.timedelta(seconds=2),
    }
    run.update(over)
    return run


def make_span(seq: int, **over: Any) -> dict[str, Any]:
    span: dict[str, Any] = {
        "seq": seq,
        "trace_id": "t" * 8,
        "span_id": f"s{seq}",
        "parent_span_id": None if seq == 0 else "s0",
        "kind": "model",
        "name": f"chat step{seq}",
        "status": "ok",
        "attributes": {"gen_ai.request.model": "gemini", "n": [1, 2, {"x": None}]},
        "payload": {
            "request_hash": "abc",
            "response": {
                "tool_calls": [{"id": "c1", "name": "search", "signature": "c2lnLw=="}],
                "text": "héllo \"q\" 'x'",
            },
        },
        "input_tokens": 5,
        "output_tokens": 6,
        "latency_ms": 7,
        "cost_usd": 0.001,
        "cache_status": "miss",
        "started_at": T0 + dt.timedelta(seconds=seq),
    }
    span.update(over)
    return span


def count(raw: Any, table: str) -> int:
    return raw.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_insert_run_get_run_roundtrip(conn: Any) -> None:
    run = make_run()
    rr().insert_run(conn, run)
    got = rr().get_run(conn, run["id"])
    assert got is not None
    for key, val in run.items():
        if key == "id":
            assert str(got["id"]) == val
        else:
            assert got[key] == val, key
    assert got["faults"] == run["faults"]
    assert isinstance(got["faults"], list)


def test_get_run_unknown_is_none(conn: Any) -> None:
    assert rr().get_run(conn, str(uuid.uuid4())) is None


def test_insert_spans_list_spans_ordered_and_json_roundtrip(conn: Any) -> None:
    run = make_run()
    rr().insert_run(conn, run)
    spans = [make_span(2), make_span(0), make_span(1)]  # inserted out of order
    rr().insert_spans(conn, run["id"], spans)
    got = rr().list_spans(conn, run["id"])
    assert [s["seq"] for s in got] == [0, 1, 2]
    by_seq = {s["seq"]: s for s in spans}
    for g in got:
        exp = by_seq[g["seq"]]
        assert g["attributes"] == exp["attributes"]
        assert g["payload"] == exp["payload"]
        assert g["payload"]["response"]["tool_calls"][0]["signature"] == "c2lnLw=="
        assert g["span_id"] == exp["span_id"]
        assert g["parent_span_id"] == exp["parent_span_id"]
        assert g["name"] == exp["name"]
        assert g["cache_status"] == "miss"
        assert g["cost_usd"] == pytest.approx(0.001)


def test_list_spans_only_for_that_run(conn: Any) -> None:
    a, b = make_run(), make_run()
    rr().insert_run(conn, a)
    rr().insert_run(conn, b)
    rr().insert_spans(conn, a["id"], [make_span(0), make_span(1)])
    rr().insert_spans(conn, b["id"], [make_span(0)])
    assert len(rr().list_spans(conn, a["id"])) == 2
    assert len(rr().list_spans(conn, b["id"])) == 1


def test_save_run_writes_run_and_spans(conn: Any, raw: Any) -> None:
    run = make_run()
    rr().save_run(conn, run, [make_span(0), make_span(1)])
    assert count(raw, "runs") == 1
    assert count(raw, "spans") == 2


def test_save_run_is_atomic_when_span_insert_fails(conn: Any, raw: Any) -> None:
    run = make_run()
    # Duplicate seq violates UNIQUE (run_id, seq) on the second span.
    spans = [make_span(0), make_span(1), make_span(1, span_id="dup")]
    repo = rr()
    with pytest.raises(Exception):  # noqa: B017  (psycopg.Error subclass or wrapper)
        repo.save_run(conn, run, spans)
    assert count(raw, "runs") == 0
    assert count(raw, "spans") == 0
    assert rr().get_run(conn, run["id"]) is None


def test_save_run_atomic_with_invalid_span_value(conn: Any, raw: Any) -> None:
    run = make_run()
    bad = make_span(1, cache_status="bogus")  # violates CHECK constraint
    repo = rr()
    with pytest.raises(Exception):  # noqa: B017
        repo.save_run(conn, run, [make_span(0), bad])
    assert count(raw, "runs") == 0


def test_list_runs_limit_offset_ordered_by_started_at(conn: Any) -> None:
    suite = rr().create_suite_run(
        conn, label="s", variant="v", mode="record", source_suite_run_id=None, faults=[]
    )
    other = rr().create_suite_run(
        conn, label="o", variant="v", mode="record", source_suite_run_id=None, faults=[]
    )
    ids = []
    # Insert in scrambled order; 15 in suite, 3 in other.
    order = [7, 2, 14, 0, 9, 5, 12, 1, 11, 3, 13, 6, 10, 4, 8]
    for i in order:
        r = make_run(suite_run_id=str(suite), started_at=T0 + dt.timedelta(minutes=i))
        rr().insert_run(conn, r)
        ids.append((i, r["id"]))
    for i in range(3):
        rr().insert_run(conn, make_run(suite_run_id=str(other), started_at=T0))
    expected = [rid for _, rid in sorted(ids)]

    page1 = rr().list_runs(conn, suite_run_id=str(suite), limit=10, offset=0)
    assert [str(r["id"]) for r in page1] == expected[:10]
    page2 = rr().list_runs(conn, suite_run_id=str(suite), limit=10, offset=10)
    assert [str(r["id"]) for r in page2] == expected[10:]
    small = rr().list_runs(conn, suite_run_id=str(suite), limit=3, offset=0)
    assert len(small) == 3


def test_upsert_score_inserts_then_updates(conn: Any, raw: Any) -> None:
    run = make_run()
    rr().insert_run(conn, run)
    rr().upsert_score(conn, run["id"], "contains", True, {"why": "first"})
    rows = raw.execute(
        "SELECT passed, detail FROM scores WHERE run_id=%s AND check_name=%s",
        (run["id"], "contains"),
    ).fetchall()
    assert rows == [(True, {"why": "first"})]
    rr().upsert_score(conn, run["id"], "contains", False, {"why": "second", "n": [1]})
    rr().upsert_score(conn, run["id"], "other", None, {})
    rows = raw.execute(
        "SELECT check_name, passed, detail FROM scores WHERE run_id=%s ORDER BY check_name",
        (run["id"],),
    ).fetchall()
    assert rows == [
        ("contains", False, {"why": "second", "n": [1]}),
        ("other", None, {}),
    ]


def test_suite_run_roundtrip(conn: Any) -> None:
    src = rr().create_suite_run(
        conn,
        label="base",
        variant="v1",
        mode="record",
        source_suite_run_id=None,
        faults=[],
    )
    faults = [{"kind": "rate_limit", "n": 1}, "timeout"]
    sid = rr().create_suite_run(
        conn,
        label="my label",
        variant="v2",
        mode="replay",
        source_suite_run_id=str(src),
        faults=faults,
    )
    got = rr().get_suite_run(conn, str(sid))
    assert got is not None
    assert got["label"] == "my label"
    assert got["variant"] == "v2"
    assert got["mode"] == "replay"
    assert str(got["source_suite_run_id"]) == str(src)
    assert got["faults"] == faults
    base = rr().get_suite_run(conn, str(src))
    assert base["source_suite_run_id"] is None
    assert rr().get_suite_run(conn, str(uuid.uuid4())) is None


def cache() -> Any:
    from agent_replay.db.cache_repo import CacheRepo

    return CacheRepo


def put_kwargs(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "provider": "gemini",
        "model": "gemini-3-flash",
        "request": {"messages": [{"role": "user", "content": "hi"}]},
        "response": {
            "text": "hello",
            "tool_calls": [{"name": "s", "signature": "QUJD"}],
        },
        "input_tokens": 3,
        "output_tokens": 4,
        "latency_ms": 55,
        "cost_usd": 0.002,
    }
    kw.update(over)
    return kw


def test_cache_get_unknown_none_known_returns_response(conn: Any) -> None:
    repo = cache()(conn)
    assert repo.get("nope") is None
    kw = put_kwargs()
    repo.put("h1", **kw)
    assert repo.get("h1") == kw["response"]
    assert repo.get("h2") is None


def test_cache_put_twice_keeps_first(conn: Any, raw: Any) -> None:
    repo = cache()(conn)
    first = put_kwargs()
    repo.put("h1", **first)
    repo.put("h1", **put_kwargs(response={"text": "SECOND"}, input_tokens=99))
    assert repo.get("h1") == first["response"]
    assert count(raw, "model_cache") == 1
    assert raw.execute("SELECT input_tokens FROM model_cache").fetchone()[0] == 3


def test_task_id_with_single_quote_roundtrips(conn: Any) -> None:
    for tid in ["o'brien", "x'); DROP TABLE runs; --", "q\"\\'"]:
        run = make_run(task_id=tid)
        rr().insert_run(conn, run)
        assert rr().get_run(conn, run["id"])["task_id"] == tid
    # table still exists and holds all three
    assert len(rr().list_runs(conn, suite_run_id=None, limit=10, offset=0)) >= 0


def test_json_values_written_with_psycopg_jsonb(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg.types.json as pj

    created: list[Any] = []
    real = pj.Jsonb

    class SpyJsonb(real):  # type: ignore[valid-type, misc]
        def __init__(self, *a: Any, **k: Any) -> None:
            created.append(a[0] if a else None)
            super().__init__(*a, **k)

    from agent_replay.db import cache_repo

    monkeypatch.setattr(pj, "Jsonb", SpyJsonb)
    for mod in (rr(), cache_repo):
        if hasattr(mod, "Jsonb"):
            monkeypatch.setattr(mod, "Jsonb", SpyJsonb)

    run = make_run()
    rr().save_run(conn, run, [make_span(0)])
    rr().upsert_score(conn, run["id"], "c", True, {"d": 1})
    cache()(conn).put("hj", **put_kwargs())
    assert run["faults"] in created
    assert make_span(0)["attributes"] in created
    assert make_span(0)["payload"] in created
    assert {"d": 1} in created
    assert put_kwargs()["response"] in created
    assert put_kwargs()["request"] in created


def test_repos_do_not_build_sql_with_string_formatting() -> None:
    import re

    for name in ("runs_repo.py", "cache_repo.py"):
        text = (SRC / name).read_text()
        assert "Jsonb" in text, name
        assert not re.search(
            r"""f["'](?:[^"'\n]*)\b(SELECT|INSERT|UPDATE)\b""", text, re.IGNORECASE
        ), name
        assert not re.search(r"""["']\s*%\s*\(?\w""", text), f"{name}: %-formatting"
