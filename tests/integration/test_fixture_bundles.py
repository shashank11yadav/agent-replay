"""Fixtures export / import / verify, through the CLI and the library.

Judge cache entries are not linked from spans, so "exactly the referenced entries" is
tested by adding an unrelated decoy cache row before export. The wiped target is a
second fresh migrated database.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row
from support import pg, scrubbed_env

from agent_replay.config import Settings
from agent_replay.db import runs_repo
from agent_replay.db.cache_repo import CacheRepo
from agent_replay.fixtures import (
    FILES,
    FixtureError,
    export_suite,
    import_bundle,
    load_bundle,
    verify_bundle,
)
from agent_replay.llm.hashing import request_hash
from agent_replay.llm.types import Message, ModelRequest

N = 3
DUMMY_KEY = "AIzaDUMMYkey-fixtures-9f3c1e7b"
DUMMY_URL_MARK = "dummy-secret-host-fixtures.invalid"


def write_tasks(tmp: Path) -> Path:
    lines = [
        json.dumps(
            {
                "id": f"f-{i}",
                "prompt": f"What is the status of order A1001 (case {i})?",
                "category": "lookup",
                "expected_text": ["shipped"],
                "must_call_tools": ["get_order"],
                "max_steps": 4,
                "rubric": f"States that order A1001 has shipped (case {i}).",
            }
        )
        for i in range(1, N + 1)
    ]
    p = tmp / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def write_script(tmp: Path, name: str, entries: list[Any]) -> Path:
    p = tmp / name
    p.write_text(json.dumps(entries))
    return p


def cli(
    db_url: str, script: Path, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env.update(
        AGENT_REPLAY_DATABASE_URL=db_url,
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_PROVIDER="scripted",
        AGENT_REPLAY_SCRIPT_PATH=str(script),
        AGENT_REPLAY_BACKOFF_BASE_S="0",
        AGENT_REPLAY_RUN_BUDGET_USD="100",
    )
    env.update(env_extra or {})
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_replay.cli import main; sys.exit(main())",
            *args,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


class Env:
    def __init__(self, db_url: str, db_url2: str, tmp: Path) -> None:
        self.db_url = db_url
        self.db_url2 = db_url2
        self.tmp = tmp
        self.tasks = write_tasks(tmp)
        pair = [
            {
                "tool_calls": [
                    {"id": "x", "name": "get_order", "args": {"order_id": "A1001"}}
                ]
            },
            {"text": "Order A1001 has shipped."},
        ]
        self.script = write_script(tmp, "agent.json", pair * (N + 3))
        verdict = {"text": '{"pass": true, "reason": "ok"}'}
        self.judge_script = write_script(tmp, "judge.json", [verdict] * (2 * N))
        self.empty_script = write_script(tmp, "empty.json", [])
        self.sid = ""

    def run(
        self, *args: str, db: str | None = None, script: Path | None = None, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        return cli(db or self.db_url, script or self.script, *args, **kw)

    def make_suite(self, **kw: Any) -> str:
        p = self.run(
            "suite",
            "record",
            "--tasks",
            str(self.tasks),
            "--variant",
            "structured",
            "--label",
            "base",
            **kw,
        )
        assert p.returncode == 0, (p.stdout, p.stderr)
        sid = last_line(p)
        s = self.run(
            "suite",
            "score",
            "--suite-run",
            sid,
            "--tasks",
            str(self.tasks),
            script=self.judge_script,
            **kw,
        )
        assert s.returncode == 0, (s.stdout, s.stderr)
        self.sid = sid
        return sid

    def export(
        self, out: Path, db: str | None = None, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "fixtures",
            "export",
            "--suite-run",
            self.sid,
            "--out",
            str(out),
            db=db,
            **kw,
        )


@pytest.fixture
def env(db_url: str, db_url2: str, tmp_path: Path) -> Env:
    return Env(db_url, db_url2, tmp_path)


@pytest.fixture
def exported(env: Env, raw: Any) -> tuple[Env, Path]:
    env.make_suite()
    # Decoy: a cache row no step of the suite references.
    raw.execute(
        "INSERT INTO model_cache (request_hash, provider, model) VALUES (%s, 'x', 'x')",
        ("decoy" + "0" * 59,),
    )
    out = env.tmp / "fx"
    p = env.export(out)
    assert p.returncode == 0, (p.stdout, p.stderr)
    return env, out


def test_export_writes_the_five_files(exported: tuple[Env, Path]) -> None:
    _, out = exported
    for name in FILES:
        assert (out / name).is_file(), name
    json.loads((out / "manifest.json").read_text())
    assert len(read_jsonl(out / "runs.jsonl")) == N
    assert len(read_jsonl(out / "scores.jsonl")) >= N * 2  # deterministic rows + judge
    assert read_jsonl(out / "spans.jsonl")


def test_cache_jsonl_has_exactly_the_referenced_entries(
    exported: tuple[Env, Path], raw: Any
) -> None:
    _, out = exported
    exported_hashes = [r["request_hash"] for r in read_jsonl(out / "cache.jsonl")]
    assert len(exported_hashes) == len(set(exported_hashes))
    assert not any(h.startswith("decoy") for h in exported_hashes)
    expected = {
        r[0]
        for r in raw.execute(
            "SELECT request_hash FROM model_cache WHERE request_hash NOT LIKE 'decoy%'"
        ).fetchall()
    }
    assert set(exported_hashes) == expected
    span_hashes = {
        s["payload"]["request_hash"]
        for s in read_jsonl(out / "spans.jsonl")
        if s["kind"] == "model"
    }
    assert span_hashes < set(
        exported_hashes
    )  # strictly more: judge entries are included
    assert len(exported_hashes) - len(span_hashes) == N  # one judge call per run


def test_export_into_a_file_fails_cleanly(env: Env) -> None:
    env.make_suite()
    out = env.tmp / "taken"
    out.write_text("not a directory")
    p = env.export(out)
    assert p.returncode == 1
    assert f"cannot write {out}" in p.stderr
    assert "Traceback" not in p.stderr


def test_export_over_a_directory_named_like_a_file_fails_cleanly(env: Env) -> None:
    env.make_suite()
    out = env.tmp / "fx"
    (out / "manifest.json").mkdir(parents=True)
    p = env.export(out)
    assert p.returncode == 1
    assert f"cannot write {out / 'manifest.json'}" in p.stderr
    assert "Traceback" not in p.stderr


def test_export_twice_is_byte_identical_with_sorted_keys(
    exported: tuple[Env, Path],
) -> None:
    env, out = exported
    out2 = env.tmp / "fx2"
    assert env.export(out2).returncode == 0
    for name in FILES:
        assert (out / name).read_bytes() == (out2 / name).read_bytes(), name
    for name in FILES[1:]:
        for ln in (out / name).read_text().splitlines():
            keys = [k for k, _ in json.loads(ln, object_pairs_hook=list)]
            assert keys == sorted(keys), name
    m = (out / "manifest.json").read_text()
    keys = [k for k, _ in json.loads(m, object_pairs_hook=list)]
    assert keys == sorted(keys)


def test_manifest_records_suite_variant_counts_and_task_file_hash(
    exported: tuple[Env, Path],
) -> None:
    env, out = exported
    m = json.loads((out / "manifest.json").read_text())
    assert env.sid in json.dumps(m)
    assert m["variant"] == "structured"
    assert m["task_count"] == N
    assert m["run_count"] == N
    assert m["tasks_sha256"] == hashlib.sha256(env.tasks.read_bytes()).hexdigest()


def snapshot(conn: Any) -> dict[str, list[Any]]:
    q = {
        "suites": "SELECT id::text, label, variant, mode, status, tasks_sha256, task_count "
        "FROM suite_runs ORDER BY id",
        "runs": "SELECT id::text, suite_run_id::text, task_id, variant, mode, status, "
        "final_output, cache_hits, model_calls FROM runs ORDER BY id",
        "spans": "SELECT id, run_id::text, seq, kind, cache_status, payload::text "
        "FROM spans ORDER BY id",
        "scores": "SELECT run_id::text, check_name, kind, passed FROM scores "
        "ORDER BY run_id, check_name",
        "cache": "SELECT request_hash FROM model_cache WHERE request_hash NOT LIKE 'decoy%' "
        "ORDER BY request_hash",
    }
    return {k: conn.execute(v).fetchall() for k, v in q.items()}


def test_import_into_empty_db_recreates_ids_and_is_idempotent(
    exported: tuple[Env, Path], raw: Any, raw2: Any
) -> None:
    env, out = exported
    p = env.run("fixtures", "import", str(out), db=env.db_url2)
    assert p.returncode == 0, (p.stdout, p.stderr)
    first = snapshot(raw2)
    assert first == snapshot(raw)
    assert len(first["runs"]) == N and first["spans"]
    again = env.run("fixtures", "import", str(out), db=env.db_url2)
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert snapshot(raw2) == first


def test_import_advances_the_span_identity_sequence(
    exported: tuple[Env, Path], raw2: Any
) -> None:
    env, out = exported
    assert env.run("fixtures", "import", str(out), db=env.db_url2).returncode == 0
    run_id = raw2.execute("SELECT id FROM runs LIMIT 1").fetchone()[0]
    raw2.execute("INSERT INTO spans (run_id, seq) VALUES (%s, 9999)", (run_id,))
    new_id = raw2.execute("SELECT id FROM spans WHERE seq = 9999").fetchone()[0]
    assert (
        new_id
        > raw2.execute("SELECT max(id) FROM spans WHERE seq <> 9999").fetchone()[0]
    )
    # A new recorded suite after the import also works (no duplicate-key error).
    p = env.run(
        "suite",
        "record",
        "--tasks",
        str(env.tasks),
        "--variant",
        "structured",
        db=env.db_url2,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)


def test_replay_after_wipe_and_import_hits_cache_for_every_model_step(
    exported: tuple[Env, Path], raw2: Any
) -> None:
    env, out = exported
    assert env.run("fixtures", "import", str(out), db=env.db_url2).returncode == 0
    # Empty script: any live call would raise, so only cache hits can succeed.
    p = env.run(
        "suite",
        "replay",
        "--source",
        env.sid,
        "--variant",
        "structured",
        "--tasks",
        str(env.tasks),
        db=env.db_url2,
        script=env.empty_script,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    new_sid = last_line(p)
    assert new_sid != env.sid
    rows = raw2.execute(
        "SELECT s.cache_status, r.status FROM spans s JOIN runs r ON r.id = s.run_id "
        "WHERE r.suite_run_id = %s AND s.kind = 'model'",
        (new_sid,),
    ).fetchall()
    assert len(rows) >= 2 * N
    assert {r[0] for r in rows} == {"hit"}
    assert {r[1] for r in rows} == {"ok"}


def copy_dir(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_verify_passes_for_a_complete_export(exported: tuple[Env, Path]) -> None:
    env, out = exported
    p = env.run("fixtures", "verify", str(out))
    assert p.returncode == 0, (p.stdout, p.stderr)


def test_verify_fails_listing_run_with_missing_model_cache_entry(
    exported: tuple[Env, Path],
) -> None:
    env, out = exported
    bad = copy_dir(out, env.tmp / "bad1")
    spans = read_jsonl(bad / "spans.jsonl")
    victim = next(s for s in spans if s["kind"] == "model")
    vrun = str(victim["run_id"])
    others = {str(r["id"]) for r in read_jsonl(bad / "runs.jsonl")} - {vrun}
    h = victim["payload"]["request_hash"]
    kept = [
        ln
        for ln in (bad / "cache.jsonl").read_text().splitlines()
        if json.loads(ln)["request_hash"] != h
    ]
    (bad / "cache.jsonl").write_text("\n".join(kept) + "\n")
    p = env.run("fixtures", "verify", str(bad))
    assert p.returncode == 1, (p.stdout, p.stderr)
    text = p.stdout + p.stderr
    assert vrun in text
    assert not any(o in text for o in others)


def test_verify_fails_when_a_judge_cache_entry_is_missing(
    exported: tuple[Env, Path],
) -> None:
    env, out = exported
    bad = copy_dir(out, env.tmp / "bad2")
    span_hashes = {
        s["payload"]["request_hash"]
        for s in read_jsonl(bad / "spans.jsonl")
        if s["kind"] == "model"
    }
    lines = (bad / "cache.jsonl").read_text().splitlines()
    judge = [ln for ln in lines if json.loads(ln)["request_hash"] not in span_hashes]
    assert len(judge) == N
    kept = [ln for ln in lines if ln != judge[0]]
    (bad / "cache.jsonl").write_text("\n".join(kept) + "\n")
    p = env.run("fixtures", "verify", str(bad))
    assert p.returncode == 1, (p.stdout, p.stderr)
    run_ids = {str(r["id"]) for r in read_jsonl(bad / "runs.jsonl")}
    assert any(rid in p.stdout + p.stderr for rid in run_ids)


def test_verify_fails_listing_run_with_missing_tool_result(
    exported: tuple[Env, Path],
) -> None:
    env, out = exported
    bad = copy_dir(out, env.tmp / "bad3")
    spans = read_jsonl(bad / "spans.jsonl")
    victim = next(s for s in spans if s["kind"] == "tool")
    vrun = str(victim["run_id"])
    others = {str(r["id"]) for r in read_jsonl(bad / "runs.jsonl")} - {vrun}
    victim["payload"]["result"] = None
    (bad / "spans.jsonl").write_text(
        "\n".join(json.dumps(s, sort_keys=True) for s in spans) + "\n"
    )
    p = env.run("fixtures", "verify", str(bad))
    assert p.returncode == 1, (p.stdout, p.stderr)
    text = p.stdout + p.stderr
    assert vrun in text
    assert not any(o in text for o in others)


def test_exported_files_contain_no_secret_values(env: Env, raw: Any) -> None:
    secret_env = {"AGENT_REPLAY_GEMINI_API_KEY": DUMMY_KEY}
    env.make_suite(env_extra=secret_env)
    out = env.tmp / "fx-secret"
    p = env.export(out, env_extra=secret_env)
    assert p.returncode == 0, (p.stdout, p.stderr)
    blob = "".join((out / n).read_text() for n in FILES)
    assert blob
    assert DUMMY_KEY not in blob
    assert env.db_url not in blob
    assert "postgres:postgres" not in blob
    assert DUMMY_URL_MARK not in blob
    assert DUMMY_KEY not in p.stdout + p.stderr


Conn = psycopg.Connection[dict[str, Any]]


@pytest.fixture
def make_db() -> Iterator[Callable[[], Conn]]:
    """Factory for extra migrated databases, all dropped after the test."""
    names: list[str] = []
    conns: list[Conn] = []

    def make() -> Conn:
        name, url = pg.create_database()
        names.append(name)
        conn = pg.connect(url, autocommit=True, row_factory=dict_row)
        conns.append(conn)
        return conn

    yield make
    for c in conns:
        c.close()
    for name in names:
        pg.drop_database(name)


def _request(text: str) -> ModelRequest:
    return ModelRequest(
        provider="scripted",
        model="m",
        system="s",
        messages=[Message(role="user", text=text)],
    )


def _cache(conn: Conn, req: ModelRequest, answer: str = "ok") -> str:
    key = request_hash(req)
    CacheRepo(conn).put(  # type: ignore[arg-type]
        key,
        provider=req.provider,
        model=req.model,
        request=req.model_dump(mode="json"),
        response={"text": answer, "tool_calls": []},
    )
    return key


def seed(
    conn: Conn, judge_passed: bool | None = True, judge_cached: bool = True
) -> str:
    """One suite, one run, a model span, a tool span, a check row and a judge row."""
    suite = str(runs_repo.create_suite_run(conn, variant="v", task_count=1))
    run_id = str(
        runs_repo.insert_run(
            conn,
            {
                "suite_run_id": suite,
                "task_id": "t",
                "variant": "v",
                "mode": "record",
                "final_output": "answer",
            },
        )
    )
    key = _cache(conn, _request("hello"))
    runs_repo.insert_spans(
        conn,
        run_id,
        [
            {
                "seq": 0,
                "kind": "model",
                "payload": {"request_hash": key, "response": {"text": "ok"}},
            },
            {
                "seq": 1,
                "kind": "tool",
                "payload": {"name": "get_order", "result": {"status": "shipped"}},
            },
        ],
    )
    judge_req = _request("judge me")
    if judge_cached:
        _cache(conn, judge_req)
    runs_repo.upsert_score(conn, run_id, "output_contains", True, {})
    runs_repo.upsert_score(
        conn,
        run_id,
        "judge",
        judge_passed,
        {"request_hash": request_hash(judge_req)},
        kind="judge",
    )
    return suite


def settings() -> Settings:
    return Settings.model_validate({})


def test_round_trip_reimport_is_noop_and_judge_entry_is_exported(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src, dst = make_db(), make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    bundle = load_bundle(tmp_path)
    assert len(bundle.cache) == 2
    assert verify_bundle(bundle) == {}
    first = import_bundle(dst, bundle)
    assert first["runs"] == 1 and first["spans"] == 2 and first["cache"] == 2
    again = import_bundle(dst, bundle)
    assert set(again.values()) == {0}


def test_export_fails_listing_run_when_judge_entry_missing(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src = make_db()
    suite = seed(src, judge_cached=False)
    run_id = str(runs_repo.list_suite_run_rows(src, suite)[0]["id"])
    with pytest.raises(FixtureError, match=run_id):
        export_suite(src, suite, tmp_path / "out", settings())
    assert not (tmp_path / "out").exists()


def test_unresponded_judge_rows_export_but_verify_reports_incomplete_baseline(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src = make_db()
    suite = seed(src, judge_passed=None, judge_cached=False)
    export_suite(src, suite, tmp_path, settings())
    run_id = str(runs_repo.list_suite_run_rows(src, suite)[0]["id"])
    gaps = verify_bundle(load_bundle(tmp_path))
    assert list(gaps) == [run_id]
    assert gaps[run_id][0].startswith("incomplete baseline: judge verdict")


def test_verify_flags_judge_entry_removed_from_cache_file(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src = make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    judge_key = request_hash(_request("judge me"))
    lines = (tmp_path / "cache.jsonl").read_text().splitlines()
    kept = [ln for ln in lines if json.loads(ln)["request_hash"] != judge_key]
    (tmp_path / "cache.jsonl").write_text("\n".join(kept) + "\n")
    # Also drop it from the manifest, leaving only the score row as evidence.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["judge_requests"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    gaps = verify_bundle(load_bundle(tmp_path))
    run_id = str(runs_repo.list_suite_run_rows(src, suite)[0]["id"])
    assert list(gaps) == [run_id]


def test_import_refuses_changed_content_and_leaves_database_unchanged(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src, dst = make_db(), make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    import_bundle(dst, load_bundle(tmp_path))
    runs = [json.loads(ln) for ln in (tmp_path / "runs.jsonl").read_text().splitlines()]
    runs[0]["final_output"] = "tampered"
    (tmp_path / "runs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in runs)
    )
    with pytest.raises(FixtureError, match="different content"):
        import_bundle(dst, load_bundle(tmp_path))
    row = dst.execute("SELECT final_output FROM runs").fetchone()
    assert row is not None and row["final_output"] == "answer"


def test_import_rolls_back_after_partial_work(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src, dst = make_db(), make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    bundle = load_bundle(tmp_path)
    other = seed(dst, judge_cached=False)
    # The suite and its run go in before the cache rows; this clash comes after them.
    _cache(dst, _request("judge me"), answer="a different verdict")
    before = dst.execute("SELECT count(*) AS n FROM model_cache").fetchone()
    with pytest.raises(FixtureError, match="cache entry"):
        import_bundle(dst, bundle)
    after = dst.execute("SELECT count(*) AS n FROM model_cache").fetchone()
    assert before == after
    suites = dst.execute("SELECT id FROM suite_runs").fetchall()
    assert [str(r["id"]) for r in suites] == [other]


def test_new_span_cannot_be_merged_into_an_existing_run(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src, dst = make_db(), make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    bundle = load_bundle(tmp_path)
    import_bundle(dst, bundle)
    dst.execute("DELETE FROM spans WHERE seq = 1")
    with pytest.raises(FixtureError, match="existing parent"):
        import_bundle(dst, bundle)


def test_cache_row_whose_request_does_not_match_its_hash_is_rejected(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    src = make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    rows = [
        json.loads(ln) for ln in (tmp_path / "cache.jsonl").read_text().splitlines()
    ]
    rows[0]["request"]["system"] = "swapped"
    (tmp_path / "cache.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    with pytest.raises(FixtureError, match="does not match its request"):
        load_bundle(tmp_path)


def test_two_imports_of_one_bundle_at_once_both_succeed(
    make_db: Callable[[], Conn], tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    src, dst = make_db(), make_db()
    suite = seed(src)
    export_suite(src, suite, tmp_path, settings())
    bundle = load_bundle(tmp_path)
    url = dst.info.dsn
    password = dst.info.password

    def run_import() -> dict[str, int]:
        with psycopg.connect(
            url, password=password, autocommit=True, row_factory=dict_row
        ) as conn:
            return import_bundle(conn, bundle)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_import(), range(2)))
    assert sorted(r["runs"] for r in results) == [0, 1]
