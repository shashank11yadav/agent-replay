"""Schema, migration runner, pool, `db migrate` and the compose db service."""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from support import ROOT, pg

from agent_replay.cli import db_cmd
from agent_replay.config import Settings
from agent_replay.db.migrate import (
    MIGRATIONS_DIR,
    MigrationError,
    apply_migrations,
    list_migrations,
)
from agent_replay.db.pool import get_pool

TABLES = {"schema_migrations", "suite_runs", "runs", "spans", "scores", "model_cache"}


@pytest.fixture
def db_url(empty_db_url: str) -> str:
    """These tests run the migrations themselves, so they start from an empty database."""
    return empty_db_url


@pytest.fixture
def conn(db_url: str) -> Iterator[Any]:
    c = pg.connect(db_url, autocommit=True)
    try:
        yield c
    finally:
        c.close()


def run_cli(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["AGENT_REPLAY_DATABASE_URL"] = url
    code = "import sys; from agent_replay.cli import main; sys.exit(main(sys.argv[1:]))"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=120,
    )


@pytest.fixture
def migrations_dir() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("agent_replay.db")
    assert spec is not None and spec.submodule_search_locations, (
        "package agent_replay.db does not exist"
    )
    return Path(next(iter(spec.submodule_search_locations))) / "migrations"


@pytest.fixture
def extra_migrations(migrations_dir: Path) -> Iterator[dict[str, str]]:
    """Collect temp migration files (name -> sql); written by test, removed after."""
    files: dict[str, str] = {}
    yield files
    for name in files:
        (migrations_dir / name).unlink(missing_ok=True)


def write_extra(migrations_dir: Path, files: dict[str, str]) -> None:
    for name, sql in files.items():
        (migrations_dir / name).write_text(sql)


def tables(conn: Any) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    ).fetchall()
    return {r[0] for r in rows}


def migrated(db_url: str) -> None:
    r = run_cli(db_url, "db", "migrate")
    assert r.returncode == 0, f"migrate failed: {r.stdout}\n{r.stderr}"


# ---- generic row filler (column names other than the constrained ones are unknown) ----
_counter = itertools.count(1)


def _value(dtype: str, udt: str, conn: Any) -> Any:
    n = next(_counter)
    if dtype in ("text", "character varying", "character"):
        return f"v{n}"
    if dtype in (
        "integer",
        "bigint",
        "smallint",
        "numeric",
        "double precision",
        "real",
    ):
        return n
    if dtype == "boolean":
        return False
    if dtype in ("json", "jsonb"):
        return "{}"
    if dtype.startswith("timestamp"):
        return "2026-01-01T00:00:00+00:00"
    if dtype == "uuid":
        return conn.execute("SELECT gen_random_uuid()").fetchone()[0]
    if dtype == "USER-DEFINED":
        return conn.execute(
            "SELECT enumlabel FROM pg_enum e JOIN pg_type t ON t.oid=e.enumtypid "
            "WHERE t.typname=%s ORDER BY enumsortorder LIMIT 1",
            (udt,),
        ).fetchone()[0]
    raise AssertionError(f"test filler cannot handle column type {dtype}")


def insert_row(conn: Any, table: str, **over: Any) -> dict[str, Any]:
    from psycopg import sql
    from psycopg.types.json import Jsonb  # noqa: F401

    cols = conn.execute(
        "SELECT column_name, data_type, udt_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=%s AND is_nullable='NO' "
        "AND column_default IS NULL AND is_identity='NO' AND is_generated='NEVER'",
        (table,),
    ).fetchall()
    fks = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT a.attname, cf.relname, af.attname FROM pg_constraint c "
            "JOIN pg_class cl ON cl.oid=c.conrelid "
            "JOIN pg_namespace n ON n.oid=cl.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=c.conkey[1] "
            "JOIN pg_class cf ON cf.oid=c.confrelid "
            "JOIN pg_attribute af ON af.attrelid=c.confrelid AND af.attnum=c.confkey[1] "
            "WHERE c.contype='f' AND cl.relname=%s AND n.nspname='public' "
            "AND array_length(c.conkey,1)=1",
            (table,),
        ).fetchall()
    }
    values: dict[str, Any] = dict(over)
    for name, dtype, udt in cols:
        if name in values:
            continue
        if name in fks:
            parent, pcol = fks[name]
            values[name] = insert_row(conn, parent)[pcol]
        else:
            values[name] = _value(dtype, udt, conn)
    names = list(values)
    q = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, names)),
        sql.SQL(", ").join(sql.Placeholder() * len(names)),
    )
    cur = conn.execute(q, [values[k] for k in names])
    desc = [d.name for d in cur.description]
    return dict(zip(desc, cur.fetchone(), strict=True))


# ---- compose ----
def _db_block(text: str) -> str:
    m = re.search(r"^  db:\s*\n((?:(?:    .*|\s*)\n?)*)", text, re.MULTILINE)
    assert m, "no `db` service under services:"
    return m.group(1)


def test_compose_db_service() -> None:
    text = (ROOT / "docker-compose.yml").read_text()
    assert not re.search(r"^version\s*:", text, re.MULTILINE), (
        "top-level version: key present"
    )
    block = _db_block(text)
    assert re.search(
        r"image:\s*[\"']?postgres:18\.6-alpine[\"']?\s*$", block, re.MULTILINE
    )
    assert re.search(r"POSTGRES_PASSWORD\s*[:=]\s*\S+", block)
    assert re.search(r"healthcheck:", block) and "pg_isready" in block
    assert re.search(r"5432", block)
    m = re.search(
        r"^\s*-\s*[\"']?([A-Za-z_][\w.-]*):/var/lib/postgresql(?:/data)?\b",
        block,
        re.MULTILINE,
    )
    assert m, "no named volume mounted at /var/lib/postgresql"
    assert not m.group(0).rstrip().endswith("/data"), (
        "mount must be /var/lib/postgresql"
    )
    assert re.search(r"/var/lib/postgresql(?!/)", m.group(0))
    top = re.search(r"^volumes:\s*\n((?:[ \t]+.*\n?)+)", text, re.MULTILINE)
    assert top and re.search(
        rf"^\s+{re.escape(m.group(1))}\s*:", top.group(1), re.MULTILINE
    ), "volume not declared in top-level volumes:"


# ---- unreachable Postgres is a failure with a message ----
def test_test_database_reachable(conn: Any) -> None:
    assert conn.execute("SELECT 1").fetchone()[0] == 1


# ---- migrate on empty db ----
def test_migrate_creates_tables_and_exits_zero(db_url: str, conn: Any) -> None:
    assert tables(conn) == set()
    r = run_cli(db_url, "db", "migrate")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert TABLES <= tables(conn)


def test_migrate_second_run_is_noop(db_url: str, conn: Any) -> None:
    migrated(db_url)
    before = conn.execute("SELECT * FROM schema_migrations").fetchall()
    r = run_cli(db_url, "db", "migrate")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "up to date" in (r.stdout + r.stderr).lower()
    assert conn.execute("SELECT * FROM schema_migrations").fetchall() == before


# ---- failing migration is atomic ----
def test_failing_migration_rolled_back(
    db_url: str, conn: Any, migrations_dir: Path, extra_migrations: dict[str, str]
) -> None:
    extra_migrations["998_bad.sql"] = (
        "CREATE TABLE mig_partial (id int);\nSELECT * FROM mig_missing_table;\n"
    )
    write_extra(migrations_dir, extra_migrations)
    r = run_cli(db_url, "db", "migrate")
    assert r.returncode != 0
    assert "998_bad" in (r.stdout + r.stderr)
    assert "mig_partial" not in tables(conn)
    if "schema_migrations" in tables(conn):
        rows = conn.execute("SELECT * FROM schema_migrations").fetchall()
        assert not any(
            "998_bad" in str(v) or "998" == str(v) for row in rows for v in row
        )


# ---- filename order, each version once ----
def test_migrations_applied_in_filename_order_recorded_once(
    db_url: str, conn: Any, migrations_dir: Path, extra_migrations: dict[str, str]
) -> None:
    # 901 depends on a table made by 900: wrong order fails.
    extra_migrations["901_child.sql"] = (
        "CREATE TABLE mig_child (pid int REFERENCES mig_parent(id));\n"
    )
    extra_migrations["900_parent.sql"] = (
        "CREATE TABLE mig_parent (id int PRIMARY KEY);\n"
    )
    write_extra(migrations_dir, extra_migrations)
    migrated(db_url)
    assert {"mig_parent", "mig_child"} <= tables(conn)
    migrated(db_url)  # re-run must not duplicate
    rows = conn.execute("SELECT * FROM schema_migrations").fetchall()
    files = sorted(migrations_dir.glob("*.sql"))
    assert len(rows) == len(files)
    for f in files:
        keys = {f.name, f.stem, f.stem.split("_")[0], str(int(f.stem.split("_")[0]))}
        hits = [row for row in rows if any(str(v) in keys for v in row)]
        assert len(hits) == 1, f"{f.name} recorded {len(hits)} times"


# ---- CHECK constraints ----
def test_runs_mode_check(db_url: str, conn: Any) -> None:
    import psycopg

    migrated(db_url)
    for mode in ("record", "replay"):
        assert insert_row(conn, "runs", mode=mode, status="ok")["mode"] == mode
    for bad in ("bogus", "Record", ""):
        with pytest.raises(psycopg.errors.CheckViolation):
            insert_row(conn, "runs", mode=bad, status="ok")


def test_runs_status_check(db_url: str, conn: Any) -> None:
    import psycopg

    migrated(db_url)
    for status in ("ok", "error", "diverged", "skipped"):
        assert (
            insert_row(conn, "runs", mode="record", status=status)["status"] == status
        )
    for bad in ("bogus", "OK", "running", ""):
        with pytest.raises(psycopg.errors.CheckViolation):
            insert_row(conn, "runs", mode="record", status=bad)


# ---- keys ----
def _pk(conn: Any, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid=i.indrelid "
        "AND a.attnum = ANY(i.indkey) WHERE i.indisprimary AND i.indrelid = %s::regclass",
        (table,),
    ).fetchall()
    return {r[0] for r in rows}


def test_primary_keys(db_url: str, conn: Any) -> None:
    migrated(db_url)
    assert _pk(conn, "scores") == {"run_id", "check_name"}
    assert _pk(conn, "model_cache") == {"request_hash"}


def test_spans_fk_and_unique_run_seq(db_url: str, conn: Any) -> None:
    import psycopg

    migrated(db_url)
    fk = conn.execute(
        "SELECT confrelid::regclass::text FROM pg_constraint "
        "WHERE contype='f' AND conrelid='spans'::regclass"
    ).fetchall()
    assert "runs" in {r[0] for r in fk}
    run = insert_row(conn, "runs", mode="record", status="ok")
    pk = _pk(conn, "runs")
    assert len(pk) == 1
    run_id = run[next(iter(pk))]
    insert_row(conn, "spans", run_id=run_id, seq=1)
    insert_row(conn, "spans", run_id=run_id, seq=2)
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_row(conn, "spans", run_id=run_id, seq=1)
    # a run id that does not exist is rejected
    other = insert_row(conn, "runs", mode="record", status="ok")[next(iter(pk))]
    conn.execute("DELETE FROM spans WHERE run_id=%s", (run_id,))
    conn.execute("DELETE FROM runs WHERE " + next(iter(pk)) + "=%s", (other,))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_row(conn, "spans", run_id=other, seq=1)


def test_scores_pk_enforced(db_url: str, conn: Any) -> None:
    import psycopg

    migrated(db_url)
    run = insert_row(conn, "runs", mode="record", status="ok")
    rid = run[next(iter(_pk(conn, "runs")))]
    insert_row(conn, "scores", run_id=rid, check_name="c1")
    insert_row(conn, "scores", run_id=rid, check_name="c2")
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_row(conn, "scores", run_id=rid, check_name="c1")


# ---- pool ----
def test_get_pool_dict_row(db_url: str) -> None:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    from agent_replay.config import Settings
    from agent_replay.db.pool import get_pool

    pool = get_pool(Settings(database_url=db_url))
    try:
        assert isinstance(pool, ConnectionPool)
        pool.open()
        pool.wait(timeout=10)
        with pool.connection() as c:
            assert c.execute("SELECT 1 AS x").fetchone() == {"x": 1}
        assert pool.kwargs.get("row_factory") is dict_row
    finally:
        pool.close()


def test_list_migrations_sorted_and_sql_only(tmp_path: Path) -> None:
    for name in ("010_b.sql", "002_a.sql", "notes.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in list_migrations(tmp_path)] == ["002_a.sql", "010_b.sql"]


def test_packaged_migrations_start_with_init() -> None:
    assert list_migrations(MIGRATIONS_DIR)[0].name == "001_init.sql"


def test_migration_error_names_file() -> None:
    err = MigrationError("003_x.sql", ValueError("boom"))
    assert "003_x.sql" in str(err) and err.name == "003_x.sql"


def test_get_pool_requires_url() -> None:
    with pytest.raises(ValueError, match="DATABASE_URL"):
        get_pool(Settings(database_url=None))


def test_concurrent_migrations_serialize(empty_db_url: str) -> None:
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda _: apply_migrations(empty_db_url), range(4)))
    assert sorted(len(r) for r in results) == [0, 0, 0, 1]


def test_db_migrate_malformed_url_hides_password(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", "postgres//user:s3cret@host/db")
    args = argparse.Namespace(db_command="migrate")
    assert db_cmd.run(args) == 1
    err = capsys.readouterr().err
    assert "s3cret" not in err and "error:" in err


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:pw123@[::1/db",
        "postgresql://u:pw123@localhost:notaport/db",
    ],
)
def test_cli_bad_url_exits_nonzero_without_leaking_password(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent_replay.cli import main

    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", url)
    code = main(["db", "migrate"])
    out = capsys.readouterr()
    assert code != 0
    assert "pw123" not in out.out + out.err
