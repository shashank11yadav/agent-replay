"""SQL repositories for suite runs, runs, spans and scores.

Every function takes an open psycopg connection (dict rows) and plain dicts keyed by column
name. All SQL uses bound parameters; JSON values are wrapped in ``Jsonb``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from agent_replay.db.cache_repo import ImportConflict

Row = dict[str, Any]

_RUN_DEFAULTS: Row = {
    "id": None,
    "suite_run_id": None,
    "source_run_id": None,
    "task_id": "",
    "variant": "",
    "status": "ok",
    "error_type": None,
    "error": None,
    "final_output": None,
    "input_tokens": 0,
    "output_tokens": 0,
    "latency_ms": 0,
    "model_calls": 0,
    "tool_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "cost_usd": 0.0,
    "started_at": None,
    "finished_at": None,
}

_SPAN_DEFAULTS: Row = {
    "trace_id": "",
    "span_id": "",
    "parent_span_id": None,
    "kind": "",
    "name": "",
    "status": "ok",
    "error_type": None,
    "input_tokens": 0,
    "output_tokens": 0,
    "latency_ms": 0,
    "cost_usd": 0.0,
    "cache_status": None,
    "started_at": None,
}


def create_suite_run(
    conn: psycopg.Connection[Row],
    *,
    label: str = "",
    variant: str = "",
    mode: str = "record",
    source_suite_run_id: str | None = None,
    faults: list[Any] | None = None,
    tasks_path: str | None = None,
    tasks_sha256: str | None = None,
    task_count: int = 0,
) -> Any:
    """Insert a suite run (status 'running') and return its id."""
    row = conn.execute(
        """
        INSERT INTO suite_runs (
            label, variant, mode, source_suite_run_id, faults, tasks_path, tasks_sha256,
            task_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            label,
            variant,
            mode,
            source_suite_run_id,
            Jsonb(faults or []),
            tasks_path,
            tasks_sha256,
            task_count,
        ),
    ).fetchone()
    assert row is not None
    return row["id"]


def set_suite_status(
    conn: psycopg.Connection[Row], suite_run_id: str, status: str
) -> None:
    """Set a suite run's status; 'running' clears finished_at, other statuses set it."""
    conn.execute(
        """
        UPDATE suite_runs
        SET status = %(status)s,
            finished_at = CASE WHEN %(status)s = 'running' THEN NULL ELSE now() END
        WHERE id = %(id)s
        """,
        {"status": status, "id": suite_run_id},
    )


def list_suite_run_rows(conn: psycopg.Connection[Row], suite_run_id: str) -> list[Row]:
    """Every run of a suite (no paging), oldest first."""
    return conn.execute(
        "SELECT * FROM runs WHERE suite_run_id = %s ORDER BY started_at, id",
        (suite_run_id,),
    ).fetchall()


def get_suite_run(conn: psycopg.Connection[Row], suite_run_id: str) -> Row | None:
    return conn.execute(
        "SELECT * FROM suite_runs WHERE id = %s", (suite_run_id,)
    ).fetchone()


def insert_run(conn: psycopg.Connection[Row], run: Row) -> Any:
    """Insert a run (``mode`` is required) and return its id."""
    params = {**_RUN_DEFAULTS, **run}
    params["faults"] = Jsonb(run.get("faults") or [])
    row = conn.execute(
        """
        INSERT INTO runs (
            id, suite_run_id, source_run_id, task_id, variant, mode, status, error_type,
            error, final_output, faults, input_tokens, output_tokens, latency_ms,
            model_calls, tool_calls, cache_hits, cache_misses, cost_usd, started_at,
            finished_at
        )
        VALUES (
            COALESCE(%(id)s::uuid, gen_random_uuid()), %(suite_run_id)s, %(source_run_id)s,
            %(task_id)s, %(variant)s, %(mode)s, %(status)s, %(error_type)s, %(error)s,
            %(final_output)s, %(faults)s, %(input_tokens)s, %(output_tokens)s,
            %(latency_ms)s, %(model_calls)s, %(tool_calls)s, %(cache_hits)s,
            %(cache_misses)s, %(cost_usd)s, COALESCE(%(started_at)s::timestamptz, now()),
            %(finished_at)s
        )
        RETURNING id
        """,
        params,
    ).fetchone()
    assert row is not None
    return row["id"]


def get_run(conn: psycopg.Connection[Row], run_id: str) -> Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()


def insert_spans(conn: psycopg.Connection[Row], run_id: str, spans: list[Row]) -> None:
    """Insert span dicts (without run_id) for a run."""
    rows = []
    for span in spans:
        params = {**_SPAN_DEFAULTS, **span, "run_id": run_id}
        params["attributes"] = Jsonb(span.get("attributes") or {})
        params["payload"] = Jsonb(span.get("payload") or {})
        rows.append(params)
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO spans (
                run_id, seq, trace_id, span_id, parent_span_id, kind, name, status,
                error_type, attributes, payload, input_tokens, output_tokens, latency_ms,
                cost_usd, cache_status, started_at
            )
            VALUES (
                %(run_id)s, %(seq)s, %(trace_id)s, %(span_id)s, %(parent_span_id)s,
                %(kind)s, %(name)s, %(status)s, %(error_type)s, %(attributes)s,
                %(payload)s, %(input_tokens)s, %(output_tokens)s, %(latency_ms)s,
                %(cost_usd)s, %(cache_status)s, COALESCE(%(started_at)s::timestamptz, now())
            )
            """,
            rows,
        )


def list_spans(conn: psycopg.Connection[Row], run_id: str) -> list[Row]:
    return conn.execute(
        "SELECT * FROM spans WHERE run_id = %s ORDER BY seq", (run_id,)
    ).fetchall()


def save_run(conn: psycopg.Connection[Row], run: Row, spans: list[Row]) -> Any:
    """Write a run and its spans atomically, even on an autocommit connection."""
    with conn.transaction():
        run_id = insert_run(conn, run)
        insert_spans(conn, str(run_id), spans)
    return run_id


def list_runs(
    conn: psycopg.Connection[Row],
    *,
    suite_run_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Row]:
    """Runs ordered by started_at; ``suite_run_id=None`` means all suites."""
    return conn.execute(
        """
        SELECT * FROM runs
        WHERE (%(suite)s::uuid IS NULL OR suite_run_id = %(suite)s::uuid)
        ORDER BY started_at, id
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"suite": suite_run_id, "limit": limit, "offset": offset},
    ).fetchall()


def upsert_score(
    conn: psycopg.Connection[Row],
    run_id: str,
    check_name: str,
    passed: bool | None,
    detail: Row,
    kind: str = "deterministic",
) -> None:
    conn.execute(
        """
        INSERT INTO scores (run_id, check_name, kind, passed, detail)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id, check_name)
        DO UPDATE SET kind = EXCLUDED.kind, passed = EXCLUDED.passed, detail = EXCLUDED.detail
        """,
        (run_id, check_name, kind, passed, Jsonb(detail)),
    )


def delete_judge_score(conn: psycopg.Connection[Row], run_id: str) -> None:
    """Remove the run's judge score row (used when re-scoring without a judge)."""
    conn.execute(
        "DELETE FROM scores WHERE run_id = %s AND check_name = 'judge' AND kind = 'judge'",
        (run_id,),
    )


def detach_run(conn: psycopg.Connection[Row], run_id: str) -> None:
    """Remove a run from its suite (suite_run_id = NULL); the run and its spans stay."""
    conn.execute("UPDATE runs SET suite_run_id = NULL WHERE id = %s", (run_id,))


_SUITE_COLUMNS = (
    "id",
    "label",
    "variant",
    "mode",
    "status",
    "tasks_path",
    "tasks_sha256",
    "task_count",
    "source_suite_run_id",
    "faults",
    "metrics",
    "created_at",
    "finished_at",
)
_RUN_COLUMNS = (
    "id",
    "suite_run_id",
    "source_run_id",
    "task_id",
    "variant",
    "mode",
    "status",
    "error_type",
    "error",
    "final_output",
    "faults",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "model_calls",
    "tool_calls",
    "cache_hits",
    "cache_misses",
    "cost_usd",
    "started_at",
    "finished_at",
)
_SPAN_COLUMNS = (
    "id",
    "run_id",
    "seq",
    "trace_id",
    "span_id",
    "parent_span_id",
    "kind",
    "name",
    "status",
    "error_type",
    "attributes",
    "payload",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
    "cache_status",
    "started_at",
)
_SCORE_COLUMNS = ("run_id", "check_name", "kind", "passed", "detail", "created_at")


def list_suite_spans(conn: psycopg.Connection[Row], suite_run_id: str) -> list[Row]:
    """Every span of the suite's runs, in id order (stable for export)."""
    return conn.execute(
        """
        SELECT s.* FROM spans s JOIN runs r ON r.id = s.run_id
        WHERE r.suite_run_id = %s ORDER BY s.id
        """,
        (suite_run_id,),
    ).fetchall()


def list_suite_score_rows(
    conn: psycopg.Connection[Row], suite_run_id: str
) -> list[Row]:
    """Full score rows (all columns) of the suite's runs."""
    return conn.execute(
        """
        SELECT s.* FROM scores s JOIN runs r ON r.id = s.run_id
        WHERE r.suite_run_id = %s ORDER BY s.run_id, s.check_name
        """,
        (suite_run_id,),
    ).fetchall()


def _norm(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return value


def _sql_where(columns: tuple[str, ...]) -> str:
    return " AND ".join(c + " = %(" + c + ")s" for c in columns)


def _import_table(
    conn: psycopg.Connection[Row],
    *,
    table: str,
    label: str,
    columns: tuple[str, ...],
    key: tuple[str, ...],
    rows: list[Row],
    json_columns: tuple[str, ...] = (),
    other_key: tuple[str, ...] = (),
    parent_is_new: Callable[[Row], bool] | None = None,
    null_ok: tuple[str, ...] = (),
    insert_as: Callable[[Row], Row] | None = None,
) -> list[Row]:
    """Insert rows that do not exist yet; return those inserted.

    A row whose key exists is accepted only when every column is identical (columns in
    ``null_ok`` may be NULL in the database). A clash on ``other_key`` (another unique
    constraint), different content, or a new row under a parent that already existed
    raises ImportConflict. Table and column names are module constants; values are bound.
    """
    cols = ", ".join(columns)
    select = "SELECT " + cols + " FROM " + table + " WHERE "
    insert = (
        "INSERT INTO "
        + table
        + " ("
        + cols
        + ") VALUES ("
        + ", ".join("%(" + c + ")s" for c in columns)
        + ")"
    )
    inserted: list[Row] = []
    for row in rows:
        ident = ", ".join(c + "=" + str(row[c]) for c in key)
        existing = conn.execute(
            select + _sql_where(key), {c: row[c] for c in key}
        ).fetchone()
        if existing is not None:
            for c in columns:
                if existing[c] is None and c in null_ok:
                    continue
                if _norm(existing[c]) != _norm(row[c]):
                    raise ImportConflict(
                        f"{label} {ident} already exists with different content"
                    )
            continue
        if other_key:
            clash = conn.execute(
                select + _sql_where(other_key), {c: row[c] for c in other_key}
            ).fetchone()
            if clash is not None:
                raise ImportConflict(f"{label} {ident} clashes with an existing row")
        if parent_is_new is not None and not parent_is_new(row):
            raise ImportConflict(
                f"{label} {ident} would be added to an existing parent row"
            )
        params = insert_as(row) if insert_as else dict(row)
        for name in json_columns:
            if params[name] is not None:
                params[name] = Jsonb(params[name])
        conn.execute(insert, params)
        inserted.append(row)
    return inserted


def import_suite_run(conn: psycopg.Connection[Row], row: Row) -> bool:
    """Insert an exported suite run with its id; True when it was new.

    ``source_suite_run_id`` is set to NULL when that suite is not in this database.
    An existing suite with the same id must be identical.
    """
    source = row.get("source_suite_run_id")
    if source is not None and get_suite_run(conn, source) is None:
        row = {**row, "source_suite_run_id": None}
    return bool(
        _import_table(
            conn,
            table="suite_runs",
            label="suite run",
            columns=_SUITE_COLUMNS,
            key=("id",),
            rows=[row],
            json_columns=("faults", "metrics"),
            null_ok=("source_suite_run_id",),
        )
    )


def import_runs(
    conn: psycopg.Connection[Row], rows: list[Row], suite_is_new: bool
) -> set[str]:
    """Insert exported runs with their ids; returns the ids that were new.

    ``source_run_id`` is restored only for new runs whose source exists in this database
    after the import; otherwise it stays NULL.
    """
    new = _import_table(
        conn,
        table="runs",
        label="run",
        columns=_RUN_COLUMNS,
        key=("id",),
        rows=rows,
        json_columns=("faults",),
        null_ok=("source_run_id",),
        parent_is_new=lambda _row: suite_is_new,
        insert_as=lambda r: {**r, "source_run_id": None},
    )
    for row in new:
        if row["source_run_id"] is not None:
            conn.execute(
                """
                UPDATE runs SET source_run_id = %(src)s
                WHERE id = %(id)s AND EXISTS (SELECT 1 FROM runs WHERE id = %(src)s)
                """,
                {"src": row["source_run_id"], "id": row["id"]},
            )
    return {str(r["id"]) for r in new}


def import_spans(
    conn: psycopg.Connection[Row], rows: list[Row], new_run_ids: set[str]
) -> int:
    """Insert exported spans, identified by (run_id, seq); the database assigns ``id``.

    The exported ``id`` is not kept: another database's spans may already use it.
    """
    columns = tuple(c for c in _SPAN_COLUMNS if c != "id")
    done = _import_table(
        conn,
        table="spans",
        label="span",
        columns=columns,
        key=("run_id", "seq"),
        rows=rows,
        json_columns=("attributes", "payload"),
        parent_is_new=lambda r: str(r["run_id"]) in new_run_ids,
        insert_as=lambda r: {c: r[c] for c in columns},
    )
    return len(done)


def import_scores(
    conn: psycopg.Connection[Row], rows: list[Row], new_run_ids: set[str]
) -> int:
    """Insert exported score rows; existing rows must be identical."""
    return len(
        _import_table(
            conn,
            table="scores",
            label="score",
            columns=_SCORE_COLUMNS,
            key=("run_id", "check_name"),
            rows=rows,
            json_columns=("detail",),
            parent_is_new=lambda r: str(r["run_id"]) in new_run_ids,
        )
    )


def list_suite_runs(
    conn: psycopg.Connection[Row], *, limit: int = 50, offset: int = 0
) -> list[Row]:
    """Suite runs, newest first."""
    return conn.execute(
        """
        SELECT * FROM suite_runs
        ORDER BY created_at DESC, id DESC
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"limit": limit, "offset": offset},
    ).fetchall()


def list_scores_for_runs(
    conn: psycopg.Connection[Row], run_ids: list[str]
) -> list[Row]:
    """Every score row of the given runs, in one query."""
    if not run_ids:
        return []
    return conn.execute(
        "SELECT * FROM scores WHERE run_id = ANY(%s::uuid[]) ORDER BY run_id, check_name",
        (run_ids,),
    ).fetchall()


def list_run_scores(conn: psycopg.Connection[Row], run_id: str) -> list[Row]:
    """Every score row of one run, ordered by check name."""
    return conn.execute(
        "SELECT * FROM scores WHERE run_id = %s ORDER BY check_name", (run_id,)
    ).fetchall()
