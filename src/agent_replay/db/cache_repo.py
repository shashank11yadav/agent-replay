"""Model response cache stored in Postgres."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

Row = dict[str, Any]


class CacheRepo:
    def __init__(self, conn: psycopg.Connection[Row]) -> None:
        self._conn = conn

    def get(self, request_hash: str) -> Row | None:
        row = self._conn.execute(
            "SELECT response FROM model_cache WHERE request_hash = %s", (request_hash,)
        ).fetchone()
        return None if row is None else row["response"]

    def put(
        self,
        request_hash: str,
        *,
        provider: str,
        model: str,
        request: Row,
        response: Row,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        """Store a response; an existing entry is kept unless ``overwrite`` is set."""
        conflict = (
            """DO UPDATE SET
                provider = EXCLUDED.provider, model = EXCLUDED.model,
                request = EXCLUDED.request, response = EXCLUDED.response,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                latency_ms = EXCLUDED.latency_ms, cost_usd = EXCLUDED.cost_usd"""
            if overwrite
            else "DO NOTHING"
        )
        self._conn.execute(
            f"""
            INSERT INTO model_cache (
                request_hash, provider, model, request, response,
                input_tokens, output_tokens, latency_ms, cost_usd
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (request_hash) {conflict}
            """,
            (
                request_hash,
                provider,
                model,
                Jsonb(request),
                Jsonb(response),
                input_tokens,
                output_tokens,
                latency_ms,
                cost_usd,
            ),
        )


_CACHE_COLUMNS = (
    "request_hash",
    "provider",
    "model",
    "request",
    "response",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
    "created_at",
)
_CACHE_JSON = ("request", "response")


def get_entries(conn: psycopg.Connection[Row], request_hashes: list[str]) -> list[Row]:
    """Full cache rows for the given hashes (unknown hashes are skipped), by hash."""
    if not request_hashes:
        return []
    return conn.execute(
        "SELECT * FROM model_cache WHERE request_hash = ANY(%s) ORDER BY request_hash",
        (request_hashes,),
    ).fetchall()


class ImportConflict(Exception):
    """An imported cache row clashes with different data already stored."""


def import_entries(conn: psycopg.Connection[Row], rows: list[Row]) -> int:
    """Insert cache rows with their original created_at.

    An existing hash is accepted only when provider, model, request and response match.
    """
    inserted = 0
    sql = (
        "INSERT INTO model_cache ("
        + ", ".join(_CACHE_COLUMNS)
        + ") VALUES ("
        + ", ".join("%(" + c + ")s" for c in _CACHE_COLUMNS)
        + ")"
    )
    for row in rows:
        existing = conn.execute(
            "SELECT provider, model, request, response FROM model_cache "
            "WHERE request_hash = %s",
            (row["request_hash"],),
        ).fetchone()
        if existing is not None:
            if any(existing[c] != row[c] for c in existing):
                raise ImportConflict(
                    f"cache entry {row['request_hash']} already exists with different content"
                )
            continue
        params = {c: row[c] for c in _CACHE_COLUMNS}
        for key in _CACHE_JSON:
            params[key] = Jsonb(params[key])
        conn.execute(sql, params)
        inserted += 1
    return inserted
