"""Throwaway Postgres databases for tests.

Every database is created on the server named by AGENT_REPLAY_TEST_DATABASE_URL (or the
local default) and dropped with FORCE afterwards, so a failed test never leaves open
connections behind. An unreachable server fails the test instead of skipping it: a green
run must mean the database tests actually ran.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


def admin_url() -> str:
    return os.environ.get("AGENT_REPLAY_TEST_DATABASE_URL") or DEFAULT_URL


def with_dbname(url: str, name: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path=f"/{name}"))


def connect(url: str, **kw: Any) -> Any:
    try:
        return psycopg.connect(url, connect_timeout=5, **kw)
    except psycopg.OperationalError as exc:
        pytest.fail(
            "Postgres is unreachable at AGENT_REPLAY_TEST_DATABASE_URL "
            f"(default {DEFAULT_URL}): {exc}. Start the compose `db` service "
            "or set the variable; this is a failure, not a skip.",
            pytrace=False,
        )


def create_database(
    prefix: str = "ar_test", *, migrate: bool = True
) -> tuple[str, str]:
    """Create a fresh database and return ``(name, url)``."""
    name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    base = admin_url()
    with connect(base, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    url = with_dbname(base, name)
    if migrate:
        from agent_replay.db.migrate import apply_migrations

        try:
            apply_migrations(url)
        except BaseException:
            drop_database(name)
            raise
    return name, url


def drop_database(name: str) -> None:
    with connect(admin_url(), autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@contextmanager
def throwaway_database(
    prefix: str = "ar_test", *, migrate: bool = True
) -> Iterator[str]:
    """Yield the URL of a new database and drop it on exit."""
    name, url = create_database(prefix, migrate=migrate)
    try:
        yield url
    finally:
        drop_database(name)
