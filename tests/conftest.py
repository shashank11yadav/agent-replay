"""Shared pytest configuration and the throwaway Postgres fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from psycopg.rows import dict_row
from support import pg


def pytest_configure() -> None:
    # Tests must never see a real key, even if the developer shell has one.
    os.environ["AGENT_REPLAY_GEMINI_API_KEY"] = ""
    # No test may start the real Claude Code CLI; claude_code tests point this at a fake.
    os.environ["AGENT_REPLAY_CLAUDE_CLI"] = ""


@pytest.fixture
def db_url() -> Iterator[str]:
    """A migrated database, one per test."""
    with pg.throwaway_database() as url:
        yield url


@pytest.fixture
def conn(db_url: str) -> Iterator[Any]:
    """Repository connection: dict rows, autocommit (repos manage own transactions)."""
    c = pg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def raw(db_url: str) -> Iterator[Any]:
    """Independent tuple-row connection for checking what was really stored."""
    c = pg.connect(db_url, autocommit=True)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def db_url2() -> Iterator[str]:
    """A second migrated database, e.g. the empty target of an import."""
    with pg.throwaway_database() as url:
        yield url


@pytest.fixture
def raw2(db_url2: str) -> Iterator[Any]:
    c = pg.connect(db_url2, autocommit=True)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def empty_db_url() -> Iterator[str]:
    """A database with no migrations applied."""
    with pg.throwaway_database(migrate=False) as url:
        yield url


@pytest.fixture
def source_db_url() -> Iterator[str]:
    """Migrated database where a scripted suite is recorded."""
    with pg.throwaway_database() as url:
        yield url


@pytest.fixture
def target_db_url() -> Iterator[str]:
    """Second migrated, empty database (the import target after a wipe)."""
    with pg.throwaway_database() as url:
        yield url


@pytest.fixture
def ci_db_url() -> Iterator[str]:
    """Empty, unmigrated database, like the workflow's fresh postgres service."""
    with pg.throwaway_database(migrate=False) as url:
        yield url


@pytest.fixture(scope="module")
def mod_source_db_url() -> Iterator[str]:
    """Migrated database shared by one module's tests (recording a suite is slow)."""
    with pg.throwaway_database() as url:
        yield url


@pytest.fixture(scope="module")
def mod_source_db_url2() -> Iterator[str]:
    with pg.throwaway_database() as url:
        yield url
