"""Unit tests for repo helpers that need no database."""

from __future__ import annotations

from typing import Any

from agent_replay.db import runs_repo
from agent_replay.db.cache_repo import CacheRepo


class FakeConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def execute(self, query: str, params: Any = None) -> FakeConn:
        self.calls.append((query, params))
        return self

    def fetchone(self) -> None:
        return None


def test_cache_get_unknown_returns_none() -> None:
    assert CacheRepo(FakeConn()).get("x") is None  # type: ignore[arg-type]


def test_get_run_binds_id_as_parameter() -> None:
    conn = FakeConn()
    assert runs_repo.get_run(conn, "o'brien") is None  # type: ignore[arg-type]
    assert conn.calls[0][1] == ("o'brien",)
    assert "o'brien" not in conn.calls[0][0]
