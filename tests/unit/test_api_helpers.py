from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from agent_replay.api import create_app
from agent_replay.api.schemas import UUID_PATTERN, to_json
from agent_replay.config import Settings


def test_to_json_converts_db_types() -> None:
    u = UUID("12345678-1234-5678-1234-567812345678")
    when = datetime(2026, 1, 1, tzinfo=UTC)
    out = to_json({"a": [u, Decimal("0.0100")], "b": (when,), "c": 1})
    assert out == {
        "a": [str(u), "0.0100"],
        "b": [when.isoformat()],
        "c": 1,
    }


def test_uuid_pattern_rejects_garbage() -> None:
    import re

    assert re.match(UUID_PATTERN, "12345678-1234-5678-1234-567812345678")
    assert not re.match(UUID_PATTERN, "not-a-uuid")


def test_validation_rejects_before_touching_db() -> None:
    # No lifespan (no pool): 422 comes from validation alone.
    app = create_app(Settings(database_url="postgresql://u:p@localhost:1/x"))
    c = TestClient(app, base_url="http://localhost")
    assert c.get("/api/runs/not-a-uuid").status_code == 422
    assert c.get("/api/runs?limit=0").status_code == 422
    assert c.get("/api/diff?left=x&right=y").status_code == 422


_DSN = "postgresql://u:p@localhost:1/x"


@pytest.mark.parametrize("origins", [["*"], ["http://a.test", "*"], [""]])
def test_cors_rejects_wildcard_and_empty(origins: list[str]) -> None:
    with pytest.raises(ValueError, match="CORS"):
        create_app(Settings(database_url=_DSN, cors_origins=origins))


def test_docs_endpoints_disabled() -> None:
    c = TestClient(create_app(Settings(database_url=_DSN)), base_url="http://localhost")
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


def test_lifespan_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))
    with pytest.raises(ValueError, match="DATABASE_URL"), TestClient(app):
        pass


@pytest.mark.parametrize(
    ("raw", "hosts"),
    [
        ("api.example.com", ["api.example.com"]),
        ("localhost, 127.0.0.1,", ["localhost", "127.0.0.1"]),
        ('["a.example", "b.example"]', ["a.example", "b.example"]),
    ],
)
def test_allowed_hosts_accepts_a_comma_list_or_json(
    monkeypatch: pytest.MonkeyPatch, raw: str, hosts: list[str]
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_ALLOWED_HOSTS", raw)
    assert Settings().allowed_hosts == hosts


def test_allowed_hosts_default_is_local_only() -> None:
    assert Settings().allowed_hosts == ["localhost", "127.0.0.1"]


def test_an_empty_allowed_hosts_list_is_refused() -> None:
    with pytest.raises(ValueError, match="ALLOWED_HOSTS"):
        create_app(Settings(database_url=_DSN, allowed_hosts=[]))
