from __future__ import annotations

import pytest

from agent_replay.cli import main
from agent_replay.config import Settings


def test_safe_dict_reports_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_GEMINI_API_KEY", "secret-value")
    data = Settings().safe_dict()
    assert data["gemini_api_key"] == "set"
    assert "secret-value" not in str(data)


def test_safe_dict_unset_key() -> None:
    assert Settings().safe_dict()["gemini_api_key"] == "unset"


def test_main_without_command_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_database_url_password_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql://bob:hunter2@db:5432/app"
    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", url)
    s = Settings()
    assert s.database_url == url
    assert s.safe_dict()["database_url"] == "postgresql://bob:***@db:5432/app"
    for text in (str(s.safe_dict()), repr(s), str(s)):
        assert "hunter2" not in text


def test_database_url_without_password_unchanged() -> None:
    from agent_replay.config import redact_url

    assert redact_url(None) is None
    assert redact_url("postgresql://db/app") == "postgresql://db/app"


def test_extra_fields_forbidden() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(bogus=1)  # type: ignore[call-arg]


def test_safe_dict_is_json_serialisable(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.setenv("AGENT_REPLAY_VIEWER_DIST", "/tmp/x")
    json.dumps(Settings().safe_dict())


def test_whitespace_api_key_is_not_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_GEMINI_API_KEY", "   ")
    assert Settings().has_gemini_key is False


def test_version_matches_pyproject() -> None:
    import tomllib
    from pathlib import Path

    import agent_replay

    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert agent_replay.__version__ == data["project"]["version"]
