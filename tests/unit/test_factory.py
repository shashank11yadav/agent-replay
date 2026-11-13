from __future__ import annotations

import pytest

from agent_replay.config import Settings
from agent_replay.llm.factory import build_provider, make_client


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        build_provider(Settings(), "bogus", cache=object())  # type: ignore[arg-type]


def test_record_mode_skips_cache_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_SCRIPT_PATH", "x")
    monkeypatch.setattr(
        "agent_replay.llm.fake.ScriptedProvider.from_file", lambda p: object()
    )
    s = Settings(provider="scripted")
    rec = build_provider(s, "record", cache=object())
    rep = build_provider(s, "replay", cache=object())
    assert (rec._read, rec._overwrite) == (False, True)
    assert (rep._read, rep._overwrite) == (True, False)


def test_judge_client_uses_the_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_replay.cli import common
    from agent_replay.llm.client import LLMClient

    built: list[LLMClient] = []

    def spy(provider: object, settings: Settings) -> LLMClient:
        client = make_client(provider, settings)  # type: ignore[arg-type]
        built.append(client)
        return client

    monkeypatch.setattr(common, "make_client", spy)
    s = Settings(max_attempts=5, backoff_base_s=0.25)
    common.build_judge(object(), s)  # type: ignore[arg-type]
    assert (built[0]._max_attempts, built[0]._backoff_base_s) == (5, 0.25)
