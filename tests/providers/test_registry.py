"""Provider registry, the factory without provider branches, and build semantics."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_replay.config import Settings
from agent_replay.llm import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ScriptedProvider,
)
from agent_replay.llm.ratelimit import RateLimitedProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("AGENT_REPLAY_"):
            monkeypatch.delenv(key)
    # The default provider is claude_code: never let it find a real `claude` on PATH.
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "")


class Sentinel:
    """A stand-in provider; identity is what the tests compare."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(text="dummy")


def make_spec(**overrides: Any) -> Any:
    from agent_replay.llm import registry

    fields: dict[str, Any] = {
        "name": "dummy",
        "build": lambda settings, mode: Sentinel(),
        "needs_credential_to_record": False,
        "default_model": "dummy-m",
        "default_judge_model": None,
        "default_prices": (5.0, 6.0),
        "model_prices": {},
        "rate_limited": False,
    }
    fields.update(overrides)
    return registry.ProviderSpec(**fields)


@pytest.fixture
def dummy() -> Iterator[Any]:
    """Register a throwaway 'dummy' spec through a helper and always clean up."""
    from agent_replay.llm import registry

    registered: list[str] = []

    def install(**overrides: Any) -> Any:
        spec = make_spec(**overrides)
        registry.register(spec)
        registered.append(spec.name)
        return spec

    yield install
    for name in registered:
        if name in registry.provider_names():
            registry.unregister(name)


def test_registry_api_and_spec_fields(dummy: Any) -> None:
    from agent_replay.llm import registry

    spec = dummy(model_prices={"dummy-m": (1.0, 2.0)}, default_judge_model="dummy-j")
    assert registry.get_provider_spec("dummy") is spec
    assert spec.name == "dummy"
    assert spec.needs_credential_to_record is False
    assert spec.default_model == "dummy-m"
    assert spec.default_judge_model == "dummy-j"
    assert spec.default_prices == (5.0, 6.0)
    assert spec.model_prices == {"dummy-m": (1.0, 2.0)}
    assert spec.rate_limited is False
    assert callable(spec.build)
    names = registry.provider_names()
    assert "dummy" in names
    assert list(names) == sorted(names)
    registry.unregister("dummy")
    assert "dummy" not in registry.provider_names()


def test_duplicate_register_and_unknown_unregister_raise(dummy: Any) -> None:
    from agent_replay.llm import registry

    dummy()
    with pytest.raises(ValueError):
        registry.register(make_spec())
    with pytest.raises(ValueError):
        registry.unregister("never-registered")


def test_builtin_providers_registered_after_factory_import() -> None:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import registry

    assert {"gemini", "scripted"} <= set(registry.provider_names())


def test_gemini_spec_defaults() -> None:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import registry

    spec = registry.get_provider_spec("gemini")
    assert spec.name == "gemini"
    assert spec.default_model == "gemini-3.5-flash-lite"
    assert spec.default_judge_model is None
    assert spec.default_prices == (0.30, 2.50)
    assert spec.rate_limited is True
    assert spec.needs_credential_to_record is True


def test_scripted_spec_defaults() -> None:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import registry

    spec = registry.get_provider_spec("scripted")
    assert spec.name == "scripted"
    assert spec.default_model == "gemini-3.5-flash-lite"
    assert spec.default_judge_model is None
    assert spec.default_prices == (0.30, 2.50)
    assert spec.rate_limited is False
    assert spec.needs_credential_to_record is False


def test_factory_wraps_rate_limited_dummy_with_shared_limiter(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_replay.llm.factory import build_live_provider, get_limiter

    inner = Sentinel()
    dummy(rate_limited=True, build=lambda settings, mode: inner)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    settings = Settings()
    for mode in ("replay", "record"):
        built = build_live_provider(settings, mode)  # type: ignore[arg-type]
        assert isinstance(built, RateLimitedProvider)
        assert built._provider is inner
        assert built._limiter is get_limiter(settings)


def test_factory_returns_unwrapped_dummy_when_not_rate_limited(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_replay.llm.factory import build_live_provider

    inner = Sentinel()
    dummy(rate_limited=False, build=lambda settings, mode: inner)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    assert build_live_provider(Settings(), "replay") is inner
    assert build_live_provider(Settings(), "record") is inner


def test_factory_passes_settings_and_mode_to_build(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_replay.llm.factory import build_live_provider

    seen: list[tuple[Any, str]] = []

    def build(settings: Any, mode: str) -> ModelProvider:
        seen.append((settings, mode))
        return Sentinel()

    dummy(build=build)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    settings = Settings()
    build_live_provider(settings, "replay")
    build_live_provider(settings, "record")
    assert [m for _, m in seen] == ["replay", "record"]
    assert all(s is settings for s, _ in seen)


def test_replay_none_passes_through_and_record_error_propagates(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_replay.llm.factory import build_live_provider

    def build(settings: Any, mode: str) -> ModelProvider | None:
        if mode == "record":
            raise ValueError("no credential for dummy")
        return None

    dummy(build=build, rate_limited=True)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    settings = Settings()
    assert build_live_provider(settings, "replay") is None  # not wrapped when None
    with pytest.raises(ValueError, match="no credential for dummy"):
        build_live_provider(settings, "record")


def test_gemini_without_key_behaves_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.llm.factory import build_live_provider

    settings = Settings(provider="gemini")
    assert build_live_provider(settings, "replay") is None
    with pytest.raises(ValueError) as info:
        build_live_provider(settings, "record")
    assert (
        str(info.value)
        == "A Gemini API key is required to record: set AGENT_REPLAY_GEMINI_API_KEY"
    )


def test_gemini_with_key_is_rate_limited_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.llm.factory import build_live_provider, get_limiter
    from agent_replay.llm.gemini import GeminiProvider

    monkeypatch.setenv("AGENT_REPLAY_GEMINI_API_KEY", "AIzaSyFAKE-KEY-0000000000")
    settings = Settings(provider="gemini")
    for mode in ("replay", "record"):
        built = build_live_provider(settings, mode)  # type: ignore[arg-type]
        assert isinstance(built, RateLimitedProvider)
        assert isinstance(built._provider, GeminiProvider)
        assert built._limiter is get_limiter(settings)


def test_scripted_without_script_path_behaves_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.llm.factory import build_live_provider

    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    settings = Settings()
    assert build_live_provider(settings, "replay") is None
    with pytest.raises(ValueError) as info:
        build_live_provider(settings, "record")
    assert (
        str(info.value)
        == "AGENT_REPLAY_SCRIPT_PATH is required for the scripted provider"
    )


def test_scripted_with_script_path_is_unwrapped_scripted_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_replay.llm.factory import build_live_provider

    script = tmp_path / "script.json"
    script.write_text('[{"text": "hi"}]', encoding="utf-8")
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    monkeypatch.setenv("AGENT_REPLAY_SCRIPT_PATH", str(script))
    settings = Settings()
    for mode in ("replay", "record"):
        built = build_live_provider(settings, mode)  # type: ignore[arg-type]
        assert isinstance(built, ScriptedProvider)


def test_unknown_provider_lists_registered_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.llm import registry
    from agent_replay.llm.factory import build_live_provider

    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "nope")
    settings = Settings()  # must still construct
    assert settings.provider == "nope"
    for mode in ("replay", "record"):
        with pytest.raises(ValueError) as info:
            build_live_provider(settings, mode)  # type: ignore[arg-type]
        message = str(info.value)
        assert "nope" in message
        for name in registry.provider_names():
            assert name in message
        assert "gemini" in message and "scripted" in message


def test_tracker_and_limiter_shared_and_used_by_build_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_replay.llm.factory import build_provider, get_limiter, get_tracker

    a, b = Settings(), Settings()
    assert get_tracker(a) is get_tracker(b)
    assert get_limiter(a) is get_limiter(b)

    script = tmp_path / "script.json"
    script.write_text('[{"text": "hi"}]', encoding="utf-8")
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    monkeypatch.setenv("AGENT_REPLAY_SCRIPT_PATH", str(script))
    s = Settings()
    chain = build_provider(s, "replay", cache=object())
    assert chain.live is not None
    assert chain.live.tracker is get_tracker(s)  # type: ignore[attr-defined]
    assert chain._tracker is get_tracker(s)
