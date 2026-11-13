"""Provider-aware settings defaults and the judge_model setting."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from support import scrubbed_env

from agent_replay.config import Settings
from agent_replay.demo.agent import DemoAgent
from agent_replay.llm import ModelRequest, ModelResponse
from agent_replay.scoring.judge import build_judge_request
from agent_replay.taskspec import TaskSpec

TASK = TaskSpec(
    id="t1",
    prompt="What is the price of product A1003?",
    category="lookup",
    expected_text=["37.95"],
    must_call_tools=["search"],
    max_steps=4,
    rubric="The answer states the price 37.95 EUR.",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("AGENT_REPLAY_"):
            monkeypatch.delenv(key)
    # The default provider is claude_code: never let it find a real `claude` on PATH.
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "")


@pytest.fixture
def dummy() -> Iterator[Any]:
    from agent_replay.llm import registry

    names: list[str] = []

    def install(**overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "name": "dummy",
            "build": lambda settings, mode: None,
            "needs_credential_to_record": False,
            "default_model": "dummy-m",
            "default_judge_model": None,
            "default_prices": (5.0, 6.0),
            "model_prices": {},
            "rate_limited": False,
        }
        fields.update(overrides)
        spec = registry.ProviderSpec(**fields)
        registry.register(spec)
        names.append(spec.name)
        return spec

    yield install
    for name in names:
        if name in registry.provider_names():
            registry.unregister(name)


def test_default_settings_values() -> None:
    s = Settings()
    assert s.provider == "claude_code"
    assert s.model == "claude-haiku-5-5"
    assert s.judge_model == "claude-sonnet-5-5"
    assert (s.price_input_per_m, s.price_output_per_m) == (0.10, 0.50)
    assert (s.judge_price_input_per_m, s.judge_price_output_per_m) == (2.00, 10.00)


def test_gemini_settings_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "gemini")
    s = Settings()
    assert s.model == "gemini-3.5-flash-lite"
    assert s.judge_model == "gemini-3.5-flash-lite"
    assert s.price_input_per_m == 0.30
    assert s.price_output_per_m == 2.50


def test_agent_model_env_applies_to_every_provider(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy()
    for provider in ("gemini", "scripted", "dummy", "nope"):
        monkeypatch.setenv("AGENT_REPLAY_PROVIDER", provider)
        monkeypatch.setenv("AGENT_REPLAY_MODEL", "x")
        assert Settings().model == "x"


def test_judge_model_env_and_defaults(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gemini: judge follows the effective agent model, as the judge used settings.model.
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "gemini")
    monkeypatch.setenv("AGENT_REPLAY_MODEL", "x")
    assert Settings().judge_model == "x"
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "j")
    s = Settings()
    assert (s.model, s.judge_model) == ("x", "j")
    monkeypatch.delenv("AGENT_REPLAY_MODEL")
    s = Settings()
    assert (s.model, s.judge_model) == ("gemini-3.5-flash-lite", "j")
    monkeypatch.delenv("AGENT_REPLAY_JUDGE_MODEL")

    # a provider with its own judge default
    dummy(default_judge_model="dummy-j")
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    s = Settings()
    assert (s.model, s.judge_model) == ("dummy-m", "dummy-j")
    monkeypatch.setenv("AGENT_REPLAY_MODEL", "x")
    s = Settings()
    assert (s.model, s.judge_model) == ("x", "dummy-j")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "j")
    assert Settings().judge_model == "j"


def test_provider_defaults_and_model_prices(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy(model_prices={"dummy-m": (1.0, 2.0)})
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    s = Settings()
    assert s.model == "dummy-m"
    assert (s.price_input_per_m, s.price_output_per_m) == (1.0, 2.0)

    # a model without a table entry falls back to the provider's default prices
    monkeypatch.setenv("AGENT_REPLAY_MODEL", "other")
    s = Settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (5.0, 6.0)

    # explicit price variables win over both
    monkeypatch.setenv("AGENT_REPLAY_PRICE_INPUT_PER_M", "9")
    monkeypatch.setenv("AGENT_REPLAY_PRICE_OUTPUT_PER_M", "11")
    s = Settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (9.0, 11.0)
    monkeypatch.delenv("AGENT_REPLAY_MODEL")
    s = Settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (9.0, 11.0)


def test_only_one_price_variable_set_overrides_only_that_one(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy(model_prices={"dummy-m": (1.0, 2.0)})
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "dummy")
    monkeypatch.setenv("AGENT_REPLAY_PRICE_OUTPUT_PER_M", "11")
    s = Settings()
    assert (s.price_input_per_m, s.price_output_per_m) == (1.0, 11.0)


def test_unregistered_provider_takes_the_default_provider_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "nope")
    s = Settings()
    assert s.provider == "nope"
    assert s.model == "claude-haiku-5-5"
    assert s.judge_model == "claude-sonnet-5-5"
    assert (s.price_input_per_m, s.price_output_per_m) == (0.10, 0.50)


def test_resolved_fields_are_always_concrete(
    dummy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy(default_judge_model=None)
    for provider in ("gemini", "scripted", "dummy", "nope"):
        monkeypatch.setenv("AGENT_REPLAY_PROVIDER", provider)
        s = Settings()
        assert isinstance(s.model, str) and s.model
        assert isinstance(s.judge_model, str) and s.judge_model
        assert isinstance(s.price_input_per_m, float)
        assert isinstance(s.price_output_per_m, float)
        assert isinstance(Settings.model_validate({}).judge_model, str)


def test_importing_config_does_not_import_google_genai() -> None:
    env = scrubbed_env()
    code = (
        "import sys; import agent_replay.config; print('google.genai' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip() == "False"


def test_judge_request_uses_judge_model_and_demo_agent_uses_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_MODEL", "agent-m")
    monkeypatch.setenv("AGENT_REPLAY_JUDGE_MODEL", "judge-m")
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    s = Settings()
    judge = build_judge_request(TASK, "answer", s)
    assert judge.model == "judge-m"
    assert judge.provider == "scripted"

    seen: list[ModelRequest] = []

    class Ctx:
        def call_model(self, request: ModelRequest) -> ModelResponse:
            seen.append(request)
            return ModelResponse(text="x")

        def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            raise AssertionError("no tool call expected")

    DemoAgent("minimal", s).run(TASK, Ctx())  # type: ignore[arg-type]
    assert seen[0].model == "agent-m"
    assert seen[0].provider == "scripted"
