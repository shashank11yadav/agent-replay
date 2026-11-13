"""claude_code registry spec, settings, CLI discovery and the session conftest."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from claude_cli_harness import FakeClaude, clean_agent_env
from support import ROOT, scrubbed_env


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    clean_agent_env(monkeypatch)


@pytest.fixture
def fake(tmp_path: Path) -> FakeClaude:
    return FakeClaude(tmp_path)


def _spec() -> Any:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import registry

    return registry.get_provider_spec("claude_code")


# ---- the registered spec ---------------------------------------------------------------


def test_claude_code_is_a_registered_provider_name() -> None:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import registry

    assert "claude_code" in registry.provider_names()


def test_claude_code_spec_fields() -> None:
    spec = _spec()
    assert spec.name == "claude_code"
    assert spec.default_model == "claude-haiku-5-5"
    assert spec.default_judge_model == "claude-sonnet-5-5"
    assert tuple(spec.default_prices) == (0.10, 0.50)
    assert {k: tuple(v) for k, v in spec.model_prices.items()} == {
        "claude-haiku-5-5": (0.10, 0.50),
        "claude-sonnet-5-5": (2.00, 10.00),
    }
    assert spec.needs_credential_to_record is False
    assert spec.rate_limited is True
    assert callable(spec.build)


# ---- settings --------------------------------------------------------------------------


def test_settings_defaults_for_the_cli_variables() -> None:
    from agent_replay.config import Settings

    s = Settings()
    assert s.claude_cli == "claude"
    assert s.claude_max_turns is None


def test_settings_read_cli_and_max_turns_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.config import Settings

    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "/opt/bin/claude-x")
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_MAX_TURNS", "4")
    s = Settings()
    assert s.claude_cli == "/opt/bin/claude-x"
    assert s.claude_max_turns == 4


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_max_turns_must_be_at_least_one(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    from pydantic import ValidationError

    from agent_replay.config import Settings

    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_MAX_TURNS", bad)
    with pytest.raises(ValidationError):
        Settings()


def test_max_turns_one_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_replay.config import Settings

    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_MAX_TURNS", "1")
    assert Settings().claude_max_turns == 1


def test_claude_code_provider_resolves_models_and_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.config import Settings

    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "claude_code")
    s = Settings()
    assert s.model == "claude-haiku-5-5"
    assert s.judge_model == "claude-sonnet-5-5"
    assert (s.price_input_per_m, s.price_output_per_m) == (0.10, 0.50)


# ---- how the CLI is found --------------------------------------------------------------


def _build(monkeypatch: pytest.MonkeyPatch, cli: str, mode: str, **env: str) -> Any:
    from agent_replay.config import Settings
    from agent_replay.llm.factory import build_live_provider

    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "claude_code")
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", cli)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return build_live_provider(Settings(), mode)  # type: ignore[arg-type]


@pytest.fixture
def no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)


def _not_found_paths(tmp_path: Path) -> dict[str, str]:
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\n", encoding="utf-8")
    plain.chmod(0o644)
    adir = tmp_path / "a-directory"
    adir.mkdir()
    return {
        "empty": "",
        "missing-path": str(tmp_path / "nope" / "claude"),
        "bare-name-not-on-path": "definitely-not-a-real-claude-cli-xyz",
        "not-executable-file": str(plain),
        "directory": str(adir),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "missing-path",
        "bare-name-not-on-path",
        "not-executable-file",
        "directory",
    ],
)
def test_replay_without_a_found_cli_is_cache_only_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_process: None, kind: str
) -> None:
    cli = _not_found_paths(tmp_path)[kind]
    assert _build(monkeypatch, cli, "replay") is None


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "missing-path",
        "bare-name-not-on-path",
        "not-executable-file",
        "directory",
    ],
)
def test_record_without_a_found_cli_names_the_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_process: None, kind: str
) -> None:
    cli = _not_found_paths(tmp_path)[kind]
    with pytest.raises(ValueError, match="AGENT_REPLAY_CLAUDE_CLI"):
        _build(monkeypatch, cli, "record")


@pytest.mark.parametrize("mode", ["record", "replay"])
def test_an_executable_path_is_found(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, mode: str
) -> None:
    provider = _build(monkeypatch, fake.path, mode)
    assert provider is not None
    assert callable(provider.generate)


def test_a_bare_name_is_searched_on_path(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    monkeypatch.setenv("PATH", str(fake.dir))
    provider = _build(monkeypatch, "claude", "replay")
    assert provider is not None


def test_a_bare_name_missing_from_path_is_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_process: None
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert _build(monkeypatch, "claude", "replay") is None


def test_unknown_provider_error_lists_claude_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_replay.config import Settings
    from agent_replay.llm.factory import build_live_provider

    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "nope")
    with pytest.raises(ValueError, match="claude_code"):
        build_live_provider(Settings(), "replay")


# ---- conftest --------------------------------------------------------------------------


def test_conftest_sets_the_cli_variable_to_an_empty_string() -> None:
    text = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert re.search(r"AGENT_REPLAY_CLAUDE_CLI", text)
    assert re.search(r"AGENT_REPLAY_CLAUDE_CLI[\"']\]\s*=\s*[\"']{2}", text)


def test_session_starts_with_an_empty_cli_setting() -> None:
    # The autouse fixture of this file removed every AGENT_REPLAY_* variable, so read the
    # value a fresh interpreter sees through the conftest hook instead.
    code = (
        "import os; import conftest; "
        "conftest.pytest_configure(); "
        "print(repr(os.environ.get('AGENT_REPLAY_CLAUDE_CLI')))"
    )
    env = scrubbed_env()
    env["AGENT_REPLAY_CLAUDE_CLI"] = "/should/be/overwritten"
    env["PYTHONPATH"] = str(ROOT / "tests")
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert out.stdout.strip() == "''"
