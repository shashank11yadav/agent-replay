"""Package metadata, the CLI entry point and Settings."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
from support import ROOT, scrubbed_env


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    for name in list(os.environ):
        if name.startswith("AGENT_REPLAY_"):
            monkeypatch.delenv(name)
    # The default provider is claude_code: never let it find a real `claude` on PATH.
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "")
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def make_settings():  # type: ignore[no-untyped-def]
    from agent_replay.config import Settings

    return Settings()


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    return subprocess.run(
        ["uv", "run", "agent-replay", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


# ---- pyproject ----------------------------------------------------------


def test_pyproject_declares_python_script_and_dev_deps() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]
    assert re.search(r">=\s*3\.12", project["requires-python"])
    assert "agent-replay" in project["scripts"]
    dev: list[str] = list(data.get("dependency-groups", {}).get("dev", []))
    dev += data.get("tool", {}).get("uv", {}).get("dev-dependencies", [])
    joined = " ".join(str(d) for d in dev).replace(" ", "")
    for pin in ("pytest==9.2.0", "ruff==0.17.3", "mypy==2.4.1"):
        assert pin in joined, f"{pin} missing from dev dependencies: {dev}"


# ---- CLI ----------------------------------------------------------------


def test_cli_version_prints_package_version() -> None:
    import agent_replay

    res = run_cli("--version")
    assert res.returncode == 0, res.stderr
    assert agent_replay.__version__
    assert agent_replay.__version__ in res.stdout + res.stderr


def test_cli_help_exits_zero_with_usage() -> None:
    res = run_cli("--help")
    assert res.returncode == 0, res.stderr
    assert "usage" in (res.stdout + res.stderr).lower()


def test_cli_unknown_command_exits_nonzero() -> None:
    res = run_cli("no-such-command")
    assert res.returncode != 0


# ---- Settings: env vars and defaults ------------------------------------


def test_settings_defaults(clean_env: pytest.MonkeyPatch) -> None:
    s = make_settings()
    assert s.provider == "claude_code"
    assert s.model == "claude-haiku-5-5"
    assert s.judge_model == "claude-sonnet-5-5"
    assert s.rpm == 60
    assert s.rpd == 5000
    assert s.run_budget_usd == 1.50
    assert s.price_input_per_m == 0.10
    assert s.price_output_per_m == 0.50
    assert s.timeout_s == 30
    assert s.max_steps == 6
    assert s.tool_timeout_s == 10
    assert s.max_attempts == 3
    assert s.backoff_base_s == 2
    assert s.seed == 7
    assert s.temperature == 0
    assert not s.cors_origins
    assert s.viewer_dist is None


def test_settings_reads_every_env_var(clean_env: pytest.MonkeyPatch) -> None:
    values = {
        "DATABASE_URL": "postgresql://u:p@h:5432/d",
        "MODEL": "other-model",
        "PROVIDER": "other",
        "RPM": "11",
        "RPD": "22",
        "RUN_BUDGET_USD": "3.5",
        "PRICE_INPUT_PER_M": "0.5",
        "PRICE_OUTPUT_PER_M": "4.5",
        "TIMEOUT_S": "45",
        "MAX_ATTEMPTS": "5",
        "BACKOFF_BASE_S": "4",
        "MAX_STEPS": "9",
        "TOOL_TIMEOUT_S": "12",
        "SEED": "99",
        "VIEWER_DIST": "/srv/dist",
    }
    for k, v in values.items():
        clean_env.setenv(f"AGENT_REPLAY_{k}", v)
    clean_env.setenv("AGENT_REPLAY_GEMINI_API_KEY", "abc")
    clean_env.setenv("AGENT_REPLAY_CORS_ORIGINS", '["http://a.example"]')
    s = make_settings()
    assert s.database_url == values["DATABASE_URL"]
    assert s.model == "other-model"
    assert s.provider == "other"
    assert (s.rpm, s.rpd) == (11, 22)
    assert s.run_budget_usd == 3.5
    assert s.price_input_per_m == 0.5
    assert s.price_output_per_m == 4.5
    assert s.timeout_s == 45
    assert s.max_attempts == 5
    assert s.backoff_base_s == 4
    assert s.max_steps == 9
    assert s.tool_timeout_s == 12
    assert s.seed == 99
    assert str(s.viewer_dist) == "/srv/dist"
    assert s.has_gemini_key is True
    assert "http://a.example" in list(s.cors_origins)


# ---- Settings: validation -----------------------------------------------


def test_budget_zero_accepted(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_REPLAY_RUN_BUDGET_USD", "0")
    assert make_settings().run_budget_usd == 0


def test_budget_float_parsing(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_REPLAY_RUN_BUDGET_USD", "0.75")
    assert make_settings().run_budget_usd == 0.75


@pytest.mark.parametrize(
    "name", ["RUN_BUDGET_USD", "PRICE_INPUT_PER_M", "PRICE_OUTPUT_PER_M"]
)
def test_negative_money_values_rejected(
    clean_env: pytest.MonkeyPatch, name: str
) -> None:
    clean_env.setenv(f"AGENT_REPLAY_{name}", "-0.01")
    with pytest.raises(ValidationError):
        make_settings()


def test_timeout_zero_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_REPLAY_TIMEOUT_S", "0")
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize("name", ["RPM", "RPD"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_rate_limits_must_be_positive(
    clean_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_env.setenv(f"AGENT_REPLAY_{name}", value)
    with pytest.raises(ValidationError):
        make_settings()


# ---- Settings: API key handling -----------------------------------------


@pytest.mark.parametrize("mode", ["unset", "empty"])
def test_no_api_key(clean_env: pytest.MonkeyPatch, mode: str) -> None:
    if mode == "empty":
        clean_env.setenv("AGENT_REPLAY_GEMINI_API_KEY", "")
    assert make_settings().has_gemini_key is False


def test_api_key_never_leaks(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_REPLAY_GEMINI_API_KEY", "sk-test-123")
    s = make_settings()
    assert s.has_gemini_key is True
    assert "sk-test-123" not in repr(s)
    assert "sk-test-123" not in str(s)
    assert "sk-test-123" not in str(s.safe_dict())


def test_dotenv_file_not_read(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("AGENT_REPLAY_MODEL=x\n")
    assert make_settings().model == "claude-haiku-5-5"


# ---- conftest ------------------------------------------------------------


def test_conftest_sets_empty_api_key_for_session() -> None:
    assert os.environ.get("AGENT_REPLAY_GEMINI_API_KEY") == ""
