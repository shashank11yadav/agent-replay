"""Harness around fake_claude_cli.py, so no test needs a real `claude` binary."""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

FAKE_SCRIPT = Path(__file__).with_name("fake_claude_cli.py")
SCENARIO_ENV = "FAKE_CLAUDE_SCENARIO"


def success_json(
    structured: Any = ..., *, usage: Any = ..., **extra: Any
) -> dict[str, Any]:
    """A `--output-format json` success object (structured_output omitted when `...`)."""
    obj: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "num_turns": 2,
        "result": "",
        "total_cost_usd": 0.0123,
    }
    if usage is ...:
        usage = {
            "input_tokens": 11,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 5,
        }
    if usage is not None:
        obj["usage"] = usage
    if structured is not ...:
        obj["structured_output"] = structured
    obj.update(extra)
    return obj


def error_json(
    text: str = "",
    *,
    status: int | None = None,
    subtype: str = "success",
    **extra: Any,
) -> dict[str, Any]:
    """A failed result: is_error true, the failure text in `result`."""
    obj: dict[str, Any] = {
        "type": "result",
        "subtype": subtype,
        "is_error": True,
        "api_error_status": status,
        "num_turns": 1,
        "result": text,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    obj.update(extra)
    return obj


def step_ok(structured: Any, **kw: Any) -> dict[str, Any]:
    return {"stdout_json": success_json(structured, **kw), "exit_code": 0}


def step_text(text: str, **kw: Any) -> dict[str, Any]:
    return step_ok({"text": text}, **kw)


class FakeClaude:
    """A launcher script in tmp_path that runs fake_claude_cli.py with this interpreter."""

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "fake-bin"
        self.dir.mkdir()
        self.launcher = self.dir / "claude"
        self.scenario = tmp_path / "scenario.json"
        self.launcher.write_text(
            f"#!{sys.executable}\n"
            "import runpy\n"
            "import sys\n"
            f"runpy.run_path({str(FAKE_SCRIPT)!r}, "
            'init_globals={"LAUNCHER_ARGV0": sys.argv[0]}, run_name="__main__")\n',
            encoding="utf-8",
        )
        self.launcher.chmod(self.launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        self.script(step_text("default reply"))

    @property
    def path(self) -> str:
        return str(self.launcher)

    @property
    def log_path(self) -> Path:
        return Path(str(self.scenario) + ".calls.jsonl")

    def script(self, *steps: dict[str, Any]) -> None:
        """Call N gets step N; the last step repeats."""
        self.scenario.write_text(json.dumps({"calls": list(steps)}), encoding="utf-8")

    def calls(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec["stdin"] = base64.b64decode(rec.pop("stdin_b64"))
            out.append(rec)
        return out

    def env(self, **extra: str) -> dict[str, str]:
        """AGENT_REPLAY_* / fake variables that point a process at this fake."""
        env = {
            "AGENT_REPLAY_PROVIDER": "claude_code",
            "AGENT_REPLAY_CLAUDE_CLI": self.path,
            SCENARIO_ENV: str(self.scenario),
        }
        env.update(extra)
        return env

    def apply(self, monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
        for key, value in self.env(**extra).items():
            monkeypatch.setenv(key, value)

    def settings(self, monkeypatch: pytest.MonkeyPatch, **extra: str) -> Any:
        from agent_replay.config import Settings

        self.apply(monkeypatch, **extra)
        return Settings()

    def provider(self, monkeypatch: pytest.MonkeyPatch, **extra: str) -> Any:
        """The bare claude_code provider (no rate limiter) built through the registry."""
        import agent_replay.llm.factory  # noqa: F401  (registers the built-ins)
        from agent_replay.llm import registry

        settings = self.settings(monkeypatch, **extra)
        provider = registry.get_provider_spec("claude_code").build(settings, "record")
        assert provider is not None
        return provider


def clean_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every AGENT_REPLAY_* variable (the session conftest sets a few)."""
    for key in list(os.environ):
        if key.startswith("AGENT_REPLAY_"):
            monkeypatch.delenv(key)
