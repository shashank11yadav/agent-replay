"""Helpers shared by the test modules."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def scrubbed_env() -> dict[str, str]:
    """The current environment without AGENT_REPLAY_* settings, for CLI subprocesses.

    No Gemini key and no Claude Code CLI: the default provider is claude_code, and a
    developer machine may have the real `claude` on PATH. A test that needs a provider sets
    it (or points AGENT_REPLAY_CLAUDE_CLI at the fake) on top of this.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_REPLAY_")}
    env["AGENT_REPLAY_GEMINI_API_KEY"] = ""
    env["AGENT_REPLAY_CLAUDE_CLI"] = ""
    return env
