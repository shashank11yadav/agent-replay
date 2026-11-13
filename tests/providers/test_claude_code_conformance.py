"""The claude_code provider against the conformance kit, with the fake CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from claude_cli_harness import (
    FakeClaude,
    clean_agent_env,
    error_json,
    step_ok,
    step_text,
)
from conformance import ProviderConformance, SeenRequest

from agent_replay.llm import (
    ModelProvider,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)


class ClaudeFakeBackend:
    def __init__(self, fake: FakeClaude, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake = fake
        self.monkeypatch = monkeypatch

    def make_provider(self) -> ModelProvider:
        provider: ModelProvider = self.fake.provider(self.monkeypatch)
        return provider

    def script_text(self, text: str, *, input_tokens: int, output_tokens: int) -> None:
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.fake.script(step_text(text, usage=usage))

    def script_tool_calls(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        step: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        reply = {"tool_calls": [{"name": n, "args": a} for n, a in calls]}
        self.fake.script(step_ok(reply, usage=usage))

    def script_error(self, error_cls: type[Exception]) -> None:
        bodies: dict[type[Exception], dict[str, Any]] = {
            RateLimitError: error_json("Request rejected (429)", status=429),
            TransientError: error_json("API Error: 503 unavailable", status=503),
            PermanentModelError: error_json("Not logged in · Please run /login"),
            QuotaExhausted: error_json("You've hit your session limit · resets 3:45pm"),
        }
        self.fake.script({"stdout_json": bodies[error_cls], "exit_code": 1})

    def seen_request(self) -> SeenRequest:
        stdin = self.fake.calls()[-1]["stdin"].decode("utf-8")
        seen = SeenRequest()
        for line in stdin.splitlines():
            if line.startswith("tool_calls: "):
                for call in json.loads(line[len("tool_calls: ") :]):
                    seen.tool_calls.append((call["name"], call["args"]))
            elif line.startswith("tool_result: "):
                result = json.loads(line[len("tool_result: ") :])
                seen.tool_results.append((result["name"], result["response"]))
        return seen


class TestClaudeCodeConformance(ProviderConformance):
    declared_errors = frozenset(
        {RateLimitError, TransientError, PermanentModelError, QuotaExhausted}
    )
    undeclared_reason = ""
    provider_name = "claude_code"
    model = "claude-haiku-5-5"

    @pytest.fixture
    def backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> ClaudeFakeBackend:
        clean_agent_env(monkeypatch)
        return ClaudeFakeBackend(FakeClaude(tmp_path), monkeypatch)


def test_all_four_error_classes_are_declared() -> None:
    assert TestClaudeCodeConformance.declared_errors == frozenset(
        {RateLimitError, TransientError, PermanentModelError, QuotaExhausted}
    )
