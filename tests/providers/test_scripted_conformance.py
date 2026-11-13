"""ScriptedProvider against the provider conformance kit, with a script file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conformance import ProviderConformance, SeenRequest

from agent_replay.llm import ModelProvider, ScriptedProvider, make_tool_call_id


class ScriptedFakeBackend:
    def __init__(self, tmp_path: Path) -> None:
        self._path = tmp_path / "script.json"
        self._provider: ScriptedProvider | None = None
        self._responses: list[dict[str, Any]] = []

    def make_provider(self) -> ModelProvider:
        self._provider = ScriptedProvider.from_file(self._path)
        return self._provider

    def _write(self, response: dict[str, Any]) -> None:
        self._responses = [response]
        self._path.write_text(json.dumps(self._responses), encoding="utf-8")

    def script_text(self, text: str, *, input_tokens: int, output_tokens: int) -> None:
        self._write(
            {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
        )

    def script_tool_calls(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        step: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self._write(
            {
                "tool_calls": [
                    {"id": make_tool_call_id(step, i), "name": n, "args": a}
                    for i, (n, a) in enumerate(calls)
                ],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def script_error(self, error_cls: type[Exception]) -> None:
        raise AssertionError("scripted declares no error classes")

    def seen_request(self) -> SeenRequest:
        assert self._provider is not None
        request = self._provider.requests[-1]
        seen = SeenRequest()
        for message in request.messages:
            seen.tool_calls.extend((c.name, c.args) for c in message.tool_calls)
            seen.tool_results.extend((r.name, r.response) for r in message.tool_results)
        return seen


class TestScriptedConformance(ProviderConformance):
    declared_errors = frozenset()
    undeclared_reason = (
        "scripted has no error injection; its only error is ScriptExhausted"
    )

    @pytest.fixture
    def backend(self, tmp_path: Path) -> ScriptedFakeBackend:
        return ScriptedFakeBackend(tmp_path)
