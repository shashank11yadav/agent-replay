"""Scripted fake provider for tests and offline runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from agent_replay.llm.types import ModelRequest, ModelResponse

_SCRIPT = TypeAdapter(list[ModelResponse])


class ScriptExhausted(Exception):
    """The scripted provider was called more times than it has responses."""


class ScriptedProvider:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    @classmethod
    def from_file(cls, path: str | Path) -> ScriptedProvider:
        p = Path(path)
        try:
            responses = _SCRIPT.validate_python(
                json.loads(p.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Invalid script file {p}: {exc}") from exc
        return cls(responses)

    @property
    def has_tool_calls(self) -> bool:
        """True when the script drives an agent (some response calls a tool)."""
        return any(r.tool_calls for r in self._responses)

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) > len(self._responses):
            raise ScriptExhausted(
                f"Script has {len(self._responses)} responses; call {len(self.requests)} has none"
            )
        return self._responses[len(self.requests) - 1].model_copy(deep=True)
