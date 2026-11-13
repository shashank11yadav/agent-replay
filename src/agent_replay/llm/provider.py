"""The model provider protocol."""

from __future__ import annotations

from typing import Protocol

from agent_replay.llm.types import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...
