"""GeminiProvider against the provider conformance kit, with a fake genai client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from conformance import ProviderConformance, SeenRequest

from agent_replay.llm import (
    ModelProvider,
    PermanentModelError,
    RateLimitError,
    TransientError,
)
from agent_replay.llm.gemini import GeminiProvider


class _Models:
    def __init__(self) -> None:
        self.response: Any = None
        self.exc: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class GeminiFakeBackend:
    def __init__(self) -> None:
        self.models = _Models()
        self.client = SimpleNamespace(models=self.models)

    def make_provider(self) -> ModelProvider:
        return GeminiProvider(api_key="AIzaSyFAKE-KEY-0000000000", client=self.client)

    @staticmethod
    def _usage(input_tokens: int, output_tokens: int) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=None,
            tool_use_prompt_token_count=None,
            total_token_count=None,
        )

    def script_text(self, text: str, *, input_tokens: int, output_tokens: int) -> None:
        self.models.response = SimpleNamespace(
            text=text,
            function_calls=None,
            candidates=[],
            usage_metadata=self._usage(input_tokens, output_tokens),
        )

    def script_tool_calls(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        step: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.models.response = SimpleNamespace(
            text=None,
            function_calls=[SimpleNamespace(name=n, args=a, id=None) for n, a in calls],
            candidates=[],
            usage_metadata=self._usage(input_tokens, output_tokens),
        )

    def script_error(self, error_cls: type[Exception]) -> None:
        from google.genai import errors

        body = {"error": {"message": "boom", "status": "X"}}
        self.models.exc = {
            RateLimitError: errors.ClientError(429, body),
            TransientError: errors.ServerError(503, body),
            PermanentModelError: errors.ClientError(400, body),
        }[error_cls]

    def seen_request(self) -> SeenRequest:
        contents = self.models.calls[-1]["contents"]
        seen = SeenRequest()
        for content in contents:
            for part in content.parts:
                if part.function_call is not None:
                    fc = part.function_call
                    seen.tool_calls.append((fc.name, dict(fc.args)))
                if part.function_response is not None:
                    fr = part.function_response
                    seen.tool_results.append((fr.name, dict(fr.response)))
        return seen


class TestGeminiConformance(ProviderConformance):
    declared_errors = frozenset({RateLimitError, TransientError, PermanentModelError})
    undeclared_reason = "QuotaExhausted comes from the rate limiter, not the provider"

    @pytest.fixture
    def backend(self) -> GeminiFakeBackend:
        return GeminiFakeBackend()
