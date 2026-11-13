"""Gemini provider: maps neutral requests/responses/errors to google-genai.

This is the only module that imports google.genai.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from agent_replay.llm.hashing import make_tool_call_id, next_step
from agent_replay.llm.types import (
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    RateLimitError,
    ToolCall,
    ToolSchema,
    TransientError,
)

_REDACTED = "[REDACTED]"


def build_client(api_key: str, timeout_s: float = 30) -> genai.Client:
    """Construct a genai.Client with a millisecond timeout and SDK retries disabled."""
    if timeout_s < 0.001:
        raise ValueError("timeout_s must be at least 0.001 (1 ms)")
    http_options = types.HttpOptions(
        timeout=int(timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def _count(usage: Any, field: str) -> int:
    value = getattr(usage, field, None)
    return int(value) if value is not None else 0


def _to_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for m in messages:
        parts: list[types.Part] = []
        if m.role == "tool":
            parts = [
                types.Part.from_function_response(name=r.name, response=r.response)
                for r in m.tool_results
            ]
        else:
            if m.text:
                parts.append(types.Part.from_text(text=m.text))
            if m.role == "model":
                parts.extend(_call_part(c) for c in m.tool_calls)
        if parts:
            contents.append(types.Content(role=m.role, parts=parts))
    return contents


def _call_part(call: ToolCall) -> types.Part:
    """A function-call part, echoing the thought signature when one was recorded."""
    if call.signature is None:
        return types.Part.from_function_call(name=call.name, args=call.args)
    return types.Part(
        function_call=types.FunctionCall(name=call.name, args=call.args),
        thought_signature=_decode_signature(call.signature),
    )


def _encode_signature(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _decode_signature(signature: str) -> bytes:
    try:
        return base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ToolCall.signature is not valid base64") from exc


def _call_signatures(response: Any) -> list[str | None]:
    """Thought signatures of the function-call parts of the first candidate, in order."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return []
    signatures: list[str | None] = []
    for part in parts:
        if getattr(part, "function_call", None) is None:
            continue
        raw = getattr(part, "thought_signature", None)
        signatures.append(_encode_signature(raw) if isinstance(raw, bytes) else None)
    return signatures


def _to_tools(tools: Sequence[ToolSchema]) -> list[Any] | None:
    if not tools:
        return None
    decls = [
        types.FunctionDeclaration(
            name=t.name, description=t.description, parameters_json_schema=t.parameters
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=decls)]


class GeminiProvider:
    def __init__(self, api_key: str, client: Any = None, timeout_s: float = 30) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is empty; set AGENT_REPLAY_GEMINI_API_KEY")
        self._api_key = SecretStr(api_key)
        self._client: Any = (
            client if client is not None else build_client(api_key, timeout_s=timeout_s)
        )

    def __repr__(self) -> str:
        return "GeminiProvider(api_key=**********)"

    __str__ = __repr__

    def _redact(self, text: str) -> str:
        raw = self._api_key.get_secret_value()
        for secret in sorted({raw, raw.strip()}, key=len, reverse=True):
            if secret:
                text = text.replace(secret, _REDACTED)
        return text

    def _scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {self._scrub(k): self._scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._scrub(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._scrub(v) for v in value)
        return value

    def _sanitize(self, exc: errors.APIError) -> None:
        """Redact the key from the error in place and drop its HTTP response."""
        exc.args = self._scrub(exc.args)
        exc.message = self._scrub(exc.message)
        exc.details = self._scrub(exc.details)
        exc.status = self._scrub(exc.status)
        exc.response = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            seed=request.seed,
            tools=_to_tools(request.tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=_to_contents(request.messages),
                config=config,
            )
        except errors.APIError as exc:
            mapped = self._map_error(exc)
            self._sanitize(exc)
            raise mapped from exc
        except httpx.TransportError as exc:
            # Not chained: the transport exception can carry the request (and key).
            message = self._redact(
                f"Gemini transport error: {type(exc).__name__}: {exc}"
            )
            raise TransientError(message) from None
        return self._to_response(response, next_step(request.messages))

    def _map_error(self, exc: errors.APIError) -> Exception:
        message = self._redact(f"Gemini API error: {exc}")
        if isinstance(exc, errors.ClientError) and exc.code == 429:
            return RateLimitError(message)
        if isinstance(exc, errors.ServerError):
            return TransientError(message)
        return PermanentModelError(message)

    @staticmethod
    def _to_response(response: Any, step: int) -> ModelResponse:
        calls = response.function_calls or []
        if not response.text and not calls:
            # A blocked or cut-off candidate; the Claude provider rejects the same shape.
            raise PermanentModelError(
                "Gemini returned no text and no function call (blocked or empty candidate)"
            )
        signatures = _call_signatures(response)
        if len(signatures) != len(calls):
            signatures = [None] * len(calls)
        tool_calls = [
            ToolCall(
                id=make_tool_call_id(step, i),
                name=c.name or "",
                args=dict(c.args or {}),
                signature=sig,
            )
            for i, (c, sig) in enumerate(zip(calls, signatures, strict=True))
        ]
        usage = response.usage_metadata
        return ModelResponse(
            text=response.text,
            tool_calls=tool_calls,
            input_tokens=_count(usage, "prompt_token_count")
            + _count(usage, "tool_use_prompt_token_count"),
            output_tokens=_count(usage, "candidates_token_count")
            + _count(usage, "thoughts_token_count"),
        )
