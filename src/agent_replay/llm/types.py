"""Provider-neutral request/response types and LLM error classes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    # Opaque base64 of the provider's thought signature; echoed back on replayed
    # history and excluded from request hashing.
    signature: str | None = None


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["user", "model", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class ModelRequest(BaseModel):
    provider: str
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSchema] = Field(default_factory=list)
    temperature: float = 0.0
    seed: int | None = None


class ModelResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cache_status: Literal["hit", "miss", "live"] = "live"
    attempts: int = 1


class RateLimitError(Exception):
    """The provider returned a rate-limit (429) response."""


class InjectedRateLimitError(RateLimitError):
    """A 429 raised by a ``rate_limit`` fault; retried like a real one."""


class TransientError(Exception):
    """A retryable provider failure."""


class PermanentModelError(Exception):
    """A non-retryable provider failure."""


class CacheMissNoLiveModel(Exception):
    """A replay needed a response that is not cached and live calls are disabled."""


class QuotaExhausted(Exception):
    """The provider quota is exhausted."""


class BudgetExceeded(Exception):
    """The per-process run budget was exceeded."""

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Run budget exceeded: spent ${spent_usd:.2f} of ${budget_usd:.2f}. "
            "Raise AGENT_REPLAY_RUN_BUDGET_USD to allow more spend."
        )
