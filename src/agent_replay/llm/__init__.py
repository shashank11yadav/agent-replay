"""LLM core: neutral types, provider protocol, request hashing, scripted provider."""

from agent_replay.llm.fake import ScriptedProvider, ScriptExhausted
from agent_replay.llm.hashing import make_tool_call_id, request_hash
from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.types import (
    BudgetExceeded,
    CacheMissNoLiveModel,
    InjectedRateLimitError,
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ToolCall,
    ToolResult,
    ToolSchema,
    TransientError,
)

__all__ = [
    "BudgetExceeded",
    "CacheMissNoLiveModel",
    "InjectedRateLimitError",
    "Message",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "PermanentModelError",
    "QuotaExhausted",
    "RateLimitError",
    "ScriptExhausted",
    "ScriptedProvider",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "TransientError",
    "make_tool_call_id",
    "request_hash",
]
