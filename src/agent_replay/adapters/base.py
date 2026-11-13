"""Adapter interface: how an agent talks to models and tools.

Note: `ToolResult` here is the tool-call outcome returned by `RunContext.call_tool`. It is
unrelated to `agent_replay.llm.ToolResult(call_id, name, response)`, which is the message
payload sent back to the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

from agent_replay.llm import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from agent_replay.taskspec import TaskSpec


class ToolResult(BaseModel):
    """Outcome of one `ctx.call_tool` call (not the llm message ToolResult)."""

    ok: bool
    data: dict[str, Any] | None
    raw: str | None
    error_type: str | None = None


class AgentOutcome(BaseModel):
    status: Literal["ok", "error"]
    final_output: str | None = None
    error: str | None = None


class RunContext(Protocol):
    def call_model(self, request: ModelRequest) -> ModelResponse: ...

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult: ...


class AgentAdapter(Protocol):
    name: str

    def run(self, task: TaskSpec, ctx: RunContext) -> AgentOutcome: ...
