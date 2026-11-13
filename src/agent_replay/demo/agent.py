"""Demo agent: a bounded tool-calling loop over the RunContext."""

from __future__ import annotations

from importlib import resources
from typing import Any

from pydantic import ValidationError

from agent_replay.adapters.base import AgentOutcome, RunContext
from agent_replay.config import Settings
from agent_replay.demo.tools import OUTPUT_MODELS, TOOL_SCHEMAS
from agent_replay.llm import (
    BudgetExceeded,
    CacheMissNoLiveModel,
    Message,
    ModelRequest,
    QuotaExhausted,
    ToolCall,
    ToolResult,
    make_tool_call_id,
)
from agent_replay.llm.hashing import next_step
from agent_replay.taskspec import TaskSpec


def available_variants() -> tuple[str, ...]:
    """Variant names are the stems of prompts/*.txt."""
    prompts = resources.files("agent_replay.demo").joinpath("prompts")
    return tuple(
        sorted(p.name[:-4] for p in prompts.iterdir() if p.name.endswith(".txt"))
    )


def load_prompt(variant: str) -> str:
    variants = available_variants()
    if variant not in variants:
        raise ValueError(
            f"unknown prompt variant {variant!r}; available: {', '.join(variants)}"
        )
    return (
        resources.files("agent_replay.demo")
        .joinpath("prompts", f"{variant}.txt")
        .read_text("utf-8")
    )


def _tool_response(
    name: str, data: dict[str, Any] | None, ok: bool, error: str | None
) -> dict[str, Any]:
    if not ok:
        return data if data is not None else {"error": error or "tool_error"}
    model = OUTPUT_MODELS.get(name)
    if data is None or model is None:
        return {"error": "malformed_tool_output"}
    if set(data) == {"error"}:
        return data
    try:
        model.model_validate(data)
    except ValidationError:
        return {"error": "malformed_tool_output"}
    return data


class DemoAgent:
    name = "demo-agent"

    def __init__(
        self, variant: str = "minimal", settings: Settings | None = None
    ) -> None:
        self._system = load_prompt(variant)
        self._settings = settings or Settings.model_validate({})

    def run(self, task: TaskSpec, ctx: RunContext) -> AgentOutcome:
        s = self._settings
        limit = min(task.max_steps, s.max_steps)
        messages = [Message(role="user", text=task.prompt)]
        tools = list(TOOL_SCHEMAS.values())
        try:
            for _ in range(limit):
                response = ctx.call_model(
                    ModelRequest(
                        provider=s.provider,
                        model=s.model,
                        system=self._system,
                        messages=list(messages),
                        tools=tools,
                        temperature=s.temperature,
                        seed=s.seed,
                    )
                )
                if not response.tool_calls:
                    return AgentOutcome(status="ok", final_output=response.text or "")
                calls = [
                    ToolCall(
                        # Same ids the providers give (hashing.next_step).
                        id=make_tool_call_id(next_step(messages), i),
                        name=c.name,
                        args=c.args,
                        signature=c.signature,
                    )
                    for i, c in enumerate(response.tool_calls)
                ]
                messages.append(
                    Message(role="model", text=response.text or "", tool_calls=calls)
                )
                results: list[ToolResult] = []
                for c in calls:
                    r = ctx.call_tool(c.name, c.args)
                    results.append(
                        ToolResult(
                            call_id=c.id,
                            name=c.name,
                            response=_tool_response(c.name, r.data, r.ok, r.error_type),
                        )
                    )
                messages.append(Message(role="tool", tool_results=results))
        except (BudgetExceeded, QuotaExhausted, CacheMissNoLiveModel):
            raise  # run-level conditions the runner maps; they must stop the suite
        except Exception as exc:  # noqa: BLE001 - any failure becomes an error outcome
            return AgentOutcome(status="error", error=type(exc).__name__)
        return AgentOutcome(status="error", error="max_steps_exceeded")
