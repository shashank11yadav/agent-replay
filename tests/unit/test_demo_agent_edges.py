from typing import Any

import pytest

from agent_replay.adapters.base import RunContext
from agent_replay.demo.agent import DemoAgent, available_variants
from agent_replay.llm import (
    BudgetExceeded,
    CacheMissNoLiveModel,
    ModelRequest,
    ModelResponse,
    QuotaExhausted,
    RateLimitError,
    ToolCall,
)
from agent_replay.taskspec import TaskSpec


class Ctx:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.n = 0

    def call_model(self, request: ModelRequest) -> ModelResponse:
        r = self.responses[self.n]
        self.n += 1
        if isinstance(r, Exception):
            raise r
        return r  # type: ignore[no-any-return]

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("no tools expected")


def _task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "id": "t1",
            "prompt": "hi",
            "category": "lookup",
            "max_steps": 3,
            "expected_text": ["x"],
            "must_call_tools": [],
            "rubric": "r",
        }
    )


@pytest.mark.parametrize(
    "exc", [BudgetExceeded(2.0, 1.5), QuotaExhausted("q"), CacheMissNoLiveModel("h")]
)
def test_run_level_errors_propagate(exc: Exception) -> None:
    ctx: RunContext = Ctx([exc])  # type: ignore[assignment]
    with pytest.raises(type(exc)):
        DemoAgent().run(_task(), ctx)


def test_rate_limit_is_an_error_outcome() -> None:
    ctx: RunContext = Ctx([RateLimitError("429")])  # type: ignore[assignment]
    out = DemoAgent().run(_task(), ctx)
    assert (out.status, out.error) == ("error", "RateLimitError")


def test_first_tool_call_id_matches_provider_numbering() -> None:
    class C(Ctx):
        def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            from agent_replay.adapters.base import ToolResult as TR

            return TR(ok=True, data={"error": "x"}, raw=None)

    seen: list[ModelRequest] = []

    class D(C):
        def call_model(self, request: ModelRequest) -> ModelResponse:
            seen.append(request)
            return super().call_model(request)

    ctx = D(
        [
            ModelResponse(tool_calls=[ToolCall(id="z", name="get_order", args={})]),
            ModelResponse(text="done"),
        ]
    )
    DemoAgent().run(_task(), ctx)  # type: ignore[arg-type]
    model_msg = next(m for m in seen[1].messages if m.role == "model")
    assert model_msg.tool_calls[0].id == "call_1_0"


def test_variants_discovered_from_prompt_files() -> None:
    assert available_variants() == (
        "minimal",
        "plan_act",
        "structured",
        "structured_regressed",
    )
