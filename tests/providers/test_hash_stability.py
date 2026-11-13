"""Request hashes stay byte-identical across provider registry changes.

The literals are golden values. If one fails, a ModelRequest field or the canonical encoding
changed, and that would invalidate every recorded fixture (see docs/decisions/ADR-004).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from agent_replay.config import Settings
from agent_replay.demo.agent import DemoAgent
from agent_replay.llm import (
    Message,
    ModelRequest,
    ModelResponse,
    ToolSchema,
    request_hash,
)
from agent_replay.scoring.judge import build_judge_request
from agent_replay.taskspec import TaskSpec

BASIC_HASH = "1606c4efc6be87194c9dd0e414b1357b8dbffedb5a4a4ea2bf73413f725f9737"
JUDGE_HASH = "7f183ab50737931e1d3ee3a0c8a5ce945c13f9e1cd00b36c839814cb5de45c20"
AGENT_HASH = "004982b097783466277fe481ba1b018c2ddab9f0140e248b8a6f729b08c99229"
JUDGE_HASH_SCRIPTED = "32f304228d060a8793e15cae3a62cf5fc84fffc0a6a905a441ba3dab2963dfd3"
AGENT_HASH_SCRIPTED = "7698d1711ff7e00200f0ae83f10767718c2e6f356511940d04cda278403e27e7"
JUDGE_HASH_CLAUDE_CODE = (
    "b3f603ce721042f542052174ef30117ea007b368c4cbb683774527042d409d32"
)
AGENT_HASH_CLAUDE_CODE = (
    "683344ac8ae74940e20f3211fc672ea3f1667f8294a1aef94455155d35936d87"
)

TASK = TaskSpec(
    id="t1",
    prompt="What is the price of product A1003?",
    category="lookup",
    expected_text=["37.95"],
    must_call_tools=["search"],
    max_steps=4,
    rubric="The answer states the price 37.95 EUR.",
)
ANSWER = "It costs 37.95 EUR."


class _Ctx:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def call_model(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text="x")

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("no tool call expected")


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("AGENT_REPLAY_"):
            monkeypatch.delenv(key)
    # The default provider is claude_code: never let it find a real `claude` on PATH.
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "")


def _first_agent_hash(settings: Settings) -> str:
    ctx = _Ctx()
    DemoAgent("minimal", settings).run(TASK, ctx)  # type: ignore[arg-type]
    return request_hash(ctx.requests[0])


def test_fixed_request_hash_is_pinned() -> None:
    req = ModelRequest(
        provider="gemini",
        model="gemini-3.5-flash-lite",
        system="s",
        messages=[Message(role="user", text="hi")],
        tools=[
            ToolSchema(
                name="t",
                description="d",
                parameters={"type": "object", "properties": {}},
            )
        ],
        temperature=0,
        seed=7,
    )
    assert request_hash(req) == BASIC_HASH


def test_judge_and_agent_hashes_pinned_under_gemini_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "gemini")
    s = Settings.model_validate({})
    assert request_hash(build_judge_request(TASK, ANSWER, s)) == JUDGE_HASH
    assert _first_agent_hash(s) == AGENT_HASH


def test_judge_and_agent_hashes_pinned_under_default_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default provider is claude_code, the provider of the committed fixtures.
    _clean_env(monkeypatch)
    s = Settings.model_validate({})
    assert s.provider == "claude_code"
    assert request_hash(build_judge_request(TASK, ANSWER, s)) == JUDGE_HASH_CLAUDE_CODE
    assert _first_agent_hash(s) == AGENT_HASH_CLAUDE_CODE


def test_judge_and_agent_hashes_pinned_under_scripted_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "scripted")
    s = Settings.model_validate({})
    assert request_hash(build_judge_request(TASK, ANSWER, s)) == JUDGE_HASH_SCRIPTED
    assert _first_agent_hash(s) == AGENT_HASH_SCRIPTED
