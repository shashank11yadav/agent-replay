"""LLM core types, request hashing, the provider protocol and the scripted provider."""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_replay import llm
from agent_replay.llm import (
    BudgetExceeded,
    CacheMissNoLiveModel,
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ScriptedProvider,
    ScriptExhausted,
    ToolCall,
    ToolSchema,
    TransientError,
    hashing,
    make_tool_call_id,
    request_hash,
)


def field_names(cls: type) -> set[str]:
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(cls.model_fields)  # type: ignore[attr-defined]


def make_message(role: str, text: str) -> Message:
    names = field_names(Message)
    key = "text" if "text" in names else "content"
    return Message(role=role, **{key: text})


def make_schema(params: dict | None = None, name: str = "lookup") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="Look something up",
        parameters=params
        or {"type": "object", "properties": {"q": {"type": "string"}}},
    )


def make_request(**over) -> ModelRequest:
    base = {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "system": "You are helpful.",
        "messages": [make_message("user", "hello")],
        "tools": [make_schema()],
        "temperature": 0.0,
        "seed": 7,
    }
    base.update(over)
    return ModelRequest(**base)


def make_response(text: str | None = "ok", **over) -> ModelResponse:
    base = {
        "text": text,
        "tool_calls": [],
        "input_tokens": 3,
        "output_tokens": 4,
        "latency_ms": 12,
        "cache_status": "live",
        "attempts": 1,
    }
    base.update(over)
    return ModelResponse(**base)


# --- types -----------------------------------------------------------------


def test_model_request_fields():
    assert {
        "provider",
        "model",
        "system",
        "messages",
        "tools",
        "temperature",
        "seed",
    } <= field_names(ModelRequest)
    req = make_request()
    assert req.provider == "gemini" and req.model == "gemini-2.5-flash"
    assert req.system == "You are helpful."
    assert req.temperature == 0.0 and req.seed == 7
    assert len(req.messages) == 1 and len(req.tools) == 1


def test_message_roles_user_model_tool():
    assert "role" in field_names(Message)
    for role in ("user", "model", "tool"):
        assert make_message(role, "x").role == role
    with pytest.raises(ValidationError):
        make_message("assistant", "x")


def test_tool_call_and_schema_fields():
    assert field_names(ToolCall) >= {"id", "name", "args"}
    tc = ToolCall(id="call_0_0", name="lookup", args={"q": "a"})
    assert (tc.id, tc.name, tc.args) == ("call_0_0", "lookup", {"q": "a"})
    assert field_names(ToolSchema) >= {"name", "description", "parameters"}
    s = make_schema()
    assert s.name == "lookup" and s.description == "Look something up"
    assert s.parameters["type"] == "object"


def test_model_response_fields():
    assert field_names(ModelResponse) >= {
        "text", "tool_calls", "input_tokens", "output_tokens",
        "latency_ms", "cache_status", "attempts",
    }  # fmt: skip
    r = make_response(
        text=None,
        tool_calls=[ToolCall(id="call_1_0", name="t", args={})],
        cache_status="hit",
    )
    assert r.text is None
    assert r.tool_calls[0].name == "t"
    assert r.cache_status == "hit"
    assert (r.input_tokens, r.output_tokens, r.latency_ms, r.attempts) == (3, 4, 12, 1)
    for status in ("hit", "miss", "live"):
        assert make_response(cache_status=status).cache_status == status
    with pytest.raises(ValidationError):
        make_response(cache_status="bogus")


# --- hash ------------------------------------------------------------------


def test_hash_is_64_char_lowercase_hex():
    h = request_hash(make_request())
    assert isinstance(h, str)
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_hash_is_deterministic():
    assert request_hash(make_request()) == request_hash(make_request())


def test_hash_ignores_dict_key_order_in_schemas_and_args():
    a = make_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
    )
    b = make_schema(
        {
            "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
            "type": "object",
        }
    )
    assert request_hash(make_request(tools=[a])) == request_hash(
        make_request(tools=[b])
    )

    def with_args(args: dict) -> ModelRequest:
        msg = make_message("model", "calling")
        names = field_names(Message)
        assert "tool_calls" in names, (
            "Message must carry tool_calls so args enter the hash"
        )
        msg = Message(
            role="model",
            tool_calls=[ToolCall(id="call_0_0", name="lookup", args=args)],
            **({"text": "x"} if "text" in names else {"content": "x"}),
        )
        return make_request(messages=[make_message("user", "hi"), msg])

    assert request_hash(with_args({"x": 1, "y": {"p": 1, "q": 2}})) == request_hash(
        with_args({"y": {"q": 2, "p": 1}, "x": 1})
    )


@pytest.mark.parametrize(
    "over",
    [
        {"provider": "other"},
        {"model": "other-model"},
        {"system": "Different system."},
        {"messages": [make_message("user", "goodbye")]},
        {"messages": [make_message("model", "hello")]},
        {"tools": [make_schema(name="other")]},
        {"tools": [make_schema({"type": "object", "properties": {"z": {"type": "string"}}})]},
        {"temperature": 0.5},
        {"seed": 8},
    ],
    ids=["provider", "model", "system", "message-text", "message-role", "tool-name",
         "tool-schema", "temperature", "seed"],
)  # fmt: skip
def test_changing_any_hashed_field_changes_hash(over):
    assert request_hash(make_request(**over)) != request_hash(make_request())


def _version_constant() -> str:
    names = [n for n in dir(hashing) if "SCHEMA_VERSION" in n.upper() and n.isupper()]
    assert names, (
        "hashing.py must expose a cache schema version constant (name contains SCHEMA_VERSION)"
    )
    return names[0]


def test_hash_includes_schema_version_constant(monkeypatch: pytest.MonkeyPatch):
    name = _version_constant()
    old = getattr(hashing, name)
    req = make_request()
    before = request_hash(req)
    new = old + 1 if isinstance(old, int) else f"{old}-changed"
    monkeypatch.setattr(hashing, name, new)
    assert request_hash(req) != before


def test_make_tool_call_id():
    assert make_tool_call_id(step=2, index=1) == "call_2_1"
    assert make_tool_call_id(step=0, index=0) == "call_0_0"


# --- provider protocol -----------------------------------------------------


def test_model_provider_is_protocol_with_generate():
    assert getattr(ModelProvider, "_is_protocol", False) is True
    assert callable(ModelProvider.generate)
    hints = typing.get_type_hints(ModelProvider.generate)
    assert hints.get("request") is ModelRequest
    assert hints.get("return") is ModelResponse


# --- scripted provider -----------------------------------------------------


def test_scripted_provider_returns_in_order_and_records_requests():
    r1, r2 = make_response("one"), make_response("two")
    p = ScriptedProvider([r1, r2])
    q1, q2 = make_request(), make_request(system="second")
    assert p.generate(q1).text == "one"
    assert p.generate(q2).text == "two"
    assert list(p.requests) == [q1, q2]


def test_scripted_provider_exhausted():
    p = ScriptedProvider([make_response("only")])
    p.generate(make_request())
    with pytest.raises(ScriptExhausted):
        p.generate(make_request())
    with pytest.raises(ScriptExhausted):
        ScriptedProvider([]).generate(make_request())


def test_scripted_provider_from_file(tmp_path: Path):
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            [
                {
                    "text": "from file",
                    "tool_calls": [
                        {"id": "call_0_0", "name": "lookup", "args": {"q": "a"}}
                    ],
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "latency_ms": 7,
                    "cache_status": "live",
                    "attempts": 1,
                },
                {
                    "text": "second",
                    "tool_calls": [],
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "latency_ms": 3,
                    "cache_status": "live",
                    "attempts": 1,
                },
            ]
        )
    )
    p = ScriptedProvider.from_file(path)
    first = p.generate(make_request())
    assert first.text == "from file"
    assert first.tool_calls[0].name == "lookup" and first.tool_calls[0].args == {
        "q": "a"
    }
    assert (first.input_tokens, first.output_tokens, first.latency_ms) == (5, 6, 7)
    assert p.generate(make_request()).text == "second"
    with pytest.raises(ScriptExhausted):
        p.generate(make_request())


def test_scripted_provider_from_file_invalid_json_names_path(tmp_path: Path):
    path = tmp_path / "broken_script.json"
    path.write_text("{not json")
    with pytest.raises(Exception) as ei:
        ScriptedProvider.from_file(path)
    assert str(path) in str(ei.value)
    assert not isinstance(ei.value, json.JSONDecodeError) or str(path) in str(ei.value)


# --- errors ----------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        RateLimitError,
        TransientError,
        PermanentModelError,
        CacheMissNoLiveModel,
        QuotaExhausted,
    ],
)
def test_error_classes_importable_and_raisable(cls):
    assert issubclass(cls, Exception)
    assert getattr(llm, cls.__name__) is cls
    assert cls.__module__ in ("agent_replay.llm.types", "agent_replay.llm.provider")
    with pytest.raises(cls):
        raise cls("boom")


def test_budget_exceeded_message_and_attrs():
    assert BudgetExceeded.__module__ in (
        "agent_replay.llm.types",
        "agent_replay.llm.provider",
    )
    e = BudgetExceeded(spent_usd=1.2, budget_usd=1.5)
    assert e.spent_usd == 1.2 and e.budget_usd == 1.5
    s = str(e)
    assert "1.20" in s and "1.50" in s and "AGENT_REPLAY_RUN_BUDGET_USD" in s
    assert issubclass(BudgetExceeded, Exception)


def _req(text: str = "hi") -> ModelRequest:
    return ModelRequest(
        provider="p", model="m", system="s", messages=[Message(role="user", text=text)]
    )


def test_hash_ignores_unset_vs_default_tools():
    assert request_hash(_req()) == request_hash(_req().model_copy(update={"tools": []}))


def test_from_file_missing_file_names_path(tmp_path):
    path = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="nope.json"):
        ScriptedProvider.from_file(path)


def test_from_file_wrong_shape_names_path(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"text": "x"}')
    with pytest.raises(ValueError, match="bad.json"):
        ScriptedProvider.from_file(path)


def _tool_req(response: dict) -> ModelRequest:
    from agent_replay.llm.types import ToolResult

    msg = Message(
        role="tool",
        tool_results=[ToolResult(call_id="call_0_0", name="search", response=response)],
    )
    return ModelRequest(provider="p", model="m", system="s", messages=[msg])


def test_tool_result_round_trips_json():
    req = _tool_req({"error": "boom"})
    assert ModelRequest.model_validate_json(req.model_dump_json()) == req


def test_tool_result_changes_hash():
    assert request_hash(_tool_req({"a": 1})) != request_hash(_tool_req({"a": 2}))


def test_scripted_provider_returns_copy():
    from agent_replay.llm import ModelResponse

    script = [ModelResponse(text="x")]
    out = ScriptedProvider(script).generate(_req())
    out.text = "mutated"
    assert script[0].text == "x"
