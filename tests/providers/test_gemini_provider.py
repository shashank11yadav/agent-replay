"""Gemini provider, fully offline; google-genai is imported lazily."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any

import pytest
from support import ROOT

from agent_replay.llm import (
    Message,
    ModelRequest,
    PermanentModelError,
    RateLimitError,
    ToolCall,
    ToolResult,
    ToolSchema,
    TransientError,
)

KEY = "AIzaSyFAKE-SECRET-KEY-1234567890"
SRC = ROOT / "src"


class FakeModels:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.models = FakeModels(response, exc)


def usage(
    p: Any = None, c: Any = None, t: Any = None, tu: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=p,
        candidates_token_count=c,
        thoughts_token_count=t,
        tool_use_prompt_token_count=tu,
        total_token_count=None,
    )


def fc(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args, id=None)


def resp(text: Any = None, calls: Any = None, u: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text, function_calls=calls, candidates=[], usage_metadata=u or usage(1, 1)
    )


def make_req(**kw: Any) -> ModelRequest:
    base: dict[str, Any] = {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "system": "You are a helper.",
        "messages": [Message(role="user", text="hi")],
        "temperature": 0.0,
        "seed": 7,
    }
    base.update(kw)
    return ModelRequest(**base)


def provider(client: Any, key: str = KEY) -> Any:
    from agent_replay.llm.gemini import GeminiProvider

    return GeminiProvider(api_key=key, client=client)


def test_only_gemini_module_imports_google_genai() -> None:
    offenders = []
    for p in SRC.rglob("*.py"):
        txt = p.read_text()
        if re.search(
            r"^\s*(from\s+google(\.genai)?\s+import|import\s+google)", txt, re.MULTILINE
        ):
            offenders.append(p.name)
    assert offenders == ["gemini.py"]


def test_accepts_injected_client_no_network() -> None:
    c = FakeClient(resp(text="ok"))
    out = provider(c).generate(make_req())
    assert out.text == "ok"
    assert len(c.models.calls) == 1


def test_generate_call_shape() -> None:
    from google.genai import types

    c = FakeClient(resp(text="ok"))
    tools = [
        ToolSchema(
            name="search",
            description="d",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
    ]
    provider(c).generate(make_req(tools=tools, temperature=0.3, seed=11))
    kw = c.models.calls[0]
    assert kw["model"] == "gemini-3.5-flash-lite"
    cfg = kw["config"]
    assert isinstance(cfg, types.GenerateContentConfig)
    assert cfg.automatic_function_calling.disable is True
    assert cfg.temperature == 0.3
    assert cfg.seed == 11
    decls = [d for t in cfg.tools for d in t.function_declarations]
    assert len(decls) == 1
    assert decls[0].name == "search"
    assert decls[0].description == "d"
    assert decls[0].parameters_json_schema == tools[0].parameters


def test_system_passed_as_system_instruction() -> None:
    c = FakeClient(resp(text="ok"))
    provider(c).generate(make_req(system="Be terse."))
    assert c.models.calls[0]["config"].system_instruction == "Be terse."


def test_function_call_maps_to_tool_call() -> None:
    c = FakeClient(resp(calls=[fc("search", {"q": "x"})]))
    out = provider(c).generate(make_req())
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert (tc.name, tc.args) == ("search", {"q": "x"})
    assert re.fullmatch(r"call_\d+_0", tc.id)


def test_text_response_has_no_tool_calls() -> None:
    for calls in (None, []):
        out = provider(FakeClient(resp(text="answer", calls=calls))).generate(
            make_req()
        )
        assert out.text == "answer"
        assert out.tool_calls == []


@pytest.mark.parametrize(
    "u,exp",
    [
        (usage(10, 5, 3, 2), (12, 8)),
        (usage(10, 5, None, None), (10, 5)),
        (usage(None, None, 4, None), (0, 4)),
        (usage(None, None, None, 7), (7, 0)),
        (usage(), (0, 0)),
    ],
)
def test_token_accounting(u: Any, exp: tuple[int, int]) -> None:
    out = provider(FakeClient(resp(text="a", u=u))).generate(make_req())
    assert (out.input_tokens, out.output_tokens) == exp


def test_history_rebuilt_with_real_types() -> None:
    from google.genai import types

    c = FakeClient(resp(text="done"))
    msgs = [
        Message(role="user", text="find x"),
        Message(
            role="model",
            text="looking",
            tool_calls=[ToolCall(id="call_1_0", name="search", args={"q": "x"})],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(call_id="call_1_0", name="search", response={"r": 1})
            ],
        ),
    ]
    provider(c).generate(make_req(messages=msgs))
    contents = c.models.calls[0]["contents"]
    assert [x.role for x in contents] == ["user", "model", "tool"]
    assert all(isinstance(x, types.Content) for x in contents)
    assert contents[0].parts[0].text == "find x"
    mparts = contents[1].parts
    assert any(p.text == "looking" for p in mparts)
    calls = [p.function_call for p in mparts if p.function_call is not None]
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert dict(calls[0].args) == {"q": "x"}
    fr = contents[2].parts[0].function_response
    assert fr.name == "search"
    assert dict(fr.response) == {"r": 1}


def test_model_turn_with_only_tool_calls_has_no_empty_text_part() -> None:
    c = FakeClient(resp(text="x"))
    msgs = [
        Message(role="user", text="q"),
        Message(role="model", tool_calls=[ToolCall(id="call_1_0", name="t", args={})]),
        Message(
            role="tool",
            tool_results=[ToolResult(call_id="call_1_0", name="t", response={})],
        ),
    ]
    provider(c).generate(make_req(messages=msgs))
    parts = c.models.calls[0]["contents"][1].parts
    assert len(parts) == 1 and parts[0].function_call.name == "t"


def test_error_mapping() -> None:
    from google.genai import errors

    def body(status: str) -> dict[str, Any]:
        return {"error": {"message": "boom", "status": status}}

    cases = [
        (errors.ClientError(429, body("RESOURCE_EXHAUSTED")), RateLimitError),
        (errors.ServerError(503, body("UNAVAILABLE")), TransientError),
        (errors.ClientError(400, body("INVALID_ARGUMENT")), PermanentModelError),
        (errors.ClientError(403, body("PERMISSION_DENIED")), PermanentModelError),
    ]
    for exc, want in cases:
        with pytest.raises(want) as ei:
            provider(FakeClient(exc=exc)).generate(make_req())
        assert type(ei.value) is want
        assert ei.value.__cause__ is exc


def test_400_is_not_rate_limit() -> None:
    from google.genai import errors

    exc = errors.ClientError(
        400, {"error": {"message": "RESOURCE_EXHAUSTED 429", "status": "X"}}
    )
    with pytest.raises(PermanentModelError):
        provider(FakeClient(exc=exc)).generate(make_req())


def test_empty_key_raises_value_error_without_echo() -> None:
    from agent_replay.llm.gemini import GeminiProvider

    for bad in ("", "   "):
        with pytest.raises(ValueError) as ei:
            GeminiProvider(api_key=bad, client=FakeClient())
        assert KEY not in str(ei.value)
    with pytest.raises(ValueError):
        GeminiProvider(api_key="", client=None)


def test_key_never_in_messages_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    from google.genai import errors

    caplog.set_level(logging.DEBUG)
    exc = errors.ClientError(
        429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    p = provider(FakeClient(exc=exc))
    with pytest.raises(RateLimitError) as ei:
        p.generate(make_req())
    assert KEY not in str(ei.value) and KEY not in repr(ei.value)
    p2 = provider(FakeClient(resp(text="fine")))
    p2.generate(make_req())
    assert KEY not in repr(p) and KEY not in str(p)
    for rec in caplog.records:
        assert KEY not in rec.getMessage()
    assert KEY not in caplog.text


def test_build_client_offline_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai
    from google.genai import types

    from agent_replay.llm.gemini import build_client

    captured: dict[str, Any] = {}
    real = genai.Client

    def spy(*a: Any, **k: Any) -> Any:
        captured.update(k)
        return real(*a, **k)

    import httpx

    def no_net(*a: Any, **k: Any) -> Any:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(httpx.Client, "send", no_net)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_net)
    monkeypatch.setattr(genai, "Client", spy)
    client = build_client(KEY, timeout_s=30)
    assert client is not None
    opts = captured["http_options"]
    assert isinstance(opts, types.HttpOptions)
    assert opts.timeout == 30000
    assert opts.retry_options.attempts == 1
    assert captured["api_key"] == KEY


def test_build_client_timeout_in_ms() -> None:
    from agent_replay.llm.gemini import build_client

    c = build_client(KEY, timeout_s=5)
    assert c is not None
