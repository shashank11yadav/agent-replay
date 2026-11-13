"""Unit tests for Gemini provider mapping internals (offline)."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from agent_replay.llm import Message, PermanentModelError, ToolCall
from agent_replay.llm.gemini import (
    GeminiProvider,
    _count,
    _to_contents,
)
from agent_replay.llm.hashing import next_step

KEY = "AIzaSyUNIT-SECRET"


def test_count_treats_none_and_missing_as_zero():
    assert _count(None, "x") == 0
    assert _count(SimpleNamespace(x=None), "x") == 0
    assert _count(SimpleNamespace(), "x") == 0
    assert _count(SimpleNamespace(x=4), "x") == 4


def test_next_step_counts_prior_model_turns():
    user = Message(role="user", text="a")
    model = Message(role="model", text="b")
    assert next_step([user]) == 1
    assert next_step([user, model, user, model]) == 3


def test_empty_model_turn_is_skipped():
    contents = _to_contents([Message(role="user", text="q"), Message(role="model")])
    assert [c.role for c in contents] == ["user"]


def test_model_turn_text_precedes_calls():
    msg = Message(
        role="model", text="t", tool_calls=[ToolCall(id="c", name="n", args={"a": 1})]
    )
    parts = _to_contents([msg])[0].parts
    assert parts is not None
    assert parts[0].text == "t"
    assert parts[1].function_call is not None
    assert parts[1].function_call.name == "n"


def test_key_scrubbed_from_error_message():
    exc = errors.ClientError(400, {"error": {"message": f"bad key {KEY}"}})

    class Models:
        def generate_content(self, **kwargs):
            raise exc

    p = GeminiProvider(api_key=KEY, client=SimpleNamespace(models=Models()))
    req_msgs = [Message(role="user", text="hi")]
    from agent_replay.llm import ModelRequest

    req = ModelRequest(provider="gemini", model="m", system="s", messages=req_msgs)
    with pytest.raises(PermanentModelError) as ei:
        p.generate(req)
    assert KEY not in str(ei.value)
    assert "[REDACTED]" in str(ei.value)


def _provider_raising(exc: Exception, key: str = KEY) -> GeminiProvider:
    class Models:
        def generate_content(self, **kwargs):
            raise exc

    return GeminiProvider(api_key=key, client=SimpleNamespace(models=Models()))


def _request():
    from agent_replay.llm import ModelRequest

    return ModelRequest(
        provider="gemini",
        model="m",
        system="s",
        messages=[Message(role="user", text="hi")],
    )


def test_thought_signature_round_trip():
    from google.genai import types

    sig = b"\x00\xffopaque-sig\x10"
    fc = types.FunctionCall(name="lookup", args={"q": 1})
    response = SimpleNamespace(
        text=None,
        function_calls=[fc],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(function_call=None, thought_signature=None),
                        SimpleNamespace(function_call=fc, thought_signature=sig),
                    ]
                )
            )
        ],
        usage_metadata=None,
    )
    out = GeminiProvider._to_response(response, 1)
    call = out.tool_calls[0]
    assert call.signature is not None
    parts = _to_contents([Message(role="model", tool_calls=[call])])[0].parts
    assert parts is not None
    assert parts[0].thought_signature == sig
    assert parts[0].function_call is not None
    assert parts[0].function_call.name == "lookup"


def test_signature_none_when_candidates_missing_or_empty():
    fc = SimpleNamespace(name="n", args={})
    for extra in ({}, {"candidates": None}, {"candidates": []}):
        resp = SimpleNamespace(
            text=None, function_calls=[fc], usage_metadata=None, **extra
        )
        assert GeminiProvider._to_response(resp, 1).tool_calls[0].signature is None
    resp = SimpleNamespace(
        text=None,
        function_calls=[fc],
        usage_metadata=None,
        candidates=[SimpleNamespace(content=None)],
    )
    assert GeminiProvider._to_response(resp, 1).tool_calls[0].signature is None


def test_unsigned_call_part_has_no_signature():
    parts = _to_contents(
        [Message(role="model", tool_calls=[ToolCall(id="c", name="n")])]
    )[0].parts
    assert parts is not None
    assert parts[0].thought_signature is None


def test_request_hash_ignores_signature():
    from agent_replay.llm import ModelRequest
    from agent_replay.llm.hashing import request_hash

    def req(sig):
        call = ToolCall(id="c", name="n", args={"a": 1}, signature=sig)
        return ModelRequest(
            provider="gemini",
            model="m",
            system="s",
            messages=[Message(role="model", tool_calls=[call])],
        )

    assert request_hash(req(None)) == request_hash(req("c2lnbmF0dXJl"))
    assert req("c2lnbmF0dXJl").messages[0].tool_calls[0].signature == "c2lnbmF0dXJl"


def test_stripped_key_redacted_from_api_error():
    from agent_replay.llm import PermanentModelError

    exc = errors.ClientError(400, {"error": {"message": f"bad key {KEY}"}})
    p = _provider_raising(exc, key=f"  {KEY}\n")
    with pytest.raises(PermanentModelError) as info:
        p.generate(_request())
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout(f"timed out {KEY}"),
        httpx.ConnectError(f"refused {KEY}"),
    ],
)
def test_transport_error_maps_to_transient(exc):
    from agent_replay.llm import TransientError

    with pytest.raises(TransientError) as info:
        _provider_raising(exc).generate(_request())
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_build_client_rejects_sub_millisecond_timeout():
    from agent_replay.llm.gemini import build_client

    with pytest.raises(ValueError):
        build_client(KEY, timeout_s=0.0005)
    assert build_client(KEY, timeout_s=0.001).models is not None


def test_chained_api_error_is_sanitized():
    import traceback

    response = httpx.Response(
        400, request=httpx.Request("POST", "https://x", headers={"x-goog-api-key": KEY})
    )
    exc = errors.ClientError(
        400, {"error": {"message": f"bad key {KEY}", "extra": [f"  {KEY}\n"]}}, response
    )
    p = _provider_raising(exc, key=f"  {KEY}\n")
    with pytest.raises(PermanentModelError) as info:
        p.generate(_request())
    cause = info.value.__cause__
    assert cause is exc
    assert cause.response is None
    for text in (
        str(cause),
        repr(cause),
        repr(cause.args),
        str(cause.message),
        str(cause.details),
        "".join(traceback.format_exception(info.value)),
    ):
        assert KEY not in text


@pytest.mark.parametrize("text", [None, ""])
def test_a_candidate_with_no_text_and_no_call_is_permanent(text):
    # e.g. a SAFETY or MAX_TOKENS finish: storing it as an empty answer would hide it.
    resp = SimpleNamespace(
        text=text, function_calls=None, usage_metadata=None, candidates=[]
    )
    with pytest.raises(PermanentModelError, match="no text and no function call"):
        GeminiProvider._to_response(resp, 1)
