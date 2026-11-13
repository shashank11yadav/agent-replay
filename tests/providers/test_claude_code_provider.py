"""The claude_code provider against a fake CLI script.

The fake CLI (fake_claude_cli.py behind a launcher in tmp_path) records argv, stdin, cwd and its
environment; no test here runs the real `claude` binary."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from claude_cli_harness import (
    FakeClaude,
    clean_agent_env,
    error_json,
    step_ok,
    step_text,
    success_json,
)

from agent_replay.llm import (
    Message,
    ModelRequest,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ToolCall,
    ToolResult,
    ToolSchema,
    TransientError,
    request_hash,
)
from agent_replay.llm.claude_code import ClaudeCodeProvider, classify_failure

SCHEMA_LITERAL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["name", "args"],
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

GET_ORDER = ToolSchema(
    name="get_order",
    description="Look up an order.",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
CAFE = ToolSchema(
    name="café",
    description="Zażółć: naïve ☕",
    parameters={"z": 1, "a": {"é": [2, 1]}},
)

SYSTEM = "You are a shop assistant."


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    clean_agent_env(monkeypatch)


@pytest.fixture
def fake(tmp_path: Path) -> FakeClaude:
    return FakeClaude(tmp_path)


def cc() -> Any:
    import agent_replay.llm.factory  # noqa: F401
    from agent_replay.llm import claude_code

    return claude_code


def req(
    messages: list[Message] | None = None,
    *,
    tools: list[ToolSchema] | None = None,
    system: str = SYSTEM,
    model: str = "claude-haiku-5-5",
    temperature: float = 0.0,
    seed: int | None = 7,
) -> ModelRequest:
    return ModelRequest(
        provider="claude_code",
        model=model,
        system=system,
        messages=messages or [Message(role="user", text="hi")],
        tools=[GET_ORDER] if tools is None else tools,
        temperature=temperature,
        seed=seed,
    )


def flags(system: str, model: str = "claude-haiku-5-5") -> list[str]:
    return [
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(SCHEMA_LITERAL, separators=(",", ":")),
        "--model",
        model,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--safe-mode",
        "--disable-slash-commands",
        "--permission-prompts",
        "none",
    ]


def system_for(tools_block: str, system: str = SYSTEM) -> str:
    return system + "\n\n## Tools\n" + tools_block + "\n\n" + cc().REPLY_INSTRUCTIONS


GET_ORDER_BLOCK = (
    "- get_order: Look up an order.\n"
    '  parameters: {"properties":{"order_id":{"type":"string"}},'
    '"required":["order_id"],"type":"object"}'
)
CAFE_BLOCK = '- café: Zażółć: naïve ☕\n  parameters: {"a":{"é":[2,1]},"z":1}'


# ---- argv ------------------------------------------------------------------------------


def test_argv_is_exactly_the_documented_list(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    provider = fake.provider(monkeypatch)
    provider.generate(req())
    (call,) = fake.calls()
    assert call["argv0"] == shutil.which(fake.path) == fake.path
    assert call["argv"] == flags(system_for(GET_ORDER_BLOCK))


def test_max_turns_is_appended_only_when_set(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req())
    fake.provider(monkeypatch, AGENT_REPLAY_CLAUDE_MAX_TURNS="3").generate(req())
    first, second = fake.calls()
    assert "--max-turns" not in first["argv"]
    assert second["argv"] == flags(system_for(GET_ORDER_BLOCK)) + ["--max-turns", "3"]


def test_forbidden_flags_are_never_passed(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch, AGENT_REPLAY_CLAUDE_MAX_TURNS="2").generate(req())
    (call,) = fake.calls()
    for flag in ("--bare", "--setting-sources", "--fallback-model"):
        assert flag not in call["argv"]


def test_the_model_flag_follows_the_request_model(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req(model="claude-sonnet-5-5"))
    (call,) = fake.calls()
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"


def test_one_subprocess_per_generate_and_no_shell(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, tmp_path: Path
) -> None:
    nasty = f"x; touch {tmp_path}/pwned #\n$(touch {tmp_path}/pwned2) 'q\" `id`"
    provider = fake.provider(monkeypatch)
    provider.generate(req(system=nasty))
    assert len(fake.calls()) == 1
    provider.generate(req(system=nasty))
    assert len(fake.calls()) == 2
    assert not (tmp_path / "pwned").exists()
    assert not (tmp_path / "pwned2").exists()
    argv = fake.calls()[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1].startswith(nasty + "\n\n## Tools\n")


def test_no_subprocess_call_uses_shell_or_a_command_string(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []
    real_init = subprocess.Popen.__init__

    def spy_init(self: Any, args: Any, *a: Any, **kw: Any) -> None:
        seen.append((args, kw))
        real_init(self, args, *a, **kw)

    provider = fake.provider(monkeypatch)
    monkeypatch.setattr(subprocess.Popen, "__init__", spy_init)
    provider.generate(req())
    assert len(seen) == 1
    args, kw = seen[0]
    assert isinstance(args, list) and all(isinstance(a, str) for a in args)
    assert not kw.get("shell")


# ---- stdin and cwd ---------------------------------------------------------------------


def test_conversation_goes_to_stdin_as_utf8_and_not_into_argv(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    text = "What is the status of order A1001? café ☕ UNIQUEUSERTEXT"
    fake.provider(monkeypatch).generate(req([Message(role="user", text=text)]))
    (call,) = fake.calls()
    assert call["stdin"] == f"[user]\n{text}".encode()
    assert not any("UNIQUEUSERTEXT" in a for a in call["argv"])


@pytest.mark.parametrize(
    "case",
    ["success", "exit_1", "non_json", "bad_shape", "timeout", "signal", "quota"],
)
def test_cwd_is_a_fresh_empty_dir_removed_on_success_and_every_error(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, case: str
) -> None:
    steps: dict[str, dict[str, Any]] = {
        "success": step_text("ok"),
        "exit_1": {"stdout_json": error_json("boom", status=500), "exit_code": 1},
        "non_json": {"stdout": "not json", "exit_code": 0},
        "bad_shape": step_ok({"unknown": 1}),
        "timeout": {"sleep": 30, "stdout": ""},
        "signal": {"kill_signal": int(signal.SIGKILL)},
        "quota": {
            "stdout_json": error_json("You've hit your session limit"),
            "exit_code": 1,
        },
    }
    step = dict(steps[case], write_file_in_cwd=True)
    fake.script(step)
    extra = {"AGENT_REPLAY_TIMEOUT_S": "0.5"} if case == "timeout" else {}
    provider = fake.provider(monkeypatch, **extra)
    if case == "success":
        provider.generate(req())
    else:  # the exact error class is asserted in the error tests
        with pytest.raises(
            (RateLimitError, TransientError, PermanentModelError, QuotaExhausted)
        ):
            provider.generate(req())
    (call,) = fake.calls()
    cwd = call["cwd"]
    assert call["cwd_listing"] == []
    assert Path(cwd).resolve() != Path.cwd().resolve()
    assert Path(cwd).is_absolute()
    assert not os.path.exists(cwd)


def test_each_call_gets_its_own_cwd(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    provider = fake.provider(monkeypatch)
    provider.generate(req())
    provider.generate(req())
    first, second = fake.calls()
    assert first["cwd"] != second["cwd"]


# ---- environment -----------------------------------------------------------------------


STRIPPED_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "AGENT_REPLAY_")
STRIPPED = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)


def test_child_environment_drops_agent_replay_and_login_overrides(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    monkeypatch.setenv("AGENT_REPLAY_GEMINI_API_KEY", "gem-secret-1")
    monkeypatch.setenv("AGENT_REPLAY_DATABASE_URL", "postgresql://u:pw-secret@h/db")
    monkeypatch.setenv("AGENT_REPLAY_ANYTHING_ELSE", "x")
    for name in STRIPPED:
        monkeypatch.setenv(name, f"{name.lower()}-secret")
    monkeypatch.setenv("KEEP_ME", "kept")
    # Only the prefixes are stripped: a variable that merely mentions CLAUDE stays.
    monkeypatch.setenv("CLAUDE_CODE_KEEP", "kept-too")
    provider = fake.provider(monkeypatch)
    expected = {
        k: v for k, v in os.environ.items() if not k.startswith(STRIPPED_PREFIXES)
    }
    provider.generate(req())
    (call,) = fake.calls()
    child = call["env"]
    assert not [k for k in child if k.startswith(STRIPPED_PREFIXES)]
    assert child["CLAUDE_CODE_KEEP"] == "kept-too"
    blob = json.dumps(child)
    for secret in (
        "gem-secret-1",
        "pw-secret",
        *(f"{n.lower()}-secret" for n in STRIPPED),
    ):
        assert secret not in blob
    assert child["KEEP_ME"] == "kept"
    for key, value in expected.items():
        if key == "LC_CTYPE":  # the interpreter may coerce it for the C locale
            continue
        assert child.get(key) == value, key


# ---- schema and rendering --------------------------------------------------------------


def test_reply_schema_constant_is_exact_and_free_of_combinators() -> None:
    schema = cc().REPLY_SCHEMA
    assert schema == SCHEMA_LITERAL
    assert "$schema" not in schema
    assert "oneOf" not in schema and "anyOf" not in schema


def test_reply_schema_is_passed_as_compact_json(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req())
    argv = fake.calls()[0]["argv"]
    raw = argv[argv.index("--json-schema") + 1]
    assert json.loads(raw) == SCHEMA_LITERAL
    assert " " not in raw and "\n" not in raw


def test_reply_instructions_is_a_non_empty_string_constant() -> None:
    assert isinstance(cc().REPLY_INSTRUCTIONS, str)
    assert cc().REPLY_INSTRUCTIONS.strip()


def test_system_prompt_with_two_tools_in_request_order(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req(tools=[GET_ORDER, CAFE]))
    argv = fake.calls()[0]["argv"]
    got = argv[argv.index("--system-prompt") + 1]
    assert got == (
        SYSTEM
        + "\n\n## Tools\n"
        + GET_ORDER_BLOCK
        + "\n"
        + CAFE_BLOCK
        + "\n\n"
        + cc().REPLY_INSTRUCTIONS
    )


def test_system_prompt_order_follows_the_request_not_the_alphabet(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req(tools=[CAFE, GET_ORDER]))
    argv = fake.calls()[0]["argv"]
    got = argv[argv.index("--system-prompt") + 1]
    assert got == system_for(CAFE_BLOCK + "\n" + GET_ORDER_BLOCK)


def test_system_prompt_without_tools_says_none(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.provider(monkeypatch).generate(req(tools=[]))
    argv = fake.calls()[0]["argv"]
    got = argv[argv.index("--system-prompt") + 1]
    assert got == SYSTEM + "\n\n## Tools\nNone.\n\n" + cc().REPLY_INSTRUCTIONS


def test_stdin_for_a_model_turn_with_text_and_tool_calls_and_tool_results(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    messages = [
        Message(role="user", text="Check A1001 and A1002."),
        Message(
            role="model",
            text="Let me check.",
            tool_calls=[
                ToolCall(
                    id="call_1_0",
                    name="get_order",
                    args={"order_id": "A1001"},
                    signature="c2lnbmF0dXJl",
                ),
                ToolCall(id="call_1_1", name="get_order", args={"order_id": "A1002"}),
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(
                    call_id="call_1_0",
                    name="get_order",
                    response={"status": "shipped", "total": 37.95},
                ),
                ToolResult(
                    call_id="call_1_1",
                    name="get_order",
                    response={"status": "pending", "items": [1, 2]},
                ),
            ],
        ),
        Message(
            role="model",
            tool_calls=[ToolCall(id="call_2_0", name="get_order", args={})],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(
                    call_id="call_2_0",
                    name="get_order",
                    response={"error": "not_found"},
                )
            ],
        ),
        Message(role="model", text="Done."),
        Message(role="model"),
        Message(role="user", text="thanks"),
    ]
    blocks = [
        "[user]\nCheck A1001 and A1002.",
        (
            "[assistant]\nLet me check."
            '\ntool_calls: [{"args":{"order_id":"A1001"},"id":"call_1_0",'
            '"name":"get_order"},'
            '{"args":{"order_id":"A1002"},"id":"call_1_1","name":"get_order"}]'
        ),
        (
            "[tool]"
            '\ntool_result: {"call_id":"call_1_0","name":"get_order",'
            '"response":{"status":"shipped","total":37.95}}'
            '\ntool_result: {"call_id":"call_1_1","name":"get_order",'
            '"response":{"items":[1,2],"status":"pending"}}'
        ),
        '[assistant]\ntool_calls: [{"args":{},"id":"call_2_0","name":"get_order"}]',
        (
            '[tool]\ntool_result: {"call_id":"call_2_0","name":"get_order",'
            '"response":{"error":"not_found"}}'
        ),
        "[assistant]\nDone.",
        "[assistant]",
        "[user]\nthanks",
    ]
    fake.provider(monkeypatch).generate(req(messages))
    (call,) = fake.calls()
    assert call["stdin"] == "\n\n".join(blocks).encode("utf-8")
    assert b"c2lnbmF0dXJl" not in call["stdin"]


def test_the_same_request_renders_to_identical_bytes(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    messages = [
        Message(role="user", text="q"),
        Message(
            role="model",
            tool_calls=[
                ToolCall(id="call_1_0", name="get_order", args={"b": 1, "a": 2})
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(
                    call_id="call_1_0", name="get_order", response={"y": 1, "x": 2}
                )
            ],
        ),
    ]
    provider = fake.provider(monkeypatch)
    provider.generate(req(messages, tools=[GET_ORDER, CAFE]))
    provider.generate(req(messages, tools=[GET_ORDER, CAFE]))
    one, two = fake.calls()
    assert one["stdin"] == two["stdin"]
    assert one["argv"] == two["argv"]


# ---- success mapping -------------------------------------------------------------------


def test_text_reply(monkeypatch: pytest.MonkeyPatch, fake: FakeClaude) -> None:
    fake.script(step_text("hello there"))
    out = fake.provider(monkeypatch).generate(req())
    assert out.text == "hello there"
    assert out.tool_calls == []


def test_tool_call_reply_has_no_text_and_stepped_ids(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script(
        step_ok(
            {
                "tool_calls": [
                    {"name": "get_order", "args": {"order_id": "A1"}},
                    {"name": "get_order", "args": {}},
                ]
            }
        )
    )
    messages = [
        Message(role="user", text="go"),
        Message(role="model", tool_calls=[ToolCall(id="call_1_0", name="x", args={})]),
        Message(
            role="tool",
            tool_results=[ToolResult(call_id="call_1_0", name="x", response={})],
        ),
        Message(role="model", text="more"),
        Message(role="user", text="again"),
    ]
    out = fake.provider(monkeypatch).generate(req(messages))
    assert out.text is None
    assert [(c.id, c.name, c.args) for c in out.tool_calls] == [
        ("call_3_0", "get_order", {"order_id": "A1"}),
        ("call_3_1", "get_order", {}),
    ]


def test_first_turn_ids_use_step_one(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script(step_ok({"text": "calling", "tool_calls": [{"name": "t", "args": {}}]}))
    out = fake.provider(monkeypatch).generate(req())
    assert out.text == "calling"
    assert [c.id for c in out.tool_calls] == ["call_1_0"]


def test_result_field_is_never_parsed_as_a_fallback(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    obj = success_json(result=json.dumps({"text": "from result"}))
    fake.script({"stdout_json": obj})
    with pytest.raises(PermanentModelError):
        fake.provider(monkeypatch).generate(req())


# ---- reply shape -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        ["a list"],
        "a string",
        {},
        {"text": ""},
        {"text": "", "tool_calls": []},
        {"tool_calls": []},
        {"text": "ok", "extra": 1},
        {"text": 5},
        {"tool_calls": [{"args": {}}]},
        {"tool_calls": [{"name": 3, "args": {}}]},
        {"tool_calls": [{"name": "t", "args": ["x"]}]},
        {"tool_calls": [{"name": "t", "args": "x"}]},
    ],
    ids=lambda b: json.dumps(b)[:40],
)
def test_invalid_reply_shape_is_a_permanent_error(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, bad: Any
) -> None:
    fake.script(step_ok(bad))
    with pytest.raises(PermanentModelError) as info:
        fake.provider(monkeypatch).generate(req())
    assert type(info.value) is PermanentModelError
    assert "invalid reply shape" in str(info.value)


# ---- line 13 and 14: usage and latency -------------------------------------------------


def test_input_tokens_sum_the_three_input_fields(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 300,
        "output_tokens": 7,
    }
    fake.script(
        step_text(
            "ok",
            usage=usage,
            modelUsage={"claude-haiku-5-5": {"inputTokens": 999, "outputTokens": 888}},
            total_cost_usd=5.0,
        )
    )
    out = fake.provider(monkeypatch).generate(req())
    assert (out.input_tokens, out.output_tokens) == (330, 7)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 4}, (4, 0)),
        ({"output_tokens": 9}, (0, 9)),
        ({"cache_read_input_tokens": 6, "output_tokens": 1}, (6, 1)),
        ({}, (0, 0)),
        (None, (0, 0)),
    ],
)
def test_missing_usage_fields_count_zero(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeClaude,
    usage: Any,
    expected: tuple[int, int],
) -> None:
    fake.script(step_text("ok", usage=usage))
    out = fake.provider(monkeypatch).generate(req())
    assert (out.input_tokens, out.output_tokens) == expected


def test_latency_is_the_process_wall_time_and_at_least_one_ms(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    provider = fake.provider(monkeypatch)
    fast = provider.generate(req())
    assert fast.latency_ms >= 1
    fake.script({**step_text("slow"), "sleep": 0.4})
    slow = provider.generate(req())
    assert slow.latency_ms >= 380
    assert slow.latency_ms < 20_000


# ---- temperature and seed --------------------------------------------------------------


def test_temperature_and_seed_change_the_hash_but_not_argv_or_stdin(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    provider = fake.provider(monkeypatch)
    a = req(temperature=0.0, seed=7)
    b = req(temperature=0.9, seed=7)
    c = req(temperature=0.0, seed=99)
    d = req(temperature=0.0, seed=None)
    for r in (a, b, c, d):
        provider.generate(r)
    calls = fake.calls()
    assert len(calls) == 4
    for other in calls[1:]:
        assert other["argv"] == calls[0]["argv"]
        assert other["stdin"] == calls[0]["stdin"]
    assert len({request_hash(r) for r in (a, b, c, d)}) == 4
    assert not any("temperature" in x or "seed" in x for x in calls[0]["argv"])


# ---- timeout and signals ---------------------------------------------------------------


def test_timeout_kills_the_process_and_raises_transient_without_a_chained_cause(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script({"sleep": 30, "stdout": ""})
    provider = fake.provider(monkeypatch, AGENT_REPLAY_TIMEOUT_S="0.5")
    started = time.monotonic()
    with pytest.raises(TransientError) as info:
        provider.generate(req(system="SECRET-SYSTEM-MARKER"))
    assert time.monotonic() - started < 20
    exc = info.value
    assert type(exc) is TransientError
    assert exc.__cause__ is None
    assert exc.__suppress_context__ or exc.__context__ is None
    assert "SECRET-SYSTEM-MARKER" not in str(exc) + repr(exc)
    assert "--output-format" not in str(exc) + repr(exc)


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM, signal.SIGHUP])
def test_a_process_ended_by_a_signal_is_transient(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, sig: signal.Signals
) -> None:
    fake.script({"kill_signal": int(sig)})
    with pytest.raises(TransientError) as info:
        fake.provider(monkeypatch).generate(req())
    assert type(info.value) is TransientError


def test_exit_code_143_is_transient(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script({"stdout": "", "exit_code": 143})
    with pytest.raises(TransientError):
        fake.provider(monkeypatch).generate(req())


# ---- classification --------------------------------------------------------------------


def _exit1(obj: dict[str, Any]) -> dict[str, Any]:
    return {"stdout_json": obj, "exit_code": 1}


def _stderr_only(text: str, code: int = 1) -> dict[str, Any]:
    return {"stdout": "", "stderr": text, "exit_code": code}


def _errors_list(
    *texts: str, subtype: str = "error_during_execution"
) -> dict[str, Any]:
    obj = error_json("", subtype=subtype)
    del obj["result"]
    obj["errors"] = list(texts)
    return _exit1(obj)


Q, R, T, P = QuotaExhausted, RateLimitError, TransientError, PermanentModelError

CLASSIFICATION: dict[str, tuple[dict[str, Any], type[Exception]]] = {
    # usage limit
    "quota-result": (
        _exit1(error_json("You've hit your session limit · resets 3:45pm")),
        Q,
    ),
    "quota-weekly-in-errors": (
        _errors_list("You've hit your weekly limit · resets Mon 12:00am"),
        Q,
    ),
    "quota-in-stderr": (_stderr_only("You've hit your Opus limit · resets 3:45pm"), Q),
    "quota-beats-429-status": (
        _exit1(error_json("You've hit your limit", status=429)),
        Q,
    ),
    "quota-beats-rate-limit-text": (
        _exit1(error_json("You've hit your limit. rate limit")),
        Q,
    ),
    "quota-beats-5xx-status": (
        _exit1(error_json("You've hit your limit", status=529)),
        Q,
    ),
    "quota-beats-overloaded": (
        _exit1(error_json("Overloaded: You've hit your limit")),
        Q,
    ),
    "quota-with-exit-0-and-is-error": (
        {"stdout_json": error_json("You've hit your session limit"), "exit_code": 0},
        Q,
    ),
    # rate limit
    "429-status": (_exit1(error_json("boom", status=429)), R),
    "429-status-empty-text": (_exit1(error_json("", status=429)), R),
    "request-rejected-text": (_exit1(error_json("Request rejected (429)")), R),
    "rate-limit-lowercase": (_exit1(error_json("hit a rate limit, slow down")), R),
    "rate-limit-mixed-case": (_exit1(error_json("Rate Limit exceeded")), R),
    "rate-limit-in-errors": (_errors_list("Request rejected (429)"), R),
    "rate-limit-in-stderr": (_stderr_only("Request rejected (429)"), R),
    "429-beats-overloaded-text": (_exit1(error_json("Overloaded", status=429)), R),
    "429-beats-login-text": (_exit1(error_json("Not logged in", status=429)), R),
    "rate-limit-text-beats-5xx-status": (
        _exit1(error_json("rate limit reached", status=500)),
        R,
    ),
    # transient
    "status-500": (_exit1(error_json("boom", status=500)), T),
    "status-503": (_exit1(error_json("boom", status=503)), T),
    "status-529": (_exit1(error_json("boom", status=529)), T),
    "status-599": (_exit1(error_json("boom", status=599)), T),
    "5xx-beats-login-text": (_exit1(error_json("Not logged in", status=503)), T),
    "overloaded-text": (
        _exit1(error_json("API Error: Repeated 529 Overloaded errors")),
        T,
    ),
    "overloaded-bare": (_exit1(error_json("Overloaded")), T),
    "api-error-5-text": (_exit1(error_json("API Error: 500 Internal server error")), T),
    "api-error-502-in-stderr": (_stderr_only("API Error: 502 Bad gateway"), T),
    "unable-to-connect": (_exit1(error_json("Unable to connect to API")), T),
    "overloaded-text-beats-400-status": (
        _exit1(error_json("Overloaded", status=400)),
        T,
    ),
    "overloaded-in-errors": (_errors_list("Overloaded"), T),
    # permanent
    "not-logged-in": (_exit1(error_json("Not logged in · Please run /login")), P),
    "login-expired": (_exit1(error_json("Login expired · Please run /login")), P),
    "failed-to-authenticate": (
        _exit1(error_json("Failed to authenticate: OAuth session expired")),
        P,
    ),
    "unknown-model": (
        _exit1(error_json("Model foo-9 is not a recognized model id", status=404)),
        P,
    ),
    "status-400": (_exit1(error_json("bad request", status=400)), P),
    "status-499": (_exit1(error_json("odd", status=499)), P),
    "status-600": (_exit1(error_json("odd", status=600)), P),
    "max-structured-output-retries": (
        _errors_list(
            "Failed to produce valid output",
            subtype="error_max_structured_output_retries",
        ),
        P,
    ),
    "max-turns": (_errors_list(subtype="error_max_turns"), P),
    "max-turns-exit-0": (
        {
            "stdout_json": error_json("", subtype="error_max_turns", is_error=False),
            "exit_code": 0,
        },
        P,
    ),
    "non-zero-exit-with-success-body": (
        {"stdout_json": success_json({"text": "fine"}), "exit_code": 1},
        P,
    ),
    "non-json-stdout": ({"stdout": "Segmentation banana", "exit_code": 0}, P),
    "empty-stdout": ({"stdout": "", "exit_code": 0}, P),
    "json-list-stdout": ({"stdout": "[1, 2]", "exit_code": 0}, P),
    "json-string-stdout": ({"stdout": '"hi"', "exit_code": 0}, P),
    "is-error-true-exit-0": (
        {"stdout_json": error_json("Not logged in"), "exit_code": 0},
        P,
    ),
    "is-error-true-with-structured-output": (
        {
            "stdout_json": success_json({"text": "fine"}, is_error=True),
            "exit_code": 0,
        },
        P,
    ),
    "subtype-not-success-with-structured-output": (
        {
            "stdout_json": success_json(
                {"text": "fine"}, subtype="error_during_execution"
            ),
            "exit_code": 0,
        },
        P,
    ),
    "missing-structured-output": ({"stdout_json": success_json(), "exit_code": 0}, P),
    # a plain-text reply is the model's words, never a CLI error message
    "prose-says-rate-limit": (
        {
            "stdout_json": success_json(result="The order API has a rate limit."),
            "exit_code": 0,
        },
        P,
    ),
    "prose-says-hit-your": (
        {
            "stdout_json": success_json(result="You've hit your daily goal!"),
            "exit_code": 0,
        },
        P,
    ),
    "prose-says-overloaded-with-exit-1": (
        {"stdout_json": success_json(result="The shop is Overloaded."), "exit_code": 1},
        P,
    ),
    "usage-error-on-stderr": (
        _stderr_only("error: unknown option '--permission-prompts'", code=1),
        P,
    ),
    "exit-2-no-output": (_stderr_only("", code=2), P),
}


@pytest.mark.parametrize("name", list(CLASSIFICATION))
def test_failure_classification(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, name: str
) -> None:
    step, expected = CLASSIFICATION[name]
    fake.script(step)
    with pytest.raises(Exception) as info:
        fake.provider(monkeypatch).generate(req())
    assert type(info.value) is expected, (name, repr(info.value))


# ---- error messages --------------------------------------------------------------------


def test_message_has_the_claude_code_prefix_and_the_failure_text(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script(_exit1(error_json("Not logged in · Please run /login")))
    with pytest.raises(PermanentModelError) as info:
        fake.provider(monkeypatch).generate(req())
    msg = str(info.value)
    assert msg.startswith("claude_code: ")
    assert msg.count(": ") >= 2  # 'claude_code: <category>: <text>'
    assert "Not logged in · Please run /login" in msg


def test_message_takes_stderr_text_when_stdout_is_empty(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script(_stderr_only("Request rejected (429) from stderr"))
    with pytest.raises(RateLimitError) as info:
        fake.provider(monkeypatch).generate(req())
    assert str(info.value).startswith("claude_code: ")
    assert "Request rejected (429) from stderr" in str(info.value)


def test_failure_text_is_cut_to_500_characters(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script(_exit1(error_json("A" * 500 + "B" * 500, status=429)))
    with pytest.raises(RateLimitError) as info:
        fake.provider(monkeypatch).generate(req())
    msg = str(info.value)
    assert msg.startswith("claude_code: ")
    assert "A" * 500 in msg
    assert "B" not in msg


def test_a_short_failure_text_is_kept_whole(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    text = "C" * 499
    fake.script(_exit1(error_json(text, status=429)))
    with pytest.raises(RateLimitError) as info:
        fake.provider(monkeypatch).generate(req())
    assert text in str(info.value)


def test_non_json_stdout_chains_the_decode_error(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script({"stdout": "this is not json", "exit_code": 0})
    with pytest.raises(PermanentModelError) as info:
        fake.provider(monkeypatch).generate(req())
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "case",
    ["exit-1", "quota", "429", "503", "non-json", "bad-shape", "timeout", "signal"],
)
def test_errors_never_leak_prompt_argv_or_environment(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, case: str
) -> None:
    sys_marker = "SYSPROMPT-MARKER-7f3a"
    user_marker = "USERTEXT-MARKER-91bc"
    env_marker = "ENVVALUE-MARKER-5d2e"
    key_marker = "sk-ant-MARKER-0c4e"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key_marker)
    steps: dict[str, dict[str, Any]] = {
        "exit-1": _stderr_only("it broke"),
        "quota": _exit1(error_json("You've hit your session limit")),
        "429": _exit1(error_json("slow down", status=429)),
        "503": _exit1(error_json("down", status=503)),
        "non-json": {"stdout": "<html>", "exit_code": 0},
        "bad-shape": step_ok({"nope": True}),
        "timeout": {"sleep": 30, "stdout": ""},
        "signal": {"kill_signal": int(signal.SIGKILL)},
    }
    fake.script(steps[case])
    extra = {"AGENT_REPLAY_LEAK_PROBE": env_marker}
    if case == "timeout":
        extra["AGENT_REPLAY_TIMEOUT_S"] = "0.5"
    provider = fake.provider(monkeypatch, **extra)
    with pytest.raises(Exception) as info:
        provider.generate(
            req(
                [Message(role="user", text=user_marker)],
                system=f"{sys_marker} be nice",
            )
        )
    exc: BaseException | None = info.value
    assert isinstance(
        exc, (RateLimitError, TransientError, PermanentModelError, QuotaExhausted)
    )
    seen = 0
    while exc is not None and seen < 10:
        blob = str(exc) + "\n" + repr(exc)
        for marker in (sys_marker, user_marker, env_marker, key_marker):
            assert marker not in blob, (marker, type(exc).__name__)
        for flag in ("--output-format", "--no-session-persistence", "--json-schema"):
            assert flag not in blob
        exc = exc.__cause__
        seen += 1


def _req() -> ModelRequest:
    return ModelRequest(
        provider="claude_code",
        model="claude-haiku-5-5",
        system="s",
        messages=[Message(role="user", text="hi")],
    )


def test_a_cli_that_cannot_start_is_permanent_and_hides_the_path(
    tmp_path: Path,
) -> None:
    missing = str(tmp_path / "gone" / "SECRET-PATH-claude")
    with pytest.raises(PermanentModelError) as info:
        ClaudeCodeProvider(missing).generate(_req())
    assert str(info.value).startswith("claude_code: ")
    assert "SECRET-PATH" not in str(info.value)
    assert info.value.__cause__ is None


def test_empty_cli_path_and_bad_max_turns_are_rejected() -> None:
    with pytest.raises(ValueError):
        ClaudeCodeProvider("")
    with pytest.raises(ValueError):
        ClaudeCodeProvider("/bin/true", max_turns=0)


def test_a_boolean_status_is_not_a_status_code() -> None:
    assert classify_failure(True, "") is PermanentModelError


_OK_REPLY = (
    '{"type":"result","subtype":"success","is_error":false,'
    '"structured_output":{"text":"ok"},"usage":{"input_tokens":1,"output_tokens":1}}'
)


def test_a_relative_cli_path_is_made_absolute_and_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_replay.llm.factory  # noqa: F401  (registers the built-ins)
    from agent_replay.config import Settings
    from agent_replay.llm import registry

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(
        f"#!/bin/sh\ncat >/dev/null\nprintf '%s' '{_OK_REPLY}'\n", encoding="utf-8"
    )
    script.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_REPLAY_PROVIDER", "claude_code")
    monkeypatch.setenv("AGENT_REPLAY_CLAUDE_CLI", "bin/claude")
    provider = registry.get_provider_spec("claude_code").build(Settings(), "record")
    assert isinstance(provider, ClaudeCodeProvider)
    argv = provider._argv(_req())
    assert Path(argv[0]).is_absolute()
    assert Path(argv[0]) == script
    assert provider.generate(_req()).text == "ok"


@pytest.mark.parametrize("field", ["model", "system"])
def test_a_nul_byte_in_argv_is_permanent_and_not_echoed(
    tmp_path: Path, field: str
) -> None:
    request = _req().model_copy(update={field: "NUL-MARKER\x00x"})
    with pytest.raises(PermanentModelError) as info:
        ClaudeCodeProvider("/bin/true").generate(request)
    assert "NUL-MARKER" not in str(info.value) + repr(info.value)
    assert info.value.__cause__ is None


def test_a_reply_without_structured_output_is_an_invalid_shape(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude
) -> None:
    fake.script({"stdout_json": success_json(result="rate limit"), "exit_code": 0})
    with pytest.raises(PermanentModelError, match="invalid reply shape"):
        fake.provider(monkeypatch).generate(req())


@pytest.mark.parametrize("model", ["--dangerous", "a b", "x" * 101, ""])
def test_an_invalid_model_id_is_refused_before_the_cli_runs(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, model: str
) -> None:
    fake.script(step_text("hi"))
    request = req().model_copy(update={"model": model})
    with pytest.raises(PermanentModelError, match="invalid model id") as info:
        fake.provider(monkeypatch).generate(request)
    if model:
        assert model not in str(info.value)
    assert fake.calls() == []


def test_a_timeout_also_kills_what_the_cli_started(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    script = tmp_path / "claude"
    script.write_text(
        f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nsleep 60\n", encoding="utf-8"
    )
    script.chmod(0o755)
    provider = ClaudeCodeProvider(str(script), timeout_s=1)
    with pytest.raises(TransientError, match="killed after 1s"):
        provider.generate(_req())
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("the CLI's child process outlived the timeout")
