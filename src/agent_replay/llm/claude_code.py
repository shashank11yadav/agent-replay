"""Claude Code CLI provider: one `claude -p` subprocess per model call (ADR-010).

The CLI's own tools are off. Our tools are described in the system prompt and the model
answers through ``--json-schema`` with one object ``{text?, tool_calls?: [{name, args}]}``,
which is validated again here. The conversation goes to stdin, never argv. The child runs in a
fresh empty temporary directory, in its own process group (a timeout kills the whole group),
with every ``AGENT_REPLAY_*``, ``ANTHROPIC_*`` and ``CLAUDE_CODE_USE_*`` variable removed from
its environment, so nothing can move the call off the subscription login (an API key, auth
token, base URL or a Bedrock/Vertex switch).

Error messages carry at most 500 characters of the CLI's failure text and never the prompt,
the system prompt, argv or environment values.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Any

from agent_replay.llm.hashing import make_tool_call_id, next_step
from agent_replay.llm.types import (
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExhausted,
    RateLimitError,
    ToolCall,
    ToolSchema,
    TransientError,
)

REPLY_SCHEMA: dict[str, Any] = {
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

REPLY_INSTRUCTIONS = (
    "## Reply format\n"
    "The conversation so far is in the user message: [user] blocks come from the user, "
    "[assistant] blocks are your earlier replies (with the tool calls you made) and [tool] "
    "blocks hold the results of those tool calls.\n"
    'Answer only through the structured output. To call tools, set "tool_calls" to a list '
    'of {"name": <a tool listed under Tools>, "args": <an object matching its parameters>}; '
    'you may add a short "text". To give your final answer, set "text" and leave out '
    '"tool_calls". Never call a tool that is not listed under Tools.'
)

_PREFIX = "claude_code"
_MAX_FAILURE_CHARS = 500
_SIGTERM_EXIT = 143
# Stripped by prefix, so a variable added in a later CLI release (another ANTHROPIC_* endpoint
# or CLAUDE_CODE_USE_* backend) cannot move the call off the subscription login either. The
# login itself lives in the CLI's own config, not in these variables.
_STRIPPED_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "AGENT_REPLAY_")
# No leading "-": the id is its own argv item after --model and must not read as a flag.
_MODEL_ID = re.compile(r"(?!-)[A-Za-z0-9._:/-]{1,100}")
_REAP_TIMEOUT_S = 5
_QUOTA_MARKERS = ("You've hit your", "You’ve hit your")
_TRANSIENT_MARKERS = ("Overloaded", "API Error: 5", "Unable to connect to API")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_SCHEMA_ARG = json.dumps(REPLY_SCHEMA, separators=(",", ":"))


def render_system(system: str, tools: Sequence[ToolSchema]) -> str:
    """Our system prompt plus the tool declarations and the reply instructions."""
    if tools:
        block = "\n".join(
            f"- {t.name}: {t.description}\n  parameters: {_canonical(t.parameters)}"
            for t in tools
        )
    else:
        block = "None."
    return system + "\n\n## Tools\n" + block + "\n\n" + REPLY_INSTRUCTIONS


def _render_message(message: Message) -> str:
    if message.role == "user":
        return "[user]\n" + message.text
    if message.role == "model":
        out = "[assistant]"
        if message.text:
            out += "\n" + message.text
        if message.tool_calls:
            calls = [
                {"id": c.id, "name": c.name, "args": c.args} for c in message.tool_calls
            ]
            out += "\ntool_calls: " + _canonical(calls)
        return out
    out = "[tool]"
    for r in message.tool_results:
        result = {"call_id": r.call_id, "name": r.name, "response": r.response}
        out += "\ntool_result: " + _canonical(result)
    return out


def render_conversation(messages: Sequence[Message]) -> str:
    """The stdin prompt: one block per message. ToolCall.signature is never rendered."""
    return "\n\n".join(_render_message(m) for m in messages)


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(_STRIPPED_PREFIXES)}


def _count(usage: Any, field: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _message(category: str, text: str) -> str:
    return f"{_PREFIX}: {category}: {text[:_MAX_FAILURE_CHARS]}"


def _failure_text(
    obj: dict[str, Any] | None, stderr: str, *, with_result: bool = True
) -> str:
    """The CLI's failure text: ``result`` (when asked), ``errors[]`` and stderr."""
    parts: list[str] = []
    if obj is not None:
        result = obj.get("result")
        if with_result and isinstance(result, str) and result.strip():
            parts.append(result.strip())
        errors = obj.get("errors")
        if isinstance(errors, list):
            parts.extend(e.strip() for e in errors if isinstance(e, str) and e.strip())
    if stderr.strip():
        parts.append(stderr.strip())
    return "\n".join(parts)


def classify_failure(status: Any, text: str) -> type[Exception]:
    """Error class for a failed run; first match wins (quota, rate limit, transient).

    Callers pass only CLI failure text, never a reply the model wrote: see ``generate``.
    """
    code = status if isinstance(status, int) and not isinstance(status, bool) else None
    if any(marker in text for marker in _QUOTA_MARKERS):
        return QuotaExhausted
    if code == 429 or "Request rejected (429)" in text or "rate limit" in text.lower():
        return RateLimitError
    if (code is not None and 500 <= code <= 599) or any(
        marker in text for marker in _TRANSIENT_MARKERS
    ):
        return TransientError
    return PermanentModelError


_CATEGORY: dict[type[Exception], str] = {
    QuotaExhausted: "usage limit",
    RateLimitError: "rate limited",
    TransientError: "transient error",
    PermanentModelError: "permanent error",
}


def _invalid_shape(detail: str) -> PermanentModelError:
    return PermanentModelError(_message("invalid reply shape", detail))


def _parse_reply(structured: Any, step: int) -> tuple[str | None, list[ToolCall]]:
    """Validate the structured output again; the reply values never go into messages."""
    if not isinstance(structured, dict):
        raise _invalid_shape("structured_output is not an object")
    if set(structured) - {"text", "tool_calls"}:
        raise _invalid_shape("unknown keys in structured_output")
    text = structured.get("text")
    if "text" in structured and not isinstance(text, str):
        raise _invalid_shape("text is not a string")
    raw_calls = structured.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise _invalid_shape("tool_calls is not a list")
    calls: list[ToolCall] = []
    for i, item in enumerate(raw_calls):
        if not isinstance(item, dict) or set(item) - {"name", "args"}:
            raise _invalid_shape(f"tool call {i} is not a {{name, args}} object")
        name, args = item.get("name"), item.get("args")
        if not isinstance(name, str):
            raise _invalid_shape(f"tool call {i} has no string name")
        if not isinstance(args, dict):
            raise _invalid_shape(f"tool call {i} args is not an object")
        calls.append(ToolCall(id=make_tool_call_id(step, i), name=name, args=args))
    if not text and not calls:
        raise _invalid_shape("neither a non-empty text nor a tool call")
    return text, calls


class ClaudeCodeProvider:
    """Runs the Claude Code CLI headless, once per ``generate`` call."""

    def __init__(
        self, cli_path: str, *, timeout_s: float = 30, max_turns: int | None = None
    ) -> None:
        if not cli_path:
            raise ValueError("cli_path is empty; set AGENT_REPLAY_CLAUDE_CLI")
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._cli = cli_path
        self._timeout_s = timeout_s
        self._max_turns = max_turns

    def __repr__(self) -> str:
        return "ClaudeCodeProvider()"

    __str__ = __repr__

    def _argv(self, request: ModelRequest) -> list[str]:
        if not _MODEL_ID.fullmatch(request.model):
            # The value is not echoed: it comes from the environment.
            raise PermanentModelError(_message("permanent error", "invalid model id"))
        argv = [
            self._cli,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            _SCHEMA_ARG,
            "--model",
            request.model,
            "--system-prompt",
            render_system(request.system, request.tools),
            "--tools",
            "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--safe-mode",
            "--disable-slash-commands",
            "--permission-prompts",
            "none",
        ]
        if self._max_turns is not None:
            argv += ["--max-turns", str(self._max_turns)]
        return argv

    def _run(self, argv: list[str], stdin: bytes) -> tuple[Any, int]:
        """Run the CLI in a fresh empty temp dir; returns (CompletedProcess, latency_ms)."""
        completed: Any = None
        start_error: str | None = None
        with tempfile.TemporaryDirectory(prefix="agent-replay-claude-") as cwd:
            started = time.monotonic()
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    env=_child_env(),
                    # Its own process group, so a timeout can stop what it started too.
                    start_new_session=True,
                )
            except OSError as exc:
                # The OSError holds the CLI path (an environment value): not chained.
                start_error = type(exc).__name__
            except ValueError:
                # e.g. a NUL byte in the model id or system prompt; the value is not echoed.
                start_error = "invalid argument"
            else:
                completed = self._communicate(proc, argv, stdin)
            latency_ms = max(1, round((time.monotonic() - started) * 1000))
        if start_error is not None:
            raise PermanentModelError(
                _message("permanent error", f"could not start the CLI ({start_error})")
            ) from None
        if completed is None:
            raise TransientError(
                _message("timeout", f"the CLI was killed after {self._timeout_s:g}s")
            ) from None
        return completed, latency_ms

    def _communicate(
        self, proc: subprocess.Popen[bytes], argv: list[str], stdin: bytes
    ) -> subprocess.CompletedProcess[bytes] | None:
        """Feed stdin and wait; None when the call timed out and its group was killed."""
        try:
            stdout, stderr = proc.communicate(input=stdin, timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return None
        except BaseException:
            _kill_group(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

    def generate(self, request: ModelRequest) -> ModelResponse:
        stdin = render_conversation(request.messages).encode("utf-8")
        completed, latency_ms = self._run(self._argv(request), stdin)

        code = completed.returncode
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if code < 0 or code == _SIGTERM_EXIT:
            raise TransientError(
                _message(
                    "transient error", f"the CLI was ended by a signal (exit {code})"
                )
            )

        stdout = completed.stdout.decode("utf-8", errors="replace")
        obj: dict[str, Any] | None = None
        decode_error: json.JSONDecodeError | None = None
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            decode_error = exc
        else:
            if isinstance(parsed, dict):
                obj = parsed

        if (
            code == 0
            and obj is not None
            and obj.get("is_error") is False
            and obj.get("subtype") == "success"
        ):
            if "structured_output" not in obj:
                # The model answered in plain text; its words are not a CLI error.
                raise _invalid_shape("structured_output is missing")
            text, calls = _parse_reply(
                obj["structured_output"], next_step(request.messages)
            )
            usage = obj.get("usage")
            return ModelResponse(
                text=text,
                tool_calls=calls,
                input_tokens=_count(usage, "input_tokens")
                + _count(usage, "cache_creation_input_tokens")
                + _count(usage, "cache_read_input_tokens"),
                output_tokens=_count(usage, "output_tokens"),
                latency_ms=latency_ms,
            )

        # Classify only a real CLI failure, and from ``result`` only when the CLI flagged
        # it as an error: otherwise ``result`` may be model prose that happens to say
        # "rate limit".
        is_error = obj is not None and obj.get("is_error") is True
        error_cls: type[Exception] = PermanentModelError
        if code != 0 or is_error:
            error_cls = classify_failure(
                obj.get("api_error_status") if obj is not None else None,
                _failure_text(obj, stderr, with_result=is_error),
            )
        text = _failure_text(obj, stderr)
        if not text:
            text = _describe_failure(code, obj, decode_error is not None)
        error = error_cls(_message(_CATEGORY[error_cls], text))
        if decode_error is not None:
            raise error from decode_error
        raise error


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's process group, then reap the child (bounded)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.communicate(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pass  # a process that left the group still holds the pipes; give up on it


def _describe_failure(code: int, obj: dict[str, Any] | None, not_json: bool) -> str:
    """A fixed description when the CLI gave no failure text (never echoes its output)."""
    if not_json:
        return f"stdout is not JSON (exit {code})"
    if obj is None:
        return f"stdout is not a JSON object (exit {code})"
    subtype = obj.get("subtype")
    shown = subtype if isinstance(subtype, str) else "?"
    if code != 0:
        return f"exit {code} (subtype {shown[:80]})"
    return f"the CLI reported an error (subtype {shown[:80]})"
