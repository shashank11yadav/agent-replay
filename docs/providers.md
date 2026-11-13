# Model providers

Every model provider is a `ProviderSpec` in the registry (`src/agent_replay/llm/registry.py`).
`llm/factory.py` looks `AGENT_REPLAY_PROVIDER` up there and has no provider-name branches.

## Adding a provider (6 steps)

1. Write `src/agent_replay/llm/<name>.py` with a class whose `generate(ModelRequest) -> ModelResponse`
   maps backend failures to the neutral errors (`RateLimitError`, `TransientError`,
   `PermanentModelError`).
2. Register a `ProviderSpec` (`register(...)` in `registry.py`): `name`, `build(settings, mode)`,
   `needs_credential_to_record`, `default_model`, `default_judge_model`, `default_prices` and
   optional `model_prices`, `rate_limited`. Take model ids and prices from the vendor's own
   models and pricing pages. Import the SDK inside `build`,
   so `config.py` stays free of it.
3. Add any settings the provider needs to `src/agent_replay/config.py`.
4. Write a fake backend and subclass the conformance kit in `tests/providers/conformance.py`
   (declare which error classes the backend can produce).
5. Run `uv run pytest -q tests/providers`, `uv run ruff check .` and `uv run mypy src`.
6. Add the provider to the table below.

Nothing else (runner, cache, budget, judge, CLI) changes.

## Settings resolution

`AGENT_REPLAY_MODEL` sets the agent model, otherwise the provider's `default_model`.
`AGENT_REPLAY_JUDGE_MODEL` sets the judge model, otherwise the provider's `default_judge_model`,
otherwise the effective agent model. Prices come from `AGENT_REPLAY_PRICE_INPUT_PER_M` /
`AGENT_REPLAY_PRICE_OUTPUT_PER_M`, otherwise `model_prices[agent model]`, otherwise
`default_prices`. Judge prices come from `AGENT_REPLAY_JUDGE_PRICE_INPUT_PER_M` /
`AGENT_REPLAY_JUDGE_PRICE_OUTPUT_PER_M`, otherwise the table entry for the judge model. When the
judge model equals the agent model, the budget tracker prices both under one key and the judge
price variables are ignored: set `AGENT_REPLAY_PRICE_*` instead.

`AGENT_REPLAY_PROVIDER` defaults to `claude_code` (`DEFAULT_PROVIDER` in `registry.py`), the
provider the committed fixtures were recorded with, so replaying them needs no provider setting.
An unregistered provider name takes the default provider's models and prices in `Settings`;
`build_live_provider` then raises a ValueError listing the registered names.

## Registered providers

| name | agent model | judge model | prices in / out per 1M USD | rate limited | needs credential to record |
|---|---|---|---|---|---|
| `gemini` | `gemini-3.5-flash-lite` | same as agent | 0.30 / 2.50 | yes | yes (`AGENT_REPLAY_GEMINI_API_KEY`) |
| `scripted` | `gemini-3.5-flash-lite` (keeps scripted fixture hashes) | same as agent | 0.30 / 2.50 | no | no (`AGENT_REPLAY_SCRIPT_PATH`) |
| `claude_code` | `claude-haiku-5-5` | `claude-sonnet-5-5` | Haiku 0.10 / 0.50, Sonnet 2.00 / 10.00 | yes | no (local Claude Code CLI login) |

## `claude_code`: the Claude Code CLI

`src/agent_replay/llm/claude_code.py` runs the locally installed and logged-in Claude Code CLI
headless, one `claude -p` subprocess per model call (ADR-010). It uses the subscription (OAuth)
login of the person running it. That login is for local runs on your own machine only.
Never use it in CI or for other people's runs. Replay never needs the CLI: CI replays a
`claude_code` baseline cache-only, without the CLI installed (`.github/workflows/regression-gate.yml`
sets `AGENT_REPLAY_CLAUDE_CLI` empty).

### How the CLI is found

`AGENT_REPLAY_CLAUDE_CLI` (default `claude`) is resolved with `shutil.which`: a bare name is
searched on `PATH`, a path must be an executable file (not a directory). An empty value is never
found. With no CLI found, replay is cache-only (the live provider is None and no process starts)
and record fails with a ValueError naming `AGENT_REPLAY_CLAUDE_CLI`. The tests set it to an empty
string in `tests/conftest.py` and point it at a fake script, so they never run the real CLI.

### Flags

Every call runs an argv list (never a shell):

| flag | why |
|---|---|
| `-p` | print (headless) mode: one answer, then exit |
| `--output-format json` | one JSON result object on stdout (`structured_output`, `is_error`, `subtype`, `usage`, `api_error_status`, `result`, `errors`) |
| `--json-schema <REPLY_SCHEMA>` | the model answers through the reply schema, returned in `structured_output` |
| `--model <request.model>` | the exact model id from the request, so the recorded model is the one asked for |
| `--system-prompt <ours>` | replaces Claude Code's own system prompt with ours plus the tool list and reply instructions |
| `--tools ""` | disables every built-in CLI tool (no file, shell or web access) |
| `--no-session-persistence` | nothing is saved to disk, no session can be resumed |
| `--strict-mcp-config` | with no `--mcp-config`, no MCP servers at all |
| `--safe-mode` | no CLAUDE.md, hooks, plugins, skills or MCP servers, but keeps the OAuth login |
| `--disable-slash-commands` | no skills or custom commands |
| `--permission-prompts none` | anything that would ask a human is denied |
| `--max-turns <n>` | only when `AGENT_REPLAY_CLAUDE_MAX_TURNS` is set (>= 1); unset by default |

Never passed: `--bare` (it refuses the OAuth login), `--setting-sources` (`--safe-mode` and the
empty working directory already cover project settings and CLAUDE.md) and `--fallback-model` (it
would change the recorded model id).
`--max-turns` is opt-in because it is missing from the installed 2.1.318 `--help` and a low
value may stop the run before the structured output is produced.

### Tool calling through the reply schema

The CLI's tools are off, so our tools are listed in the system prompt (`## Tools`, one entry per
tool with its canonical JSON parameters, or `None.`). The model replies with one object
`{"text"?: string, "tool_calls"?: [{"name": string, "args": object}]}` (no top-level `oneOf`).
The reply is validated again in Python; a bad shape is a `PermanentModelError`. Tool-call ids are
`make_tool_call_id(step, index)`. The conversation is rendered deterministically to stdin
(`[user]`, `[assistant]` with `tool_calls: <json>`, `[tool]` with `tool_result: <json>` blocks);
thought signatures are never rendered. The `result` text is never parsed as a fallback.

Tokens: input = `usage.input_tokens + usage.cache_creation_input_tokens +
usage.cache_read_input_tokens`, output = `usage.output_tokens` (missing fields count 0;
`modelUsage` and `total_cost_usd` are not used). `latency_ms` is the subprocess wall time,
including CLI startup. temperature and seed have no CLI flag: they stay in the request hash but
are not sent.

### Process isolation

- The prompt goes to stdin as UTF-8, never into argv.
- The working directory is a fresh empty temporary directory, removed after every call (also on
  errors and timeouts), so no project CLAUDE.md, `.claude/` or `.mcp.json` is read.
- The child environment is the parent's with every variable starting with `AGENT_REPLAY_`,
  `ANTHROPIC_` or `CLAUDE_CODE_USE_` stripped. The Gemini key and the database URL never reach
  the CLI, and nothing can move the call off the subscription login (an API key, auth token,
  base URL, or a Bedrock or Vertex switch). The login itself lives in the CLI's own config.
- The child starts in its own process group. A call that runs past `AGENT_REPLAY_TIMEOUT_S` gets
  the whole group killed (anything the CLI started goes too) and raises `TransientError`.
- The model id must match `(?!-)[A-Za-z0-9._:/-]{1,100}`, so it can never be read as a flag;
  anything else is a `PermanentModelError` before a process starts.

### Errors

A signal (negative exit code or 143) is a `TransientError`. A run that exits 0 with
`is_error: false` and subtype `success` is a reply; without `structured_output` it is a
`PermanentModelError` (bad reply shape). Only a real CLI failure, a non-zero exit or
`is_error: true`, is classified, from `api_error_status` and the failure text (`errors[]`,
stderr, and `result` only when `is_error` is true, since otherwise `result` may be the model's
own prose), first match wins:
"You've hit your" → `QuotaExhausted`; status 429, "Request rejected (429)" or "rate limit" →
`RateLimitError`; status 5xx, "Overloaded", "API Error: 5" or "Unable to connect to API" →
`TransientError`; anything else (not logged in, unknown model, `error_max_turns`,
`error_max_structured_output_retries`, non-JSON output, missing `structured_output`, bad reply
shape) → `PermanentModelError`. Messages are `claude_code: <category>: ` plus at most 500
characters of the CLI's failure text, never the prompt, argv or environment values.

### Confirmed live

The smoke check on 2026-11-12 with CLI 2.1.318 (`evals/RUNBOOK.md`) confirmed:

- The flag combination `--json-schema` with `--tools ""` and `--safe-mode` returns
  `structured_output`, and tool calls came through the reply schema end to end.
- A Max login accepted `claude-haiku-5-5` and `claude-sonnet-5-5`.
- The `usage` field names gave non-zero token counts. A missing field still counts 0.

Not exercised live, and kept out of the default path for that reason: `--max-turns` (opt-in) and
`--setting-sources` (never passed). `ANTHROPIC_API_KEY` and the other variables matched by the
prefixes under process isolation are removed from the child environment, so they cannot switch
the CLI to per-token billing or another endpoint.

Flag behaviour is documented in the
[CLI reference](https://code.claude.com/docs/en/cli-reference) and
[headless mode](https://code.claude.com/docs/en/headless) pages, error texts in
[errors](https://code.claude.com/docs/en/errors), and the subscription terms in
[legal and compliance](https://code.claude.com/docs/en/legal-and-compliance).
