# ADR-010: Provider registry, and the Claude Code CLI (`claude_code`) as the live provider

- Date: 2026-11-12
- Status: accepted. Supersedes the provider choice and the same-model judge in ADR-003.

## Context
Live runs should not need a paid API key, and the model layer should take another provider
without touching the runner, cache, budget or judge. A Claude Max subscription is available
through the locally logged-in Claude Code CLI (`claude -p`, OAuth login, no API key), so the
committed baseline and the deliberate regression are recorded with it. Gemini stays supported.

Facts this rests on (Claude Code CLI 2.1.318;
[headless mode](https://code.claude.com/docs/en/headless),
[CLI reference](https://code.claude.com/docs/en/cli-reference),
[errors](https://code.claude.com/docs/en/errors)):
- `claude -p` reads the prompt from stdin. `--output-format json` returns one result object
  with `structured_output` (when `--json-schema` is given), `is_error`, `subtype`, `usage`,
  `api_error_status`, `result` and `errors[]`.
- `--bare` refuses the OAuth login. `--safe-mode` keeps it and drops CLAUDE.md, hooks, MCP
  servers, plugins and skills. `--tools ""` disables the built-in tools.
- There is no temperature, top_p or seed flag.
- Usage limits surface as text ("You've hit your session limit ..."), and the CLI retries
  transient failures itself.
- The [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance) reserves
  subscription OAuth for ordinary individual use; products for other people should use API keys.
  So the subscription login is used only for local runs by the account owner, never in CI.
- Model ids `claude-haiku-5-5` and `claude-sonnet-5-5`, with
  [API list prices](https://platform.claude.com/docs/en/about-claude/pricing) of $0.10 / $0.50 and
  $2.00 / $10.00 per 1M tokens.

Before this change `llm/factory.py` hard-coded `gemini` and `scripted`, and the judge reused the
agent model.

## Decision
1. Provider registry (`llm/registry.py`). Each provider registers a `ProviderSpec`: name;
   `build(settings, mode)`, which returns the live provider or None (replay without a credential
   or tool) and raises ValueError in record mode when it cannot record;
   `needs_credential_to_record`; default agent model; default judge model (None means the agent
   model); default prices plus optional per-model prices; `rate_limited`. The factory has no
   provider-name branches, and an unknown name gives a ValueError listing the registered names.
   The defaults are plain data, so `config.py` resolves them without importing any SDK.
2. Settings resolution keeps every effective Gemini value unchanged. `AGENT_REPLAY_MODEL` and
   `AGENT_REPLAY_JUDGE_MODEL` override the provider defaults; for Gemini the judge default is the
   agent model, as before. `scripted` mirrors Gemini's defaults so its fixtures keep their
   hashes. Every field of `ModelRequest` stays the same, and a test pins the request hashes.
3. Per-model prices in one shared budget. The per-process `BudgetTracker` prices each response
   with its request's model, so Sonnet judge calls are counted at Sonnet rates and Haiku agent
   calls at Haiku rates, both drawing on one budget.
4. The `claude_code` provider (`llm/claude_code.py`):
   - One subprocess per call, an argv list (never a shell), the prompt on stdin, and a fresh
     empty temporary working directory (no project CLAUDE.md, `.claude/` or `.mcp.json`). Flags:
     `-p --output-format json --json-schema <schema> --model <id> --system-prompt <ours>
     --tools "" --no-session-persistence --strict-mcp-config --safe-mode
     --disable-slash-commands --permission-prompts none`.
   - Not passed: `--bare` (refuses OAuth); `--setting-sources` (`--safe-mode` and the empty
     working directory already cover CLAUDE.md and project settings); `--fallback-model` (it
     would change the recorded model id). `--max-turns` is opt-in through
     `AGENT_REPLAY_CLAUDE_MAX_TURNS`, because the installed `--help` does not list it and a low
     value can stop the run before the structured output arrives.
   - Child environment: the parent's minus every variable starting with `AGENT_REPLAY_` (the
     Gemini key and the database URL never reach the CLI), `ANTHROPIC_` (API key, auth token,
     base URL and the rest) or `CLAUDE_CODE_USE_` (Bedrock, Vertex and other backends), so
     `claude_code` always means the logged-in subscription, whose login lives in the CLI's own
     config. The child runs in its own process group, and a timeout kills the whole group.
   - Tool calling: the CLI's tools are off, so our tools are described in the system prompt and
     the model answers through `--json-schema` with one object `{text?, tool_calls?: [{name,
     args}]}`. The schema has no top-level `oneOf`, and the reply is validated again in Python.
     History and tool declarations are rendered deterministically.
   - Results come from documented fields only. `structured_output` is required, and the `result`
     text is never parsed as a fallback. Usage is `input_tokens + cache_creation_input_tokens +
     cache_read_input_tokens` in and `output_tokens` out; a missing field counts 0.
   - Errors map to the neutral classes: usage limit to `QuotaExhausted` (not retried; the suite
     stops and can resume); 429 to `RateLimitError`; 5xx, overload, network, timeout or signal to
     `TransientError`; auth, unknown model, schema-retry exhaustion and bad output to
     `PermanentModelError`. Only a real CLI failure (non-zero exit or `is_error: true`) is
     classified from its text, so a reply that happens to mention a rate limit stays a reply.
   - "Found" means `shutil.which(AGENT_REPLAY_CLAUDE_CLI)` (default `claude`) resolves. With no
     CLI found, replay is cache-only, the same as Gemini without a key, and record fails with a
     clear message.
   - The rate limiter stays on with the shared `AGENT_REPLAY_RPM` / `AGENT_REPLAY_RPD`. Max
     limits are not published as RPM, so the limiter is a cheap runaway cap; calls are serial
     anyway.
   - Defaults: agent `claude-haiku-5-5`, judge `claude-sonnet-5-5`. `claude_code` is also the
     default provider, since the committed fixtures were recorded with it.
5. Baselines pin their provider. The request hash includes provider and model, so the fixture
   manifest records `provider`, `agent_model` and `judge_model` (taken from the exported cache
   entries). The regression-gate workflow reads them from the base commit's manifest and sets
   `AGENT_REPLAY_PROVIDER`, `AGENT_REPLAY_MODEL` and `AGENT_REPLAY_JUDGE_MODEL` for replay and
   score. It sets `AGENT_REPLAY_CLAUDE_CLI` to empty and never installs the CLI, so CI is always
   cache-only for a `claude_code` baseline. The compose `seed` service only imports, and import
   re-hashes from stored requests, so it needs no provider setting.

## Alternatives considered
- The Anthropic API with an API key: per-token billing. It can be added later as another
  registered provider.
- `--bare`: the cleanest isolation, but it refuses OAuth, so it cannot use the subscription.
- The Agent SDK instead of the CLI: another dependency, while the CLI is already installed and
  logged in.
- A top-level `oneOf` reply schema (text or tool calls): one object with optional fields plus
  Python validation is simpler and avoids depending on how the CLI handles `oneOf`.
- Provider and model as fixed env in the workflow: simpler, but it drifts from the baseline when
  someone re-records with another provider (every request misses, SKIPPED forever). Reading the
  manifest keeps them tied.
- One price pair per process: the judge would be priced at Haiku rates, about 20 times too low,
  and the judge dominates the notional cost.

## Consequences
- Cost: a Max subscription has no per-token bill. The run budget becomes a notional
  API-equivalent estimate at list prices, but it still stops runaway commands at
  `AGENT_REPLAY_RUN_BUDGET_USD`. The binding limits are the account's session and weekly usage
  limits, which surface as `quota_exhausted`.
- Determinism: temperature and seed stay in `ModelRequest` and in the hash but are not sent, so
  live re-recordings vary and replay determinism comes from the cache alone.
- Judge: Sonnet 5.5 judges a Haiku 5.5 agent. That reduces self-judging bias but does not remove
  it, since both are the same family. Gemini still judges with its agent model.
- Latency: a live `claude_code` step's `latency_ms` is the subprocess wall time, including CLI
  startup, internal CLI retries and structured-output re-prompts. It is not comparable with
  Gemini latencies, which is one more reason the baseline manifest pins the provider. Cached
  steps keep their recorded latency (ADR-004).
- Subprocess per call: startup time on every call, but no shared state, simple crash isolation
  and timeouts enforced by killing the process group. Our retries stack on the CLI's own,
  bounded by `AGENT_REPLAY_TIMEOUT_S` per attempt.
- CI: a `claude_code` baseline replays cache-only. Unchanged PRs are gated fully; a PR that
  changes a model request is SKIPPED unless it carries its candidate recordings (ADR-011).
- Prompt exposure: the system prompt is on argv, visible in the local process list. It holds
  only the synthetic-domain prompt and tool schemas.
- The flag combination (`--json-schema` with `--tools ""` and `--safe-mode` on a Max login) is
  documented flag by flag rather than together. The smoke check before the first suite confirmed
  it: `structured_output` was present, both model ids were accepted and usage tokens were
  non-zero (`evals/RUNBOOK.md`).
- Adding a provider is a module, a `ProviderSpec` and a conformance-kit subclass
  (`docs/providers.md`).
