# Design

agent-replay records every model call and tool call of an agent run as OpenTelemetry-style spans
in Postgres, replays the run with tool results stubbed from the recording and unchanged model
requests served from a hash-keyed cache, scores each run (deterministic checks plus an LLM judge)
and fails a pull request when pass rate, tokens or p95 latency regress past a threshold. A small
React viewer steps through a run and diffs two runs.

It runs locally or on a small free-tier host, with a read-only API. It ships one demo agent and a
small adapter interface for plugging in others. The reasons behind each choice are in the ADRs in
`docs/decisions/`.

## Glossary

- Run: one execution of one task by the agent. Mode `record` (live model, real tools) or
  `replay` (model through the cache first, tools stubbed from a source run).
- Step: one model call (span `generate_content <model>`) or one tool call
  (span `execute_tool <tool>`), children of the run's root span `invoke_agent demo-agent`,
  ordered by an integer `seq` from 0.
- Suite run: one run of every task in a task file with one prompt variant.
- Variant: a system prompt file in `src/agent_replay/demo/prompts/<variant>.txt`. The shipped
  variant is named in `evals/suite.json` (currently `structured`).
- Request hash: the cache key of a model request (below).
- Fixtures: a suite run exported as JSONL so it can be committed and imported elsewhere.

## Data flow

Record. The runner opens the root span and calls `agent.run(task, RecordingContext)`.
`call_model` builds a provider-neutral request; the client sends it through the cache (write
only) to the live provider, stores the response under its request hash and emits a model span.
`call_tool` runs the real tool (synthetic and deterministic) and emits a tool span. When the
agent returns, the run row, its totals and all spans are written in one transaction, so a failed
run never leaves orphan spans.

Replay. Same agent, `ReplayContext`. Tool calls are answered by stubs built from the source
run's tool spans. Model calls go through
`LLMClient -> [FaultInjectingProvider] -> CacheFirstProvider -> BudgetGuardProvider -> live provider`.
`CacheFirstProvider` answers first from the responses the source run itself received (its model
spans, by request hash), then from the shared cache. The cache is keyed by request alone, so a
later recording that sent the same request may have replaced the entry; the source run's spans
still hold what it got. A hit returns the response with its recorded latency. A miss calls the
live provider if one is available (a Gemini key, or a Claude Code CLI on `PATH`); otherwise it
raises `CacheMissNoLiveModel` and the run is `skipped`. No network call or subprocess happens
then.

CI gate. The regression-gate workflow imports `evals/baseline/` into a Postgres service
container, reads `provider`, `agent_model` and `judge_model` from the baseline manifest,
optionally imports carried candidate recordings from the PR, runs `suite replay` with the PR's
variant and `suite score`, then `gate` (report to the step summary) and `gate-check` for the
exit code. A separate job posts the report as a PR comment. CI never has the Claude Code CLI, so
a `claude_code` baseline is gated cache-only.

## Request hash

SHA-256 over canonical JSON (sorted keys, compact separators, UTF-8) of
`{"schema_version": 1, "request": <request>}`, where the request holds provider, model, system
prompt, messages, tool schemas, temperature and seed. Nothing else enters the hash: no
timestamps, no run ids. Tool-call ids inside messages are deterministic (`call_<step>_<index>`)
and opaque provider signatures are stripped before hashing, so an identical conversation always
hashes identically. Changing the prompt, the model, a tool schema or the provider changes the
hash, which is the signal the gate relies on (ADR-004).

## Replay and tool stubs

- Tool results always come from the source run's tool spans, matched by tool name plus
  canonical JSON arguments; repeated identical calls are served in recorded order. Tools are
  never executed in replay.
- A tool call with no recorded match returns `{"error": "unrecorded_tool_call"}` to the agent,
  the span gets `error.type=unrecorded_tool_call` and the run is `diverged` unless it later errors.
- With the same variant, code, provider and models and no live provider, every model step is a
  hit and the replay reproduces the source run's tokens, latency, tool calls and final output.
- Latency is virtual: a hit reports the latency recorded with the cache entry, a live call its
  measured latency, and client retry backoff is added to the step. Run latency is the sum of its
  steps. That is why the `rate_limit:2` replay shows a higher p95 with zero live calls.

Run status is `ok`, `diverged`, `skipped` or `error`. Errors carry an `error_type`: the
exception class for a failed tool or agent, `max_steps_exceeded`, `injected_rate_limit` (a
`rate_limit` fault outlasted the retries), or a stop code. A stop code means the provider or the
run budget failed, not the agent: `rate_limit` (429 after all attempts), `transient_error` (5xx
or timeout after all attempts), `quota_exhausted` or `budget_exceeded`
(`agent_replay/codes.py`). A failed run is data: `record` and `replay` store it, print its id
and exit 0. They exit 1 when the run was `skipped` or ended on a stop code (on `quota_exhausted`
or `budget_exceeded` the run is stored and the command reports the stop instead of the id).

## Faults (replay only)

| spec | effect |
|---|---|
| `tool_timeout:<tool\|*>` | the stub returns `{"error": "tool_timeout"}`, `error.type=timeout`, latency equal to the tool timeout |
| `malformed_tool_output:<tool\|*>` | the stub returns a payload that fails the tool's output model; the demo agent passes `{"error": "malformed_tool_output"}` to the model instead of crashing |
| `rate_limit:<n>` | the first `n` model attempts raise a 429 below the client, so the client retries with backoff; more than the retry limit ends the run as `injected_rate_limit`, an ordinary error rather than a stop code, so a fault suite still replays every task |

Injected 429s never reach the cache, the budget guard or the live rate limiter. Faults are stored
on the run row. `record` has no `--fault` flag, and an unknown spec exits non-zero listing the
valid ones.

## Scoring

`score --run <id> --tasks <file>` stores one `scores` row per check:

- `expected_text`: every listed string appears in the final answer, case-insensitive;
- `must_call_tools`: every listed tool was called;
- `max_steps`: model calls within the task's limit;
- `completed`: the run status is `ok`;
- `judge`: given the task, a one-line rubric and the final answer, the judge model returns
  `{"pass": bool, "reason": str}`, parsed strictly with pydantic. The answer is embedded as a JSON
  string literal so it cannot close its own delimiter.

A task passes only when every deterministic check passes and the judge says pass. Judge calls go
through the same client, cache and run budget as agent calls, so an unchanged answer costs
nothing on replay. Judge rows without a verdict have `passed = null` and a `detail.verdict`:
`judge_error` (unparseable reply or a failed call; counts as a failed task), `judge_skipped` (cache
miss, no live provider) or `budget_exceeded` / `quota_exhausted`. An empty final answer fails the
judge without a call.

The judge model is `AGENT_REPLAY_JUDGE_MODEL`, else the provider's default judge model, else the
agent model. For `claude_code` that is `claude-sonnet-5-5` judging a `claude-haiku-5-5` agent;
for `gemini` the judge is the agent model. Requests carry temperature 0 and seed 7, which
`claude_code` cannot send.

## Suites and metrics

`suite record` records every task of a task file into a new suite run; `--resume <id>` skips
tasks already recorded (in a new process with a fresh budget). `suite replay --source <id>`
replays every run of a source suite into a new suite run, with optional `--variant` and
`--fault`. Neither scores: `suite score --suite-run <id>` does that. A stop code stores the run
that hit it, stops the suite, prints how many tasks remain (and the `--resume` command for
`suite record`) and exits 1.

Without `--tasks`, `suite record --resume`, `suite replay` and `suite score` read the task file
path stored on the suite run. That value may come from imported fixtures, so it is used only
when it is relative and stays under the current directory; otherwise the command asks for
`--tasks`.

Per suite: task count, pass rate, deterministic and judge pass rates, mean input, output and
total tokens per task, p95 run latency, mean model calls, and counts of `error`, `diverged` and
`skipped` runs. p95 is `statistics.quantiles(latencies, n=100, method="inclusive")[94]`, the
single value for one run and null for none. Token means and p95 include error and skipped runs.

## Gate

`gate --baseline <id> --candidate <id> --thresholds evals/gate.json --out-dir <dir>` writes
`report.md` and `verdict.json`, appends the report to `$GITHUB_STEP_SUMMARY` when set, and exits
0 whenever it produced a report (2 if it could not). `gate-check <dir>/verdict.json` exits 1 for
FAIL or ERROR and 0 for PASS or SKIPPED.

Thresholds (`evals/gate.json`): pass rate may drop at most 5 percentage points, mean total
tokens may rise at most 15 %, p95 latency may rise at most 20 %. A delta exactly at a threshold
passes. A rise from a zero baseline fails; a missing p95 is N/A and cannot fail on its own.

Verdict precedence is ERROR > SKIPPED > FAIL > PASS:

- ERROR when the comparison would be meaningless, and the report names the cause:
  - a candidate run stopped on a stop code (`rate_limit`, `transient_error`, `quota_exhausted`,
    `budget_exceeded`);
  - a candidate judge row has `quota_exhausted` or `budget_exceeded`, or `judge_error` from a
    failed provider call (an unparseable reply is an ordinary fail);
  - the candidate's tasks differ from the baseline's: a baseline task with no candidate run, a
    task the baseline does not have, or a task with more than one run;
  - a candidate run lacks deterministic or judge scores (so a `--no-judge` scoring is ERROR);
  - the baseline is empty, not fully scored, has more than one run for a task, has a skipped
    run or has a run stopped on a stop code. The report blames the baseline, and a baseline
    task whose run was stopped (and so never replayed) does not count as a missing candidate run.
- SKIPPED when a candidate run is `skipped` or a judge row is `judge_skipped`: a changed request
  needed a live model that was not available. The report says how many requests changed.
- FAIL when any metric is past its threshold, otherwise PASS.

The report has one row per metric (baseline, candidate, delta, threshold, status) and lists the
tasks that flipped pass to fail and fail to pass.

The workflow runs on `pull_request` only. The `gate` job has `contents: read` and no secrets.
It takes the baseline fixtures, the task file and `evals/gate.json` from the PR base commit and
ignores the PR's own copies of them. Steps after `suite replay` run
with `if: ${{ !cancelled() }}` so an interrupted replay still publishes an ERROR report. Each of
the replay and score steps runs with `AGENT_REPLAY_RUN_BUDGET_USD=0.75`. The job uploads the
report with any notices (carried recordings, thresholds taken from the PR) above it. A separate
`comment` job, which runs no code from the PR, is the only one with `pull-requests: write`: it
skips fork PRs and posts the report with `gh pr comment ... || echo`, so a failed comment never
fails the run. Third-party actions in both workflows are pinned to commit SHAs, and Dependabot
proposes updates for actions, npm, uv and Docker.

## Carried candidate recordings

A PR that changes model requests would miss the cache in CI and get SKIPPED. It can instead
carry a fixtures export of its candidate in `evals/candidate/`: a replay of the baseline suite
with the PR's variant, recorded and scored locally with the baseline's provider and models
(ADR-011). The job imports it after the baseline, from the PR head only, and refuses before
importing when:

- `evals/candidate`, its manifest or anything inside it is a symlink;
- `provider`, `agent_model` or `judge_model` differs from the baseline manifest, or a value fails
  `^[A-Za-z0-9._-]{1,100}$`;
- `suite_run.source_suite_run_id` is not the baseline's `suite_run_id` (stale recordings).

At import, any row that clashes with a baseline row fails the job, so carried recordings can add
cache entries but never replace one. The replay is then cache-only, the verdict counts, and the
step summary says that CI checked the recorded evidence without re-running the model. CI checks
consistency, not authenticity: a maintainer reviews `evals/candidate/` like code or reruns the
gate locally with a live provider, and keeps CODEOWNERS and branch protection on `.github/`,
`evals/` and `src/`.

## Fixtures

`fixtures export --suite-run <id> --out <dir>` writes, one JSON object per line with sorted keys:

- `manifest.json`: suite run row, variant, task file hash, run count, and `provider`,
  `agent_model` and `judge_model` taken from the exported cache entries;
- `runs.jsonl`, `spans.jsonl`, `scores.jsonl`;
- `cache.jsonl`: every cache entry the suite's model steps and judge rows refer to (the judge
  stores its request hash in its score row).

Re-exporting an unchanged suite from the same database gives identical bytes. Span ids are
reassigned on import, so a re-export from another database differs only in span ids. Fixtures
never contain an API key or any environment value.

`fixtures import <dir>` treats the files as untrusted: every row is validated, unknown fields
are rejected, and the manifest's `run_count` and `task_count` must match the files. Import runs
in one transaction. An existing row with the same key (the id; for spans, the run id and `seq`)
must be identical (a no-op); a differing row or a new row under an existing suite or run raises
`ImportConflict` and nothing is imported. Span ids are assigned by the target database. Each
cache row's request is re-hashed and must match its key, so an entry can only answer the request
it holds. Source ids that point at rows missing from the target database become null.

`fixtures verify <dir>` exits 0 only if every model step and judged run has its cache entry,
each cache entry agrees with the span that used it (text and tool calls), every run has
deterministic and judge scores with a judge verdict, no run is skipped or stopped, and every
tool span has a result; otherwise it lists the gaps and exits 1.

## Model client

All model calls go through `llm/client.py` over a small `ModelProvider` protocol. Providers come
from the registry in `llm/registry.py`: `claude_code` (the default, and the provider the
committed fixtures were recorded with), `gemini` and the offline `scripted` fake. See
`docs/providers.md` and ADR-010.

- The client retries 429 and 5xx with exponential backoff (base 2 s, 3 attempts by default) and
  raises other errors, `QuotaExhausted` and `BudgetExceeded` at once. It logs one line per call:
  model, request hash, cache status, tokens, latency, attempts and estimated cost.
- The rate limiter takes requests per minute and per day from config (`AGENT_REPLAY_RPM`,
  `AGENT_REPLAY_RPD`); the per-day budget counts this process only and raises `QuotaExhausted`.
- The run budget guard estimates each live call's cost from its usage and its model's prices.
  Before a live call it raises `BudgetExceeded` if spend has reached the budget or spend plus the
  largest call seen so far would pass it. The budget is per process
  (`AGENT_REPLAY_RUN_BUDGET_USD`, default 1.50) and shared by agent and judge calls. Cache hits
  and injected 429s cost nothing. With `claude_code` on a subscription the figure is notional.

## HTTP API

Read-only FastAPI app, all routes under `/api`:

| route | returns |
|---|---|
| `GET /api/health` | `{"status": "ok"}` when the database answers |
| `GET /api/suite-runs?limit=` | suite runs, newest first |
| `GET /api/suite-runs/{id}/summary` | suite metrics |
| `GET /api/runs?suite_run_id=&limit=&offset=` | runs (`limit` 1 to 200, default 50), each with `passed`: true, false, or null when not fully scored |
| `GET /api/runs/{id}` | the run, its spans in `seq` order and its scores |
| `GET /api/diff?left=&right=` | the run diff |

A malformed id gives 422, an unknown id 404, and any write method 405. CORS allows only the
origins in `AGENT_REPLAY_CORS_ORIGINS` (never `*`), GET only, no credentials. A request whose
`Host` is not in `AGENT_REPLAY_ALLOWED_HOSTS` (default `localhost,127.0.0.1`) gets 400 before it
reaches a route, which guards against DNS rebinding. Every response carries
`Content-Security-Policy: default-src 'self'; connect-src 'self' <CORS origins>;
frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.
Database connections set a 5 s `statement_timeout`. When `AGENT_REPLAY_VIEWER_DIST` points at a
built viewer, it is mounted at `/` after the API router (ADR-009).

The run diff aligns steps by `seq`, reports per step whether the content changed and the token
and latency deltas, lists unmatched trailing steps as added or removed, and gives the first
divergent step and run-level deltas. Cache status is not content, so record versus replay of the
same run shows no change.

## Viewer

React and TypeScript, built with Vite. A run list per suite with each run's pass verdict (the
API's `passed`), a run page that steps through spans with Previous and Next or the arrow keys,
and a diff page. API responses are validated with zod; an invalid response shows an error. A
network error or a 502, 503 or 504 before the API has first answered shows a "waking up" state
and is retried once after 30 s, which covers a hosted API that is still starting; after the
first successful response such errors show at once. Each page aborts its requests (and any
pending retry) when it unmounts or its route changes. Routes live in the URL hash (`#/runs`,
`#/suites/<id>`, `#/runs/<id>`, `#/runs/<id>/<step>` with steps counted from 1,
`#/diff/<a>/<b>`), so any static host works and the selected suite and step survive a reload.
The API base is `VITE_API_URL` if set, else same-origin `/api`.

## Defaults

All are environment settings with the `AGENT_REPLAY_` prefix: provider `claude_code`, max
agent steps 6, tool timeout 10 s, model request timeout 30 s, max model attempts 3, backoff base
2 s, RPM 60, RPD 5000, run budget 1.50 USD, temperature 0, seed 7, allowed hosts
`localhost,127.0.0.1`.
