# agent-replay

Record an LLM agent run once, replay it offline as often as you like, score it, and block the pull request that makes the agent worse.

agent-replay stores every model call and tool call of an agent run in Postgres, replays runs deterministically from a request-hash cache, scores them with deterministic checks and an LLM judge, and compares a candidate suite with a committed baseline on pass rate, tokens and p95 latency. A GitHub Actions workflow runs that comparison on every pull request with no API key and no live model call. A React viewer steps through a run and diffs two runs side by side.

![Suite runs in the viewer](docs/images/viewer-runs.png)

## Why

Changing a prompt, a model or a tool schema changes how an agent behaves, and a unit test on the final text does not show it. A prompt can keep the answers right while doubling the output tokens, or make the agent slower, or quietly break one task in five. Running the whole task set live on every pull request is slow, costs money and is not repeatable.

agent-replay turns that into a number you can gate on. Record a baseline once, commit it as readable fixtures, and every later change is replayed against the same tool results and scored the same way. The measured case below shows why one metric is not enough: a deliberately worse prompt used *fewer* tokens and still failed the gate on pass rate and latency.

## Features

- **Record.** Every model call and tool call becomes an OpenTelemetry-style span with tokens, latency and the request hash, written to Postgres in one transaction per run.
- **Replay.** Tool results come from the recording, matched by tool name and canonical arguments. Model responses come from the source run's own spans first, then from a cache keyed by a SHA-256 request hash. A changed prompt, model or tool schema misses the cache, and that miss is the signal. A tool call the recording never saw marks the run `diverged`.
- **Scoring.** Four deterministic checks (`expected_text`, `must_call_tools`, `max_steps`, `completed`) plus an LLM judge with its own model setting (in the committed runs Sonnet judges Haiku). A task passes only when all of them pass. The answer is embedded in the judge prompt as a JSON string literal, so it cannot break out of its delimiter.
- **Fault injection.** Replay-only faults (`tool_timeout:<tool|*>`, `malformed_tool_output:<tool|*>`, `rate_limit:<n>`) test how the agent and the client behave when tools or the provider misbehave.
- **Run diff.** Two runs are aligned step by step with token, latency and pass/fail deltas, and the first divergence is highlighted.
- **CI regression gate.** PASS, FAIL, SKIPPED or ERROR against thresholds in `evals/gate.json`. The baseline, the task file and the thresholds come from the pull request's base commit: the gate ignores the PR's own copies. A PR that changes a model request can carry its own recordings and still be gated offline.
- **Viewer.** React and TypeScript, served by the API. Suite and run pages with deep links, step navigation with the arrow keys, a diff page, light and dark themes, and a strict Content Security Policy.
- **Pluggable providers.** A provider is one `ProviderSpec` in a registry. Three ship: `claude_code` (the Claude Code CLI), `gemini` (the Google GenAI SDK) and `scripted` (an offline fake for tests). Adding one touches no runner, cache, budget or judge code, and a shared conformance test kit checks each provider.

## Results

All numbers were recorded on 2026-11-12 with the `claude_code` provider: agent `claude-haiku-5-5`, judge `claude-sonnet-5-5`, 30 synthetic tasks per suite. The baseline and regression suites are committed as fixtures under `evals/`, and the tests recompute the baseline row and the regression table from them.

### Comparing prompt variants

| Variant | Pass rate | Deterministic pass rate | Judge pass rate | Mean total tokens | p95 latency (ms) | Notional USD |
|---|---|---|---|---|---|---|
| minimal | 80.0% | 80.0% | 86.7% | 7792.1 | 8991 | 0.2607 |
| structured (baseline) | 73.3% | 73.3% | 83.3% | 7676.5 | 8408 | 0.2601 |
| plan\_act | 66.7% | 66.7% | 83.3% | 7236.2 | 11249 | 0.2624 |

Source: [evals/results/approaches.md](evals/results/approaches.md). USD is the API list-price equivalent of the tokens used.

### The gate catching a regression

`structured_regressed` is a deliberately weaker version of the baseline prompt. It was recorded live, then gated offline from the committed fixtures with no model available.

| metric | baseline | candidate | delta | threshold | status |
|---|---|---|---|---|---|
| pass rate | 73.3% | 66.7% | -6.7 pp | drop at most 5 pp | FAIL |
| mean total tokens | 7676.5 | 7583.4 | -1.2 % | rise at most 15 % | PASS |
| p95 latency (ms) | 8408.5 | 12353.5 | +46.9 % | rise at most 20 % | FAIL |

Verdict **FAIL**, and `gate-check` exits 1. Mean output tokens roughly doubled (583.3 to 1145.2), but the regressed prompt made fewer model calls (59 against 67), so total tokens went down. A token-only gate would have let this change through. Five tasks flipped from pass to fail. Source: [evals/results/regression.md](evals/results/regression.md).

The same comparison of the baseline against its own offline replay gives **PASS** with zero deltas.

### Fault replays

Each fault was replayed over the 30 baseline runs ([evals/results/faults.md](evals/results/faults.md)). Under `rate_limit:2` every injected 429 was retried with backoff and served from the cache: 29 runs ended `ok`, the pass rate matched the baseline at 73.3%, and no new live call was made. Under the tool faults every run ended in a classified state (`ok`, a clean `max_steps_exceeded` error, or `diverged`). None crashed.

### Cost and speed

The three variants, the three fault replays and a smoke check made 312 Haiku calls (mean 2.6 s) and 131 Sonnet judge calls (mean 5.4 s), about $1.20 at list prices. No command hit the per-process run budget. CI replays the committed fixtures cache-only and spends nothing. Details: [evals/RUNBOOK.md](evals/RUNBOOK.md).

## Architecture

```mermaid
flowchart LR
  subgraph cli["agent-replay CLI"]
    REC[record / replay / suite] --> RT[Runtime: recording and replay contexts, tool stubs, faults]
    RT --> AGENT[Demo agent: bounded tool-calling loop]
    AGENT --> CLIENT[LLM client: retries and backoff]
    CLIENT --> CACHE[Recorded spans, then request-hash cache]
    CACHE --> GUARD[Run budget guard and rate limiter]
    GUARD --> PROV[Provider registry: claude_code, gemini, scripted]
    SCORE[score: deterministic checks and LLM judge]
    GATE[gate / gate-check / compare]
    FIX[fixtures export / import / verify]
  end
  RT --> DB[(Postgres: runs, spans, scores, model_cache)]
  SCORE --> DB
  GATE --> DB
  FIX <--> DB
  FIX <--> FILES[evals/ JSONL fixtures]
  DB --> API[FastAPI, read-only, /api]
  API --> VIEW[React viewer]
  CI[GitHub Actions regression gate] --> FIX
  CI --> GATE
```

A recorded run is a tree of spans: the root `invoke_agent` span, then `generate_content <model>` and `execute_tool <tool>` children ordered by step. Replay rebuilds the same agent with a replay context: tools are stubbed from the source run, and each model request passes through the LLM client, any injected faults, the cache and the budget guard before it can reach a live provider. A request that hits the cache never reaches the provider, so a fully cached suite replays with no network and no CLI. Suites can be exported as JSONL fixtures, committed, and imported anywhere. Import is transactional, re-hashes every cache request and refuses any row that clashes with an existing one.

The design notes are in [docs/design.md](docs/design.md), and each major choice has a decision record in [docs/decisions/](docs/decisions/).

## The regression gate in CI

[`.github/workflows/regression-gate.yml`](.github/workflows/regression-gate.yml) runs on every pull request:

1. Check out the PR base commit and take `evals/baseline/`, `evals/tasks/tasks.jsonl` and `evals/gate.json` from it.
2. Check the baseline with `fixtures verify`, import it into a Postgres service container and read the provider and models from its manifest.
3. If the PR carries `evals/candidate/`, check that it matches the baseline (provider, models, source suite, no symlinks) and import it.
4. Replay the baseline with the PR's prompt variant from `evals/suite.json`, cache-only, and score it.
5. Run `gate`, write the report to the step summary, and fail the job on FAIL or ERROR with `gate-check`.
6. A separate job with write permission posts the report as a PR comment, so the job that runs PR code only ever has read access.

Verdicts, in precedence order:

| Verdict | When |
|---|---|
| ERROR | a run stopped on a quota, a rate limit, a transient provider failure or the run budget, the judge failed on a provider error, the candidate's task set differs from the baseline's or has duplicates, either suite is not fully scored, or the baseline has duplicate tasks, skipped runs or runs stopped on a stop code |
| SKIPPED | a needed model call had no live model available |
| FAIL | pass rate, mean total tokens or p95 latency moved past its threshold |
| PASS | none of the above |

A pull request that changes a model request misses the cache in CI, because CI has no model. Instead of being skipped, it can carry a fixtures export of its candidate in `evals/candidate/`, recorded locally ([ADR-011](docs/decisions/ADR-011-carried-candidate-recordings.md)). The `demo/prompt-regression` branch does exactly that with the regressed prompt, and the gate reports FAIL on it.

## Quick start

Requires Docker.

```sh
docker compose up --build --wait
```

This builds the image, starts Postgres, applies the migrations, imports the committed baseline and regression fixtures, and serves the API and the viewer on http://127.0.0.1:8000. No API key and no model login are needed to browse the runs. Stop it with `docker compose down -v`. If port 5432 is taken on your machine, set `AGENT_REPLAY_DB_PORT` to another port first.

![Diff of the same task under two prompts](docs/images/viewer-diff.png)

## Try it

Run the regression gate offline on the committed fixtures. This needs Python 3.12, [uv](https://docs.astral.sh/uv/) and a Postgres server. It creates a throwaway database, never calls a model, and drops the database when it finishes.

```sh
make setup
make db          # Postgres from docker compose
make gate-demo   # baseline replay: PASS, prompt regression: FAIL
```

The same steps by hand, against a database you choose:

```sh
export AGENT_REPLAY_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres
export AGENT_REPLAY_CLAUDE_CLI=        # no CLI: a cache miss is skipped, never sent
uv run agent-replay db migrate
uv run agent-replay fixtures import evals/baseline
uv run agent-replay fixtures import evals/regression
BASE=$(uv run python -c 'import json; print(json.load(open("evals/baseline/manifest.json"))["suite_run_id"])')
REG=$(uv run python -c 'import json; print(json.load(open("evals/regression/manifest.json"))["suite_run_id"])')

PASS_DIR=$(mktemp -d); FAIL_DIR=$(mktemp -d)

# Replay the baseline from its own recordings, score it, and gate it: PASS
OFFLINE=$(uv run agent-replay suite replay --source "$BASE" --tasks evals/tasks/tasks.jsonl --label offline | tail -n 1)
uv run agent-replay suite score --suite-run "$OFFLINE" --tasks evals/tasks/tasks.jsonl
uv run agent-replay gate --baseline "$BASE" --candidate "$OFFLINE" --thresholds evals/gate.json --out-dir "$PASS_DIR"
uv run agent-replay gate-check "$PASS_DIR/verdict.json"

# Gate the committed prompt regression: FAIL, exit code 1
uv run agent-replay gate --baseline "$BASE" --candidate "$REG" --thresholds evals/gate.json --out-dir "$FAIL_DIR"
uv run agent-replay gate-check "$FAIL_DIR/verdict.json"

# Compare the two suites side by side (a Markdown table)
uv run agent-replay compare --suite-run "$BASE" --suite-run "$REG" --out "$FAIL_DIR/compare.md"
```

Each `gate` writes `report.md` (the Markdown that CI posts on the pull request) and `verdict.json` to its output directory. The quick-start stack shows the same two suites in the viewer: open a task and diff it against the other suite's run.

Recording new runs needs a live provider: a logged-in [Claude Code CLI](https://code.claude.com/docs/en/cli-reference) for `claude_code`, or `AGENT_REPLAY_GEMINI_API_KEY` for `gemini`. The full live procedure, from a one-task smoke check to recording, scoring and exporting a suite, is in [evals/RUNBOOK.md](evals/RUNBOOK.md).

## Engineering

- Tests: 1,168 Python tests (unit, integration against real Postgres, provider conformance, CLI subprocess and workflow-script tests) and 112 viewer tests with Vitest and Testing Library. The integration tests create and drop throwaway databases (one per test, or per module for the slower recordings), and the workflow tests execute the regression gate's own shell steps.
- Checks: `ruff` lint and format, `mypy --strict` on the package, TypeScript strict mode. CI runs all of them, builds the viewer and builds the Docker image. Actions are pinned to commit SHAs, with Dependabot configured for actions, npm, uv and Docker.
- Request hashing: Golden hash tests pin the request hash for each provider, so a refactor cannot silently invalidate committed fixtures.
- Hardening: Read-only API with a host allowlist, CORS that refuses `*`, a Content Security Policy and a per-statement timeout. Every SQL value is a bound parameter. Fixture files are opened without following symlinks, size-capped and schema-checked with unknown fields rejected. The Claude Code CLI runs in an empty temporary directory with its own tools disabled, a scrubbed environment and its whole process group killed on timeout. Live runs are capped by a per-process notional budget and a client-side rate limiter.
- Decisions: Eleven architecture decision records in [docs/decisions/](docs/decisions/) cover the stack, storage, the model client, the replay cache, the CI gate, hosting, tracing, the eval design, viewer serving, the provider registry and carried recordings.
- Write-up: [docs/case-study.md](docs/case-study.md) walks through the design, the results and the bugs found along the way.

## Tech stack

| Area | Tools |
|---|---|
| Language and packaging | Python 3.12, uv, `uv_build` |
| Storage | PostgreSQL 18, psycopg 3 with a connection pool, plain SQL migrations |
| Tracing | OpenTelemetry SDK, GenAI semantic-convention attributes |
| API | FastAPI, Uvicorn, Pydantic 2 |
| Models | Claude Code CLI (`claude -p`), Google GenAI SDK, a scripted fake |
| Viewer | React 19, TypeScript, Vite, Zod, Vitest |
| Delivery | Docker multi-stage image (non-root), Docker Compose, GitHub Actions |

## Project layout

```
src/agent_replay/
  cli/        CLI commands: db, record, replay, score, suite, gate, gate-check, compare, fixtures
  runtime/    recording and replay contexts, tool stubs, fault injection, run lifecycle
  llm/        provider registry, client, cache, budget guard, rate limiter, request hashing, providers
  scoring/    deterministic checks and the LLM judge
  suite/      record, replay and score a whole task file (resumable)
  report/     suite metrics, the gate, Markdown reports
  db/         migrations and repositories
  api/        read-only FastAPI app
  adapters/   interface for plugging in another agent
  demo/       the demo agent, its three tools, prompt variants and synthetic data
  fixtures.py export, import and verify of JSONL fixtures
  diff.py     step-aligned run diff
viewer/       React viewer and its tests
evals/        tasks, gate thresholds, committed baseline and regression fixtures, result tables, runbook
tests/        unit, integration, providers, evals, repo and docs tests
docs/         design notes, decision records, provider guide, case study
deploy/       optional free-tier hosting guide
scripts/      offline-gate.sh
```

## Development

```sh
make setup        # uv sync and npm ci
make db           # Postgres for the tests (AGENT_REPLAY_TEST_DATABASE_URL, default localhost:5432)
make check        # lint, typecheck, Python tests, viewer tests, viewer build
make test-fast    # unit and provider tests, about 20 seconds
make image        # Docker build
```

Settings are `AGENT_REPLAY_*` environment variables read by [`src/agent_replay/config.py`](src/agent_replay/config.py), and no `.env` file is read. The ones you are most likely to touch:

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_REPLAY_DATABASE_URL` | unset | database for the CLI and the API |
| `AGENT_REPLAY_PROVIDER` | `claude_code` | `claude_code`, `gemini` or `scripted` |
| `AGENT_REPLAY_MODEL`, `AGENT_REPLAY_JUDGE_MODEL` | from the provider | agent and judge models |
| `AGENT_REPLAY_CLAUDE_CLI` | `claude` | CLI name or path, empty means cache-only |
| `AGENT_REPLAY_GEMINI_API_KEY` | empty | live Gemini calls |
| `AGENT_REPLAY_RUN_BUDGET_USD` | 1.50 | notional spend limit per process |
| `AGENT_REPLAY_RPM`, `AGENT_REPLAY_RPD` | 60, 5000 | client-side rate limiter |
| `AGENT_REPLAY_TIMEOUT_S`, `AGENT_REPLAY_MAX_ATTEMPTS` | 30, 3 | per-call timeout and retries |
| `AGENT_REPLAY_CORS_ORIGINS`, `AGENT_REPLAY_ALLOWED_HOSTS` | none, `localhost,127.0.0.1` | API origins and host allowlist |

To add a model provider, follow [docs/providers.md](docs/providers.md). To host the API and viewer on free tiers, see [deploy/DEPLOY.md](deploy/DEPLOY.md).

## Author

Built by Shashank Yadav. More of my work is at [shashankyadav.dev](https://shashankyadav.dev).

## License

[MIT](LICENSE)
