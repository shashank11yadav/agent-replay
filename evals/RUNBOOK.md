# Reproducing the results

These are the commands behind `evals/baseline/`, `evals/regression/` and `evals/results/*.md`,
with what each step measured on 2026-11-12. Live steps use the default provider, `claude_code`:
each model call runs the Claude Code CLI (`claude -p`) logged in on the local machine, so no API
key is involved. The offline steps need no CLI and no key at all.

## Cost and limits

- Live calls ran on a Claude Max subscription, which is not billed per token.
- Every USD figure here is notional: the API-equivalent cost at the `claude_code` default prices
  per 1M input / output tokens. Agent `claude-haiku-5-5`: $0.10 / $0.50. Judge
  `claude-sonnet-5-5`: $2.00 / $10.00.
- Each command is one process with its own notional run budget,
  `AGENT_REPLAY_RUN_BUDGET_USD` (default 1.50). When the estimate reaches it, the CLI reports a
  usage or rate limit, or a call still fails after its retries (5xx or timeout), the command
  stores what it has, prints the remaining tasks and exits 1. No command in the recorded runs
  stopped.
- A usage-limit stop ("You've hit your session limit" and similar) ends the command as
  `quota_exhausted`. Resume after the stated reset time. Only `suite record` resumes
  (`suite record --resume <suite id>`); rerun a stopped `suite replay` or `suite score`.
  Without `--tasks`, a resume reads the task file path stored on the suite, which works only
  when it is relative and you run from the same directory (the repository root here).
- "Not logged in" or "Login expired" ends runs as `error`. Run `claude`, use `/login`, rerun.

## Environment

Use a Postgres database reserved for these runs. With the compose `db` service running (`make db`
starts it; it listens on `127.0.0.1` at `AGENT_REPLAY_DB_PORT`, default 5432):

```bash
docker compose exec db createdb -U postgres agent_replay_live
```

| Variable | Use |
|---|---|
| `AGENT_REPLAY_PROVIDER` | leave unset; the default `claude_code` is what the fixtures hold |
| `AGENT_REPLAY_DATABASE_URL` | the database for these runs |
| `AGENT_REPLAY_RUN_BUDGET_USD` | optional, default 1.50 |
| `AGENT_REPLAY_TIMEOUT_S` | per-call timeout, default 30 |
| `AGENT_REPLAY_CLAUDE_CLI` | unset for live steps; an empty string forces cache-only replay |
| `AGENT_REPLAY_MODEL`, `AGENT_REPLAY_JUDGE_MODEL` | leave unset; the defaults `claude-haiku-5-5` and `claude-sonnet-5-5` are what the fixtures hold |

```bash
uv run agent-replay db migrate
```

## Smoke check

One task, before any suite, to prove the CLI flag combination, both model ids and the usage
fields. CLI version used: 2.1.318 (`claude --version`).

```bash
SMOKE=$(uv run agent-replay record --task T01 --tasks evals/tasks/tasks.jsonl --variant structured | tee /dev/stderr | tail -n 1) && echo "smoke: $SMOKE"
```

```bash
uv run agent-replay score --run "$SMOKE" --tasks evals/tasks/tasks.jsonl
```

Measured: the run ended `ok` with 2 model calls and 1 tool call, and the tool call came through
the reply schema. Both models were accepted; all 4 deterministic checks and the judge passed.
Tokens: 6517 input and 531 output. Wall time 6.86 s for the record, about 3.4 s per call
including CLI startup, well inside the 30 s timeout. Notional agent cost $0.00092.

## Record and score the three variants

The shipped variant in `evals/suite.json` is `structured`, and it is the baseline. Each suite
has 30 tasks.

```bash
STRUCTURED=$(uv run agent-replay suite record --tasks evals/tasks/tasks.jsonl --variant structured --label structured | tee /dev/stderr | tail -n 1) && echo "structured: $STRUCTURED"
```

```bash
uv run agent-replay suite score --suite-run "$STRUCTURED" --tasks evals/tasks/tasks.jsonl
```

```bash
MINIMAL=$(uv run agent-replay suite record --tasks evals/tasks/tasks.jsonl --variant minimal --label minimal | tee /dev/stderr | tail -n 1) && echo "minimal: $MINIMAL"
```

```bash
uv run agent-replay suite score --suite-run "$MINIMAL" --tasks evals/tasks/tasks.jsonl
```

```bash
PLAN_ACT=$(uv run agent-replay suite record --tasks evals/tasks/tasks.jsonl --variant plan_act --label plan_act | tee /dev/stderr | tail -n 1) && echo "plan_act: $PLAN_ACT"
```

```bash
uv run agent-replay suite score --suite-run "$PLAN_ACT" --tasks evals/tasks/tasks.jsonl
```

If a record stops, resume it with the id it printed, then score it:

```bash
uv run agent-replay suite record --resume "$STRUCTURED"
```

Measured:

| Command | Live calls | Notional USD |
|---|---|---|
| `suite record`, structured | 67 Haiku | $0.0300 |
| `suite record`, minimal | 68 Haiku | $0.0305 |
| `suite record`, plan_act | 66 Haiku | $0.0294 |
| `suite score`, structured | 29 Sonnet | $0.2301 |
| `suite score`, minimal | 29 Sonnet | $0.2302 |
| `suite score`, plan_act | 30 Sonnet | $0.2330 |

A run with no final answer gets no judge call, which is why two suites made 29 judge calls.
Record plus score took about 5 to 6 minutes per variant.

## Compare the approaches

`compare` reads the database only.

```bash
uv run agent-replay compare --suite-run "$MINIMAL" --suite-run "$STRUCTURED" --suite-run "$PLAN_ACT" --out evals/results/approaches.md
```

Results: `evals/results/approaches.md`.

## Fault replays of the baseline

One replay of the baseline per fault, each scored. `rate_limit:2` injects two 429s per run and
the retries hit the cache, so it makes no live calls. The two tool faults change tool results, so
the CLI is called live from the changed step on: at most 112 agent calls per replay (the sum of
`max_steps` over the 30 tasks), plus up to 30 judge calls in `suite score`.

```bash
F_TIMEOUT=$(uv run agent-replay suite replay --source "$STRUCTURED" --tasks evals/tasks/tasks.jsonl --label fault-tool-timeout --fault 'tool_timeout:*' | tee /dev/stderr | tail -n 1) && echo "tool_timeout: $F_TIMEOUT"
```

```bash
uv run agent-replay suite score --suite-run "$F_TIMEOUT" --tasks evals/tasks/tasks.jsonl
```

```bash
F_MALFORMED=$(uv run agent-replay suite replay --source "$STRUCTURED" --tasks evals/tasks/tasks.jsonl --label fault-malformed --fault 'malformed_tool_output:*' | tee /dev/stderr | tail -n 1) && echo "malformed_tool_output: $F_MALFORMED"
```

```bash
uv run agent-replay suite score --suite-run "$F_MALFORMED" --tasks evals/tasks/tasks.jsonl
```

```bash
F_RATE=$(uv run agent-replay suite replay --source "$STRUCTURED" --tasks evals/tasks/tasks.jsonl --label fault-rate-limit --fault 'rate_limit:2' | tee /dev/stderr | tail -n 1) && echo "rate_limit: $F_RATE"
```

```bash
uv run agent-replay suite score --suite-run "$F_RATE" --tasks evals/tasks/tasks.jsonl
```

```bash
uv run agent-replay compare --suite-run "$STRUCTURED" --suite-run "$F_TIMEOUT" --suite-run "$F_MALFORMED" --suite-run "$F_RATE" --out evals/results/faults.md
```

Measured new live agent cost: `tool_timeout:*` $0.0221, `malformed_tool_output:*` $0.0182,
`rate_limit:2` $0.0000. Wall time 4 m 26 s, 4 m 33 s and 3 m 02 s. Results:
`evals/results/faults.md`. No run crashed or ended in an unclassified state.

## Export the baseline fixtures

```bash
uv run agent-replay fixtures export --suite-run "$STRUCTURED" --out evals/baseline
```

```bash
uv run agent-replay fixtures verify evals/baseline
```

```bash
uv run python -c 'import json; m = json.load(open("evals/baseline/manifest.json")); print(m["provider"], m["agent_model"], m["judge_model"], m["variant"], m["run_count"])'
```

Expected: `claude_code claude-haiku-5-5 claude-sonnet-5-5 structured 30`.

## Offline check

This is what the CI gate does on every PR: import the baseline into a fresh database, replay it
with the CLI disabled, score it and gate it against itself. No live call is made.

`make gate-demo` runs this check and a gate of the committed regression suite in one go
(`scripts/offline-gate.sh`, with `AGENT_REPLAY_DATABASE_URL` naming a Postgres server; the
Makefile points it at the compose `db`). It works in a throwaway database that it drops
afterwards, unsets every `AGENT_REPLAY_*` variable, blanks the CLI and the Gemini key, takes the
provider and models from the baseline manifest, prints the two verdicts (baseline replay PASS,
prompt regression FAIL) and exits 0 only when both are as expected. The steps below do the same
by hand.

`$STRUCTURED` is the baseline's suite id. If the shell variable is gone, read it from the
manifest:

```bash
STRUCTURED=$(uv run python -c 'import json; print(json.load(open("evals/baseline/manifest.json"))["suite_run_id"])')
```

Create a fresh database and point `AGENT_REPLAY_DATABASE_URL` at it for the rest of this section:

```bash
docker compose exec db createdb -U postgres agent_replay_offline
```

```bash
AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay db migrate
```

```bash
AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay fixtures import evals/baseline
```

```bash
OFFLINE=$(AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay suite replay --source "$STRUCTURED" --tasks evals/tasks/tasks.jsonl --label offline-check | tee /dev/stderr | tail -n 1) && echo "offline: $OFFLINE"
```

```bash
AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay suite score --suite-run "$OFFLINE" --tasks evals/tasks/tasks.jsonl
```

```bash
AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay gate --baseline "$STRUCTURED" --candidate "$OFFLINE" --thresholds evals/gate.json --out-dir "$TMPDIR/gate-offline"
```

```bash
AGENT_REPLAY_CLAUDE_CLI= uv run agent-replay gate-check "$TMPDIR/gate-offline/verdict.json"
```

Measured: PASS, `gate-check` exit 0. Pass rate 73.3% vs 73.3%, mean tokens 7676.5 vs 7676.5,
p95 latency 8408.5 vs 8408.5 ms, all deltas 0. 29 ok runs and 1 error, as in the baseline, and
0 skipped. The import held 1 suite run, 30 runs, 96 cache entries, 136 spans and 150 scores.

## Deliberate prompt regression

The regressed prompt differs from the baseline, so every agent call of this replay is live.
Run it against the live database, with `$STRUCTURED` set as above.

```bash
REGRESSED=$(uv run agent-replay suite replay --source "$STRUCTURED" --variant structured_regressed --tasks evals/tasks/tasks.jsonl --label structured_regressed | tee /dev/stderr | tail -n 1) && echo "regressed: $REGRESSED"
```

```bash
uv run agent-replay suite score --suite-run "$REGRESSED" --tasks evals/tasks/tasks.jsonl
```

```bash
uv run agent-replay fixtures export --suite-run "$REGRESSED" --out evals/regression
```

```bash
uv run agent-replay fixtures verify evals/regression
```

Measured: 59 live Haiku agent calls and 30 Sonnet judge calls, notional $0.0365 agent plus
$0.2788 judge, $0.3153 in all. 29 ok runs and 1 diverged (T24), 0 skipped. The manifest names
`claude_code`, `claude-haiku-5-5`, `claude-sonnet-5-5`, variant `structured_regressed` and 30
runs, and its source suite is the baseline.

### Gate it offline

In another fresh database (`$REG_DB`), with the CLI disabled, import the baseline first and the
regression second. This is what CI does when a PR carries `evals/regression/` as
`evals/candidate/`: the second import adds the candidate suite and the cache entries the
baseline does not have, and refuses any row that clashes with a baseline row.

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay db migrate
```

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay fixtures import evals/baseline
```

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay fixtures import evals/regression
```

Replay the baseline with the regressed prompt, cache-only, as the CI job would:

```bash
PR=$(AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay suite replay --source "$STRUCTURED" --variant structured_regressed --tasks evals/tasks/tasks.jsonl --label pr | tee /dev/stderr | tail -n 1) && echo "pr: $PR"
```

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay suite score --suite-run "$PR" --tasks evals/tasks/tasks.jsonl
```

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay gate --baseline "$STRUCTURED" --candidate "$PR" --thresholds evals/gate.json --out-dir "$TMPDIR/gate-regression"
```

```bash
AGENT_REPLAY_CLAUDE_CLI= AGENT_REPLAY_DATABASE_URL="$REG_DB" uv run agent-replay gate-check "$TMPDIR/gate-regression/verdict.json"
```

Measured: the regression import added 1 suite run, 30 runs, 89 cache entries, 118 spans and 150
scores. The cache-only replay had 29 ok runs and 1 diverged, 0 skipped. The verdict is FAIL and
`gate-check` exits 1: pass rate 73.3% to 66.7% (-6.7 pp, FAIL), mean total tokens 7676.5 to
7583.4 (-1.2 %, PASS), p95 latency 8408.5 to 12353.5 ms (+46.9 %, FAIL). Pass to fail: T11, T15,
T19, T20, T23. Fail to pass: T26, T28, T30. Gating `$REGRESSED` directly instead of `$PR` gives
the same verdict. Results: `evals/results/regression.md`.

The workflow's `run:` steps were also run in order with `bash -eo pipefail` on a checkout that
carried `evals/regression/` as `evals/candidate/` and set `evals/suite.json` to
`structured_regressed`, with an empty `AGENT_REPLAY_CLAUDE_CLI`, no secrets and a fresh database
(the PR-comment step needs a GitHub token and was left out). Every step exited 0 up to the gate
check, which exited 1. The step summary held "Regression gate: FAIL" and this notice:

> Notice: this PR carries candidate recordings in evals/candidate/; CI checked the recorded
> evidence and did not re-run the model; review evals/candidate/ by hand (or run the gate locally
> with a live provider) before trusting a PASS.

## Totals (all live runs except the regressed suite)

The smoke check, the three variants and the three fault replays made 312 Haiku calls (mean
2627 ms, p95 3776 ms) and 131 Sonnet judge calls (mean 5394 ms, p95 7137 ms), about $1.20
notional (Haiku $0.13, Sonnet $1.07). The regressed suite is counted above. No usage-limit or
rate-limit stop happened, and every `suite record`, `suite replay` and `suite score` exited 0.
