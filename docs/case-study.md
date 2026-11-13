# Case study: agent-replay

## The problem

When I change a prompt, a model or a tool schema in an LLM agent, I want a number before merge that says whether the agent got worse. Running the whole task set live on every pull request costs money and quota, and live runs are not repeatable. A test that only checks the final text misses a prompt that makes the agent slower or longer-winded.

agent-replay records each model call and tool call of an agent run into Postgres, replays the run offline, scores it, and fails a pull request when pass rate, mean total tokens or p95 latency move past a threshold against a committed baseline.

## The design

- Record and replay cache. Every model call is stored with its tokens, latency and a request hash: SHA-256 over the canonical JSON of the cache schema version, provider, model, system instruction, messages, tool schemas, temperature and seed (ADR-004). Replay reads the cache first and calls a live model only on a miss, so a changed prompt, model or tool schema misses the cache, and that is the signal. Tool results always come from the recording, matched by tool name and canonical arguments, so replay never runs the real tools. A tool call that was never recorded gets `unrecorded_tool_call` and the run is marked `diverged`.
- Deterministic checks plus a judge on a different model. Each run gets four deterministic checks (`expected_text`, `must_call_tools`, `max_steps`, `completed`) and an LLM judge, and a task passes only when all of them pass. The agent runs on `claude-haiku-5-5` and the judge on `claude-sonnet-5-5`, a different model, so the agent's model never grades its own answers (ADR-010).
- Fault injection. Replay-only faults: `tool_timeout:<tool|*>`, `malformed_tool_output:<tool|*>` and `rate_limit:<n>`. The results are reported, not gated.
- The CI regression gate. `.github/workflows/regression-gate.yml` runs on pull requests. It takes the baseline fixtures, the task file and `evals/gate.json` from the PR base commit, replays the baseline with the variant in `evals/suite.json`, scores it, and returns PASS, FAIL, SKIPPED or ERROR. Thresholds: pass rate may drop at most 5 pp, mean total tokens may rise at most 15 %, p95 latency at most 20 % (ADR-005). The job that runs the PR's code is read-only and holds no secrets. A separate job that runs no PR code posts the report as a PR comment.
- A provider registry and the `claude_code` provider. Every provider is a `ProviderSpec` in `src/agent_replay/llm/registry.py`, and adding one touches no runner, cache, budget or judge code (`docs/providers.md`, ADR-010). The `claude_code` provider runs the locally installed and logged-in Claude Code CLI, one `claude -p` subprocess per model call, in an empty temporary directory, with the CLI's own tools turned off and the reply forced through a JSON schema. Live runs use my Max subscription login, so they are not billed per token, and that login never runs in CI. `gemini` and the offline `scripted` fake are the other two providers.
- Carried candidate recordings (ADR-011). CI has no CLI, so a pull request that changes a model request would miss the cache and get SKIPPED. A PR can instead carry a fixtures export of its candidate in `evals/candidate/`. The job imports it after the baseline, replays cache-only and gates the result. It checks that the directory comes from the PR head, that provider and models match the baseline manifest, that the source suite is the baseline suite and that there are no symlinks, and the import refuses any row that clashes with a baseline row.
- Where the trust sits. With carried recordings CI checks the recorded evidence rather than re-running the model, so the design keeps the reference points out of the PR. The baseline, the task file and the thresholds come from the base commit. `evals/candidate/` is plain JSONL that a maintainer reviews like code, or checks with a local rerun on a live provider. Every verdict that rests on carried recordings carries a notice saying so above the report. And the gate becomes a required check together with CODEOWNERS and branch protection on `.github/`, `evals/` and `src/`.

## Decisions and trade-offs

- One subprocess per model call: a fresh working directory, no shared state and a kill on timeout, at the cost of CLI startup on every call. Measured over all live runs, a Haiku call took 2.6 s on average and a Sonnet judge call 5.4 s, startup included. I used the CLI rather than the Agent SDK because the CLI was already installed and logged in, and the SDK would add a dependency.
- Subscription login instead of an API key: no per-token bill for live runs, but CI cannot use it, so CI replays from the committed cache.
- Fixtures as readable JSONL, so a reviewer can read a fixture diff, at the cost of larger files than a database dump.
- Baseline, task file and thresholds come from the PR base commit, and the gate ignores the PR's own copies. Together with CODEOWNERS and branch protection on `.github/`, `evals/` and `src/`, that keeps the reference fixed. A re-baselining PR is gated against the old baseline.
- A read-only API, no accounts, no multi-tenancy.
- Synthetic order and product data, so every expected value is recomputable (ADR-008).

## Results

All numbers were recorded on 2026-11-12 with provider `claude_code`, 30 synthetic tasks per variant. USD is notional (API-equivalent list prices), because nothing was billed. The full tables are in `evals/results/`.

Prompt variants ([approaches.md](../evals/results/approaches.md)): `minimal` passed 80.0% of tasks, the `structured` baseline 73.3% and `plan_act` 66.7%. `structured` had the lowest p95 latency of the three (8408 ms) and `plan_act` the lowest mean total tokens (7236.2).

Deliberate regression ([regression.md](../evals/results/regression.md)). The candidate is `structured_regressed`, a weaker version of the baseline prompt, recorded live and gated offline from the committed fixtures.

| metric | before | after | delta | threshold | status |
|---|---|---|---|---|---|
| pass rate | 73.3% | 66.7% | -6.7 pp | drop at most 5 pp | FAIL |
| mean total tokens | 7676.5 | 7583.4 | -1.2 % | rise at most 15 % | PASS |
| p95 latency (ms) | 8408.5 | 12353.5 | +46.9 % | rise at most 20 % | FAIL |

The verdict is FAIL and `gate-check` exits 1. Tokens did not trip the gate: mean output tokens roughly doubled (583.3 to 1145.2), but the regressed prompt made fewer model calls (59 against 67). A token-only gate would have passed this change. Five tasks flipped from pass to fail (T11, T15, T19, T20, T23) and three from fail to pass (T26, T28, T30).

Fault replays of the baseline ([faults.md](../evals/results/faults.md)), 30 runs each. Both tool faults made most runs diverge (20 for `tool_timeout:*`, 22 for `malformed_tool_output:*`), because after the fault the live model asked for calls the recording does not have. `rate_limit:2` made no new live calls: the injected 429s were retried and every retry hit the cache, and its p95 rose from 8408 to 14408 ms only because backoff time is added to step latency. No run crashed or ended in an unclassified state.

Cost and latency ([RUNBOOK](../evals/RUNBOOK.md)): the smoke check, the three variants and the three fault replays made 312 Haiku calls (mean 2627 ms, p95 3776 ms) and 131 Sonnet judge calls (mean 5394 ms, p95 7137 ms), about $1.20 notional. The regressed suite was 59 Haiku and 30 Sonnet calls, $0.3153 notional. No command stopped on a budget or usage limit. Recording and scoring one variant took about 5 to 6 minutes.

## What went wrong, and how it was found

These are faults found while building and checking the system, each fixed.

- A relative CLI path. `AGENT_REPLAY_CLAUDE_CLI` set to a relative path passed the "found" check but failed every call, because each call runs in a temporary directory. It is now resolved to an absolute path. A NUL byte in an argv value now becomes a permanent model error.
- Record versus replay looked like a change. The first diff compared whole payloads, so every model step differed because of the cache status field. It now compares a content allowlist.
- Unscored runs counted as passes. A suite with no scores had a deterministic pass rate that looked fine. An unscored or empty suite now gives ERROR.
- Resume left an extra run. `suite record --resume` left the stopped run next to its replacement, giving N+1 runs in a suite. Stopped runs are now detached.
- Fixture import dropped judge entries silently. Import is now strict and transactional: a differing row, a clash or a new row under an existing parent rolls back and exits 1. Cache rows are re-hashed on load.
- Judge input injection. The answer is now embedded in the judge prompt as a JSON string literal, so an answer cannot close its own delimiter.
- The gate trusted the PR. The first gate workflow failed to fetch the base commit on private repositories, failed every PR before a baseline existed, and read the task file from the PR, which would let a PR loosen its own labels. Base files now come from a pinned checkout of the base commit, a missing baseline is a green job with a notice, and actions are pinned to commit SHAs.
- A seed step hid a failed import. In the compose seed, a semicolon masked a failed baseline import. It now runs with `set -e`.
- A blank page on GitHub Pages. The viewer had no Vite `base`, so a project site requested assets from the domain root. `deploy/DEPLOY.md` now builds with `--base=/<repo>/`, checked with a local build.
- Labels that human text could not match. Float-formatted labels such as `2700.0` failed answers formatted as money, and some comparison labels were echoed in the prompt. The dataset labels were corrected and recomputed from the tool functions.
- A `*` CORS origin was accepted. `create_app` now raises on a `*` or blank origin.
- The token gate hole. The regression run above showed that a worse prompt can use fewer total tokens. That is why the gate checks pass rate and p95 latency as well, and why the metrics have to be read together.

## Hardening after the first version

A second pass over the finished system tightened the places where a wrong answer could pass quietly:

- Replay serves each response from the source run's own spans before the shared cache, so a later recording of the same request cannot change what a replay sees.
- The gate gives ERROR for more causes: a run that still failed after its retries (`transient_error`), a judge call that failed at the provider, a candidate whose tasks differ from the baseline's or repeat, and runs missing deterministic or judge scores.
- `fixtures verify` also checks that every cache entry agrees with the span that used it, that every run is fully scored, and that the manifest's run and task counts match the files.
- The API answers only the host names in `AGENT_REPLAY_ALLOWED_HOSTS`, sends a Content Security Policy, `nosniff` and `no-referrer` on every response, and sets a 5 s statement timeout. The viewer runs zod without `eval`, so the policy needs no exception.
- The Claude Code CLI runs in its own process group, so a timeout kills everything it started. The variables that could move it off the subscription login are stripped, and a failure is classified only from the CLI's error output, never from the model's reply.
- CI actions are pinned to commit SHAs, with Dependabot updates. The PR comment moved to its own job, so the job that runs PR code is read-only.
- `make gate-demo` (`scripts/offline-gate.sh`) checks both committed verdicts, baseline PASS and regression FAIL, offline in a throwaway database.
