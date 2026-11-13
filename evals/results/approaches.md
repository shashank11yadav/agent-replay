# Approach comparison

Run on 2026-11-12 with provider `claude_code` (the Claude Code CLI on a Max subscription).
Agent model `claude-haiku-5-5`, judge model `claude-sonnet-5-5`. 30 tasks per variant. Baseline:
`structured`, suite run 80ed9cf4-bc86-48ce-a93e-82ee9deef657 (shipped in `evals/suite.json` and
exported to `evals/baseline`).

Pass rate means all deterministic checks passed and the judge passed. Judge pass rate is judge
passes over 30 tasks; a run with no answer gets no judge call (29 judged for minimal and
structured). USD is notional (API-equivalent list prices, not billed), agent and judge calls
together.

| Variant | Pass rate | Deterministic pass rate | Judge pass rate | Mean total tokens | p95 latency (ms) | Notional USD |
|---|---|---|---|---|---|---|
| minimal | 80.0% | 80.0% | 86.7% | 7792.1 | 8991 | 0.2607 |
| structured (baseline) | 73.3% | 73.3% | 83.3% | 7676.5 | 8408 | 0.2601 |
| plan\_act | 66.7% | 66.7% | 83.3% | 7236.2 | 11249 | 0.2624 |

Minimal and structured each had one error (task T30, `max_steps_exceeded`); plan\_act had none.

## Suite comparison (compare output)

| Suite run | Variant | Pass rate | Mean total tokens | p95 latency (ms) | Errors | Diverged | Skipped |
|---|---|---|---|---|---|---|---|
| 3bcf931d-7b63-4635-b2bd-429dd0d9f3a5 | minimal | 80.0% | 7792.1 | 8991 | 1 | 0 | 0 |
| 80ed9cf4-bc86-48ce-a93e-82ee9deef657 | structured | 73.3% | 7676.5 | 8408 | 1 | 0 | 0 |
| a330d9ff-9df8-4a26-afcc-44723c71df57 | plan\_act | 66.7% | 7236.2 | 11249 | 0 | 0 | 0 |

Token means and p95 latency include error and skipped runs.
