# Deliberate prompt regression

The baseline is the `structured` prompt. The candidate is the `structured_regressed` prompt, a
deliberately worse version of it. Both ran the same 30 tasks with the `claude_code` provider
(agent `claude-haiku-5-5`, judge `claude-sonnet-5-5`) on 2026-11-12. The candidate was recorded
by replaying the baseline suite with the new prompt, so every agent call was live. The offline
gate then compared the two from the fixtures in `evals/baseline/` and `evals/regression/`, with
the Claude Code CLI disabled.

Verdict: **FAIL**. `gate-check` exited 1.

| metric | before (baseline, structured) | after (candidate, structured_regressed) | delta | threshold | status |
|---|---|---|---|---|---|
| pass rate | 73.3% | 66.7% | -6.7 pp | drop at most 5 pp | FAIL |
| mean total tokens | 7676.5 | 7583.4 | -1.2 % | rise at most 15 % | PASS |
| p95 latency (ms) | 8408.5 | 12353.5 | +46.9 % | rise at most 20 % | FAIL |
| notional USD, agent | 0.0300 | 0.0365 | +0.0065 | none | |
| notional USD, judge | 0.2301 | 0.2788 | +0.0487 | none | |
| notional USD, total | 0.2601 | 0.3153 | +0.0552 (+21 %) | none | |

Thresholds come from `evals/gate.json`: pass rate may drop at most 5 percentage points, mean
total tokens may rise at most 15 %, and p95 latency may rise at most 20 %. Notional USD is what
the calls would cost at list prices. Nothing was billed; the runs used a subscription.

The pass rate and p95 latency tripped the gate. The pass rate fell 6.7 points against a limit of
5, and p95 latency rose 46.9 % against a limit of 20 %. Mean total tokens did not trip: they fell
1.2 %, even though mean output tokens roughly doubled (583.3 to 1145.2), because the regressed
prompt made fewer model calls (59 against 67).

Other numbers from the same runs: deterministic pass rate 73.3% to 66.7%, judge pass rate
83.3% to 66.7%. Five tasks flipped from pass to fail (T11, T15, T19, T20, T23) and three from
fail to pass (T26, T28, T30). The candidate suite had 29 ok runs and 1 diverged run (T24), with
0 skipped.
