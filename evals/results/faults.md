# Fault replays of the baseline

Run on 2026-11-12 with provider `claude_code`, agent `claude-haiku-5-5`, judge
`claude-sonnet-5-5`. Each fault is replayed over the structured baseline (suite run
80ed9cf4-bc86-48ce-a93e-82ee9deef657), 30 runs each. Share ended ok or clean error is the share
of the 30 runs that finished `ok` or as a classified `error` (here `max_steps_exceeded`). A run
that diverged (after the injected fault the live model asked for a tool call the recording does
not have) is in neither group. No run crashed or ended in an unclassified state.

| Fault | Share ended ok or clean error | Ended ok | Clean errors | Diverged | Skipped | Pass rate | Mean total tokens | p95 latency (ms) | New live agent notional USD |
|---|---|---|---|---|---|---|---|---|---|
| tool\_timeout:\* | 33.3% | 4 | 6 | 20 | 0 | 0.0% | 9893.3 | 47298 | 0.0221 |
| malformed\_tool\_output:\* | 26.7% | 6 | 2 | 22 | 0 | 0.0% | 8852.8 | 9149 | 0.0182 |
| rate\_limit:2 | 100.0% | 29 | 1 | 0 | 0 | 73.3% | 7676.5 | 14408 | 0.0000 |

The `rate_limit:2` replay made no new live calls: the injected 429s were retried with backoff
and every retry hit the cache. Its one error is the same task as in the baseline.

## Suite comparison (compare output)

| Suite run | Variant | Pass rate | Mean total tokens | p95 latency (ms) | Errors | Diverged | Skipped |
|---|---|---|---|---|---|---|---|
| 80ed9cf4-bc86-48ce-a93e-82ee9deef657 | structured | 73.3% | 7676.5 | 8408 | 1 | 0 | 0 |
| 93f0eb06-81a4-48a0-b259-f5de80975cd0 | structured | 0.0% | 9893.3 | 47298 | 6 | 20 | 0 |
| 0da793cb-9e54-4e5e-8f17-261f2105cdc3 | structured | 0.0% | 8852.8 | 9149 | 2 | 22 | 0 |
| fe30231f-7e30-4fe5-abc4-2cfa35c7cd04 | structured | 73.3% | 7676.5 | 14408 | 1 | 0 | 0 |

Token means and p95 latency include error and skipped runs.
