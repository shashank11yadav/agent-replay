# ADR-004: Replay: hash-keyed model cache, recorded tool stubs, virtual latency

- Date: 2026-11-02
- Status: accepted

## Context
A CI gate that calls the live model for every task on every PR is slow, costs money or quota,
and gives different answers each time. A 30-task suite with the judge is 89 to 96 model calls
(measured: 67 agent and 29 judge calls for the baseline, 59 and 30 for the regressed suite,
`evals/RUNBOOK.md`). The gate
needs the unchanged part of a run to be free and identical, and only the changed part to cost
anything.

## Decision
- Hash: SHA-256 over canonical JSON (sorted keys, compact separators) of the cache schema
  version and the neutral request (provider, model, system, messages, tools, temperature,
  seed). Tool-call ids are deterministic (`call_<step>_<index>`), never random.
- Cache: `model_cache(request_hash PK, provider, model, request, response, latency_ms)`. Record
  mode writes; replay reads, then writes on live misses. Judge calls use the same cache.
- Replay reads the source run's own responses first (its model spans, by request hash) and the
  shared cache second. The cache is keyed by request alone, so a later recording of the same
  request may have replaced the entry, but the source run's spans still hold what it received.
- Tool stubs: built from the source run's `execute_tool` spans, keyed by tool name plus
  canonical arguments, first in first out for repeats. An unmatched call returns
  `{"error": "unrecorded_tool_call"}` and marks the run `diverged`. Tools are never executed in
  replay.
- No live provider: a miss raises `CacheMissNoLiveModel`, the run becomes `skipped`, and no
  network call is made.
- Virtual latency: a cache hit reports the latency recorded with the entry; a live call reports
  its measured latency; client backoff time is added. Run latency is the sum of step latencies.

## Alternatives considered
- Full live replay with a fixed seed: costly, and not identical (seeds are best effort, and the
  Claude Code CLI has none).
- Wall-clock latency on replay: cache hits take about 0 ms, so p95 would collapse and every
  changed prompt would look slower.
- Matching tool calls by position instead of arguments: a reordered but correct agent would get
  wrong results without any error.

## Consequences
- A prompt change that alters the first request makes every later step of that run a miss, so
  a request-changing PR costs about one live suite plus its judge calls.
- The cache must be committed with the baseline (fixtures) for CI to run offline.
- A candidate's p95 mixes recorded and fresh latencies; the fresh ones carry network noise.
