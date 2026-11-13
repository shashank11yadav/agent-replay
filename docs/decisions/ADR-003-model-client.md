# ADR-003: Model client: provider interface, rate limiter and run budget guard

- Date: 2026-11-02
- Status: accepted. The provider choice (Gemini only) and the same-model judge are superseded
  by ADR-010.

## Context
Every model call has to be recordable, cacheable, retried the same way and bounded in cost,
whatever the backend. Live calls cost money or quota, so a runaway command must stop early, and
a provider's own spend controls act too late to stop one process.

## Decision
- A `ModelProvider` protocol over provider-neutral request and response types. Only the
  provider module imports its SDK (`llm/gemini.py` for `google.genai`). Automatic tool execution
  in the SDK is off, so every tool call is visible to the recorder.
- One client (`llm/client.py`) owns retries, backoff, timeouts and the per-call log line. It
  retries 429 and 5xx with exponential backoff and raises everything else at once.
- The live provider is wrapped as `BudgetGuardProvider(RateLimitedProvider(provider))`, and the
  whole live chain sits behind the cache, so cache hits never reach the guard or the limiter.
- Rate limiter: requests per minute and per day from config (defaults 60 and 5000). The per-day
  count is per process and raises `QuotaExhausted` instead of sleeping until the next day.
- Run budget guard: each live call's cost is estimated from its usage,
  `input_tokens * price_in / 1e6 + output_tokens * price_out / 1e6`. Spend is cumulative per
  process and shared by agent and judge calls. Before a live call the guard raises
  `BudgetExceeded` if spend has reached the budget or spend plus the largest call cost seen so
  far would pass it. The budget is `AGENT_REPLAY_RUN_BUDGET_USD`, default 1.50.
  `BudgetExceeded` is handled like `QuotaExhausted`: not retried, run error `budget_exceeded`,
  the suite stops and can resume, non-zero exit, gate ERROR.
- Errors are mapped once per provider to neutral classes: `RateLimitError`, `TransientError`,
  `PermanentModelError`, `QuotaExhausted`. For Gemini, input tokens are prompt plus tool-use
  prompt tokens and output tokens are candidate plus thinking tokens, since thinking is billed
  as output.

## Alternatives considered
- Relying on the provider's project spend cap alone: it lags, can overshoot, and does not stop
  one runaway command early. The in-process guard is cheap and testable.
- A pre-call cost estimate from a token count of the request: needs an extra counting call per
  request. Projecting from the largest call so far needs nothing extra.
- A provider abstraction library: a small protocol is enough and keeps the hash input under
  this project's control.

## Consequences
- The neutral types are the hash input, so switching provider or model invalidates the cache
  by design.
- The guard overshoots by at most one call above the largest seen, per process. It does not see
  other processes or the real bill.
- The CI job runs replay and score as separate processes, so it sets
  `AGENT_REPLAY_RUN_BUDGET_USD=0.75` on each to keep one job under the default budget.
