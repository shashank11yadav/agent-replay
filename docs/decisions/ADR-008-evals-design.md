# ADR-008: Evals: synthetic order-support domain, prompt-strategy comparison

- Date: 2026-11-02
- Status: accepted

## Context
The gate needs a task set with labels it can trust: 30 scripted tasks, deterministic checks plus
an LLM judge, a baseline, at least two other approaches to compare, and a deliberate prompt
regression to show the gate failing. Every live suite costs subscription usage or money, so the
set stays small. Measured on the `claude_code` provider, one suite with its judge calls is about
$0.26 notional at API list prices (`evals/results/approaches.md`).

## Decision
- Domain: a synthetic order-support agent with three deterministic tools over JSON files in
  `src/agent_replay/demo/data/`: `get_order(order_id)`, `search_products(query)` and
  `convert_currency(amount, from_currency, to_currency)` (fixed rates, 2-decimal rounding).
  Boring on purpose: every right answer is computable, so deterministic checks carry most of the
  weight, and the repo holds no personal data.
- 30 tasks: 10 single-tool lookups, 10 multi-tool tasks, 5 search or compare tasks, 5 not-found
  tasks.
- Variants (system prompts, same model and parameters): `structured` is the baseline and the
  shipped prompt; `minimal` and `plan_act` are the approaches compared against it;
  `structured_regressed` is the deliberate regression.
- A dataset test recomputes every expected value from the data through the tool functions.
- Fault replays: all 30 tasks replayed from the baseline under each of `tool_timeout:*`,
  `malformed_tool_output:*` and `rate_limit:2`. Reported, not gated.

## Alternatives considered
- Public benchmarks (tau-bench style tasks): higher cost per suite, licence checks, and labels
  that cannot be recomputed from local data.
- Comparing two models instead of two prompts: the question a prompt PR asks is "did this prompt
  change hurt", and a model comparison would double the recording cost.

## Consequences
- Results show prompt effects on one small model, not general agent quality.
- With 30 tasks one flip is 3.3 pp, and two flips (6.7 pp) exceed the 5 pp threshold, so the
  thresholds are set with that granularity in mind.
