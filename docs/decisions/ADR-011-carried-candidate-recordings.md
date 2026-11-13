# ADR-011: PRs that change model requests carry their candidate recordings

- Date: 2026-11-12
- Status: accepted. Amends ADR-005 and ADR-010.

## Context
The committed baseline is recorded with `claude_code`, the Claude Code CLI on a Max
subscription login (ADR-010). That login is for local use by the account owner only and never
runs in CI ([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)). The
regression-gate workflow (`.github/workflows/regression-gate.yml`) therefore replays a
`claude_code` baseline cache-only. Any PR that changes a model request, such as a prompt edit,
misses the cache and gets SKIPPED, never FAIL, so the gate says nothing on exactly the PRs it
exists for.

Facts this rests on, from the code as built:
- `fixtures import` runs in one transaction. Identical existing rows are a no-op; a differing row
  raises `ImportConflict` and nothing is imported.
- Import re-hashes cache entries from their stored requests, so an entry can only answer the
  request it contains.
- The workflow takes the baseline fixtures, the task file and the gate thresholds from the PR base
  commit, and reads provider and models from the baseline manifest through files in `gate-ids/`.

## Decision
1. A PR that changes model requests may carry a fixtures export of its candidate in
   `evals/candidate/`. The candidate is a replay of the baseline suite with the PR's variant,
   recorded locally with the baseline's provider and models and scored. It uses the same format
   as `evals/baseline/`.
2. The gate job imports `evals/candidate/` after the baseline when
   `evals/candidate/manifest.json` exists. The directory comes from the PR head only and is never
   copied from the base commit. The job then replays and scores cache-only as before, so the
   verdict is the one the recorded evidence gives.
3. Refusals, each failing the job with a clear error before anything is imported:
   - a symlink at `evals/candidate`, at its manifest or anywhere inside it;
   - `provider`, `agent_model` or `judge_model` differing from the baseline manifest (missing
     counts as empty), or a non-empty value failing `^[A-Za-z0-9._-]{1,100}$`;
   - `suite_run.source_suite_run_id` not equal to the baseline's `suite_run_id`;
   - and, at import, any row that clashes with a baseline row (`ImportConflict`). Carried
     recordings can add cache entries but never override baseline ones.
4. Every use of carried recordings adds a notice to the step summary: CI checked the recorded
   evidence and did not re-run the model, so review `evals/candidate/` by hand or run the gate
   locally with a live provider before trusting a PASS.
5. After merge: while `evals/baseline/` is unchanged, a merged candidate directory is extra cache
   rows, each reviewed with its PR. The re-baselining PR deletes it. A stale directory left
   behind fails the source check with a message saying to delete or re-record it.

## Alternatives considered
- A local-only FAIL: CI shows SKIPPED for every request-changing PR, and the real verdict
  exists only on the machine of whoever ran the gate locally. Rejected.
- A second baseline recorded with `gemini`, with the Gemini key as a CI secret. CI could call
  a live model on misses. Rejected: two baselines to keep in sync, and the committed numbers would
  no longer come from one provider. It also brings real per-token spend and nondeterministic live
  calls into every prompt PR, needs a secret that fork PRs never get (so forks would still be
  SKIPPED), and a Gemini verdict would not say anything about the `claude_code` agent being
  shipped.
- Running the Claude Code CLI in CI on the subscription: not allowed (legal page, ADR-010).
- Letting carried recordings override baseline cache entries (upsert). Rejected: a PR could
  then rewrite the evidence it is compared against.
- Taking `evals/candidate/` from the base commit too: meaningless (a base-commit candidate
  belongs to an earlier PR) and it would hide the PR's own evidence.

## Consequences
- The verdict on a PR with carried recordings is only as trustworthy as those recordings. CI
  verifies consistency, not authenticity: the recordings must match the baseline's provider,
  models and suite and must not clash with it, but the response content cannot be checked without
  re-running the model. A PR author could fabricate good agent or judge responses and get a PASS.
  A fabricated FAIL only hurts its author. The base-commit baseline, task file and thresholds
  still stop the PR from changing what it is compared against or the labels it is scored
  with. A maintainer reviews `evals/candidate/` like code (`cache.jsonl` responses, `runs.jsonl`
  final answers) or reruns the gate locally with a live provider. As before, the gate is not a
  trust boundary on its own: under `pull_request` the PR also controls the workflow and `src/`,
  so CODEOWNERS and branch protection on `.github/`, `evals/` and `src/` are required.
- Contributors who change a prompt must record locally (live model access with the baseline's
  provider) before CI can give a verdict; without that the PR is SKIPPED as before.
- PR diffs grow by one fixture export (30 runs of JSONL), which is readable and reviewable.
- The deliberate regression in `evals/regression/`, carried as `evals/candidate/` with
  `evals/suite.json` set to `structured_regressed`, gives FAIL from the recorded evidence alone
  (`evals/RUNBOOK.md`).
