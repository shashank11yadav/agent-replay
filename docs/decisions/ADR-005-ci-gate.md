# ADR-005: CI gate: committed fixtures, pull_request only

- Date: 2026-11-02
- Status: accepted. Amended by ADR-011 (carried candidate recordings).

## Context
The gate runs PR code, so it must not run with write access or secrets an untrusted PR could
read. On `pull_request`, fork PRs get no secrets and a read-only `GITHUB_TOKEN`, and a missing
secret is an empty string. `pull_request_target` runs with write access and must not run PR
code. `$GITHUB_STEP_SUMMARY` takes Markdown up to 1 MiB, and
`gh pr comment --edit-last --create-if-none` keeps one comment per PR up to date. Sources:
[events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
[workflow commands](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands),
[gh pr comment](https://cli.github.com/manual/gh_pr_comment).

## Decision
- Baseline: one suite run of the shipped variant (`evals/suite.json`), exported to
  `evals/baseline/` (runs, spans, scores and cache as JSONL, plus a manifest) and committed.
  Re-baselining is a deliberate PR that re-records and re-exports.
- Workflow `regression-gate.yml` on `pull_request` only. The `gate` job runs the PR's code with
  `contents: read` and no secrets. A separate `comment` job, which runs no code from the PR, is
  the only one with `pull-requests: write`.
- Baseline fixtures, task file and thresholds come from the PR base commit, so a PR cannot
  change what it is compared against or how it is scored.
- Gate job steps: migrate, `fixtures verify` on the baseline, import fixtures, `suite replay`
  with the PR's variant, `suite score`, `gate` (exits 0 once a report exists; writes report.md,
  verdict.json and the step summary), build the comment (notices above the report) and upload it
  as an artifact, `gate-check` (exit 1 on FAIL or ERROR). Steps after replay use
  `if: ${{ !cancelled() }}` rather than `always()`, which also runs on cancel, so an interrupted
  replay still publishes an ERROR report.
- Verdicts: PASS, FAIL, SKIPPED (a needed live call had no provider) and ERROR (a run stopped on
  a rate limit, a transient provider error, quota or the run budget; a judge call that failed at
  the provider; a candidate whose tasks differ from the baseline's; an incomplete candidate or
  baseline; or a baseline with duplicate tasks, skipped runs or stopped runs). ERROR is distinct
  from SKIPPED so a broken run never looks like a missing key.
- The CI job has the same run budget guard as any command (ADR-003); it normally spends nothing
  because every unchanged request is served from the committed fixtures.
- Fork PRs and missing keys: if every request hits the cache the gate runs fully; any needed live
  call makes the verdict SKIPPED with a notice and exit 0.
- Comment job guard: `github.event.pull_request.head.repo.full_name == github.repository`, plus
  `gh pr comment ... || echo "<notice>"`. The guard comes from a community answer rather than
  the reference docs, so the `|| echo` fallback makes sure a wrong guard or a read-only token can
  never fail the run. The step summary always carries the report.
- Third-party actions are pinned to commit SHAs; Dependabot proposes updates.

## Alternatives considered
- `pull_request_target` to comment on forks: runs untrusted code with write access. Rejected.
- A live suite on every PR: cost, quota and nondeterminism.
- Committing a Postgres dump instead of JSONL: binary, and diffs cannot be reviewed.

## Consequences
- Fixture size grows with the suite (30 runs of small JSONL), and reviewers can read fixture
  diffs.
- The gate is not a trust boundary on its own: under `pull_request` the PR controls the
  workflow and `src/`, so CODEOWNERS and branch protection on `.github/`, `evals/` and `src/`
  are needed before it is a required check.
