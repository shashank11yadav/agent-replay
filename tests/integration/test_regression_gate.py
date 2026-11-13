"""regression-gate.yml shape, plus an offline run of the same command sequence
against fixtures exported from a scripted-provider suite.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from support import ROOT, scrubbed_env
from support.workflow import (
    COMMENT_JOB,
    P_BUILD_COMMENT,
    P_CHECK,
    P_COMMENT,
    P_GATE,
    P_IMPORT,
    P_MIGRATE,
    P_REPLAY,
    P_SCORE,
    P_VERIFY,
    WF,
    Checkout,
    find_step,
    index_of,
    job,
    load_wf,
    run_of,
    steps,
)

CANCELLED = "${{ !cancelled() }}"
GUARD = "github.event.pull_request.head.repo.full_name == github.repository"


# --- workflow shape -------------------------------------------------


def test_triggers_pull_request_only_never_pull_request_target() -> None:
    data = load_wf()
    on = data.get("on")
    trig = (
        {on}
        if isinstance(on, str)
        else set(on)
        if isinstance(on, (list, dict))
        else None
    )
    assert trig == {"pull_request"}, f"triggers: {on!r}"
    assert "pull_request_target" not in WF.read_text()


def test_only_the_comment_job_may_write_to_the_pull_request() -> None:
    data = load_wf()
    assert data.get("permissions") == {"contents": "read"}
    assert job(data).get("permissions") == {"contents": "read"}
    assert job(data, COMMENT_JOB).get("permissions") == {"pull-requests": "write"}
    # The comment job runs no PR code: no checkout, no uv, only the downloaded report.
    comment_steps = steps(data, COMMENT_JOB)
    assert not [s for s in comment_steps if "checkout" in str(s.get("uses", ""))]
    assert not [s for s in comment_steps if "uv run" in run_of(s)]


def test_postgres_service_and_database_url_on_localhost_5432() -> None:
    data = load_wf()
    j = job(data)
    services = j.get("services")
    assert isinstance(services, dict)
    pg = next(
        (v for v in services.values() if str(v.get("image")) == "postgres:18.6-alpine"),
        None,
    )
    assert pg is not None, f"no postgres:18.6-alpine service: {services}"
    assert pg.get("env", {}).get("POSTGRES_PASSWORD")
    opts = str(pg.get("options"))
    assert "--health-cmd" in opts and "pg_isready" in opts
    assert [str(p) for p in pg.get("ports", [])] == ["5432:5432"]

    cli_steps = [s for s in steps(data) if "agent-replay" in run_of(s)]
    assert len(cli_steps) >= 6
    for s in cli_steps:
        env = {
            **(data.get("env") or {}),
            **(j.get("env") or {}),
            **(s.get("env") or {}),
        }
        url = str(env.get("AGENT_REPLAY_DATABASE_URL", ""))
        assert url.startswith("postgresql://"), f"{run_of(s)!r}: url={url!r}"
        assert re.search(r"(localhost|127\.0\.0\.1):5432/", url), url


def test_no_model_key_or_other_secret_reaches_the_gate_job() -> None:
    data = load_wf()
    assert "GEMINI" not in WF.read_text().upper()
    for s in steps(data):
        for value in (s.get("env") or {}).values():
            assert "secrets." not in str(value), s
        assert "secrets." not in run_of(s)
    (holder,) = [
        s
        for s in steps(data, COMMENT_JOB)
        if "secrets." in json.dumps(s.get("env") or {})
    ]
    assert holder["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"


def test_replay_and_score_set_run_budget_075_via_env() -> None:
    data = load_wf()
    for pat in (P_REPLAY, P_SCORE):
        s = find_step(data, pat)
        val = (s.get("env") or {}).get("AGENT_REPLAY_RUN_BUDGET_USD")
        assert val is not None and str(val) == "0.75", f"{pat}: {val!r}"
        assert "RUN_BUDGET" not in run_of(s)


def test_steps_run_in_the_specified_order() -> None:
    data = load_wf()
    order = [
        index_of(data, p)
        for p in (
            P_MIGRATE,
            P_VERIFY,
            P_IMPORT,
            P_REPLAY,
            P_SCORE,
            P_GATE,
            P_BUILD_COMMENT,
            P_CHECK,
        )
    ]
    assert order == sorted(order) and len(set(order)) == 8, order
    assert job(data, COMMENT_JOB).get("needs") == "gate"
    assert re.search(
        r"uv run agent-replay gate-check gate-out/verdict\.json\s*$",
        run_of(find_step(data, P_CHECK)),
    )


def _verify_checkout(tmp_path: Path) -> Checkout:
    baseline = {
        "dir": ROOT / "evals" / "baseline",
        "tasks": ROOT / "evals" / "tasks" / "tasks.jsonl",
    }
    # The verify step reads only files, so the database URL is never used.
    return Checkout(tmp_path, baseline, "postgresql://127.0.0.1:1/unused")


def test_verify_step_passes_the_committed_baseline(tmp_path: Path) -> None:
    co = _verify_checkout(tmp_path)
    p = co.run_step(find_step(load_wf(), P_VERIFY))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "ok:" in p.stdout


def test_verify_step_stops_the_job_on_an_incomplete_baseline(tmp_path: Path) -> None:
    co = _verify_checkout(tmp_path)
    runs = co.dir / "evals" / "baseline" / "runs.jsonl"
    lines = runs.read_text().splitlines()
    first = json.loads(lines[0])
    first["status"] = "skipped"
    runs.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    p = co.run_step(find_step(load_wf(), P_VERIFY))
    assert p.returncode != 0
    assert "incomplete baseline: run skipped" in p.stdout


def test_verify_step_is_skipped_when_no_baseline_is_committed(tmp_path: Path) -> None:
    co = _verify_checkout(tmp_path)
    (co.dir / "gate-ids" / "skip").touch()
    p = co.run_step(find_step(load_wf(), P_VERIFY))
    assert p.returncode == 0, p.stderr
    assert co.calls() == []


def test_command_forms_replay_score_gate() -> None:
    data = load_wf()
    suite = json.loads((ROOT / "evals" / "suite.json").read_text())
    replay = run_of(find_step(data, P_REPLAY))
    assert "--source" in replay
    m = re.search(r"--variant[ =](\S+)", replay)
    assert m, replay
    variant = m.group(1).strip("'\"")
    # Literal variant must match evals/suite.json; otherwise it must be read from it.
    assert variant == suite["variant"] or "evals/suite.json" in replay, replay
    assert "--tasks" in replay and "evals/tasks/tasks.jsonl" in replay
    score = run_of(find_step(data, P_SCORE))
    assert "--suite-run" in score
    assert "--tasks" in score and "evals/tasks/tasks.jsonl" in score
    gate = run_of(find_step(data, P_GATE))
    for flag in ("--baseline", "--candidate"):
        assert flag in gate
    assert "--thresholds evals/gate.json" in gate
    assert "--out-dir gate-out" in gate


def test_if_conditions_use_not_cancelled_and_never_always() -> None:
    data = load_wf()
    assert "always()" not in WF.read_text()
    for pat in (P_SCORE, P_GATE, P_BUILD_COMMENT, P_CHECK):
        assert find_step(data, pat).get("if") == CANCELLED, pat
    comment_if = str(job(data, COMMENT_JOB).get("if"))
    assert "!cancelled()" in comment_if
    assert GUARD in comment_if


def test_pr_comment_step_command_token_and_number_via_env() -> None:
    step = find_step(load_wf(), P_COMMENT, COMMENT_JOB)
    run = run_of(step)
    expected = (
        'gh pr comment "$PR_NUMBER" --edit-last '
        "--create-if-none -F gate-out/comment.md || echo "
    )
    assert expected in run, run
    assert "${{" not in run
    tail = run.split("|| echo", 1)[1].strip()
    assert len(tail) > 2 and tail[0] in "\"'", f"missing notice text: {tail!r}"
    env = step.get("env") or {}
    assert env.get("GH_TOKEN") == "${{ secrets.GITHUB_TOKEN }}"
    assert env.get("PR_NUMBER") == "${{ github.event.pull_request.number }}"


def test_comment_puts_the_notices_above_the_report(tmp_path: Path) -> None:
    (tmp_path / "gate-ids").mkdir()
    (tmp_path / "gate-out").mkdir()
    (tmp_path / "gate-ids" / "notices").write_text("> Notice: carried recordings\n\n")
    (tmp_path / "gate-out" / "report.md").write_text("## Regression gate: PASS\n")
    build = find_step(load_wf(), P_BUILD_COMMENT)
    p = subprocess.run(
        ["bash", "-e", "-c", run_of(build)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    comment = (tmp_path / "gate-out" / "comment.md").read_text()
    assert comment == "> Notice: carried recordings\n\n## Regression gate: PASS\n"
    (tmp_path / "gate-ids" / "notices").unlink()
    subprocess.run(["bash", "-e", "-c", run_of(build)], cwd=tmp_path, check=True)
    assert (tmp_path / "gate-out" / "comment.md").read_text().startswith("## Regr")


def test_gate_step_publishes_summary_through_the_gate_command() -> None:
    gate = run_of(find_step(load_wf(), P_GATE))
    assert "GITHUB_STEP_SUMMARY" not in gate, "the gate command appends it itself"
    assert ">>" not in gate


# --- offline end-to-end ---------------------------------------------

N = 3


def _env(db_url: str, extra: dict[str, str]) -> dict[str, str]:
    env = scrubbed_env()
    env.pop("GITHUB_STEP_SUMMARY", None)
    env.update(
        AGENT_REPLAY_DATABASE_URL=db_url,
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_BACKOFF_BASE_S="0",
        AGENT_REPLAY_RUN_BUDGET_USD="0.75",
    )
    env.update(extra)
    return env


def cli(
    db_url: str, *args: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_replay.cli import main; sys.exit(main())",
            *args,
        ],
        env=_env(db_url, extra or {}),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


def ok(p: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert p.returncode == 0, (p.args[3:], p.stdout, p.stderr)
    return p


def test_same_command_sequence_without_api_key_on_exported_fixtures_gives_pass(
    source_db_url: str, ci_db_url: str, tmp_path: Path
) -> None:
    variant = json.loads((ROOT / "evals" / "suite.json").read_text())["variant"]
    thresholds = ROOT / "evals" / "gate.json"

    # Scripted-provider baseline suite (stands in for the committed baseline).
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"g-{i}",
                    "prompt": f"What is the status of order A1001 (case {i})?",
                    "category": "lookup",
                    "expected_text": ["shipped"],
                    "must_call_tools": ["get_order"],
                    "max_steps": 4,
                    "rubric": f"States that order A1001 has shipped (case {i}).",
                }
            )
            for i in range(1, N + 1)
        )
        + "\n"
    )
    pair = [
        {
            "tool_calls": [
                {"id": "x", "name": "get_order", "args": {"order_id": "A1001"}}
            ]
        },
        {"text": "Order A1001 has shipped."},
    ]
    agent_script = tmp_path / "agent.json"
    agent_script.write_text(json.dumps(pair * (N + 3)))
    judge_script = tmp_path / "judge.json"
    judge_script.write_text(
        json.dumps([{"text": '{"pass": true, "reason": "ok"}'}] * (2 * N))
    )
    scripted = {"AGENT_REPLAY_PROVIDER": "scripted"}

    rec_args = f"suite record --tasks {tasks} --variant {variant} --label base"
    rec = ok(
        cli(
            source_db_url,
            *rec_args.split(),
            extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(agent_script)},
        )
    )
    base_sid = last_line(rec)
    score_args = f"suite score --suite-run {base_sid} --tasks {tasks}"
    ok(
        cli(
            source_db_url,
            *score_args.split(),
            extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(judge_script)},
        )
    )
    fixtures = tmp_path / "baseline"
    export_args = f"fixtures export --suite-run {base_sid} --out {fixtures}"
    ok(cli(source_db_url, *export_args.split()))

    # The CI sequence on a fresh database with no API key. The request hash includes
    # the provider name, so scripted-recorded fixtures replay under the scripted
    # provider with an empty script: any live call would raise, only cache hits pass.
    empty_script = tmp_path / "empty.json"
    empty_script.write_text("[]")
    offline = {**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(empty_script)}
    ok(cli(ci_db_url, "db", "migrate"))
    ok(cli(ci_db_url, "fixtures", "import", str(fixtures)))
    manifest = json.loads((fixtures / "manifest.json").read_text())
    assert manifest["suite_run_id"] == base_sid
    replay_args = (
        f"suite replay --source {base_sid} --variant {variant} "
        f"--tasks {tasks} --label pr"
    )
    cand = last_line(ok(cli(ci_db_url, *replay_args.split(), extra=offline)))
    assert cand != base_sid
    cand_score = f"suite score --suite-run {cand} --tasks {tasks}"
    ok(cli(ci_db_url, *cand_score.split(), extra=offline))
    out = tmp_path / "gate-out"
    summary = tmp_path / "summary.md"
    gate_args = (
        f"gate --baseline {base_sid} --candidate {cand} "
        f"--thresholds {thresholds} --out-dir {out}"
    )
    ok(cli(ci_db_url, *gate_args.split(), extra={"GITHUB_STEP_SUMMARY": str(summary)}))
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"] == "PASS", (verdict, (out / "report.md").read_text())
    assert (out / "report.md").read_text().strip() in summary.read_text()
    check = ok(cli(ci_db_url, "gate-check", str(out / "verdict.json")))
    assert "PASS" in check.stdout
