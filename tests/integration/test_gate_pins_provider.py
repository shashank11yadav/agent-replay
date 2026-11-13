"""regression-gate.yml pins provider and models from the baseline manifest.

Besides reading the YAML, these tests run the workflow's own `run:` scripts with bash in a
scratch checkout, with a `uv` shim (support.workflow) that runs the real CLI and can log
the AGENT_REPLAY_* environment of every call. The manifest-reading steps sit between the
baseline import step and the replay step.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from support.suites import base_env, build_suite, connect, export, ok, retag
from support.workflow import (
    P_CHECK,
    P_GATE,
    P_IMPORT,
    P_MIGRATE,
    P_REPLAY,
    P_SCORE,
    WF,
    Checkout,
    derive_steps,
    find_step,
    index_of,
    job,
    load_wf,
    run_of,
    steps,
)

BASELINE_ID_RE = re.compile(r"^[0-9a-f-]{36}$")


# --- static: YAML shape ------------------------------------------------------------


def _derive_text(data: dict[str, Any]) -> str:
    i, r = index_of(data, P_IMPORT), index_of(data, P_REPLAY)
    return "\n".join(run_of(s) for s in steps(data)[i:r])


def test_workflow_reads_validates_and_writes_provider_and_models_after_import() -> None:
    data = load_wf()
    text = _derive_text(data)
    assert "manifest.json" in text
    for key in ("provider", "agent_model", "judge_model"):
        assert key in text, key
    for f in ("gate-ids/provider", "gate-ids/model", "gate-ids/judge_model"):
        assert f in text, f
    assert "[A-Za-z0-9._-]{1,100}" in text, "the name pattern must be checked"
    assert "GITHUB_STEP_SUMMARY" in text, "a missing key adds a summary notice"
    # the reading happens after the import command itself
    imp = run_of(find_step(data, P_IMPORT))
    assert re.search(P_IMPORT, text)
    assert text.index("gate-ids/provider") > text.index("fixtures import") or (
        "gate-ids/provider" in imp
    )


def test_replay_and_score_set_provider_and_models_from_shell_variables() -> None:
    data = load_wf()
    for pat in (P_REPLAY, P_SCORE):
        step = find_step(data, pat)
        run = run_of(step)
        for var in (
            "AGENT_REPLAY_PROVIDER",
            "AGENT_REPLAY_MODEL",
            "AGENT_REPLAY_JUDGE_MODEL",
        ):
            assert var in run, f"{pat}: {var} not set in the step script"
        for f in ("gate-ids/provider", "gate-ids/model", "gate-ids/judge_model"):
            assert f in run, f"{pat}: does not read {f}"
        assert "${{" not in run, "no expression interpolation in the script"
        for k, v in (step.get("env") or {}).items():
            if k in (
                "AGENT_REPLAY_PROVIDER",
                "AGENT_REPLAY_MODEL",
                "AGENT_REPLAY_JUDGE_MODEL",
            ):
                raise AssertionError(f"{pat}: {k} set through step env: {v!r}")
    text = WF.read_text()
    assert "GITHUB_ENV" not in text, (
        "values go through the step's shell, not GITHUB_ENV"
    )
    assert "AGENT_REPLAY_PROVIDER" not in (job(data).get("env") or {})
    assert "AGENT_REPLAY_MODEL" not in (job(data).get("env") or {})


def test_manifest_values_never_pass_through_expressions() -> None:
    data = load_wf()
    for s in steps(data):
        run = run_of(s)
        if "gate-ids/" in run or "manifest" in run:
            assert not re.search(r"\$\{\{[^}]*(steps|fromJSON|hashFiles)", run), run
        for v in (s.get("env") or {}).values():
            assert not re.search(r"steps\.|fromJSON|hashFiles", str(v)), v


def test_job_env_empties_the_claude_cli_and_workflow_never_uses_it() -> None:
    data = load_wf()
    env = job(data).get("env") or {}
    assert "AGENT_REPLAY_CLAUDE_CLI" in env, sorted(env)
    assert env["AGENT_REPLAY_CLAUDE_CLI"] in ("", None), env["AGENT_REPLAY_CLAUDE_CLI"]
    for s in steps(data):
        v = (s.get("env") or {}).get("AGENT_REPLAY_CLAUDE_CLI", "")
        assert v in ("", None), "no step may point the CLI back on"
    text = WF.read_text()
    assert "claude -p" not in text
    assert "@anthropic-ai/claude-code" not in text


def test_third_party_actions_keep_pinned_commit_shas() -> None:
    uses = [str(s["uses"]) for s in steps(load_wf()) if "uses" in s]
    assert uses
    for u in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u.split(" ")[0]), u


# --- executed: the workflow's own scripts ------------------------------------------


@pytest.fixture(scope="module")
def real_baseline(mod_source_db_url: str, tmp_path_factory: Any) -> dict[str, Any]:
    """Scripted baseline exported as-is (provider 'scripted', default models)."""
    tmp = tmp_path_factory.mktemp("real_baseline")
    sid, tasks = build_suite(mod_source_db_url, tmp)
    out = tmp / "baseline"
    ok(export(mod_source_db_url, sid, out))
    return {"sid": sid, "tasks": tasks, "dir": out}


@pytest.fixture(scope="module")
def custom_baseline(mod_source_db_url2: str, tmp_path_factory: Any) -> dict[str, Any]:
    """A suite whose cache entries are retagged prov.x / agent-m.1 / judge-m_2."""
    tmp = tmp_path_factory.mktemp("custom_baseline")
    sid, tasks = build_suite(mod_source_db_url2, tmp)
    with connect(mod_source_db_url2) as c:
        rows = c.execute(
            "SELECT payload->>'request_hash' FROM spans s JOIN runs r ON r.id = s.run_id "
            "WHERE r.suite_run_id = %s AND s.kind = 'model'",
            (sid,),
        ).fetchall()
        a = sorted({r[0] for r in rows if r[0]})
        j = [
            r[0]
            for r in c.execute(
                "SELECT detail->>'request_hash' FROM scores sc JOIN runs r "
                "ON r.id = sc.run_id WHERE r.suite_run_id = %s "
                "AND sc.check_name = 'judge'",
                (sid,),
            ).fetchall()
        ]
        assert a and j
        retag(c, a, "prov.x", "agent-m.1")
        retag(c, j, "prov.x", "judge-m_2")
    out = tmp / "baseline"
    ok(export(mod_source_db_url2, sid, out))
    m = json.loads((out / "manifest.json").read_text())
    assert m.get("provider") == "prov.x", m
    return {"sid": sid, "tasks": tasks, "dir": out}


def run_until_replay(co: Checkout) -> list[subprocess.CompletedProcess[str]]:
    """migrate, import (+ derive steps); stops at the first failing step."""
    data = load_wf()
    done = []
    for s in [find_step(data, P_MIGRATE), *derive_steps(data)]:
        p = co.run_step(s)
        done.append(p)
        if p.returncode != 0:
            break
    return done


def test_workflow_writes_gate_ids_files_and_sets_env_on_replay_and_score(
    custom_baseline: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = Checkout(tmp_path, custom_baseline, ci_db_url)
    for p in run_until_replay(co):
        assert p.returncode == 0, (p.stdout, p.stderr)
    gi = co.dir / "gate-ids"
    assert (gi / "provider").read_text().strip() == "prov.x"
    assert (gi / "model").read_text().strip() == "agent-m.1"
    assert (gi / "judge_model").read_text().strip() == "judge-m_2"
    assert BASELINE_ID_RE.match((gi / "baseline").read_text().strip())
    data = load_wf()
    extra = {"TEST_UV_STUB": "1"}
    ok(co.run_step(find_step(data, P_REPLAY), extra))
    ok(co.run_step(find_step(data, P_SCORE), extra))
    for prefix in ("agent-replay suite replay", "agent-replay suite score"):
        e = co.call_env(prefix)
        assert e.get("AGENT_REPLAY_PROVIDER") == "prov.x", (prefix, e)
        assert e.get("AGENT_REPLAY_MODEL") == "agent-m.1", (prefix, e)
        assert e.get("AGENT_REPLAY_JUDGE_MODEL") == "judge-m_2", (prefix, e)
        assert e.get("AGENT_REPLAY_CLAUDE_CLI") == "", (prefix, e)


@pytest.mark.parametrize("missing", ["provider", "agent_model", "judge_model"])
def test_missing_manifest_key_writes_no_file_adds_notice_and_sets_no_variable(
    missing: str, custom_baseline: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = Checkout(tmp_path, custom_baseline, ci_db_url)
    co.edit_manifest(**{missing: None})
    for p in run_until_replay(co):
        assert p.returncode == 0, (p.stdout, p.stderr)
    fname = {
        "provider": "provider",
        "agent_model": "model",
        "judge_model": "judge_model",
    }[missing]
    var = {
        "provider": "AGENT_REPLAY_PROVIDER",
        "agent_model": "AGENT_REPLAY_MODEL",
        "judge_model": "AGENT_REPLAY_JUDGE_MODEL",
    }[missing]
    gi = co.dir / "gate-ids"
    assert not (gi / fname).exists()
    notice = co.summary.read_text()
    assert missing in notice, f"summary notice should name {missing!r}: {notice!r}"
    present = {"provider": "prov.x", "model": "agent-m.1", "judge_model": "judge-m_2"}
    for f, val in present.items():
        if f != fname:
            assert (gi / f).read_text().strip() == val
    ok(co.run_step(find_step(load_wf(), P_REPLAY), {"TEST_UV_STUB": "1"}))
    e = co.call_env("agent-replay suite replay")
    assert var not in e, f"{var} must stay unset without its file: {e}"


BAD = [
    ("provider", "a b"),
    ("provider", "x;touch pwned"),
    ("provider", "$(touch pwned)"),
    ("provider", "`touch pwned`"),
    ("provider", "good\ntouch pwned"),
    ("agent_model", "a" * 101),
    ("agent_model", "a/b"),
    ("judge_model", "x$(touch pwned)"),
    ("judge_model", "good\nbad"),
]


@pytest.mark.parametrize(("key", "value"), BAD)
def test_invalid_manifest_value_fails_the_job_with_a_clear_error_and_runs_nothing(
    key: str,
    value: str,
    custom_baseline: dict[str, Any],
    ci_db_url: str,
    tmp_path: Path,
) -> None:
    co = Checkout(tmp_path, custom_baseline, ci_db_url)
    co.edit_manifest(**{key: value})
    done = run_until_replay(co)
    failed = [p for p in done if p.returncode != 0]
    assert failed, "no step failed on an invalid manifest value"
    msg = (failed[-1].stderr + failed[-1].stdout).lower()
    assert key in msg or key.split("_")[0] in msg, msg
    fname = {"provider": "provider", "agent_model": "model"}.get(key, key)
    assert not (co.dir / "gate-ids" / fname).exists()
    assert not (co.dir / "pwned").exists(), "manifest text was executed by the shell"
    assert not [a for a, _ in co.calls() if a.startswith("agent-replay suite")]


# --- executed: whole sequence, line 8 ----------------------------------------------


def test_command_sequence_with_provider_unset_gets_pass_from_manifest_values(
    real_baseline: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    """Same sequence as the offline end-to-end test in test_regression_gate.py, but run through the workflow's own scripts
    with AGENT_REPLAY_PROVIDER unset and an empty scripted script: any live call raises,
    so PASS needs every request to hit the cache, which needs provider and models from
    the manifest (the request hash includes both)."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    base = real_baseline
    co = Checkout(tmp_path, base, ci_db_url)
    data = load_wf()
    seq = [
        find_step(data, P_MIGRATE),
        *derive_steps(data),
        find_step(data, P_REPLAY),
        find_step(data, P_SCORE),
        find_step(data, P_GATE),
        find_step(data, P_CHECK),
    ]
    extra = {"AGENT_REPLAY_SCRIPT_PATH": str(empty)}
    assert "AGENT_REPLAY_PROVIDER" not in base_env(ci_db_url, extra)
    for s in seq:
        p = co.run_step(s, extra)
        assert p.returncode == 0, (run_of(s), p.stdout, p.stderr)
    verdict = json.loads((co.dir / "gate-out" / "verdict.json").read_text())
    assert verdict["verdict"] == "PASS", verdict
    assert (co.dir / "gate-ids" / "provider").read_text().strip() == "scripted"
    for a, e in co.calls():
        if a.startswith("agent-replay suite replay"):
            assert e.get("AGENT_REPLAY_PROVIDER") == "scripted", e


def test_without_the_manifest_values_the_same_sequence_does_not_pass(
    real_baseline: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    """Control for the PASS test: with the keys removed the default provider is used, the
    requests miss the cache and the verdict is not PASS (so PASS above is earned)."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    co = Checkout(tmp_path, real_baseline, ci_db_url)
    co.edit_manifest(provider=None, agent_model=None, judge_model=None)
    data = load_wf()
    extra = {"AGENT_REPLAY_SCRIPT_PATH": str(empty)}
    for s in [find_step(data, P_MIGRATE), *derive_steps(data)]:
        ok(co.run_step(s, extra))
    co.run_step(find_step(data, P_REPLAY), extra)
    co.run_step(find_step(data, P_SCORE), extra)
    co.run_step(find_step(data, P_GATE), extra)
    verdict_file = co.dir / "gate-out" / "verdict.json"
    if verdict_file.exists():
        assert json.loads(verdict_file.read_text())["verdict"] != "PASS"
