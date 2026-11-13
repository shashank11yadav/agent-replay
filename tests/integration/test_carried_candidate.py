"""The regression gate imports carried candidate recordings (evals/candidate/).

Static tests read .github/workflows/regression-gate.yml. Executed tests run the workflow's
own `run:` scripts with bash in a scratch checkout (support.workflow.Checkout) against
throwaway databases. The new step sits between the baseline import step and the replay
step, and is the only step whose script mentions 'evals/candidate'.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from support import ROOT
from support.suites import (
    base_env,
    build_suite,
    cli,
    connect,
    export,
    last_line,
    ok,
)
from support.workflow import (
    COMMENT_JOB,
    P_CHECK,
    P_COMMENT,
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
    load_wf,
    run_of,
    steps,
)

KEYS = ["provider", "agent_model", "judge_model"]
N = 3
OTHER_UUID = "99999999-8888-7777-6666-555555555555"


# --- helpers -----------------------------------------------------------------------


def cand_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in steps(data) if "evals/candidate" in run_of(s)]


def cand_step(data: dict[str, Any]) -> dict[str, Any]:
    hits = cand_steps(data)
    assert len(hits) == 1, f"expected exactly one step handling evals/candidate: {hits}"
    return hits[0]


def edit_json(path: Path, fn: Callable[[dict[str, Any]], None]) -> None:
    m = json.loads(path.read_text())
    fn(m)
    path.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")


def counts(url: str) -> tuple[int, int]:
    with connect(url) as c:
        s = c.execute("SELECT count(*) FROM suite_runs").fetchone()[0]
        m = c.execute("SELECT count(*) FROM model_cache").fetchone()[0]
    return int(s), int(m)


def run_sequence(
    co: Checkout, extra_steps: list[dict[str, Any]] | None = None
) -> list[tuple[dict[str, Any], subprocess.CompletedProcess[str]]]:
    """migrate, then the derive steps (import, manifest read, new step); stops at the
    first failing step."""
    data = load_wf()
    done = []
    for s in [find_step(data, P_MIGRATE), *derive_steps(data), *(extra_steps or [])]:
        p = co.run_step(s)
        done.append((s, p))
        if p.returncode != 0:
            break
    return done


def assert_candidate_step_failed(
    done: list[tuple[dict[str, Any], subprocess.CompletedProcess[str]]],
) -> str:
    """The sequence stopped at the candidate step with a non-zero exit; returns its output."""
    assert done[-1][1].returncode != 0, "no step failed"
    failing, p = done[-1]
    assert "evals/candidate" in run_of(failing), (
        "the failing step is not the candidate step",
        run_of(failing),
        p.stderr,
    )
    for _, q in done[:-1]:
        assert q.returncode == 0
    return (p.stderr + p.stdout).lower()


@pytest.fixture(scope="module")
def world(mod_source_db_url: str, tmp_path_factory: Any) -> dict[str, Any]:
    """Scripted baseline (variant structured) and a carried candidate recorded from it with
    variant structured_regressed under a scripted live provider whose answers regress."""
    tmp = tmp_path_factory.mktemp("carried_world")
    src = mod_source_db_url
    sid, tasks = build_suite(src, tmp)
    baseline_dir = tmp / "baseline"
    ok(export(src, sid, baseline_dir))

    bad_agent = tmp / "bad_agent.json"
    bad_agent.write_text(
        json.dumps([{"text": "Sorry, I could not find that order."}] * (N + 3))
    )
    bad_judge = tmp / "bad_judge.json"
    bad_judge.write_text(
        json.dumps([{"text": '{"pass": false, "reason": "wrong"}'}] * (2 * N))
    )
    scripted = {"AGENT_REPLAY_PROVIDER": "scripted"}
    rep = ok(
        cli(
            src,
            "suite",
            "replay",
            "--source",
            sid,
            "--variant",
            "structured_regressed",
            "--tasks",
            str(tasks),
            "--label",
            "pr",
            extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(bad_agent)},
        )
    )
    cand = last_line(rep)
    ok(
        cli(
            src,
            "suite",
            "score",
            "--suite-run",
            cand,
            "--tasks",
            str(tasks),
            extra={**scripted, "AGENT_REPLAY_SCRIPT_PATH": str(bad_judge)},
        )
    )
    cand_dir = tmp / "candidate"
    ok(export(src, cand, cand_dir))
    direct_out = tmp / "direct-out"
    cli(
        src,
        "gate",
        "--baseline",
        sid,
        "--candidate",
        cand,
        "--thresholds",
        str(ROOT / "evals" / "gate.json"),
        "--out-dir",
        str(direct_out),
    )
    direct = json.loads((direct_out / "verdict.json").read_text())
    return {
        "sid": sid,
        "cand": cand,
        "tasks": tasks,
        "dir": baseline_dir,
        "cand_dir": cand_dir,
        "direct": direct,
    }


def make_checkout(
    world: dict[str, Any],
    ci_db_url: str,
    tmp: Path,
    candidate: bool = True,
    variant: str = "structured_regressed",
) -> Checkout:
    tmp.mkdir(parents=True, exist_ok=True)
    co = Checkout(tmp, world, ci_db_url)
    (co.dir / "evals" / "suite.json").write_text(
        json.dumps({"variant": variant}) + "\n"
    )
    if candidate:
        shutil.copytree(world["cand_dir"], co.dir / "evals" / "candidate")
    return co


# --- static: workflow shape ---------------------------------------------------------


def test_one_step_after_manifest_read_and_before_replay_imports_the_candidate() -> None:
    data = load_wf()
    step = cand_step(data)
    idx = steps(data).index(step)
    importers = [
        s for s in steps(data) if "fixtures import evals/candidate" in run_of(s)
    ]
    assert importers == [step]
    assert index_of(data, P_IMPORT) < idx < index_of(data, P_REPLAY)
    readers = [
        i
        for i, s in enumerate(steps(data))
        if "evals/candidate" not in run_of(s)
        and "gate-ids/judge_model" in run_of(s)
        and "agent_model" in run_of(s)
        and i < index_of(data, P_REPLAY)
    ]
    assert readers and max(readers) < idx, (readers, idx)
    run = run_of(step)
    assert "gate-ids/skip" in run, "the step honours gate-ids/skip"
    assert "exit 0" in run
    assert "uv run agent-replay fixtures import evals/candidate" in run
    assert run.index("uv run agent-replay fixtures import evals/candidate") > max(
        run.find(k) for k in ("agent_model", "judge_model", "source_suite_run_id")
    ), "the import comes after the checks"


def test_candidate_directory_is_never_taken_from_the_base_commit() -> None:
    data = load_wf()
    copiers = [
        s for s in steps(data) if ".gate-base" in run_of(s) and "cp -" in run_of(s)
    ]
    assert len(copiers) == 1, copiers
    assert "evals/candidate" not in run_of(copiers[0])
    for s in steps(data):
        if s is cand_step(data):
            continue
        assert "evals/candidate" not in run_of(s) or ".gate-base" not in run_of(s)
    cs = run_of(cand_step(data))
    assert ".gate-base" not in cs


def test_new_step_has_no_expressions_and_no_secrets() -> None:
    step = cand_step(load_wf())
    assert "${{" not in run_of(step)
    env = step.get("env") or {}
    for k, v in env.items():
        assert "secrets." not in str(v), (k, v)
        assert "${{" not in str(v), (k, v)
        assert "KEY" not in k.upper() and "TOKEN" not in k.upper(), k
    assert "GEMINI" not in run_of(step).upper()
    assert "GITHUB_ENV" not in run_of(step)


def test_new_step_contains_none_of_the_baseline_lookup_commands() -> None:
    run = run_of(cand_step(load_wf()))
    for text in (
        "uv run agent-replay db migrate",
        "uv run agent-replay fixtures import evals/baseline",
        "uv run agent-replay suite replay",
        "uv run agent-replay suite score",
        "uv run agent-replay gate",
        "gh pr comment",
    ):
        assert text not in run, text
    data = load_wf()
    for pat in (P_MIGRATE, P_IMPORT, P_REPLAY, P_SCORE, P_GATE, P_CHECK):
        find_step(data, pat)
    find_step(data, P_COMMENT, COMMENT_JOB)


def test_header_comment_describes_carried_recordings_trust_caveat_and_post_merge_rule() -> (
    None
):
    text = WF.read_text()
    header = text.split("\non:\n", 1)[0]
    flat = " ".join(
        ln.strip().lstrip("#").strip()
        for ln in header.splitlines()
        if ln.strip().startswith("#")
    ).lower()
    flat = re.sub(r"\s+", " ", flat)
    for needle in (
        "evals/candidate",
        "carried",
        "fabricated",
        "base commit",
        "live provider",
        "delete",
        "re-baselin",
    ):
        assert needle in flat, f"header comment lacks {needle!r}"
    assert re.search(r"(does not|did not|not) re-run the model", flat), flat
    assert "review" in flat and "locally" in flat
    assert re.search(r"merge", flat)
    assert re.search(r"extra cache rows", flat)
    assert "thresholds" in flat and "tasks" in flat


def test_pins_permissions_and_forbidden_strings_are_kept() -> None:
    data = load_wf()
    every = steps(data) + steps(data, COMMENT_JOB)
    uses = [str(s["uses"]) for s in every if "uses" in s]
    assert len(uses) == 6
    for u in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u.split(" ")[0]), u
    assert data.get("permissions") == {"contents": "read"}
    text = WF.read_text()
    for bad in (
        "pull_request_target",
        "always()",
        "claude -p",
        "@anthropic-ai/claude-code",
    ):
        assert bad not in text, bad


# --- executed: no candidate / skip / base tree only -----------------------------------


def test_gate_ids_skip_exits_zero_with_a_notice_and_does_nothing_else(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)  # a valid candidate is present
    data = load_wf()
    ok(co.run_step(find_step(data, P_MIGRATE)))
    (co.dir / "gate-ids" / "skip").write_text("")
    p = co.run_step(cand_step(data))
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "notice" in (p.stdout + p.stderr).lower()
    assert not (co.dir / "gate-ids" / "carried").exists()
    assert counts(ci_db_url) == (0, 0)
    assert not any("fixtures import" in a for a, _ in co.calls())


def test_no_candidate_path_exits_zero_writes_nothing_and_leaves_the_summary_alone(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    done = run_sequence(co)
    for s, p in done:
        assert p.returncode == 0, (run_of(s), p.stdout, p.stderr)
    assert any(s == cand_step(load_wf()) for s, _ in done), "the new step did not run"
    assert not (co.dir / "gate-ids" / "carried").exists()
    assert co.summary.read_text().strip() == ""
    assert counts(ci_db_url)[0] == 1


def test_without_evals_candidate_a_request_changing_variant_is_skipped_and_gate_check_0(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    data = load_wf()
    cand_step(data)
    seq = [
        find_step(data, P_MIGRATE),
        *derive_steps(data),
        find_step(data, P_REPLAY),
        find_step(data, P_SCORE),
        find_step(data, P_GATE),
    ]
    assert "AGENT_REPLAY_PROVIDER" not in base_env(ci_db_url)
    assert "AGENT_REPLAY_SCRIPT_PATH" not in base_env(ci_db_url)
    for s in seq:
        co.run_step(s)
    verdict = json.loads((co.dir / "gate-out" / "verdict.json").read_text())
    assert verdict["verdict"] == "SKIPPED", verdict
    chk = co.run_step(find_step(data, P_CHECK))
    assert chk.returncode == 0, (chk.stdout, chk.stderr)


def test_candidate_only_in_the_base_tree_is_not_imported(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    data = load_wf()
    base = co.dir / ".gate-base"
    (base / "evals" / "tasks").mkdir(parents=True)
    shutil.copytree(world["dir"], base / "evals" / "baseline")
    shutil.copy(world["tasks"], base / "evals" / "tasks" / "tasks.jsonl")
    shutil.copy(ROOT / "evals" / "gate.json", base / "evals" / "gate.json")
    shutil.copytree(world["cand_dir"], base / "evals" / "candidate")
    copier = next(
        s for s in steps(data) if ".gate-base" in run_of(s) and "cp -" in run_of(s)
    )
    p = co.run_step(copier)
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert not (co.dir / "evals" / "candidate").exists()
    for s, q in run_sequence(co):
        assert q.returncode == 0, (run_of(s), q.stdout, q.stderr)
    assert counts(ci_db_url)[0] == 1, "an extra suite run was imported"
    assert not (co.dir / "gate-ids" / "carried").exists()
    assert "evals/candidate" not in co.summary.read_text()


def test_a_base_without_thresholds_leaves_a_notice_for_the_pr_comment(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    base = co.dir / ".gate-base"
    (base / "evals" / "tasks").mkdir(parents=True)
    shutil.copytree(world["dir"], base / "evals" / "baseline")
    shutil.copy(world["tasks"], base / "evals" / "tasks" / "tasks.jsonl")
    data = load_wf()
    copier = next(
        s for s in steps(data) if ".gate-base" in run_of(s) and "cp -" in run_of(s)
    )
    p = co.run_step(copier)
    assert p.returncode == 0, (p.stdout, p.stderr)
    notices = (co.dir / "gate-ids" / "notices").read_text()
    assert "no evals/gate.json" in notices
    assert notices.startswith("> Notice:")


# --- executed: refusals ---------------------------------------------------------------


def test_symlinked_evals_candidate_is_refused(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    real = tmp_path / "real_candidate"
    shutil.copytree(world["cand_dir"], real)
    os.symlink(real, co.dir / "evals" / "candidate")
    msg = assert_candidate_step_failed(run_sequence(co))
    assert "evals/candidate" in msg
    assert counts(ci_db_url)[0] == 1
    assert not (co.dir / "gate-ids" / "carried").exists()


def test_symlinked_manifest_is_refused(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    real = tmp_path / "manifest_real.json"
    manifest = co.dir / "evals" / "candidate" / "manifest.json"
    shutil.move(manifest, real)
    os.symlink(real, manifest)
    msg = assert_candidate_step_failed(run_sequence(co))
    assert "evals/candidate" in msg
    assert counts(ci_db_url)[0] == 1


@pytest.mark.parametrize("name", ["cache.jsonl", "extra.txt", "sub"])
def test_any_symlink_inside_evals_candidate_is_refused(
    name: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    target = tmp_path / "outside"
    cdir = co.dir / "evals" / "candidate"
    if name == "cache.jsonl":
        shutil.copy(cdir / name, target)
        (cdir / name).unlink()
        os.symlink(target, cdir / name)
    elif name == "extra.txt":
        target.write_text("x")
        os.symlink(target, cdir / name)
    else:
        target.mkdir()
        os.symlink(target, cdir / name)
    msg = assert_candidate_step_failed(run_sequence(co))
    assert "evals/candidate" in msg
    assert counts(ci_db_url)[0] == 1
    assert not (co.dir / "gate-ids" / "carried").exists()


@pytest.mark.parametrize("shape", ["absent", "directory"])
def test_manifest_that_is_not_a_regular_file_fails_saying_it_is_missing(
    shape: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    manifest = co.dir / "evals" / "candidate" / "manifest.json"
    manifest.unlink()
    if shape == "directory":
        manifest.mkdir()
    msg = assert_candidate_step_failed(run_sequence(co))
    assert "manifest" in msg and "missing" in msg, msg
    assert counts(ci_db_url)[0] == 1
    assert not (co.dir / "gate-ids" / "carried").exists()


# --- executed: manifest checks --------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_differing_provider_or_model_fails_naming_the_key_before_any_import(
    key: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    cm = co.dir / "evals" / "candidate" / "manifest.json"
    current = json.loads(cm.read_text()).get(key)
    edit_json(cm, lambda m: m.__setitem__(key, (current or "") + "-x"))
    done = run_sequence(co)
    msg = assert_candidate_step_failed(done)
    assert key in msg, msg
    assert counts(ci_db_url)[0] == 1
    assert not (co.dir / "gate-ids" / "carried").exists()
    assert not any(
        a.startswith("agent-replay fixtures import evals/candidate")
        for a, _ in co.calls()
    )


@pytest.mark.parametrize("key", KEYS)
def test_candidate_missing_a_key_the_baseline_has_fails_naming_the_key(
    key: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    cm = co.dir / "evals" / "candidate" / "manifest.json"
    assert json.loads(cm.read_text()).get(key), "test data: baseline pins the key"
    edit_json(cm, lambda m: m.pop(key, None))
    msg = assert_candidate_step_failed(run_sequence(co))
    assert key in msg, msg
    assert counts(ci_db_url)[0] == 1


@pytest.mark.parametrize("key", KEYS)
def test_baseline_missing_a_key_the_candidate_has_fails_naming_the_key(
    key: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    co.edit_manifest(**{key: None})
    msg = assert_candidate_step_failed(run_sequence(co))
    assert key in msg, msg
    assert counts(ci_db_url)[0] == 1


@pytest.mark.parametrize("key", KEYS)
def test_key_missing_on_both_sides_is_accepted(
    key: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    co.edit_manifest(**{key: None})
    edit_json(
        co.dir / "evals" / "candidate" / "manifest.json", lambda m: m.pop(key, None)
    )
    for s, p in run_sequence(co):
        assert p.returncode == 0, (run_of(s), p.stdout, p.stderr)
    assert (co.dir / "gate-ids" / "carried").exists()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    "value",
    ["a b", "x;touch pwned", "$(touch pwned)", "`touch pwned`", "a" * 101, "a/b"],
)
def test_candidate_value_failing_the_name_pattern_fails_and_is_never_executed(
    key: str, value: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    edit_json(
        co.dir / "evals" / "candidate" / "manifest.json",
        lambda m: m.__setitem__(key, value),
    )
    msg = assert_candidate_step_failed(run_sequence(co))
    assert key in msg, msg
    assert not (co.dir / "pwned").exists(), "manifest text was executed by the shell"
    assert counts(ci_db_url)[0] == 1


@pytest.mark.parametrize("bad", ["other_uuid", "null", "absent"])
def test_wrong_or_null_source_suite_run_id_fails_before_any_import(
    bad: str, world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    cm = co.dir / "evals" / "candidate" / "manifest.json"
    assert (
        json.loads(cm.read_text())["suite_run"]["source_suite_run_id"] == world["sid"]
    )

    def change(m: dict[str, Any]) -> None:
        if bad == "other_uuid":
            m["suite_run"]["source_suite_run_id"] = OTHER_UUID
        elif bad == "null":
            m["suite_run"]["source_suite_run_id"] = None
        else:
            m["suite_run"].pop("source_suite_run_id", None)

    edit_json(cm, change)
    msg = assert_candidate_step_failed(run_sequence(co))
    assert "evals/candidate" in msg
    assert "another baseline" in msg, msg
    assert "delete" in msg and "re-record" in msg, msg
    assert counts(ci_db_url)[0] == 1
    assert not (co.dir / "gate-ids" / "carried").exists()


# --- executed: import -----------------------------------------------------------------


def test_conflicting_cache_entry_fails_imports_nothing_and_keeps_the_baseline_entry(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path)
    cache_file = co.dir / "evals" / "candidate" / "cache.jsonl"
    base_rows = [
        json.loads(ln)
        for ln in (world["dir"] / "cache.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    victim = base_rows[0]
    fabricated = {
        **victim,
        "response": {"text": "FABRICATED: order A1001 has shipped."},
    }
    assert fabricated["response"] != victim["response"]
    with cache_file.open("a") as fh:
        fh.write(json.dumps(fabricated, sort_keys=True) + "\n")

    done = run_sequence(co)
    assert_candidate_step_failed(done)
    suites, cache_rows = counts(ci_db_url)
    assert suites == 1, "a candidate suite run was imported"
    with connect(ci_db_url) as c:
        row = c.execute(
            "SELECT response FROM model_cache WHERE request_hash = %s",
            (victim["request_hash"],),
        ).fetchone()
    assert row is not None and row[0] == victim["response"]
    # nothing at all from the candidate: the cache holds exactly the baseline's entries
    assert cache_rows == len(base_rows)
    assert not (co.dir / "gate-ids" / "carried").exists()


def test_successful_import_writes_carried_and_announces_it_in_the_summary(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    probe = make_checkout(world, ci_db_url, tmp_path / "probe", candidate=False)
    ok(probe.run_step(find_step(load_wf(), P_MIGRATE)))
    ok(probe.run_step(find_step(load_wf(), P_IMPORT)))
    before = counts(ci_db_url)
    assert before[0] == 1
    co2 = make_checkout(world, ci_db_url, tmp_path / "second")
    for s, p in run_sequence(co2):
        assert p.returncode == 0, (run_of(s), p.stdout, p.stderr)
    assert (co2.dir / "gate-ids" / "carried").exists()
    suites, cache_rows = counts(ci_db_url)
    assert suites == 2
    assert cache_rows > before[1]
    notice = re.sub(r"\s+", " ", co2.summary.read_text().lower())
    for phrase in (
        "evals/candidate/",
        "did not re-run the model",
        "review evals/candidate/ by hand",
        "or run the gate locally with a live provider",
    ):
        assert phrase in notice, (phrase, notice)
    assert any(
        a.startswith("agent-replay fixtures import evals/candidate")
        for a, _ in co2.calls()
    )


# --- executed: end to end -------------------------------------------------------------


def test_carried_candidate_gives_the_verdict_of_gate_run_directly_and_gate_check_fails(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    direct = world["direct"]
    assert direct["verdict"] == "FAIL", direct
    co = make_checkout(world, ci_db_url, tmp_path)
    data = load_wf()
    base = base_env(ci_db_url)
    assert "AGENT_REPLAY_PROVIDER" not in base
    assert "AGENT_REPLAY_SCRIPT_PATH" not in base
    assert not base.get("AGENT_REPLAY_GEMINI_API_KEY")
    seq = [
        find_step(data, P_MIGRATE),
        *derive_steps(data),
        find_step(data, P_REPLAY),
        find_step(data, P_SCORE),
        find_step(data, P_GATE),
    ]
    for s in seq:
        p = co.run_step(s)
        assert p.returncode == 0, (run_of(s), p.stdout, p.stderr)
    verdict = json.loads((co.dir / "gate-out" / "verdict.json").read_text())
    assert verdict["verdict"] == direct["verdict"] == "FAIL", (verdict, direct)
    assert (co.dir / "gate-ids" / "carried").exists()
    cand_id = (co.dir / "gate-ids" / "candidate").read_text().strip()
    with connect(ci_db_url) as c:
        statuses = [
            r[0]
            for r in c.execute(
                "SELECT status FROM runs WHERE suite_run_id = %s", (cand_id,)
            ).fetchall()
        ]
    assert len(statuses) == N, statuses
    assert not [s for s in statuses if "skip" in str(s).lower()], statuses
    chk = co.run_step(find_step(data, P_CHECK))
    assert chk.returncode == 1, (chk.stdout, chk.stderr)
    assert "carries candidate recordings" in co.summary.read_text().lower() or (
        "did not re-run the model" in co.summary.read_text().lower()
    )
    notices = (co.dir / "gate-ids" / "notices").read_text()
    assert "did not re-run the model" in notices


def test_same_setup_without_the_candidate_directory_is_skipped_and_gate_check_exits_zero(
    world: dict[str, Any], ci_db_url: str, tmp_path: Path
) -> None:
    co = make_checkout(world, ci_db_url, tmp_path, candidate=False)
    data = load_wf()
    cand_step(data)
    for s in [
        find_step(data, P_MIGRATE),
        *derive_steps(data),
        find_step(data, P_REPLAY),
        find_step(data, P_SCORE),
        find_step(data, P_GATE),
    ]:
        co.run_step(s)
    verdict = json.loads((co.dir / "gate-out" / "verdict.json").read_text())
    assert verdict["verdict"] == "SKIPPED", verdict
    assert co.run_step(find_step(data, P_CHECK)).returncode == 0
    assert not (co.dir / "gate-ids" / "carried").exists()
