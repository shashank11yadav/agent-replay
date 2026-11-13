"""Regression gate through the CLI: verdicts, Markdown report, step summary, gate-check, compare."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from support import scrubbed_env

from agent_replay.db import runs_repo

MAIN = "import sys\nfrom agent_replay.cli import main\nsys.exit(main())\n"
THRESHOLDS = {
    "max_pass_rate_drop_pp": 5,
    "max_token_increase_pct": 15,
    "max_p95_increase_pct": 20,
}
N = 10


def cli(
    db_url: str, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_env()
    env.pop("GITHUB_STEP_SUMMARY", None)
    env["AGENT_REPLAY_DATABASE_URL"] = db_url
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", MAIN, *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def make_suite(
    conn: Any,
    specs: list[dict[str, Any]],
    *,
    variant: str = "v",
    judge_verdict: str | None = None,
) -> str:
    """specs: task, passed (default True), status, error, tokens (total), latency."""
    suite = runs_repo.create_suite_run(conn, variant=variant, task_count=len(specs))
    for i, s in enumerate(specs):
        tokens = s.get("tokens", 100)
        err = s.get("error")
        run_id = runs_repo.save_run(
            conn,
            {
                "suite_run_id": suite,
                "task_id": s["task"],
                "variant": variant,
                "mode": "replay",
                "status": s.get("status", "error" if err else "ok"),
                "error_type": err,
                "error": err,
                "input_tokens": tokens * 6 // 10,
                "output_tokens": tokens - tokens * 6 // 10,
                "latency_ms": s.get("latency", 1000),
                "model_calls": 2,
                "started_at": f"2026-01-01T00:00:{i:02d}+00:00",
            },
            [],
        )
        ok = s.get("passed", True)
        runs_repo.upsert_score(conn, run_id, "completed", ok, {})
        verdict = s.get("verdict") or judge_verdict
        if verdict:
            runs_repo.upsert_score(
                conn, run_id, "judge", None, {"verdict": verdict}, kind="judge"
            )
        else:
            runs_repo.upsert_score(
                conn,
                run_id,
                "judge",
                ok,
                {"verdict": "pass" if ok else "fail"},
                kind="judge",
            )
    return str(suite)


def specs(n: int = N, **over: Any) -> list[dict[str, Any]]:
    return [{"task": f"t{i}", **over} for i in range(n)]


def with_overrides(
    base: list[dict[str, Any]], idx: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [{**s, **idx.get(i, {})} for i, s in enumerate(base)]


def run_gate(
    db_url: str,
    tmp: Path,
    base: str,
    cand: str,
    thresholds: dict[str, Any] | None = None,
    env_extra: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], str]:
    tpath = tmp / "gate.json"
    tpath.write_text(json.dumps(thresholds or THRESHOLDS))
    out = tmp / "out"
    p = cli(
        db_url, "gate", "--baseline", base, "--candidate", cand,
        "--thresholds", str(tpath), "--out-dir", str(out),
        env_extra=env_extra,
    )  # fmt: skip
    assert p.returncode == 0, f"gate exit {p.returncode}: {p.stderr}"
    verdict = json.loads((out / "verdict.json").read_text())
    return p, verdict, (out / "report.md").read_text()


def rows(report: str) -> list[str]:
    return [ln for ln in report.splitlines() if ln.strip().startswith("|")]


def row_for(report: str, word: str) -> str:
    found = [r for r in rows(report) if word in r.lower()]
    assert found, f"no table row containing {word!r}:\n{report}"
    return found[0]


# --- load_thresholds -------------------------------------------------------------


def _val(t: Any, key: str) -> Any:
    return t[key] if isinstance(t, dict) else getattr(t, key)


def test_load_thresholds_reads_three_values(tmp_path: Path) -> None:
    from agent_replay.report.gate import load_thresholds

    p = tmp_path / "g.json"
    p.write_text(json.dumps(THRESHOLDS))
    t = load_thresholds(p)
    assert _val(t, "max_pass_rate_drop_pp") == 5
    assert _val(t, "max_token_increase_pct") == 15
    assert _val(t, "max_p95_increase_pct") == 20


@pytest.mark.parametrize("missing", list(THRESHOLDS))
def test_load_thresholds_rejects_missing(tmp_path: Path, missing: str) -> None:
    from agent_replay.report.gate import load_thresholds

    p = tmp_path / "g.json"
    p.write_text(json.dumps({k: v for k, v in THRESHOLDS.items() if k != missing}))
    with pytest.raises(ValueError):
        load_thresholds(p)


@pytest.mark.parametrize("key", list(THRESHOLDS))
def test_load_thresholds_rejects_negative(tmp_path: Path, key: str) -> None:
    from agent_replay.report.gate import load_thresholds

    p = tmp_path / "g.json"
    p.write_text(json.dumps({**THRESHOLDS, key: -1}))
    with pytest.raises(ValueError):
        load_thresholds(p)


# --- gate() function -------------------------------------------------------------


def test_gate_function_returns_verdict(conn: Any, tmp_path: Path) -> None:
    from agent_replay.report.gate import gate, load_thresholds

    p = tmp_path / "g.json"
    p.write_text(json.dumps(THRESHOLDS))
    th = load_thresholds(p)
    base = make_suite(conn, specs())
    same = make_suite(conn, specs())
    worse = make_suite(conn, with_overrides(specs(), {0: {"passed": False}}))
    assert str(getattr(gate(conn, base, same, th), "verdict", "")).endswith("PASS")
    assert str(getattr(gate(conn, base, worse, th), "verdict", "")).endswith("FAIL")


# --- verdict rules (via CLI) -----------------------------------------------------


def test_identical_suites_pass(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, specs())
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "PASS"
    assert "FAIL" not in "\n".join(rows(report))


def test_pass_rate_drop_fails_and_names_metric(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, with_overrides(specs(), {0: {"passed": False}}))  # -10pp
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "FAIL"
    assert "FAIL" in row_for(report, "pass rate")
    assert "FAIL" not in row_for(report, "token")
    assert "FAIL" not in row_for(report, "p95")


def test_pass_rate_drop_equal_to_threshold_passes(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(
        conn, with_overrides(specs(), {0: {"passed": False}})
    )  # exactly 10pp
    _, v, _ = run_gate(
        db_url, tmp_path, b, c, {**THRESHOLDS, "max_pass_rate_drop_pp": 10}
    )
    assert v["verdict"] == "PASS"


def test_pass_rate_improvement_passes(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, with_overrides(specs(), {0: {"passed": False}}))
    c = make_suite(conn, specs())
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "PASS"


def test_token_increase_over_threshold_fails(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(tokens=100))
    c = make_suite(conn, specs(tokens=116))
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "FAIL"
    assert "FAIL" in row_for(report, "token")
    assert "FAIL" not in row_for(report, "pass rate")


def test_token_increase_exactly_threshold_passes(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(tokens=100))
    c = make_suite(conn, specs(tokens=115))
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "PASS"


def test_p95_increase_over_threshold_fails(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(latency=1000))
    c = make_suite(conn, specs(latency=1201))
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "FAIL"
    assert "FAIL" in row_for(report, "p95")
    assert "FAIL" not in row_for(report, "token")


def test_p95_increase_exactly_threshold_passes(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(latency=1000))
    c = make_suite(conn, specs(latency=1200))
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "PASS"


def test_decreases_pass(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs(tokens=100, latency=1000))
    c = make_suite(conn, specs(tokens=50, latency=500))
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "PASS"


# --- SKIPPED ----------------------------------------------------------------------


def test_skipped_run_gives_skipped_with_notice_first(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(
        conn,
        with_overrides(
            specs(), {i: {"status": "skipped", "passed": False} for i in range(3)}
        ),
    )
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "SKIPPED"
    first = report.splitlines()[0]
    assert "3" in first
    assert "maintainer" in first.lower()
    assert "live model provider" in first.lower()
    assert not first.lstrip().startswith("|")


def test_skipped_beats_fail(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs(tokens=100))
    c = make_suite(
        conn,
        with_overrides(specs(tokens=500), {0: {"status": "skipped", "passed": False}}),
    )
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "SKIPPED"


# --- ERROR ------------------------------------------------------------------------

_CAUSES = [
    ("rate_limit", ["rate limit"], []),
    ("quota_exhausted", ["quota"], []),
    ("budget_exceeded", ["budget"], ["AGENT_REPLAY_RUN_BUDGET_USD"]),
]


def _rerun_hint(report: str) -> bool:
    low = report.lower()
    return any(w in low for w in ("re-run", "rerun", "run again", "re run"))


@pytest.mark.parametrize(("code", "words", "exact"), _CAUSES)
def test_run_error_code_gives_error_and_names_cause(
    conn: Any,
    db_url: str,
    tmp_path: Path,
    code: str,
    words: list[str],
    exact: list[str],
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, with_overrides(specs(), {4: {"error": code}}))
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"
    for w in words:
        assert w in report.lower()
    for e in exact:
        assert e in report
    assert _rerun_hint(report)


@pytest.mark.parametrize(("code", "words", "exact"), _CAUSES[1:])
def test_judge_verdict_gives_error_and_names_cause(
    conn: Any,
    db_url: str,
    tmp_path: Path,
    code: str,
    words: list[str],
    exact: list[str],
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, with_overrides(specs(), {2: {"verdict": code}}))
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"
    for w in words:
        assert w in report.lower()
    for e in exact:
        assert e in report
    assert _rerun_hint(report)


def test_fewer_candidate_runs_gives_error(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(10))
    c = make_suite(conn, specs(9))
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"


def test_error_beats_skipped_and_fail(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs(tokens=100))
    c = make_suite(
        conn,
        with_overrides(
            specs(tokens=500),
            {0: {"status": "skipped", "passed": False}, 1: {"error": "rate_limit"}},
        ),
    )
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"


def test_fewer_runs_beats_skipped(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs(10))
    c = make_suite(conn, with_overrides(specs(9), {0: {"status": "skipped"}}))
    _, v, _ = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"


def test_stopped_baseline_run_is_blamed_on_the_baseline(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, with_overrides(specs(), {0: {"error": "rate_limit"}}))
    c = make_suite(conn, specs())
    _, v, report = run_gate(db_url, tmp_path, b, c)
    assert v["verdict"] == "ERROR"
    assert "- Baseline:" in report
    assert "candidate row(s) stopped" not in report
    assert "Unexpected tasks" not in report


# --- CLI files and exit codes -------------------------------------------------------


def test_gate_writes_files_and_exits_zero_for_every_verdict(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs())
    cands = {
        "PASS": make_suite(conn, specs()),
        "FAIL": make_suite(conn, specs(tokens=500)),
        "SKIPPED": make_suite(
            conn, with_overrides(specs(), {0: {"status": "skipped"}})
        ),
        "ERROR": make_suite(
            conn, with_overrides(specs(), {0: {"error": "rate_limit"}})
        ),
    }
    for expected, cand in cands.items():
        d = tmp_path / expected
        d.mkdir()
        _, v, report = run_gate(db_url, d, b, cand)
        assert v["verdict"] == expected
        assert (d / "out" / "report.md").is_file()
        assert report.strip()


def test_gate_unknown_baseline_exits_2(conn: Any, db_url: str, tmp_path: Path) -> None:
    c = make_suite(conn, specs())
    tpath = tmp_path / "g.json"
    tpath.write_text(json.dumps(THRESHOLDS))
    p = cli(
        db_url, "gate", "--baseline", str(uuid.uuid4()), "--candidate", c,
        "--thresholds", str(tpath), "--out-dir", str(tmp_path / "out"),
    )  # fmt: skip
    assert p.returncode == 2
    assert "Traceback" not in p.stderr
    assert "usage:" not in p.stderr  # a real error, not an argparse rejection


def test_gate_unknown_candidate_exits_2(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs())
    tpath = tmp_path / "g.json"
    tpath.write_text(json.dumps(THRESHOLDS))
    p = cli(
        db_url, "gate", "--baseline", b, "--candidate", str(uuid.uuid4()),
        "--thresholds", str(tpath), "--out-dir", str(tmp_path / "out"),
    )  # fmt: skip
    assert p.returncode == 2
    assert "usage:" not in p.stderr


# --- step summary -------------------------------------------------------------------


def test_step_summary_is_appended(conn: Any, db_url: str, tmp_path: Path) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, specs())
    summary = tmp_path / "summary.md"
    summary.write_text("earlier content\n")
    _, _, report = run_gate(
        db_url, tmp_path, b, c, env_extra={"GITHUB_STEP_SUMMARY": str(summary)}
    )
    text = summary.read_text()
    assert text.startswith("earlier content\n")
    assert report.strip() in text


def test_step_summary_unset_does_not_fail(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs())
    c = make_suite(conn, specs())
    p, v, _ = run_gate(db_url, tmp_path, b, c)
    assert p.returncode == 0
    assert v["verdict"] == "PASS"


# --- report content -------------------------------------------------------------------


def test_report_table_columns_rows_and_flips(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    base_specs = [{"task": f"steady-{i}"} for i in range(8)] + [
        {"task": "flip-down"},
        {"task": "flip-up", "passed": False},
    ]
    cand_specs = [{"task": f"steady-{i}"} for i in range(8)] + [
        {"task": "flip-down", "passed": False},
        {"task": "flip-up"},
    ]
    b = make_suite(conn, base_specs)
    c = make_suite(conn, cand_specs)
    _, _, report = run_gate(db_url, tmp_path, b, c)
    table = rows(report)
    header = [h.strip().lower() for h in table[0].strip().strip("|").split("|")]
    assert header == [
        "metric", "baseline", "candidate", "delta", "threshold", "status",
    ]  # fmt: skip
    row_for(report, "pass rate")
    row_for(report, "token")
    row_for(report, "p95")
    # separator + header + 3 metric rows
    assert len(table) == 5
    last_table_line = max(i for i, ln in enumerate(report.splitlines()) if ln in table)
    tail = "\n".join(report.splitlines()[last_table_line + 1 :])
    assert "flip-down" in tail
    assert "flip-up" in tail
    assert tail.index("flip-down") < tail.index("flip-up")
    assert "steady-" not in tail


# --- gate-check -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "code"), [("PASS", 0), ("SKIPPED", 0), ("FAIL", 1), ("ERROR", 1)]
)
def test_gate_check_exit_codes(
    db_url: str, tmp_path: Path, verdict: str, code: int
) -> None:
    f = tmp_path / "verdict.json"
    f.write_text(json.dumps({"verdict": verdict}))
    p = cli(db_url, "gate-check", str(f))
    assert p.returncode == code, p.stderr
    assert verdict in p.stdout


def test_gate_then_gate_check_end_to_end(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    b = make_suite(conn, specs(tokens=100))
    c = make_suite(conn, specs(tokens=500))
    run_gate(db_url, tmp_path, b, c)
    p = cli(db_url, "gate-check", str(tmp_path / "out" / "verdict.json"))
    assert p.returncode == 1
    assert "FAIL" in p.stdout


# --- compare --------------------------------------------------------------------------------


def test_compare_writes_one_row_per_suite(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    a = make_suite(conn, specs(4, tokens=100, latency=1111), variant="alpha-v")
    b = make_suite(
        conn,
        with_overrides(specs(4, tokens=200, latency=2222), {0: {"passed": False}}),
        variant="bravo-v",
    )
    c = make_suite(
        conn,
        with_overrides(
            specs(4, tokens=300, latency=3333), {0: {"error": "rate_limit"}}
        ),
        variant="charlie-v",
    )
    out = tmp_path / "cmp.md"
    p = cli(
        db_url, "compare", "--suite-run", a, "--suite-run", b, "--suite-run", c,
        "--out", str(out),
    )  # fmt: skip
    assert p.returncode == 0, p.stderr
    table = rows(out.read_text())
    header = table[0].lower()
    for col in ("variant", "pass rate", "token", "p95", "error", "diverged"):
        assert col in header
    body = table[2:]
    assert len(body) == 3
    for line, variant, lat in zip(
        body, ["alpha-v", "bravo-v", "charlie-v"], ["1111", "2222", "3333"], strict=True
    ):
        assert variant in line
        assert lat in line
    assert "100" in body[0]
    assert "200" in body[1]
    assert "300" in body[2]
    assert "75" in body[1]  # pass rate 3/4


def test_compare_unknown_suite_fails(conn: Any, db_url: str, tmp_path: Path) -> None:
    a = make_suite(conn, specs(2))
    out = tmp_path / "cmp.md"
    p = cli(
        db_url, "compare", "--suite-run", a, "--suite-run", str(uuid.uuid4()),
        "--out", str(out),
    )  # fmt: skip
    assert p.returncode != 0
    assert "Traceback" not in p.stderr
    assert "usage:" not in p.stderr
