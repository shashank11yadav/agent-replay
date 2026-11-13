"""Committed baseline fixtures and the result tables recorded with claude_code.

Everything is checked from files and offline commands. The offline test imports
evals/baseline into a throwaway database, replays it with the claude_code provider and
the CLI disabled (AGENT_REPLAY_CLAUDE_CLI is empty), scores and gates. No live model is
called. Numbers are not hard-coded except the fixed manifest values and the prices;
tables are checked against the committed baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from support import ROOT, pg, scrubbed_env

EVALS = ROOT / "evals"
BASELINE = EVALS / "baseline"
RUNBOOK = EVALS / "RUNBOOK.md"
APPROACHES = EVALS / "results" / "approaches.md"
FAULTS = EVALS / "results" / "faults.md"
TASKS = EVALS / "tasks" / "tasks.jsonl"
AGENT = "claude-haiku-5-5"
JUDGE = "claude-sonnet-5-5"
FAULT_KINDS = ["tool_timeout", "malformed_tool_output", "rate_limit"]
VARIANTS = ["minimal", "structured", "plan_act"]


# --- helpers -----------------------------------------------------------------------------


def manifest() -> dict[str, Any]:
    return json.loads((BASELINE / "manifest.json").read_text())


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Markdown tables as (header cells, body rows); backslash escapes removed."""
    out: list[tuple[list[str], list[list[str]]]] = []
    cur: list[list[str]] = []

    def flush() -> None:
        if len(cur) >= 2:
            body = [
                r
                for r in cur[1:]
                if not all(re.fullmatch(r":?-+:?", c) for c in r if c)
            ]
            out.append((cur[0], body))
        cur.clear()

    for line in text.splitlines():
        if line.strip().startswith("|"):
            inner = line.strip().strip("|")
            cells = [c.strip().replace("\\", "") for c in re.split(r"(?<!\\)\|", inner)]
            cur.append(cells)
        else:
            flush()
    flush()
    return out


def col(header: list[str], *words: str, avoid: tuple[str, ...] = ()) -> int | None:
    for i, h in enumerate(header):
        low = h.lower()
        if all(w in low for w in words) and not any(a in low for a in avoid):
            return i
    return None


def num(cell: str) -> float:
    m = re.search(r"-?\d[\d,]*\.?\d*", cell)
    assert m, f"no number in cell {cell!r}"
    return float(m.group(0).replace(",", ""))


# --- database fixtures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def scratch_db() -> Iterator[str]:
    with pg.throwaway_database("ar_test_baseline", migrate=False) as url:
        yield url


def env_for(url: str) -> dict[str, str]:
    env = scrubbed_env()
    env.pop("GITHUB_STEP_SUMMARY", None)
    env.update(
        AGENT_REPLAY_DATABASE_URL=url,
        AGENT_REPLAY_PROVIDER="claude_code",
        AGENT_REPLAY_CLAUDE_CLI="",
        AGENT_REPLAY_GEMINI_API_KEY="",
        AGENT_REPLAY_BACKOFF_BASE_S="0",
    )
    return env


def cli(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_replay.cli import main; sys.exit(main())",
            *args,
        ],
        env=env_for(url),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def ok(p: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert p.returncode == 0, (p.args[3:], p.stdout, p.stderr)
    return p


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, f"empty stdout; stderr={p.stderr!r}"
    return lines[-1].strip()


@pytest.fixture(scope="module")
def offline(scratch_db: str, tmp_path_factory: Any) -> dict[str, Any]:
    """Import the baseline, replay it offline, score it, gate it."""
    from agent_replay.db.migrate import apply_migrations

    apply_migrations(scratch_db)
    ok(cli(scratch_db, "fixtures", "import", str(BASELINE)))
    base_id = manifest()["suite_run_id"]
    rep = ok(
        cli(
            scratch_db,
            "suite",
            "replay",
            "--source",
            base_id,
            "--tasks",
            str(TASKS),
            "--label",
            "t29-offline-check",
        )
    )
    cand_id = last_line(rep)
    ok(cli(scratch_db, "suite", "score", "--suite-run", cand_id, "--tasks", str(TASKS)))
    out_dir = tmp_path_factory.mktemp("baseline_gate")
    gate = ok(
        cli(
            scratch_db,
            "gate",
            "--baseline",
            base_id,
            "--candidate",
            cand_id,
            "--thresholds",
            str(EVALS / "gate.json"),
            "--out-dir",
            str(out_dir),
        )
    )
    check = cli(scratch_db, "gate-check", str(out_dir / "verdict.json"))
    return {
        "base_id": base_id,
        "cand_id": cand_id,
        "gate": gate,
        "check": check,
        "verdict": json.loads((out_dir / "verdict.json").read_text()),
        "url": scratch_db,
    }


# --- baseline manifest and fixtures -------------------------------------------------------


def test_manifest_names_a_prompt_variant_and_thirty_runs() -> None:
    # evals/suite.json may name another variant on a prompt-change branch; the baseline
    # keeps the variant it was recorded with.
    m = manifest()
    assert m["variant"] == "structured"
    prompts = ROOT / "src" / "agent_replay" / "demo" / "prompts"
    assert (prompts / f"{m['variant']}.txt").is_file()
    assert m["run_count"] == 30
    rows = [
        ln for ln in (BASELINE / "runs.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(rows) == 30


def test_manifest_task_hash_matches_task_file() -> None:
    digest = hashlib.sha256(TASKS.read_bytes()).hexdigest()
    m = manifest()
    assert m["tasks_sha256"] == digest
    assert m["suite_run"]["tasks_sha256"] == digest


def test_manifest_records_provider_and_model_ids() -> None:
    m = manifest()
    assert m["provider"] == "claude_code"
    assert m["agent_model"] == AGENT
    assert m["judge_model"] == JUDGE


# --- offline replay -----------------------------------------------------------------------


def test_offline_replay_metrics_equal_recorded_suite(offline: dict[str, Any]) -> None:
    v = offline["verdict"]
    for key in (
        "task_count",
        "pass_rate",
        "deterministic_pass_rate",
        "judge_pass_rate",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "total_tokens",
        "p95_latency_ms",
    ):
        assert v["candidate"][key] == v["baseline"][key], key
    assert v["candidate"]["task_count"] == 30


def test_offline_replay_has_zero_skipped_runs(offline: dict[str, Any]) -> None:
    v = offline["verdict"]
    assert v["candidate"]["skipped_count"] == 0
    assert v["skipped_requests"] == 0
    with psycopg.connect(offline["url"], autocommit=True) as c:
        n = c.execute(
            "SELECT count(*) FROM runs WHERE suite_run_id = %s AND status = 'skipped'",
            (offline["cand_id"],),
        ).fetchone()[0]
        total = c.execute(
            "SELECT count(*) FROM runs WHERE suite_run_id = %s", (offline["cand_id"],)
        ).fetchone()[0]
    assert n == 0
    assert total == 30


def test_offline_gate_passes_against_baseline(offline: dict[str, Any]) -> None:
    assert offline["verdict"]["verdict"] == "PASS"
    assert "PASS" in offline["gate"].stdout
    assert offline["check"].returncode == 0, (
        offline["check"].stdout,
        offline["check"].stderr,
    )


# --- approaches.md ------------------------------------------------------------------------


def approach_tables() -> list[tuple[list[str], list[list[str]]]]:
    assert APPROACHES.is_file()
    found = [
        t for t in tables(APPROACHES.read_text()) if col(t[0], "variant") is not None
    ]
    assert found, "approaches.md has no table with a Variant column"
    return found


def variant_row(variant: str) -> tuple[list[str], list[str]]:
    for header, body in approach_tables():
        vc = col(header, "variant")
        assert vc is not None
        rows = [
            r
            for r in body
            if len(r) > vc and r[vc].lower().split("(")[0].strip() == variant
        ]
        if rows:
            assert len(rows) == 1, f"more than one row for {variant}"
            return header, rows[0]
    pytest.fail(f"no row for variant {variant} in approaches.md")


@pytest.mark.parametrize("variant", VARIANTS)
def test_approaches_has_one_row_per_variant(variant: str) -> None:
    variant_row(variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_approaches_row_has_all_metric_columns(variant: str) -> None:
    header, row = variant_row(variant)
    needed = {
        "pass rate": col(header, "pass", avoid=("deterministic", "judge")),
        "deterministic pass rate": col(header, "deterministic"),
        "judge pass rate": col(header, "judge"),
        "mean tokens": col(header, "token"),
        "p95 latency": col(header, "p95"),
        "notional USD": col(header, "usd"),
    }
    for name, idx in needed.items():
        assert idx is not None, f"no {name} column in {header}"
        assert idx < len(row) and re.search(r"\d", row[idx]), (
            f"{name} empty for {variant}"
        )
    assert "notional" in " ".join(header).lower(), (
        "USD column must be labelled notional"
    )
    assert num(row[needed["notional USD"]]) > 0  # type: ignore[index]


def test_approaches_notional_usd_is_positive_for_every_variant() -> None:
    for variant in VARIANTS:
        header, row = variant_row(variant)
        idx = col(header, "usd")
        assert idx is not None, f"no USD column in {header}"
        assert num(row[idx]) > 0, variant


def test_approaches_baseline_row_matches_the_committed_baseline(
    offline: dict[str, Any],
) -> None:
    """The shipped variant's row must agree with the metrics of evals/baseline."""
    m = offline["verdict"]["baseline"]
    shipped = manifest()["variant"]
    header, row = variant_row(shipped)
    pr = col(header, "pass", avoid=("deterministic", "judge"))
    det = col(header, "deterministic")
    jd = col(header, "judge")
    tk = col(header, "token")
    p95 = col(header, "p95")
    assert None not in (pr, det, jd, tk, p95), header
    assert abs(num(row[pr]) - m["pass_rate"] * 100) < 0.06  # type: ignore[index]
    assert abs(num(row[det]) - m["deterministic_pass_rate"] * 100) < 0.06  # type: ignore[index]
    assert abs(num(row[jd]) - m["judge_pass_rate"] * 100) < 0.06  # type: ignore[index]
    assert abs(num(row[tk]) - m["mean_total_tokens"]) <= 0.51  # type: ignore[index]
    assert abs(num(row[p95]) - m["p95_latency_ms"]) <= 0.5  # type: ignore[index]


def test_approaches_states_date_provider_and_model_ids() -> None:
    assert APPROACHES.is_file()
    text = APPROACHES.read_text()
    assert re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text), "date (YYYY-MM-DD)"
    assert "claude_code" in text
    assert AGENT in text and JUDGE in text


def test_approaches_lists_the_baseline_suite_run_id() -> None:
    assert APPROACHES.is_file()
    assert manifest()["suite_run_id"] in APPROACHES.read_text()


# --- faults.md ----------------------------------------------------------------------------


def fault_tables() -> list[tuple[list[str], list[list[str]]]]:
    assert FAULTS.is_file()
    return tables(FAULTS.read_text())


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_faults_has_one_row_per_fault_kind_with_share(kind: str) -> None:
    hits: list[tuple[list[str], list[str]]] = []
    for header, body in fault_tables():
        hits += [(header, r) for r in body if any(kind in c.lower() for c in r)]
    assert len(hits) == 1, (
        f"expected exactly one row for fault kind {kind}, got {len(hits)}"
    )
    header, row = hits[0]
    share_col = col(header, "ok", avoid=("token",))
    assert share_col is not None, (
        f"no column for the share ended ok or clean error: {header}"
    )
    low = header[share_col].lower()
    assert "error" in low, (
        f"share column must mention clean error: {header[share_col]!r}"
    )
    cell = row[share_col]
    assert "%" in cell, f"share must be a percentage, got {cell!r}"
    assert 0.0 <= num(cell) <= 100.0


def test_faults_share_is_consistent_with_error_columns() -> None:
    """A share cannot exceed 100% minus the share of diverged and skipped runs."""
    for header, body in fault_tables():
        share_col = col(header, "ok", avoid=("token",))
        div = col(header, "diverged")
        skip = col(header, "skipped")
        if share_col is None or div is None or skip is None:
            continue
        for row in body:
            if "%" not in row[share_col]:
                continue
            bad = num(row[div]) + num(row[skip])
            assert num(row[share_col]) <= 100.0 - bad / 30 * 100 + 0.06, row


# --- no environment values under evals/ ---------------------------------------------------


def evals_files() -> list[Path]:
    return [p for p in EVALS.rglob("*") if p.is_file()]


def test_evals_contains_no_environment_values_or_machine_paths() -> None:
    assert RUNBOOK.is_file(), (
        "evals/RUNBOOK.md does not exist (scan needs the full set)"
    )
    needles: dict[str, str] = {}
    home = str(Path.home())
    if len(home) > 1:
        needles["home directory"] = home
    needles["temp directory"] = tempfile.gettempdir()
    for var in ("TMPDIR", "HOME"):
        val = os.environ.get(var, "").rstrip("/")
        if len(val) > 1:
            needles[var] = val
    for var in ("AGENT_REPLAY_GEMINI_API_KEY", "AGENT_REPLAY_DATABASE_URL"):
        val = os.environ.get(var, "")
        if len(val) >= 8:
            needles[var + " value"] = val
    needles["dev database credentials"] = "postgres:postgres@"
    patterns = {
        "macOS user path": r"/Users/[A-Za-z0-9._-]+/",
        "linux home path": r"/home/[A-Za-z0-9._-]+/",
        "macOS temp dir": r"/(?:private/)?var/folders/",
        "tmp path": r"(?<![A-Za-z0-9_.-])/tmp/",
        "Google API key": r"AIza[0-9A-Za-z_-]{20,}",
        "database URL with password": r"postgres(?:ql)?://[^\s:@/]+:[^\s@/]+@",
    }
    offenders: list[str] = []
    for p in evals_files():
        data = p.read_text(errors="replace")
        rel = str(p.relative_to(ROOT))
        offenders += [f"{rel}: {k}" for k, v in needles.items() if v in data]
        offenders += [
            f"{rel}: {k}" for k, rx in patterns.items() if re.search(rx, data)
        ]
    assert not offenders, offenders
