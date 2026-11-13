"""Read the GitHub workflow files and run their steps in a scratch checkout.

PyYAML is not a dependency, so the workflows are read with a small block-YAML subset
parser (mappings, sequences, scalars, flow lists, block scalars, comments). That is
enough for the two workflow files, which stay block-style.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from support import ROOT
from support.suites import base_env

WF = ROOT / ".github" / "workflows" / "regression-gate.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

# Patterns that pick out one regression-gate step by its `run:` script. P_COMMENT is in the
# comment job; the others are in the gate job.
P_MIGRATE = r"uv run agent-replay db migrate\s*$"
P_VERIFY = r"uv run agent-replay fixtures verify evals/baseline\b"
P_IMPORT = r"uv run agent-replay fixtures import evals/baseline\b"
P_REPLAY = r"uv run agent-replay suite replay\b"
P_SCORE = r"uv run agent-replay suite score\b"
P_GATE = r"uv run agent-replay gate(?![-\w])"
P_COMMENT = r"gh pr comment\b"
P_CHECK = r"uv run agent-replay gate-check\b"
P_BUILD_COMMENT = r"> gate-out/comment\.md"
GATE_JOB = "gate"
COMMENT_JOB = "comment"

FAKE_CANDIDATE = "11111111-2222-3333-4444-555555555555"


# --- YAML subset parser ------------------------------------------------------------


def _scalar(s: str) -> Any:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_scalar(p) for p in inner.split(",")] if inner else []
    if s in ("{}",):
        return {}
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "~", ""):
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def _strip_comment(line: str) -> str:
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _mini_load(text: str) -> Any:
    raw = text.splitlines()
    lines: list[tuple[int, str, int]] = []
    for n, ln in enumerate(raw):
        if ln.strip() and not ln.strip().startswith("#"):
            lines.append((len(ln) - len(ln.lstrip()), ln.strip(), n))
    pos = 0

    def block_scalar(indent: int, style: str) -> str:
        nonlocal pos
        start = lines[pos - 1][2] + 1 if pos else 0
        end = lines[pos][2] if pos < len(lines) else len(raw)
        while pos < len(lines) and lines[pos][0] > indent:
            pos += 1
            end = lines[pos][2] if pos < len(lines) else len(raw)
        block = raw[start:end]
        # Comments after the block, at or left of its key, belong to the YAML.
        while block and (
            not block[-1].strip()
            or (
                block[-1].lstrip().startswith("#")
                and len(block[-1]) - len(block[-1].lstrip()) <= indent
            )
        ):
            block.pop()
        body = [ln.strip() for ln in block]
        return ("\n" if style.startswith("|") else " ").join(body).strip()

    def parse(indent: int) -> Any:
        nonlocal pos
        ind, content, _ = lines[pos]
        if content.startswith("- ") or content == "-":
            seq: list[Any] = []
            while (
                pos < len(lines)
                and lines[pos][0] == ind
                and (lines[pos][1].startswith("- ") or lines[pos][1] == "-")
            ):
                _, c, n = lines[pos]
                rest = c[1:].strip()
                if not rest:
                    pos += 1
                    seq.append(parse(ind + 1))
                elif re.match(r"^[^'\"\s][^:]*:(\s|$)", rest) and not rest.startswith(
                    "${{"
                ):
                    lines[pos] = (ind + 2, rest, n)
                    seq.append(parse(ind + 2))
                else:
                    pos += 1
                    seq.append(_scalar(_strip_comment(rest)))
            return seq
        mp: dict[str, Any] = {}
        while pos < len(lines) and lines[pos][0] == ind:
            _, c, _ = lines[pos]
            m = re.match(r"^([^:'\"]+|'[^']*'|\"[^\"]*\"):(?:\s+(.*))?$", c)
            if not m:
                raise ValueError(f"unparsable line: {c!r}")
            key = _scalar(m.group(1))
            val = _strip_comment(m.group(2) or "")
            pos += 1
            if val[:1] in ("|", ">"):
                mp[key] = block_scalar(ind, val)
            elif val:
                mp[key] = _scalar(val)
            elif pos < len(lines) and (
                lines[pos][0] > ind
                or (lines[pos][0] == ind and lines[pos][1].startswith("- "))
            ):
                mp[key] = parse(lines[pos][0])
            else:
                mp[key] = None
        return mp

    return parse(0)


def load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"{path} does not exist"
    data = _mini_load(path.read_text())
    assert isinstance(data, dict)
    return data


# --- regression-gate.yml -----------------------------------------------------------


def load_wf() -> dict[str, Any]:
    return load_yaml(WF)


def job(data: dict[str, Any], name: str = GATE_JOB) -> dict[str, Any]:
    jobs = data.get("jobs")
    assert isinstance(jobs, dict) and set(jobs) == {GATE_JOB, COMMENT_JOB}, (
        f"expected a gate and a comment job: {jobs}"
    )
    j = jobs[name]
    assert isinstance(j, dict)
    return j


def steps(data: dict[str, Any], name: str = GATE_JOB) -> list[dict[str, Any]]:
    s = job(data, name).get("steps")
    assert isinstance(s, list) and s
    return s


def run_of(step: dict[str, Any]) -> str:
    return str(step.get("run") or "")


def find_step(
    data: dict[str, Any], pattern: str, name: str = GATE_JOB
) -> dict[str, Any]:
    hits = [s for s in steps(data, name) if re.search(pattern, run_of(s))]
    assert len(hits) == 1, f"expected exactly one step matching {pattern!r}: {hits}"
    return hits[0]


def index_of(data: dict[str, Any], pattern: str, name: str = GATE_JOB) -> int:
    idx = [i for i, s in enumerate(steps(data, name)) if re.search(pattern, run_of(s))]
    assert len(idx) == 1, f"expected exactly one step matching {pattern!r}: {idx}"
    return idx[0]


def derive_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Steps from the baseline import (included) up to the replay (excluded)."""
    return steps(data)[index_of(data, P_IMPORT) : index_of(data, P_REPLAY)]


# --- running workflow steps --------------------------------------------------------

# Stands in for `uv`: `uv run agent-replay ...` runs the real CLI and `uv run python ...`
# the test interpreter. TEST_UV_LOG records the AGENT_REPLAY_* environment of every
# call; TEST_UV_STUB fakes `suite replay` / `suite score`.
UV_SHIM = """#!/bin/sh
[ "$1" = run ] && shift
if [ -n "$TEST_UV_LOG" ]; then
  { echo "CALL $*"; env | grep '^AGENT_REPLAY_' | sort; echo END; } >> "$TEST_UV_LOG"
fi
if [ -n "$TEST_UV_STUB" ]; then
  case "$*" in
    "agent-replay suite replay"*) echo "%(cand)s"; exit 0;;
    "agent-replay suite score"*) exit 0;;
  esac
fi
case "$1" in
  python) shift; exec "%(py)s" "$@";;
  agent-replay) shift
    exec "%(py)s" -c "import sys; from agent_replay.cli import main; sys.exit(main())" "$@";;
esac
echo "unexpected uv invocation: $*" >&2
exit 97
"""


class Checkout:
    """A scratch directory laid out like the workflow's working tree."""

    def __init__(self, tmp: Path, baseline: dict[str, Any], db_url: str) -> None:
        self.dir = tmp / "checkout"
        self.dir.mkdir()
        self.db_url = db_url
        self.log = tmp / "uv.log"
        self.summary = tmp / "summary.md"
        self.summary.write_text("")
        bindir = tmp / "bin"
        bindir.mkdir()
        shim = bindir / "uv"
        shim.write_text(UV_SHIM % {"py": sys.executable, "cand": FAKE_CANDIDATE})
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        self.bindir = bindir
        (self.dir / "evals" / "tasks").mkdir(parents=True)
        shutil.copytree(baseline["dir"], self.dir / "evals" / "baseline")
        shutil.copy(baseline["tasks"], self.dir / "evals" / "tasks" / "tasks.jsonl")
        # The PR's variant is the baseline's own, whatever this branch's suite.json names.
        variant = json.loads((Path(baseline["dir"]) / "manifest.json").read_text())[
            "variant"
        ]
        (self.dir / "evals" / "suite.json").write_text(json.dumps({"variant": variant}))
        shutil.copy(ROOT / "evals" / "gate.json", self.dir / "evals" / "gate.json")
        (self.dir / "gate-ids").mkdir()
        self.baseline = baseline

    @property
    def manifest_path(self) -> Path:
        return self.dir / "evals" / "baseline" / "manifest.json"

    def edit_manifest(self, **changes: Any) -> None:
        m = json.loads(self.manifest_path.read_text())
        for k, v in changes.items():
            if v is None:
                m.pop(k, None)
            else:
                m[k] = v
        self.manifest_path.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")

    def run_step(
        self, step: dict[str, Any], extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = base_env(self.db_url, extra)
        env["PATH"] = f"{self.bindir}{os.pathsep}{env.get('PATH', '')}"
        env["GITHUB_STEP_SUMMARY"] = str(self.summary)
        env["TEST_UV_LOG"] = str(self.log)
        merged = {**(job(load_wf()).get("env") or {}), **(step.get("env") or {})}
        for k, v in merged.items():
            if "${{" in str(v) or k == "AGENT_REPLAY_DATABASE_URL":
                continue
            env[k] = "" if v is None else str(v)
        env["AGENT_REPLAY_GEMINI_API_KEY"] = ""
        env.update(extra or {})
        return subprocess.run(
            [shutil.which("bash") or "bash", "-e", "-c", run_of(step)],
            cwd=self.dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def calls(self) -> list[tuple[str, dict[str, str]]]:
        out: list[tuple[str, dict[str, str]]] = []
        if not self.log.exists():
            return out
        cur: tuple[str, dict[str, str]] | None = None
        for line in self.log.read_text().splitlines():
            if line.startswith("CALL "):
                cur = (line[5:], {})
            elif line == "END" and cur is not None:
                out.append(cur)
                cur = None
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                cur[1][k] = v
        return out

    def call_env(self, prefix: str) -> dict[str, str]:
        hits = [e for a, e in self.calls() if a.startswith(prefix)]
        assert len(hits) == 1, (prefix, [a for a, _ in self.calls()])
        return hits[0]
