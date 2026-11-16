"""Build site/data.js, the data behind the project page, from the committed fixtures.

The page shows one task (T11) step by step under the baseline prompt and the regressed prompt,
the 30-task board and the gate table. Everything comes from evals/, so the page cannot drift
from the recordings: tests/repo/test_site.py rebuilds the file and compares.

Run: uv run python scripts/site_data.py
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
PROMPTS = ROOT / "src" / "agent_replay" / "demo" / "prompts"
OUT = ROOT / "site" / "data.js"
EXAMPLE_TASK = "T11"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _suite(name: str) -> dict[str, Any]:
    folder = EVALS / name
    runs = {r["task_id"]: r for r in _jsonl(folder / "runs.jsonl")}
    spans: dict[str, list[dict[str, Any]]] = {}
    for span in _jsonl(folder / "spans.jsonl"):
        spans.setdefault(span["run_id"], []).append(span)
    scores: dict[str, list[dict[str, Any]]] = {}
    for score in _jsonl(folder / "scores.jsonl"):
        scores.setdefault(score["run_id"], []).append(score)
    return {"runs": runs, "spans": spans, "scores": scores}


def _passed(suite: dict[str, Any], run_id: str) -> bool:
    checks = suite["scores"].get(run_id, [])
    return bool(checks) and all(c["passed"] for c in checks)


def _step(span: dict[str, Any]) -> dict[str, Any]:
    payload = span.get("payload") or {}
    if span["kind"] == "model":
        response = payload.get("response") or {}
        return {
            "kind": "model",
            "ms": span["latency_ms"],
            "tokens_in": span["input_tokens"],
            "tokens_out": span["output_tokens"],
            "text": response.get("text", ""),
            "asks_for": [c["name"] for c in response.get("tool_calls", [])],
            "request_hash": payload.get("request_hash"),
        }
    return {
        "kind": "tool",
        "ms": span["latency_ms"],
        "tool": payload.get("name"),
        "args": payload.get("args"),
        "result": payload.get("result"),
    }


def _run(suite: dict[str, Any], task_id: str) -> dict[str, Any]:
    run = suite["runs"][task_id]
    spans = sorted(suite["spans"][run["id"]], key=lambda s: s["seq"])
    checks = sorted(suite["scores"][run["id"]], key=lambda c: c["check_name"])
    return {
        "answer": run["final_output"],
        "ms": run["latency_ms"],
        "tokens_in": run["input_tokens"],
        "tokens_out": run["output_tokens"],
        "model_calls": run["model_calls"],
        "tool_calls": run["tool_calls"],
        "steps": [_step(s) for s in spans if s["kind"] in ("model", "tool")],
        "checks": [
            {
                "name": c["check_name"],
                "passed": c["passed"],
                "reason": c["detail"].get("reason"),
            }
            for c in checks
        ],
    }


def _gate_rows() -> list[dict[str, str]]:
    """The three gated rows of the committed regression table."""
    rows = []
    for line in (
        (EVALS / "results" / "regression.md").read_text(encoding="utf-8").splitlines()
    ):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[5] in ("PASS", "FAIL"):
            keys = ("metric", "baseline", "candidate", "delta", "threshold", "status")
            rows.append(dict(zip(keys, cells, strict=True)))
    return rows


def _prompt_change() -> list[dict[str, str]]:
    old = (PROMPTS / "structured.txt").read_text(encoding="utf-8").splitlines()
    new = (
        (PROMPTS / "structured_regressed.txt").read_text(encoding="utf-8").splitlines()
    )
    out = []
    for line in difflib.ndiff(old, new):
        if line[:2] in ("- ", "+ "):
            text = re.sub(r"^\d+\.\s*", "", line[2:])
            out.append({"op": "removed" if line[0] == "-" else "added", "text": text})
    # A rule that only moved up a number is not a change.
    removed = {c["text"] for c in out if c["op"] == "removed"}
    added = {c["text"] for c in out if c["op"] == "added"}
    return [c for c in out if c["text"] not in removed & added]


def build() -> str:
    tasks = {t["id"]: t for t in _jsonl(EVALS / "tasks" / "tasks.jsonl")}
    base, reg = _suite("baseline"), _suite("regression")
    board = []
    for task_id, task in tasks.items():
        b, r = base["runs"][task_id], reg["runs"][task_id]
        board.append(
            {
                "id": task_id,
                "question": task["prompt"],
                "baseline": _passed(base, b["id"]),
                "regressed": _passed(reg, r["id"]),
                "baseline_ms": b["latency_ms"],
                "regressed_ms": r["latency_ms"],
                "baseline_answer": b["final_output"],
                "regressed_answer": r["final_output"],
            }
        )
    task = tasks[EXAMPLE_TASK]
    data = {
        "task": {
            "id": EXAMPLE_TASK,
            "question": task["prompt"],
            "rubric": task["rubric"],
        },
        "baseline": _run(base, EXAMPLE_TASK),
        "regressed": _run(reg, EXAMPLE_TASK),
        "prompt_change": _prompt_change(),
        "board": board,
        "gate": _gate_rows(),
    }
    body = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True)
    return f"// Built by scripts/site_data.py from evals/. Do not edit by hand.\nwindow.REPLAY = {body};\n"


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
