"""Run diff: align two loaded runs by step ``seq`` and compute deltas.

A loaded run is a run row dict plus ``"spans"`` (span row dicts) and optionally ``"scores"``
(score row dicts), as returned by ``agent_replay.db.runs_repo``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agent_replay.scoring.task_pass import task_passed

Row = dict[str, Any]

_COUNTERS = ("input_tokens", "output_tokens", "latency_ms", "model_calls", "tool_calls")


class StepDiff(BaseModel):
    seq: int
    kind: str
    name: str
    status: Literal["same", "changed", "added", "removed"]
    changed: bool
    input_tokens_delta: int
    output_tokens_delta: int
    latency_ms_delta: int


class Delta(BaseModel):
    left: int
    right: int
    delta: int


class PassDelta(BaseModel):
    left: bool | None
    right: bool | None


class RunDiff(BaseModel):
    first_divergence: int | None
    steps: list[StepDiff]
    deltas: dict[str, Delta | PassDelta]


def _num(row: Row | None, key: str) -> int:
    if row is None:
        return 0
    return int(row.get(key) or 0)


def _response_content(response: Any) -> Any:
    """Response content only: no cache status, latency, attempts, usage or signatures."""
    if not isinstance(response, dict):
        return response
    calls = [
        {"name": c.get("name"), "args": c.get("args")}
        for c in response.get("tool_calls") or []
        if isinstance(c, dict)
    ]
    return {
        "text": response.get("text"),
        "tool_calls": calls,
        "finish_reason": response.get("finish_reason"),
    }


def _content(span: Row) -> dict[str, Any]:
    payload = span.get("payload") or {}
    if not isinstance(payload, dict):
        return {"payload": payload}
    if span.get("kind") == "model":
        return {
            "request_hash": payload.get("request_hash"),
            "response": _response_content(payload.get("response")),
        }
    return {
        "args": payload.get("args"),
        "result": payload.get("result"),
        "fault": payload.get("fault"),
        "raised": payload.get("raised"),
        "response": payload.get("response"),
    }


def _same(left: Row, right: Row) -> bool:
    return (
        left.get("kind") == right.get("kind")
        and left.get("name") == right.get("name")
        and _content(left) == _content(right)
    )


def _by_seq(run: Row) -> dict[int, Row]:
    out: dict[int, Row] = {}
    for s in run.get("spans") or []:
        seq = int(s["seq"])
        if seq in out:
            raise ValueError(f"duplicate span seq {seq} within one run")
        out[seq] = s
    return out


def diff_runs(left: Row, right: Row) -> RunDiff:
    """Align the two runs' spans by ``seq`` and report per-step and run-level deltas."""
    lspans, rspans = _by_seq(left), _by_seq(right)

    steps: list[StepDiff] = []
    first: int | None = None
    for seq in sorted(lspans.keys() | rspans.keys()):
        lsp, rsp = lspans.get(seq), rspans.get(seq)
        ref = rsp if rsp is not None else lsp
        assert ref is not None
        if lsp is None:
            status: Literal["same", "changed", "added", "removed"] = "added"
        elif rsp is None:
            status = "removed"
        else:
            status = "same" if _same(lsp, rsp) else "changed"
        if status != "same" and first is None:
            first = seq
        steps.append(
            StepDiff(
                seq=seq,
                kind=str(ref.get("kind") or ""),
                name=str(ref.get("name") or ""),
                status=status,
                changed=status != "same",
                input_tokens_delta=_num(rsp, "input_tokens")
                - _num(lsp, "input_tokens"),
                output_tokens_delta=_num(rsp, "output_tokens")
                - _num(lsp, "output_tokens"),
                latency_ms_delta=_num(rsp, "latency_ms") - _num(lsp, "latency_ms"),
            )
        )

    deltas: dict[str, Delta | PassDelta] = {}
    for key in _COUNTERS:
        lv, rv = _num(left, key), _num(right, key)
        deltas[key] = Delta(left=lv, right=rv, delta=rv - lv)
    deltas["task_pass"] = PassDelta(
        left=task_passed(left.get("scores") or []),
        right=task_passed(right.get("scores") or []),
    )
    return RunDiff(first_divergence=first, steps=steps, deltas=deltas)
