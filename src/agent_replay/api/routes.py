"""Read-only routes, all mounted under /api by the app factory."""

from __future__ import annotations

import logging
from typing import Annotated, Any, cast

import psycopg
from fastapi import APIRouter, HTTPException, Path, Query, Request
from psycopg_pool import ConnectionPool

from agent_replay.api.schemas import UUID_PATTERN, json_dict, json_list
from agent_replay.db import runs_repo
from agent_replay.diff import diff_runs
from agent_replay.report.metrics import suite_metrics
from agent_replay.scoring.task_pass import task_passed

log = logging.getLogger(__name__)

router = APIRouter()

RunId = Annotated[str, Path(pattern=UUID_PATTERN)]


DictPool = ConnectionPool[psycopg.Connection[dict[str, Any]]]


def _pool(request: Request) -> DictPool:
    return cast(DictPool, request.app.state.pool)


def _load_run(pool: DictPool, run_id: str) -> dict[str, Any]:
    with pool.connection() as conn:
        run = runs_repo.get_run(conn, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            **run,
            "spans": runs_repo.list_spans(conn, run_id),
            "scores": runs_repo.list_run_scores(conn, run_id),
        }


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    try:
        with _pool(request).connection() as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as exc:
        log.error("health check failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="database unavailable") from None
    return {"status": "ok"}


@router.get("/suite-runs")
def suite_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Any]:
    with _pool(request).connection() as conn:
        return json_list(runs_repo.list_suite_runs(conn, limit=limit, offset=offset))


@router.get("/suite-runs/{suite_run_id}/summary")
def suite_summary(request: Request, suite_run_id: RunId) -> dict[str, Any]:
    with _pool(request).connection() as conn:
        if runs_repo.get_suite_run(conn, suite_run_id) is None:
            raise HTTPException(status_code=404, detail="suite run not found")
        metrics = suite_metrics(conn, suite_run_id)
    return json_dict(metrics.model_dump())


@router.get("/runs")
def runs(
    request: Request,
    suite_run_id: Annotated[str | None, Query(pattern=UUID_PATTERN)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Any]:
    """Runs in start order; ``passed`` is the task-pass verdict (null when not fully scored)."""
    with _pool(request).connection() as conn:
        rows = runs_repo.list_runs(
            conn, suite_run_id=suite_run_id, limit=limit, offset=offset
        )
        scores = runs_repo.list_scores_for_runs(conn, [str(r["id"]) for r in rows])
    by_run: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        by_run.setdefault(str(score["run_id"]), []).append(score)
    return json_list(
        [{**r, "passed": task_passed(by_run.get(str(r["id"]), []))} for r in rows]
    )


@router.get("/runs/{run_id}")
def run_detail(request: Request, run_id: RunId) -> dict[str, Any]:
    return json_dict(_load_run(_pool(request), run_id))


@router.get("/diff")
def diff(
    request: Request,
    left: Annotated[str, Query(pattern=UUID_PATTERN)],
    right: Annotated[str, Query(pattern=UUID_PATTERN)],
) -> dict[str, Any]:
    pool = _pool(request)
    result = diff_runs(_load_run(pool, left), _load_run(pool, right))
    return json_dict(result.model_dump())
