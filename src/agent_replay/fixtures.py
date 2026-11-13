"""Fixtures: export a suite run as JSONL, import it elsewhere, verify it is replayable.

Layout of a fixture directory (all UTF-8, one JSON object per line, keys sorted):
``manifest.json``, ``runs.jsonl``, ``spans.jsonl``, ``scores.jsonl``, ``cache.jsonl``.

Export reads only database rows, never settings values (the settings object is used for the
judge model name only). Export is deterministic for a given database: re-exporting an
unchanged suite gives identical bytes, because rows are ordered by key, datetimes are UTC ISO
strings and nothing records the export time. Span ids are reassigned on import, so a re-export
of the same suite from another database differs only in its span ids.

Judge cache entries are not linked from spans. The judge stores its request hash in the
judge score row (``detail.request_hash``); export reads it there. Older rows without it are
recomputed with ``build_judge_request`` from the final answer and the root span's task
snapshot. Any judge row whose verdict came from a model response (passed true/false, or an
unparseable reply) must have its cache entry: export fails, listing the run ids, when it is
missing, and ``verify`` checks the same from ``scores.jsonl`` and ``manifest.judge_requests``.
Judge rows that never got a response (skipped, budget, quota, transport errors) have no
entry, so ``verify`` reports them (and skipped or stop-code runs) as an incomplete baseline.

Imported files are untrusted but their cache responses are trusted data once imported: a
replay serves them as model output. Each cache row's request is re-hashed and must match its
key, but the response cannot be checked, so a committed baseline must come from reviewed files
(do not import fixtures from an unreviewed PR in a job that holds secrets). ``verify`` also
checks that each cache entry agrees with the recorded span that used it.
Import is all-or-nothing and conflict-checked: an existing row with the same key (id; for
spans, run id and step number) must be identical, and rows are never merged into an existing
suite or run. Span ids are assigned by the target database.

Import treats the files as untrusted data: every row is validated, unknown fields are
rejected, nothing in them is evaluated or used as a path. A suite's ``source_suite_run_id`` and
a run's ``source_run_id`` that point at rows missing from the target database are set to NULL.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import psycopg
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from agent_replay import jsonio
from agent_replay.codes import JUDGE_ERROR, STOP_CODES, UNPARSEABLE_VERDICT
from agent_replay.config import Settings
from agent_replay.db import cache_repo, runs_repo
from agent_replay.llm.hashing import request_hash
from agent_replay.llm.types import ModelRequest
from agent_replay.scoring.judge import build_judge_request
from agent_replay.taskspec import TaskSpec

Row = dict[str, Any]

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
RUNS = "runs.jsonl"
SPANS = "spans.jsonl"
SCORES = "scores.jsonl"
CACHE = "cache.jsonl"
FILES = (MANIFEST, RUNS, SPANS, SCORES, CACHE)


class FixtureError(Exception):
    """A user-facing failure; the message is safe to print."""


def _uid(value: Any) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise ValueError("not a UUID") from None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime needs a timezone")
    return value.astimezone(UTC)


Count = Annotated[int, Field(ge=0, le=2**31 - 1)]
Uid = Annotated[str, BeforeValidator(_uid)]
Stamp = Annotated[datetime, AfterValidator(_utc)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False, allow_inf_nan=False)


class SuiteRow(_Strict):
    id: Uid
    label: str
    variant: str
    mode: str
    status: str
    tasks_path: str | None
    tasks_sha256: str | None
    task_count: Count
    source_suite_run_id: Uid | None
    faults: list[Any]
    metrics: Any
    created_at: Stamp
    finished_at: Stamp | None


class RunRow(_Strict):
    id: Uid
    suite_run_id: Uid | None
    source_run_id: Uid | None
    task_id: str
    variant: str
    mode: str
    status: str
    error_type: str | None
    error: str | None
    final_output: str | None
    faults: list[Any]
    input_tokens: Count
    output_tokens: Count
    latency_ms: Count
    model_calls: Count
    tool_calls: Count
    cache_hits: Count
    cache_misses: Count
    cost_usd: float
    started_at: Stamp
    finished_at: Stamp | None


class SpanRow(_Strict):
    id: Annotated[int, Field(gt=0, lt=2**53)]
    run_id: Uid
    seq: Count
    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: str
    name: str
    status: str
    error_type: str | None
    attributes: dict[str, Any]
    payload: dict[str, Any]
    input_tokens: Count
    output_tokens: Count
    latency_ms: Count
    cost_usd: float
    cache_status: str | None
    started_at: Stamp


class ScoreRow(_Strict):
    run_id: Uid
    check_name: str
    kind: str
    passed: bool | None
    detail: dict[str, Any]
    created_at: Stamp


class CacheRow(_Strict):
    request_hash: str
    provider: str
    model: str
    request: dict[str, Any]
    response: dict[str, Any]
    input_tokens: Count
    output_tokens: Count
    latency_ms: Count
    cost_usd: float
    created_at: Stamp


class Manifest(_Strict):
    format_version: int
    suite_run_id: Uid
    variant: str
    task_count: Count
    run_count: Count
    tasks_sha256: str | None
    suite_run: SuiteRow
    judge_requests: dict[Uid, str]
    # Pinned by export from the cache entries (absent in older manifests).
    provider: str | None = None
    agent_model: str | None = None
    judge_model: str | None = None


class Bundle(_Strict):
    manifest: Manifest
    runs: list[RunRow]
    spans: list[SpanRow]
    scores: list[ScoreRow]
    cache: list[CacheRow]


def clean(text: str, limit: int = 80) -> str:
    """Printable, single-line text: control characters dropped, '::' defused."""
    kept = "".join(ch for ch in text if ch.isprintable())[:limit]
    return kept.replace("::", ": :")


def dumps(obj: Any) -> str:
    """Stable single-line JSON: sorted keys, fixed separators, no NaN."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _jsonl(rows: list[Row]) -> str:
    return "".join(dumps(r) + "\n" for r in rows)


def _dump(model: _Strict) -> Row:
    return model.model_dump(mode="json")


def collect_hashes(spans: list[SpanRow]) -> dict[str, list[str]]:
    """Model request hashes per run id, for model spans that stored a response."""
    found: dict[str, list[str]] = {}
    for span in spans:
        payload = span.payload
        if span.kind != "model" or payload.get("response") is None:
            continue
        key = payload.get("request_hash")
        if isinstance(key, str) and key:
            found.setdefault(span.run_id, []).append(key)
    return found


def judge_needs_entry(score: ScoreRow) -> bool:
    """True when the judge row's verdict came from a model response (so it was cached)."""
    if score.passed is not None:
        return True
    detail = score.detail
    return (
        detail.get("verdict") == JUDGE_ERROR
        and detail.get("error") == UNPARSEABLE_VERDICT
    )


def judge_hash(run: RunRow, spans: list[SpanRow], settings: Settings) -> str | None:
    """Request hash of the run's judge call, or None when it would not have made one."""
    answer = run.final_output or ""
    if not answer.strip():
        return None
    for span in spans:
        snapshot = span.payload.get("task") if span.kind == "agent" else None
        if isinstance(snapshot, dict):
            try:
                task = TaskSpec.model_validate(snapshot)
            except ValidationError:
                return None
            return request_hash(build_judge_request(task, answer, settings))
    return None


def pin_models(
    entries: list[CacheRow], agent_keys: set[str], judge_keys: set[str]
) -> tuple[str | None, str | None, str | None]:
    """(provider, agent_model, judge_model) of the exported entries.

    Raises FixtureError when agent entries differ in provider or model, judge entries
    differ in model, or the judge provider differs from the agent provider.
    """
    agent = [e for e in entries if e.request_hash in agent_keys]
    judge = [e for e in entries if e.request_hash in judge_keys]
    providers = {e.provider for e in agent}
    models = {e.model for e in agent}
    judge_models = {e.model for e in judge}
    if len(providers) > 1:
        raise FixtureError(
            "agent cache entries use several providers: " + _names(providers)
        )
    if len(models) > 1:
        raise FixtureError("agent cache entries use several models: " + _names(models))
    if len(judge_models) > 1:
        raise FixtureError(
            "judge cache entries use several models: " + _names(judge_models)
        )
    judge_providers = {e.provider for e in judge}
    if len(judge_providers) > 1 or (
        providers and judge_providers and providers != judge_providers
    ):
        raise FixtureError(
            "judge and agent cache entries use different providers: "
            + _names(providers | judge_providers)
        )
    provider = next(iter(providers | judge_providers), None)
    return (
        provider,
        next(iter(models), None),
        next(iter(judge_models), None),
    )


def _names(values: set[str]) -> str:
    return ", ".join(clean(v) for v in sorted(values))


def export_suite(
    conn: psycopg.Connection[Row], suite_run_id: str, out: Path, settings: Settings
) -> Manifest:
    """Write the five fixture files for a suite run into ``out``."""
    suite_row = runs_repo.get_suite_run(conn, suite_run_id)
    if suite_row is None:
        raise FixtureError(f"suite run not found: {suite_run_id}")
    suite = SuiteRow.model_validate(suite_row)
    runs = sorted(
        (
            RunRow.model_validate(r)
            for r in runs_repo.list_suite_run_rows(conn, suite.id)
        ),
        key=lambda r: (r.started_at, r.id),
    )
    spans = [
        SpanRow.model_validate(r) for r in runs_repo.list_suite_spans(conn, suite.id)
    ]
    scores = [
        ScoreRow.model_validate(r)
        for r in runs_repo.list_suite_score_rows(conn, suite.id)
    ]

    wanted = {h for hashes in collect_hashes(spans).values() for h in hashes}
    spans_by_run: dict[str, list[SpanRow]] = {}
    for span in spans:
        spans_by_run.setdefault(span.run_id, []).append(span)
    runs_by_id = {r.id: r for r in runs}
    judge_keys: dict[str, str] = {}
    needed: set[str] = set()
    for score in scores:
        if score.check_name != "judge" or score.kind != "judge":
            continue
        run = runs_by_id[score.run_id]
        key = score.detail.get("request_hash")
        if not isinstance(key, str):
            key = judge_hash(run, spans_by_run.get(run.id, []), settings)
        if key is None:
            continue
        judge_keys[run.id] = key
        if judge_needs_entry(score):
            needed.add(run.id)
    entries = [
        CacheRow.model_validate(r)
        for r in cache_repo.get_entries(conn, sorted(wanted | set(judge_keys.values())))
    ]
    present = {e.request_hash for e in entries}
    missing = sorted(rid for rid in needed if judge_keys[rid] not in present)
    if missing:
        raise FixtureError(
            "judge cache entry missing for runs "
            + ", ".join(missing)
            + " (scored with different settings or tasks than this export uses?)"
        )
    judge_requests = {
        rid: key for rid, key in sorted(judge_keys.items()) if key in present
    }
    judge_used = set(judge_requests.values())
    entries = [e for e in entries if e.request_hash in wanted | judge_used]
    provider, agent_model, judge_model = pin_models(entries, wanted, judge_used)

    manifest = Manifest(
        format_version=FORMAT_VERSION,
        suite_run_id=suite.id,
        variant=suite.variant,
        task_count=suite.task_count,
        run_count=len(runs),
        tasks_sha256=suite.tasks_sha256,
        suite_run=suite,
        judge_requests=judge_requests,
        provider=provider,
        agent_model=agent_model,
        judge_model=judge_model,
    )
    out.mkdir(parents=True, exist_ok=True)
    files = {
        MANIFEST: json.dumps(_dump(manifest), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        RUNS: _jsonl([_dump(r) for r in runs]),
        SPANS: _jsonl([_dump(s) for s in spans]),
        SCORES: _jsonl([_dump(s) for s in scores]),
        CACHE: _jsonl([_dump(e) for e in entries]),
    }
    for name, text in files.items():
        _write_replacing(out / name, text.encode("utf-8"))
    return manifest


def _write_replacing(path: Path, data: bytes) -> None:
    """Write through a temp file and rename it over ``path``.

    The rename replaces an existing symlink at ``path`` instead of writing through it.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


MAX_FILE_BYTES = 50 * 1024 * 1024
_IMPORT_LOCK = 0x6172_696D  # any fixed key; only fixture imports take it


def _read_text(path: Path) -> str:
    """Read a regular, size-capped UTF-8 file (symlinks, FIFOs and devices are refused).

    The checks run on the opened descriptor, so the file cannot be swapped between the
    check and the read.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if path.is_symlink():
            raise FixtureError(f"{path.name} is a symlink") from None
        raise FixtureError(f"cannot read {path.name} ({type(exc).__name__})") from None
    with os.fdopen(fd, "rb") as fh:
        try:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise FixtureError(f"{path.name} is not a regular file")
            if st.st_size > MAX_FILE_BYTES:
                raise FixtureError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
            data = fh.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise FixtureError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
            return data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise FixtureError(
                f"cannot read {path.name} ({type(exc).__name__})"
            ) from None


def _read_lines(path: Path, model: type[_Strict]) -> list[Any]:
    rows: list[Any] = []
    for lineno, line in jsonio.jsonl_lines(_read_text(path)):
        try:
            rows.append(model.model_validate(jsonio.loads(line)))
        except (ValueError, ValidationError, RecursionError) as exc:
            detail = _first_error(exc)
            raise FixtureError(f"{path.name}:{lineno}: invalid row{detail}") from None
    return rows


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return f" ({first['type']} at {where})" if where else f" ({first['type']})"
    return " (not valid JSON)"


def load_bundle(directory: Path) -> Bundle:
    """Read and validate a fixture directory; raises FixtureError on any bad file."""
    if not directory.is_dir():
        raise FixtureError(f"not a directory: {directory}")
    try:
        manifest = Manifest.model_validate(
            jsonio.loads(_read_text(directory / MANIFEST))
        )
    except (ValueError, ValidationError, RecursionError) as exc:
        raise FixtureError(f"{MANIFEST}: invalid manifest{_first_error(exc)}") from None
    if manifest.format_version != FORMAT_VERSION:
        raise FixtureError(f"unsupported fixture format {manifest.format_version}")
    bundle = Bundle(
        manifest=manifest,
        runs=_read_lines(directory / RUNS, RunRow),
        spans=_read_lines(directory / SPANS, SpanRow),
        scores=_read_lines(directory / SCORES, ScoreRow),
        cache=_read_lines(directory / CACHE, CacheRow),
    )
    _check_references(bundle)
    return bundle


def _check_references(bundle: Bundle) -> None:
    suite_id = bundle.manifest.suite_run_id
    if bundle.manifest.suite_run.id != suite_id:
        raise FixtureError("manifest suite_run does not match suite_run_id")
    run_ids = {r.id for r in bundle.runs}
    if len(run_ids) != len(bundle.runs):
        raise FixtureError(f"{RUNS}: duplicate run id")
    if len(bundle.runs) != bundle.manifest.run_count:
        raise FixtureError(
            f"{MANIFEST}: run_count is {bundle.manifest.run_count} but {RUNS} "
            f"has {len(bundle.runs)} runs"
        )
    if bundle.manifest.task_count != bundle.manifest.suite_run.task_count:
        raise FixtureError(f"{MANIFEST}: task_count does not match suite_run")
    if any(r.suite_run_id != suite_id for r in bundle.runs):
        raise FixtureError(f"{RUNS}: run does not belong to the manifest's suite")
    if any(s.run_id not in run_ids for s in bundle.spans):
        raise FixtureError(f"{SPANS}: span refers to a run not in {RUNS}")
    if any(s.run_id not in run_ids for s in bundle.scores):
        raise FixtureError(f"{SCORES}: score refers to a run not in {RUNS}")
    if any(r not in run_ids for r in bundle.manifest.judge_requests):
        raise FixtureError(f"{MANIFEST}: judge_requests names an unknown run")
    for entry in bundle.cache:
        try:
            actual = request_hash(ModelRequest.model_validate(entry.request))
        except ValidationError:
            actual = ""
        if actual != entry.request_hash:
            raise FixtureError(
                f"{CACHE}: entry {clean(entry.request_hash)} does not match its request"
            )


def _content(response: Any) -> tuple[Any, list[tuple[Any, Any]]]:
    """What a replay acts on: the text and the (name, args) of each tool call."""
    if not isinstance(response, dict):
        return None, []
    calls = response.get("tool_calls")
    pairs = [
        (c.get("name"), c.get("args"))
        for c in (calls if isinstance(calls, list) else [])
        if isinstance(c, dict)
    ]
    return response.get("text"), pairs


def verify_bundle(bundle: Bundle) -> dict[str, list[str]]:
    """Gaps per run id (sorted): missing or disagreeing cache entries, unscored runs and
    tool spans without a result."""
    cache = {e.request_hash: e for e in bundle.cache}
    gaps: dict[str, list[str]] = {}

    def add(run_id: str, text: str) -> None:
        gaps.setdefault(run_id, []).append(clean(text))

    for run_id, hashes in collect_hashes(bundle.spans).items():
        for key in hashes:
            if key not in cache:
                add(run_id, f"missing model cache entry {key}")
    for span in bundle.spans:
        response = span.payload.get("response")
        recorded = span.payload.get("request_hash")
        entry = cache.get(recorded) if isinstance(recorded, str) else None
        if (
            span.kind == "model"
            and response is not None
            and entry is not None
            and _content(entry.response) != _content(response)
        ):
            add(span.run_id, f"cache entry {recorded} disagrees with span {span.seq}")
    scored: dict[str, set[str]] = {}
    for score in bundle.scores:
        scored.setdefault(score.run_id, set()).add(
            "judge" if score.kind == "judge" else "deterministic"
        )
    for run in bundle.runs:
        kinds = scored.get(run.id, set())
        if "deterministic" not in kinds:
            add(run.id, "incomplete baseline: run has no deterministic scores")
        if "judge" not in kinds:
            add(run.id, "incomplete baseline: run has no judge score")
    judge_keys = dict(bundle.manifest.judge_requests)
    for score in bundle.scores:
        recorded = score.detail.get("request_hash")
        if (
            score.check_name == "judge"
            and score.kind == "judge"
            and isinstance(recorded, str)
            and judge_needs_entry(score)
        ):
            judge_keys[score.run_id] = recorded
    for score in bundle.scores:
        unanswered = score.passed is None and not judge_needs_entry(score)
        if score.check_name == "judge" and score.kind == "judge" and unanswered:
            verdict = score.detail.get("verdict")
            add(score.run_id, f"incomplete baseline: judge verdict {verdict}")
    for run in bundle.runs:
        if run.status == "skipped":
            add(run.id, "incomplete baseline: run skipped")
        if run.error_type in STOP_CODES:
            add(run.id, f"incomplete baseline: run stopped ({run.error_type})")
    for run_id, key in judge_keys.items():
        if key not in cache:
            add(run_id, f"missing judge cache entry {key}")
    for span in bundle.spans:
        payload = span.payload
        if (
            span.kind == "tool"
            and not payload.get("raised")
            and payload.get("result") is None
        ):
            add(span.run_id, f"missing tool result for {payload.get('name', '?')}")
    return {k: gaps[k] for k in sorted(gaps)}


def import_bundle(conn: psycopg.Connection[Row], bundle: Bundle) -> dict[str, int]:
    """Insert the bundle in one transaction; identical existing rows are left alone.

    Raises FixtureError (and rolls back) when an existing row differs.
    """
    try:
        with conn.transaction():
            # Imports check before they insert; one at a time keeps two of them from
            # both deciding a row is new.
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_IMPORT_LOCK,))
            new_suite = runs_repo.import_suite_run(
                conn, _plain(bundle.manifest.suite_run)
            )
            new_runs = runs_repo.import_runs(
                conn, [_plain(r) for r in bundle.runs], new_suite
            )
            counts = {
                "suite_runs": int(new_suite),
                "runs": len(new_runs),
                "cache": cache_repo.import_entries(
                    conn, [_plain(e) for e in bundle.cache]
                ),
                "spans": runs_repo.import_spans(
                    conn, [_plain(s) for s in bundle.spans], new_runs
                ),
                "scores": runs_repo.import_scores(
                    conn, [_plain(s) for s in bundle.scores], new_runs
                ),
            }
    except cache_repo.ImportConflict as exc:
        raise FixtureError(
            "import refused, nothing changed: " + clean(str(exc), 300)
        ) from None
    return counts


def _plain(model: _Strict) -> Row:
    return model.model_dump()
