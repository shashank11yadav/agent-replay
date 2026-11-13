"""Tracer setup and span helpers built on a local OTel TracerProvider.

Each ``new_tracer()`` call gets its own provider and ``CollectingSpanExporter``;
the global tracer provider is never set and no atexit hook is registered, so a
provider is freed with its tracer and exporter (or shut down explicitly via
``new_tracer_with_provider()``). Spans are converted to plain, JSON-serialisable
dicts as they end (``SimpleSpanProcessor`` is synchronous); each dict carries a
``seq`` integer in span START order. Never put API keys or other secrets in
span attributes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
    format_span_id,
    format_trace_id,
)

from agent_replay import semconv

_STATUS_NAMES = {
    StatusCode.UNSET: "unset",
    StatusCode.OK: "ok",
    StatusCode.ERROR: "error",
}


def _plain(value: Any) -> Any:
    """Turn OTel attribute values (tuples for sequences) into JSON types."""
    if isinstance(value, tuple | list):
        return list(value)
    return value


def span_to_dict(span: ReadableSpan, seq: int | None = None) -> dict[str, Any]:
    """Convert a finished span into the dict shape stored by the recorder."""
    ctx = span.context
    if ctx is None:
        raise ValueError(f"span {span.name!r} has no span context")
    attributes = {k: _plain(v) for k, v in (span.attributes or {}).items()}
    error_type = attributes.get(semconv.ERROR_TYPE)
    return {
        "seq": seq,
        "trace_id": format_trace_id(ctx.trace_id),
        "span_id": format_span_id(ctx.span_id),
        "parent_span_id": (
            format_span_id(span.parent.span_id) if span.parent is not None else None
        ),
        "name": span.name,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": _STATUS_NAMES[span.status.status_code],
        "error_type": str(error_type) if error_type is not None else None,
        "attributes": attributes,
    }


class CollectingSpanExporter(SpanExporter):
    """Keeps every exported span as a plain dict.

    ``spans`` is in end order; each dict's ``seq`` (0, 1, ...) is the span's
    start order, so ``sorted(spans, key=lambda s: s["seq"])`` gives step order.
    """

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._start_seq: dict[int, int] = {}
        self._next_seq = 0

    def _take_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def note_start(self, span_id: int) -> None:
        """Record that a span started; called by the provider's processor."""
        self._start_seq[span_id] = self._take_seq()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            seq = self._start_seq.pop(s.context.span_id, None) if s.context else None
            self.spans.append(span_to_dict(s, self._take_seq() if seq is None else seq))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def clear(self) -> None:
        self.spans.clear()
        self._start_seq.clear()
        self._next_seq = 0


class _StartOrderProcessor(SpanProcessor):
    """Tells the exporter when each span starts, for start-order ``seq``."""

    def __init__(self, exporter: CollectingSpanExporter) -> None:
        self._exporter = exporter

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        ctx = span.get_span_context()
        self._exporter.note_start(ctx.span_id)


def new_tracer_with_provider() -> tuple[TracerProvider, Tracer, CollectingSpanExporter]:
    """Like ``new_tracer`` but also returns the provider, for explicit shutdown."""
    exporter = CollectingSpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_StartOrderProcessor(exporter))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, provider.get_tracer(__name__), exporter


def new_tracer() -> tuple[Tracer, CollectingSpanExporter]:
    """Return a tracer on a fresh local provider plus its collecting exporter."""
    _, tracer, exporter = new_tracer_with_provider()
    return tracer, exporter


class StepSpan:
    """Handle on a live span so values known only after the step can be set."""

    def __init__(self, span: Span) -> None:
        self.span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)

    def set_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        self.span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

    def set_result(self, result: Any) -> None:
        """Store a tool result as a JSON string."""
        self.span.set_attribute(
            semconv.GEN_AI_TOOL_CALL_RESULT, json.dumps(result, sort_keys=True)
        )

    def set_error_type(self, error_type: str) -> None:
        """Set ``error.type``; an exception leaving the span will not overwrite it."""
        self.span.set_attribute(semconv.ERROR_TYPE, error_type)


def _has_error_type(span: Span) -> bool:
    attributes = getattr(span, "attributes", None)
    return attributes is not None and semconv.ERROR_TYPE in attributes


@contextmanager
def _span(tracer: Tracer, name: str, attributes: dict[str, Any]) -> Iterator[StepSpan]:
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=True,
        set_status_on_exception=True,
    ) as span:
        try:
            yield StepSpan(span)
        except BaseException as exc:
            if not _has_error_type(span):
                span.set_attribute(semconv.ERROR_TYPE, type(exc).__name__)
            if not isinstance(exc, Exception):
                # The OTel API only marks ``Exception`` subclasses as errors.
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def agent_span(tracer: Tracer, name: str) -> AbstractContextManager[StepSpan]:
    """Root span ``invoke_agent <name>`` for one agent run."""
    return _span(
        tracer,
        f"{semconv.OP_INVOKE_AGENT} {name}",
        {semconv.GEN_AI_OPERATION_NAME: semconv.OP_INVOKE_AGENT},
    )


def model_span(
    tracer: Tracer,
    model: str,
    *,
    provider: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> AbstractContextManager[StepSpan]:
    """Model-call span ``generate_content <model>``.

    Usage counts left as ``None`` are not set; call ``set_usage`` on the
    yielded handle once the response is known.
    """
    attributes: dict[str, Any] = {
        semconv.GEN_AI_OPERATION_NAME: semconv.OP_GENERATE_CONTENT,
        semconv.GEN_AI_PROVIDER_NAME: provider,
        semconv.GEN_AI_REQUEST_MODEL: model,
    }
    if input_tokens is not None:
        attributes[semconv.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attributes[semconv.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    return _span(tracer, f"{semconv.OP_GENERATE_CONTENT} {model}", attributes)


def tool_span(
    tracer: Tracer,
    name: str,
    *,
    call_id: str,
    arguments: Any,
    result: Any = None,
) -> AbstractContextManager[StepSpan]:
    """Tool-call span ``execute_tool <name>``; arguments/result stored as JSON.

    A ``None`` result is not set; call ``set_result`` on the yielded handle
    once the tool has returned.
    """
    attributes: dict[str, Any] = {
        semconv.GEN_AI_OPERATION_NAME: semconv.OP_EXECUTE_TOOL,
        semconv.GEN_AI_TOOL_NAME: name,
        semconv.GEN_AI_TOOL_CALL_ID: call_id,
        semconv.GEN_AI_TOOL_CALL_ARGUMENTS: json.dumps(arguments, sort_keys=True),
    }
    if result is not None:
        attributes[semconv.GEN_AI_TOOL_CALL_RESULT] = json.dumps(result, sort_keys=True)
    return _span(tracer, f"{semconv.OP_EXECUTE_TOOL} {name}", attributes)
