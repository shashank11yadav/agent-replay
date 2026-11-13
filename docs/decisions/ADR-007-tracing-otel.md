# ADR-007: Tracing: OTel SDK with a collecting exporter and pinned semconv names

- Date: 2026-11-02
- Status: accepted

## Context
Runs should be stored as OpenTelemetry-style spans so the data reads like any other agent trace
and could be exported later. opentelemetry-sdk 1.45.2 provides `TracerProvider`,
`SimpleSpanProcessor`, custom `SpanExporter` and `ReadableSpan`. The
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
are in Development status, so names may change. Instrumentations leave content capture off by
default, so a replay tool has to store request and response content itself.

## Decision
- Use the real OTel SDK with a per-run local `TracerProvider`, a `SimpleSpanProcessor`
  (synchronous, so no span is lost) and a `CollectingSpanExporter` that turns each
  `ReadableSpan` into a plain dict. The runner writes the dicts with the run row in one
  transaction at the end of the run.
- Span names and attribute keys come only from `semconv.py`. Full requests, responses, tool
  arguments and results go in the span's own `payload` (JSONB), not in OTel attributes beyond the
  semconv keys.
- No OTLP export and no global tracer provider.

## Alternatives considered
- Hand-rolled span structs: lose the OTel shape and its exception and status handling.
- Exporting through OTLP to a collector: out of scope; this is not a tracing platform.
- Writing to Postgres inside `export()`: database I/O on every span end, and a run is harder to
  make atomic.

## Consequences
- If semconv names change, only `semconv.py` and a migration of stored keys change.
- Trace and span ids are ints on `SpanContext`; they are stored as hex text via
  `format_trace_id` (32 chars) and `format_span_id` (16 chars).
