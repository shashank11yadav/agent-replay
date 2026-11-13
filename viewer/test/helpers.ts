import { vi } from "vitest";

export const SUITE_A = "aaaaaaaa-0000-4000-8000-000000000001";
export const SUITE_B = "aaaaaaaa-0000-4000-8000-000000000002";
export const RUN_1 = "bbbbbbbb-0000-4000-8000-000000000001";
export const RUN_2 = "bbbbbbbb-0000-4000-8000-000000000002";
export const RUN_3 = "bbbbbbbb-0000-4000-8000-000000000003";

const TS = "2026-01-01T00:00:00+00:00";

export function suiteRun(id: string, label: string): Record<string, unknown> {
  return {
    id,
    label,
    variant: "v1",
    mode: "record",
    status: "ok",
    tasks_path: null,
    tasks_sha256: null,
    task_count: 2,
    source_suite_run_id: null,
    faults: [],
    metrics: null,
    created_at: TS,
    finished_at: TS,
  };
}

export function span(seq: number, over: Record<string, unknown>): Record<string, unknown> {
  return {
    id: seq + 1,
    run_id: RUN_1,
    seq,
    trace_id: "tr",
    span_id: `sp${seq}`,
    parent_span_id: null,
    kind: "model",
    name: "chat",
    status: "ok",
    error_type: null,
    attributes: {},
    payload: {},
    input_tokens: 0,
    output_tokens: 0,
    latency_ms: 0,
    cost_usd: 0,
    cache_status: null,
    started_at: TS,
    ...over,
  };
}

export function runSummary(
  id: string,
  suite: string | null,
  over: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    suite_run_id: suite,
    source_run_id: null,
    task_id: "task-" + id.slice(-1),
    variant: "v1",
    mode: "record",
    status: "ok",
    error_type: null,
    error: null,
    final_output: "done",
    faults: [],
    input_tokens: 10,
    output_tokens: 5,
    latency_ms: 100,
    model_calls: 1,
    tool_calls: 1,
    cache_hits: 0,
    cache_misses: 1,
    cost_usd: 0.001,
    started_at: TS,
    finished_at: TS,
    ...over,
  };
}

/** A row of GET /api/runs: the run summary plus the task verdict (null when not fully scored). */
export function runListRow(
  id: string,
  suite: string | null,
  passed: boolean | null,
  over: Record<string, unknown> = {},
): Record<string, unknown> {
  return { ...runSummary(id, suite, over), passed };
}

export function score(runId: string, passed: boolean | null): Record<string, unknown> {
  return {
    run_id: runId,
    check_name: "contains",
    kind: "deterministic",
    passed,
    detail: {},
    created_at: TS,
  };
}

export function judge(runId: string, passed: boolean | null): Record<string, unknown> {
  return {
    run_id: runId,
    check_name: "judge",
    kind: "judge",
    passed,
    detail: {},
    created_at: TS,
  };
}

/** Run with four spans: tool, model (text), tool (faulted), model (tool calls). */
export function stepRun(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    ...runSummary(RUN_1, SUITE_A, over),
    spans: [
      span(0, {
        kind: "tool",
        name: "execute_tool get_order",
        latency_ms: 77,
        payload: {
          name: "get_order",
          call_id: "c1",
          args: { order_id: "ORD-ARG-42" },
          result: { status: "shipped-RESULT-9" },
          fault: null,
          raised: false,
          latency_ms: 77,
        },
      }),
      span(1, {
        kind: "model",
        name: "chat gemini-test",
        input_tokens: 111,
        output_tokens: 222,
        latency_ms: 333,
        cache_status: "hit",
        payload: {
          request: {
            provider: "gemini",
            model: "gemini-test",
            system: "be helpful",
            messages: [
              { role: "user", text: "REQUEST-MSG-where-is-it", tool_calls: [], tool_results: [] },
            ],
            tools: [],
          },
          response: {
            text: "RESPONSE-TEXT-on-its-way",
            tool_calls: [],
            input_tokens: 111,
            output_tokens: 222,
            latency_ms: 333,
            cache_status: "hit",
          },
          cache_status: "hit",
          latency_ms: 333,
        },
      }),
      span(2, {
        kind: "tool",
        name: "execute_tool refund",
        status: "error",
        error_type: "TimeoutError",
        latency_ms: 4444,
        payload: {
          name: "refund",
          call_id: "c2",
          args: { amount: 555 },
          result: { error: "tool_timeout" },
          fault: "tool_timeout:refund",
          raised: false,
          latency_ms: 4444,
        },
      }),
      span(3, {
        kind: "model",
        name: "chat gemini-test",
        input_tokens: 12,
        output_tokens: 13,
        latency_ms: 14,
        cache_status: "miss",
        payload: {
          request: {
            provider: "gemini",
            model: "gemini-test",
            system: "be helpful",
            messages: [],
            tools: [],
          },
          response: {
            text: null,
            tool_calls: [{ id: "c9", name: "lookup_TOOLCALL", args: { q: "ZZ-CALL-ARG" } }],
            input_tokens: 12,
            output_tokens: 13,
            latency_ms: 14,
            cache_status: "miss",
          },
          cache_status: "miss",
          latency_ms: 14,
        },
      }),
    ],
    scores: [score(RUN_1, true), judge(RUN_1, true)],
  };
}

type Handler = (url: URL) => { status?: number; body: unknown } | undefined;

/** Stub global fetch; routes by path (query-aware via the handler). Unmatched requests 404. */
export function stubApi(handler: Handler) {
  const fn = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    const hit = handler(url);
    const status = hit?.status ?? (hit ? 200 : 404);
    return new Response(JSON.stringify(hit?.body ?? {}), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

/** The abort signal of every request made through a stubApi mock. */
export function requestSignals(fn: ReturnType<typeof stubApi>): (AbortSignal | null | undefined)[] {
  return fn.mock.calls.map((c) => c[1]?.signal);
}

// A run shaped like the real GET /api/runs/{id} JSON (see src/agent_replay/api/routes.py).
export const RUN_ID = "11111111-1111-4111-8111-111111111111";

export function validRun(): Record<string, unknown> {
  return {
    id: RUN_ID,
    suite_run_id: "22222222-2222-4222-8222-222222222222",
    source_run_id: null,
    task_id: "t1",
    variant: "v1",
    mode: "record",
    status: "ok",
    error_type: null,
    error: null,
    final_output: "done",
    faults: [],
    input_tokens: 20,
    output_tokens: 10,
    latency_ms: 120,
    model_calls: 1,
    tool_calls: 1,
    cache_hits: 0,
    cache_misses: 1,
    cost_usd: 0.002,
    started_at: "2026-01-01T00:00:00+00:00",
    finished_at: "2026-01-01T00:00:01+00:00",
    spans: [
      {
        id: 1,
        run_id: RUN_ID,
        seq: 0,
        trace_id: "tr",
        span_id: "sp0",
        parent_span_id: null,
        kind: "model",
        name: "chat",
        status: "ok",
        error_type: null,
        attributes: {},
        payload: { text: "hello" },
        input_tokens: 5,
        output_tokens: 5,
        latency_ms: 10,
        cost_usd: 0.001,
        cache_status: "miss",
        started_at: "2026-01-01T00:00:00+00:00",
      },
      {
        id: 2,
        run_id: RUN_ID,
        seq: 1,
        trace_id: "tr",
        span_id: "sp1",
        parent_span_id: null,
        kind: "tool",
        name: "get_order",
        status: "ok",
        error_type: null,
        attributes: {},
        payload: { args: { id: "A1" }, result: "shipped" },
        input_tokens: 0,
        output_tokens: 0,
        latency_ms: 20,
        cost_usd: 0,
        cache_status: null,
        started_at: "2026-01-01T00:00:00+00:00",
      },
    ],
    scores: [
      {
        run_id: RUN_ID,
        check_name: "contains",
        kind: "deterministic",
        passed: true,
        detail: { why: "ok" },
        created_at: "2026-01-01T00:00:02+00:00",
      },
    ],
  };
}
