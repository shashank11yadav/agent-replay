import type * as z from "zod";
import {
  RunDiffSchema,
  RunListSchema,
  RunSchema,
  SuiteRunListSchema,
  type Run,
  type RunDiff,
  type RunListItem,
  type SuiteRun,
} from "./schemas";

export type LoadState = "loading" | "waking" | "ready";

export interface FetchOptions {
  onState?: (state: LoadState) => void;
  /** Aborts the request and any wait for its retry (pages pass their effect's signal). */
  signal?: AbortSignal;
}

/** Delay before the single retry of a failed request while the host may be cold-starting. */
export const RETRY_DELAY_MS = 30_000;

/**
 * Set after the first successful response. A host that has answered is awake, so later
 * failures are reported at once instead of after a 30 s wait.
 */
let hostAnswered = false;

/** Statuses a proxy in front of a sleeping host answers with while it wakes. */
const GATEWAY_STATUSES = new Set([502, 503, 504]);

/** The response body did not match the expected schema; `issues` lists the failing paths. */
export class ValidationError extends Error {
  readonly issues: string[];

  constructor(issues: string[]) {
    super(`Invalid API response: ${issues.join(", ")}`);
    this.name = "ValidationError";
    this.issues = issues;
  }
}

/** The server answered with a non-2xx status. */
export class HttpError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(`API request failed with status ${status}`);
    this.name = "HttpError";
    this.status = status;
  }
}

export function apiBase(): string {
  return import.meta.env.VITE_API_URL || "/api";
}

function abortError(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError(signal));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError(signal as AbortSignal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function getJson<S extends z.ZodType>(
  path: string,
  schema: S,
  opts: FetchOptions = {},
): Promise<z.infer<S>> {
  const url = `${apiBase()}${path}`;
  const { signal } = opts;
  opts.onState?.("loading");
  let res: Response | null = null;
  try {
    res = await fetch(url, { signal });
  } catch (e) {
    if (hostAnswered || signal?.aborted) throw e;
  }
  // Before the host has ever answered, a network failure (res is null) or a gateway error
  // usually means it is cold-starting. Wait, then retry exactly once.
  if (res === null || (!hostAnswered && GATEWAY_STATUSES.has(res.status))) {
    opts.onState?.("waking");
    await sleep(RETRY_DELAY_MS, signal);
    opts.onState?.("loading");
    res = await fetch(url, { signal });
  }
  if (!res.ok) throw new HttpError(res.status);
  hostAnswered = true;
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new ValidationError(["(body is not valid JSON)"]);
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    throw new ValidationError(
      parsed.error.issues.map((i) => (i.path.length ? i.path.join(".") : "(root)")),
    );
  }
  opts.onState?.("ready");
  return parsed.data;
}

const enc = encodeURIComponent;

export function fetchRun(id: string, opts?: FetchOptions): Promise<Run> {
  return getJson(`/runs/${enc(id)}`, RunSchema, opts);
}

export function fetchRuns(
  params: { suiteRunId?: string; limit?: number; offset?: number } = {},
  opts?: FetchOptions,
): Promise<RunListItem[]> {
  const q = new URLSearchParams();
  if (params.suiteRunId) q.set("suite_run_id", params.suiteRunId);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  const qs = q.toString();
  return getJson(`/runs${qs ? `?${qs}` : ""}`, RunListSchema, opts);
}

export function fetchSuiteRuns(
  params: { limit?: number } = {},
  opts?: FetchOptions,
): Promise<SuiteRun[]> {
  const qs = params.limit === undefined ? "" : `?limit=${params.limit}`;
  return getJson(`/suite-runs${qs}`, SuiteRunListSchema, opts);
}

export function fetchDiff(left: string, right: string, opts?: FetchOptions): Promise<RunDiff> {
  return getJson(`/diff?left=${enc(left)}&right=${enc(right)}`, RunDiffSchema, opts);
}
