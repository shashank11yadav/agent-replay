import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RUN_ID, validRun } from "../../test/helpers";
import type { HttpError, ValidationError } from "./client";

// The client remembers that the host has answered, so every test starts from a fresh module.
let client: typeof import("./client");
beforeEach(async () => {
  vi.resetModules();
  client = await import("./client");
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

describe("apiBase", () => {
  it("returns /api when VITE_API_URL is unset", () => {
    vi.stubEnv("VITE_API_URL", "");
    expect(client.apiBase()).toBe("/api");
  });

  it("returns VITE_API_URL when set", () => {
    vi.stubEnv("VITE_API_URL", "https://api.example.test/api");
    expect(client.apiBase()).toBe("https://api.example.test/api");
  });
});

describe("fetchRun", () => {
  it("fetches {base}/runs/{id} and returns validated data", async () => {
    vi.stubEnv("VITE_API_URL", "");
    const f = vi.fn().mockResolvedValue(jsonResponse(validRun()));
    vi.stubGlobal("fetch", f);
    const run = await client.fetchRun(RUN_ID);
    expect(run.id).toBe(RUN_ID);
    expect(run.spans).toHaveLength(2);
    const url = String(f.mock.calls[0]?.[0]);
    expect(url).toContain(`/api/runs/${RUN_ID}`);
  });

  it("rejects with ValidationError listing issue paths when spans is missing", async () => {
    const bad = validRun();
    delete bad.spans;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(bad)));
    const err = await client.fetchRun(RUN_ID).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(client.ValidationError);
    expect(err).toBeInstanceOf(Error);
    expect((err as ValidationError).issues).toContain("spans");
  });

  it("never returns partial data on invalid input", async () => {
    const bad = { ...validRun(), spans: "nope" };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(bad)));
    await expect(client.fetchRun(RUN_ID)).rejects.toBeInstanceOf(client.ValidationError);
  });

  it("retries a first network failure once after a delay, reporting 'waking'", async () => {
    vi.useFakeTimers();
    const f = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(jsonResponse(validRun()));
    vi.stubGlobal("fetch", f);
    const states: string[] = [];
    const p = client.fetchRun(RUN_ID, { onState: (s: string) => states.push(s) });
    await vi.advanceTimersByTimeAsync(0);
    expect(f).toHaveBeenCalledTimes(1); // retry is delayed, not immediate
    expect(states).toContain("waking");
    await vi.advanceTimersByTimeAsync(120_000);
    const run = await p;
    expect(f).toHaveBeenCalledTimes(2);
    expect(run.id).toBe(RUN_ID);
    expect(states.indexOf("waking")).toBeLessThan(states.length);
    expect(states[states.length - 1]).not.toBe("waking");
  });

  it("gives up after exactly one retry", async () => {
    vi.useFakeTimers();
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", f);
    const p = client.fetchRun(RUN_ID).then(
      () => "resolved",
      (e: unknown) => e,
    );
    await vi.advanceTimersByTimeAsync(300_000);
    const out = await p;
    expect(out).toBeInstanceOf(Error);
    expect(f).toHaveBeenCalledTimes(2);
  });

  it.each([502, 503, 504])("retries once after a delay when the first answer is %i", async (status) => {
    vi.useFakeTimers();
    const f = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "waking" }, status))
      .mockResolvedValueOnce(jsonResponse(validRun()));
    vi.stubGlobal("fetch", f);
    const states: string[] = [];
    const p = client.fetchRun(RUN_ID, { onState: (s: string) => states.push(s) });
    await vi.advanceTimersByTimeAsync(0);
    expect(f).toHaveBeenCalledTimes(1);
    expect(states).toContain("waking");
    await vi.advanceTimersByTimeAsync(120_000);
    expect((await p).id).toBe(RUN_ID);
    expect(f).toHaveBeenCalledTimes(2);
  });

  it("reports the status when the retry also gets a gateway error", async () => {
    vi.useFakeTimers();
    const f = vi.fn().mockImplementation(async () => jsonResponse({}, 503));
    vi.stubGlobal("fetch", f);
    const p = client.fetchRun(RUN_ID).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(120_000);
    const err = await p;
    expect(err).toBeInstanceOf(client.HttpError);
    expect((err as HttpError).status).toBe(503);
    expect(f).toHaveBeenCalledTimes(2);
  });

  it.each([404, 500])("does not retry a %i", async (status) => {
    const f = vi.fn().mockResolvedValue(jsonResponse({}, status));
    vi.stubGlobal("fetch", f);
    const err = await client.fetchRun(RUN_ID).catch((e: unknown) => e);
    expect((err as HttpError).status).toBe(status);
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("rejects with ValidationError when a 2xx body is not JSON", async () => {
    vi.stubEnv("VITE_API_URL", "");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>oops</html>", { status: 200 })));
    const err = await client.fetchRun("x").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(client.ValidationError);
    expect((err as ValidationError).issues).toEqual(["(body is not valid JSON)"]);
  });

  it("does not report 'waking' when the first request succeeds", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(validRun())));
    const states: string[] = [];
    await client.fetchRun(RUN_ID, { onState: (s: string) => states.push(s) });
    expect(states).not.toContain("waking");
  });

  it("does not wait and retry once the host has answered", async () => {
    vi.useFakeTimers();
    const f = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(validRun()))
      .mockResolvedValueOnce(jsonResponse({}, 503));
    vi.stubGlobal("fetch", f);
    await client.fetchRun(RUN_ID);
    const states: string[] = [];
    const err = await client.fetchRun(RUN_ID, { onState: (s: string) => states.push(s) }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(client.HttpError);
    expect((err as HttpError).status).toBe(503);
    expect(f).toHaveBeenCalledTimes(2);
    expect(states).not.toContain("waking");
  });

  it("passes the abort signal to fetch", async () => {
    const f = vi.fn().mockResolvedValue(jsonResponse(validRun()));
    vi.stubGlobal("fetch", f);
    const ac = new AbortController();
    await client.fetchRun(RUN_ID, { signal: ac.signal });
    expect((f.mock.calls[0]?.[1] as RequestInit | undefined)?.signal).toBe(ac.signal);
  });

  it("stops waiting for the retry when aborted", async () => {
    vi.useFakeTimers();
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", f);
    const ac = new AbortController();
    const p = client.fetchRun(RUN_ID, { signal: ac.signal }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(0);
    ac.abort();
    const err = await p;
    expect((err as Error).name).toBe("AbortError");
    await vi.advanceTimersByTimeAsync(120_000);
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("does not retry a request that was aborted", async () => {
    const ac = new AbortController();
    const f = vi.fn().mockImplementation(async () => {
      ac.abort();
      throw new DOMException("The operation was aborted.", "AbortError");
    });
    vi.stubGlobal("fetch", f);
    const err = await client.fetchRun(RUN_ID, { signal: ac.signal }).catch((e: unknown) => e);
    expect((err as Error).name).toBe("AbortError");
    expect(f).toHaveBeenCalledTimes(1);
  });
});
