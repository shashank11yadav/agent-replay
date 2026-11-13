import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RunPage } from "../../src/pages/RunPage";
import App from "../../src/App";
import { RUN_1, requestSignals, stepRun, stubApi } from "../helpers";

afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

// The run page loads GET /api/runs/<id> and keeps the current step in the URL, so it is
// rendered through App: Previous / Next change the hash and App passes the new step back down.
function load(over: Record<string, unknown> = {}, hash = `#/runs/${RUN_1}`) {
  stubApi((url) => (url.pathname.endsWith(`/runs/${RUN_1}`) ? { body: stepRun(over) } : undefined));
  window.location.hash = hash;
  render(<App />);
}

const text = () => document.body.textContent ?? "";
const stepNumber = () => Number(/step\s*(\d+)\s*of/i.exec(text())?.[1]);

/** Click Next and wait for the hash change to reach the page. */
async function next() {
  const before = stepNumber();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  await waitFor(() => expect(stepNumber()).toBe(before + 1));
}

describe("RunPage", () => {
  it("aborts its request when it unmounts", async () => {
    const fn = stubApi(() => undefined);
    const { unmount } = render(<RunPage id={RUN_1} />);
    await waitFor(() => expect(fn).toHaveBeenCalled());
    const signals = requestSignals(fn);
    expect(signals.every((s) => s instanceof AbortSignal && !s.aborted)).toBe(true);
    unmount();
    expect(signals.every((s) => s?.aborted)).toBe(true);
  });

  it("shows step 1 of N with a heading naming the span", async () => {
    load();
    expect(await screen.findByRole("heading", { name: /execute_tool get_order/ })).toBeTruthy();
    expect(text()).toMatch(/step\s*1\s*of\s*4/i);
  });

  it("Next shows the next step; Previous goes back", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    await next();
    expect(await screen.findByRole("heading", { name: /chat gemini-test/ })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /execute_tool get_order/ })).toBeNull();
    expect(text()).toMatch(/step\s*2\s*of\s*4/i);
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    expect(await screen.findByRole("heading", { name: /execute_tool get_order/ })).toBeTruthy();
  });

  it("Previous is disabled on the first step and Next on the last", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    const prev = () => screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement;
    const nxt = () => screen.getByRole("button", { name: "Next" }) as HTMLButtonElement;
    expect(prev().disabled).toBe(true);
    expect(nxt().disabled).toBe(false);
    await next();
    await next();
    await next();
    expect(text()).toMatch(/step\s*4\s*of\s*4/i);
    expect(nxt().disabled).toBe(true);
    expect(prev().disabled).toBe(false);
  });

  it("opens the step named in the URL", async () => {
    load({}, `#/runs/${RUN_1}/3`);
    expect(await screen.findByRole("heading", { name: /execute_tool refund/ })).toBeTruthy();
    expect(text()).toMatch(/step\s*3\s*of\s*4/i);
  });

  it("shows the last step when the URL names a step past the end", async () => {
    load({}, `#/runs/${RUN_1}/99`);
    await screen.findByRole("heading", { name: /chat gemini-test/ });
    expect(text()).toMatch(/step\s*4\s*of\s*4/i);
  });

  it("Next puts the step in the URL without adding a history entry", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    const before = window.history.length;
    await next();
    await screen.findByRole("heading", { name: /chat gemini-test/ });
    expect(window.location.hash).toBe(`#/runs/${RUN_1}/2`);
    expect(window.history.length).toBe(before);
  });

  it("ArrowRight and ArrowLeft move between steps", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(await screen.findByRole("heading", { name: /chat gemini-test/ })).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(await screen.findByRole("heading", { name: /execute_tool get_order/ })).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(window.location.hash).toBe(`#/runs/${RUN_1}/1`);
  });

  it("ignores arrow keys pressed with a modifier", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    fireEvent.keyDown(window, { key: "ArrowRight", altKey: true });
    expect(window.location.hash).toBe(`#/runs/${RUN_1}`);
  });

  it("model step shows request messages, response text, tokens, latency and cache status", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    await next();
    await screen.findByRole("heading", { name: /chat gemini-test/ });
    const t = text();
    expect(t).toContain("REQUEST-MSG-where-is-it");
    expect(t).toContain("RESPONSE-TEXT-on-its-way");
    expect(t).toContain("111");
    expect(t).toContain("222");
    expect(t).toContain("333");
    expect(t).toMatch(/hit/);
  });

  it("model step shows tool calls when the response has them", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    await next();
    await next();
    await next();
    await screen.findByRole("heading", { name: /chat gemini-test/ });
    const t = text();
    expect(t).toContain("lookup_TOOLCALL");
    expect(t).toContain("ZZ-CALL-ARG");
    expect(t).toMatch(/miss/);
    expect(t).not.toContain("RESPONSE-TEXT-on-its-way");
  });

  it("tool step shows arguments, result and latency_ms, with no fault when absent", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    const t = text();
    expect(t).toContain("ORD-ARG-42");
    expect(t).toContain("shipped-RESULT-9");
    expect(t).toContain("77");
    expect(t).not.toContain("tool_timeout");
    expect(t).not.toContain("TimeoutError");
  });

  it("tool step shows the fault and error type when present", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    await next();
    await next();
    await screen.findByRole("heading", { name: /execute_tool refund/ });
    const t = text();
    expect(t).toContain("tool_timeout:refund");
    expect(t).toContain("TimeoutError");
    expect(t).toContain("555");
    expect(t).toContain("4444");
  });

  it("model step with status error shows its error type", async () => {
    const base = stepRun() as { spans: Record<string, unknown>[] };
    const spans = base.spans.map((s) =>
      s.seq === 1 ? { ...s, status: "error", error_type: "rate_limit" } : s,
    );
    stubApi((url) =>
      url.pathname.endsWith(`/runs/${RUN_1}`) ? { body: { ...base, spans } } : undefined,
    );
    window.location.hash = `#/runs/${RUN_1}`;
    render(<App />);
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    expect(text()).not.toContain("rate_limit");
    await next();
    await screen.findByRole("heading", { name: /chat gemini-test/ });
    expect(text()).toContain("rate_limit");
  });

  it.each(["diverged", "error", "skipped"])("shows the run status %s in the heading", async (status) => {
    load({ status });
    const h1 = await screen.findByRole("heading", { level: 1, name: new RegExp(`run task-1 ${status}`, "i") });
    expect(h1.querySelector(`.badge-${status}`)?.textContent).toBe(status);
    // A badge is not a live region; only loading messages are.
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("announces loading in a live region under the page heading", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    window.location.hash = `#/runs/${RUN_1}`;
    render(<App />);
    const status = screen.getByRole("status");
    expect(status.getAttribute("aria-live")).toBe("polite");
    expect(status.textContent).toMatch(/loading/i);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("titles the tab with the task id", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    expect(document.title).toBe("Run task-1 · agent-replay");
  });

  it("has one h1 and puts the site header before main", async () => {
    load();
    await screen.findByRole("heading", { name: /execute_tool get_order/ });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const header = screen.getByRole("banner");
    const main = screen.getByRole("main");
    expect(main.contains(header)).toBe(false);
    expect(header.compareDocumentPosition(main) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("an API error shows role=alert instead of an empty page", async () => {
    stubApi(() => ({ status: 500, body: { detail: "boom" } }));
    render(<RunPage id={RUN_1} />);
    const alert = await screen.findByRole("alert");
    expect((alert.textContent ?? "").trim().length).toBeGreaterThan(0);
  });

  it("an invalid body shows role=alert", async () => {
    stubApi(() => ({ body: { id: 1 } }));
    render(<RunPage id={RUN_1} />);
    expect(await screen.findByRole("alert")).toBeTruthy();
  });

  it("App routes #/runs/<id> to the run page", async () => {
    stubApi((url) => (url.pathname.endsWith(`/runs/${RUN_1}`) ? { body: stepRun() } : undefined));
    window.location.hash = `#/runs/${RUN_1}`;
    render(<App />);
    expect(await screen.findByRole("heading", { name: /execute_tool get_order/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Next" })).toBeTruthy();
  });
});
