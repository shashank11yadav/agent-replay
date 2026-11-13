import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { DiffPage } from "../../src/pages/DiffPage";
import { RunPage } from "../../src/pages/RunPage";
import App from "../../src/App";
import { requestSignals, runSummary, span, stubApi } from "../helpers";

afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

const LEFT = "cccccccc-0000-4000-8000-000000000001";
const RIGHT = "cccccccc-0000-4000-8000-000000000002";
const SRC = "cccccccc-0000-4000-8000-000000000009";

// DiffPage loads GET /api/diff and shows one table row per aligned step, plus the run-level
// deltas as signed numbers.

function step(seq: number, name: string, status: string, over: Record<string, unknown> = {}) {
  return {
    seq,
    kind: name.startsWith("chat") ? "model" : "tool",
    name,
    status,
    changed: status !== "same",
    input_tokens_delta: 0,
    output_tokens_delta: 0,
    latency_ms_delta: 0,
    ...over,
  };
}

const delta = (left: number, right: number) => ({ left, right, delta: right - left });

function diffBody(over: Record<string, unknown> = {}) {
  return {
    first_divergence: 1,
    steps: [
      step(0, "execute_tool SAME-STEP", "same"),
      step(1, "chat CHANGED-STEP", "changed", { input_tokens_delta: 7 }),
      step(2, "execute_tool ADDED-STEP", "added"),
      step(3, "execute_tool REMOVED-STEP", "removed"),
    ],
    deltas: {
      input_tokens: delta(100, 220),
      output_tokens: delta(50, 20),
      latency_ms: delta(1000, 1250),
      model_calls: delta(2, 3),
      tool_calls: delta(5, 3),
      task_pass: { left: true, right: false },
    },
    ...over,
  };
}

function load(body: unknown = diffBody()) {
  const fn = stubApi((url) =>
    url.pathname.endsWith("/diff") &&
    url.searchParams.get("left") === LEFT &&
    url.searchParams.get("right") === RIGHT
      ? { body }
      : undefined,
  );
  render(<DiffPage left={LEFT} right={RIGHT} />);
  return fn;
}

const text = () => document.body.textContent ?? "";
const rowFor = (name: string) => {
  const rows = screen.getAllByRole("row").filter((r) => (r.textContent ?? "").includes(name));
  expect(rows.length).toBe(1);
  return rows[0]?.textContent ?? "";
};

describe("DiffPage", () => {
  it("aborts the diff and both run requests when it unmounts", async () => {
    const fn = stubApi(() => undefined);
    const { unmount } = render(<DiffPage left={LEFT} right={RIGHT} />);
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(3));
    const signals = requestSignals(fn);
    expect(signals.every((s) => s instanceof AbortSignal && !s.aborted)).toBe(true);
    unmount();
    expect(signals.every((s) => s?.aborted)).toBe(true);
  });

  it("shows the two runs side by side, one row per aligned step", async () => {
    const fn = load();
    await screen.findByText(/SAME-STEP/);
    for (const n of ["SAME-STEP", "CHANGED-STEP", "ADDED-STEP", "REMOVED-STEP"]) rowFor(n);
    const urls = fn.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes("/diff?") && u.includes(LEFT) && u.includes(RIGHT))).toBe(
      true,
    );
    expect(text()).toContain(LEFT);
    expect(text()).toContain(RIGHT);
  });

  it("marks changed rows 'changed', added 'added', removed 'removed'; same rows unlabelled", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    expect(rowFor("CHANGED-STEP")).toMatch(/changed/i);
    const added = rowFor("ADDED-STEP");
    expect(added).toMatch(/added/i);
    expect(added).not.toMatch(/removed|changed/i);
    const removed = rowFor("REMOVED-STEP");
    expect(removed).toMatch(/removed/i);
    expect(removed).not.toMatch(/added|changed/i);
    expect(rowFor("SAME-STEP")).not.toMatch(/changed|added|removed|first divergence/i);
  });

  it("labels only the first divergent step 'first divergence'", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    expect(rowFor("CHANGED-STEP")).toMatch(/first divergence/i);
    for (const n of ["SAME-STEP", "ADDED-STEP", "REMOVED-STEP"]) {
      expect(rowFor(n)).not.toMatch(/first divergence/i);
    }
  });

  it("says the runs match when there is no divergence", async () => {
    load(
      diffBody({
        first_divergence: null,
        steps: [step(0, "execute_tool SAME-STEP", "same"), step(1, "chat SAME-TWO", "same")],
      }),
    );
    await screen.findByText(/SAME-STEP/);
    expect(text()).toMatch(/runs match/i);
    for (const r of screen.getAllByRole("row")) {
      expect(r.textContent ?? "").not.toMatch(/first divergence|changed|added|removed/i);
    }
  });

  it("does not claim a match when the runs diverge", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    expect(text()).not.toMatch(/runs match/i);
  });

  it("shows signed run-level deltas for tokens, latency, model calls and tool calls", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    const t = text();
    expect(t).toContain("+120"); // input tokens
    expect(t).toMatch(/[-−]30/); // output tokens
    expect(t).toContain("+250"); // latency_ms
    expect(t).toContain("+1"); // model calls
    expect(t).toMatch(/[-−]2/); // tool calls
    expect(t).toMatch(/input/i);
    expect(t).toMatch(/output/i);
    expect(t).toMatch(/latency/i);
    expect(t).toMatch(/model calls/i);
    expect(t).toMatch(/tool calls/i);
  });

  it("writes every run-level change as left → right", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    const items = within(screen.getByRole("list", { name: /run-level deltas/i })).getAllByRole("listitem");
    for (const li of items) expect(li.textContent).toMatch(/ → /);
    expect(text()).toContain("100 → 220");
    expect(text()).toContain("pass → fail");
  });

  it("titles the tab and keeps one h1", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    expect(document.title).toBe("Diff · agent-replay");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("shows pass/fail for both runs", async () => {
    load();
    await screen.findByText(/SAME-STEP/);
    expect(text()).toMatch(/pass/i);
    expect(text()).toMatch(/fail/i);
  });

  it("an unknown run id shows an error with role 'alert'", async () => {
    stubApi(() => undefined); // every request 404s
    render(<DiffPage left={LEFT} right={RIGHT} />);
    const alert = await screen.findByRole("alert");
    expect((alert.textContent ?? "").length).toBeGreaterThan(0);
  });

  it("an invalid response shows an alert rather than a blank page", async () => {
    load({ nonsense: true });
    expect(await screen.findByRole("alert")).toBeTruthy();
  });

  it("App renders the diff page for #/diff/<a>/<b>", async () => {
    const fn = stubApi((url) => (url.pathname.endsWith("/diff") ? { body: diffBody() } : undefined));
    window.location.hash = `#/diff/${LEFT}/${RIGHT}`;
    render(<App />);
    await screen.findByText(/SAME-STEP/);
    expect(fn.mock.calls.some((c) => String(c[0]).includes("/diff?"))).toBe(true);
    expect(text()).not.toMatch(/placeholder/i);
  });
});

describe("RunPage diff link", () => {
  function loadRun(source: string | null) {
    const run = {
      ...runSummary(RIGHT, null, { source_run_id: source, mode: "replay" }),
      spans: [span(0, { name: "chat x" })],
      scores: [],
    };
    stubApi((url) => (url.pathname.endsWith(`/runs/${RIGHT}`) ? { body: run } : undefined));
    render(<RunPage id={RIGHT} />);
  }

  it("a replayed run links 'Diff with source' to #/diff/<source>/<id>", async () => {
    loadRun(SRC);
    const link = await screen.findByRole("link", { name: /diff with source/i });
    expect(link.getAttribute("href")).toBe(`#/diff/${SRC}/${RIGHT}`);
  });

  it("a run without a source has no such link", async () => {
    loadRun(null);
    await screen.findByRole("heading", { name: /chat x/ });
    expect(screen.queryByRole("link", { name: /diff with source/i })).toBeNull();
  });
});
