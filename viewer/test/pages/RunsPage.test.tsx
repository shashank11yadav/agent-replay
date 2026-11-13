import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "../../src/App";
import { RunsPage } from "../../src/pages/RunsPage";
import {
  RUN_1,
  RUN_2,
  RUN_3,
  SUITE_A,
  SUITE_B,
  requestSignals,
  runListRow,
  stubApi,
  suiteRun,
} from "../helpers";

const RUN_4 = "bbbbbbbb-0000-4000-8000-000000000004";

afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

// The run list rows carry the server's verdict in `passed`, so the page needs no per-run fetch.
const ROWS: Record<string, Record<string, unknown>> = {
  [RUN_1]: runListRow(RUN_1, SUITE_A, true, {
    input_tokens: 700,
    output_tokens: 534,
    latency_ms: 4321,
  }),
  [RUN_2]: runListRow(RUN_2, SUITE_A, false, { status: "diverged", variant: "v2", task_id: "task-two" }),
  [RUN_3]: runListRow(RUN_3, SUITE_B, true, { task_id: "other-suite-task" }),
  [RUN_4]: runListRow(RUN_4, SUITE_A, null, { task_id: "task-unscored" }),
};

function api(suites = [suiteRun(SUITE_A, "nightly-alpha"), suiteRun(SUITE_B, "nightly-beta")]) {
  return stubApi((url) => {
    if (url.pathname.endsWith("/suite-runs")) return { body: suites };
    if (url.pathname.endsWith("/runs")) {
      const sid = url.searchParams.get("suite_run_id");
      return { body: Object.values(ROWS).filter((r) => !sid || r.suite_run_id === sid) };
    }
    // Run detail, used only when a test follows a row link to the run page.
    const id = /\/runs\/([^/]+)$/.exec(url.pathname)?.[1];
    if (id && ROWS[id]) return { body: { ...ROWS[id], spans: [], scores: [] } };
    return undefined;
  });
}

function rowOf(runId: string): HTMLElement {
  const link = screen
    .getAllByRole("link")
    .find((a) => a.getAttribute("href") === `#/runs/${runId}`);
  expect(link, `link to #/runs/${runId}`).toBeTruthy();
  const row = link!.closest("tr");
  expect(row, "table row").toBeTruthy();
  return row as HTMLElement;
}

describe("RunsPage", () => {
  it("aborts its requests when it unmounts", async () => {
    const fn = api();
    // With a suite selected the page asks for the suite list and that suite's runs.
    const { unmount } = render(<RunsPage suite={SUITE_A} />);
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));
    const signals = requestSignals(fn);
    expect(signals.every((s) => s instanceof AbortSignal && !s.aborted)).toBe(true);
    unmount();
    expect(signals.every((s) => s?.aborted)).toBe(true);
  });

  it("lists suite runs as links to #/suites/<id>", async () => {
    api();
    render(<RunsPage />);
    const alpha = await screen.findByRole("link", { name: /nightly-alpha/ });
    expect(alpha.getAttribute("href")).toBe(`#/suites/${SUITE_A}`);
    expect(screen.getByRole("link", { name: /nightly-beta/ }).getAttribute("href")).toBe(
      `#/suites/${SUITE_B}`,
    );
    expect(screen.queryByRole("region", { name: /runs in suite/i })).toBeNull();
  });

  it("asks for one suite run more than it shows and says when more exist", async () => {
    const many = Array.from({ length: 51 }, (_, i) =>
      suiteRun(`aaaaaaaa-0000-4000-8000-${String(i).padStart(12, "0")}`, `suite-${i}`),
    );
    const fn = api(many);
    render(<RunsPage />);
    await screen.findByRole("link", { name: /suite-0 / });
    expect(screen.queryByRole("link", { name: /suite-50 / })).toBeNull();
    expect(screen.getByText(/more exist/i)).toBeTruthy();
    const url = new URL(String(fn.mock.calls[0]?.[0]), "http://localhost");
    expect(url.searchParams.get("limit")).toBe("51");
  });

  it("does not claim more suite runs exist when all fit", async () => {
    api();
    render(<RunsPage />);
    await screen.findByRole("link", { name: /nightly-alpha/ });
    expect(screen.queryByText(/more exist/i)).toBeNull();
  });

  it("has one h1 and titles the tab with the selected suite", async () => {
    api();
    render(<RunsPage suite={SUITE_B} />);
    await screen.findByRole("link", { name: /nightly-beta/ });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    await waitFor(() => expect(document.title).toBe("Suite nightly-beta · agent-replay"));
  });

  it("announces loading suite runs in a live region", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    render(<RunsPage />);
    expect(screen.getByRole("status").textContent).toMatch(/loading suite runs/i);
    expect(document.title).toBe("Suite runs · agent-replay");
  });

  it("marks the suite from the URL as current", async () => {
    api();
    render(<RunsPage suite={SUITE_B} />);
    const beta = await screen.findByRole("link", { name: /nightly-beta/ });
    expect(beta.getAttribute("aria-current")).toBe("page");
    expect(screen.getByRole("link", { name: /nightly-alpha/ }).getAttribute("aria-current")).toBeNull();
  });

  it("shows rows for the selected suite with task, variant, status, pass/fail, tokens, latency", async () => {
    api();
    render(<RunsPage suite={SUITE_A} />);
    await waitFor(() => rowOf(RUN_1));

    const r1 = rowOf(RUN_1);
    expect(r1.textContent).toContain("task-1");
    expect(r1.textContent).toContain("v1");
    expect(r1.textContent).toMatch(/ok/i);
    expect(r1.textContent).toMatch(/pass/i);
    const flat = (r1.textContent ?? "").replace(/[,\s]/g, "");
    expect(flat).toContain("1234"); // total tokens 700 + 534; thousands separators tolerated
    expect(flat).toContain("4321");

    const r2 = rowOf(RUN_2);
    expect(r2.textContent).toContain("task-two");
    expect(r2.textContent).toContain("v2");
    expect(r2.textContent).toMatch(/diverged/i);
  });

  it.each([
    ["pass", RUN_1, /fail|not scored/i],
    ["fail", RUN_2, /pass|not scored/i],
    ["not scored", RUN_4, /pass|fail/i],
  ])("shows %s from the run list's passed field", async (label, id, other) => {
    api();
    render(<RunsPage suite={SUITE_A} />);
    await waitFor(() => rowOf(id));
    const verdict = rowOf(id).querySelectorAll("td")[3]?.textContent ?? "";
    expect(verdict).toBe(label);
    expect(verdict).not.toMatch(other);
  });

  it("fetches no run details to show the verdicts", async () => {
    const fn = api();
    render(<RunsPage suite={SUITE_A} />);
    await waitFor(() => rowOf(RUN_1));
    const paths = fn.mock.calls.map((c) => new URL(String(c[0]), "http://localhost").pathname);
    expect(paths.filter((p) => /\/runs\/[^/]+$/.test(p))).toEqual([]);
  });

  it("only shows runs of the selected suite, and following a suite link switches suites", async () => {
    api();
    window.location.hash = `#/suites/${SUITE_A}`;
    render(<App />);
    await waitFor(() => rowOf(RUN_1));
    expect(screen.queryByText(/other-suite-task/)).toBeNull();
    fireEvent.click(screen.getByRole("link", { name: /nightly-beta/ }));
    await waitFor(() => rowOf(RUN_3));
    expect(window.location.hash).toBe(`#/suites/${SUITE_B}`);
    expect(screen.queryByText(/task-two/)).toBeNull();
  });

  it("Back from a run returns to the same suite", async () => {
    api();
    window.location.hash = `#/suites/${SUITE_B}`;
    render(<App />);
    await waitFor(() => rowOf(RUN_3));
    fireEvent.click(within(rowOf(RUN_3)).getByRole("link"));
    await screen.findByRole("heading", { name: /run other-suite-task/i });
    window.history.back();
    await waitFor(() => rowOf(RUN_3));
    expect(window.location.hash).toBe(`#/suites/${SUITE_B}`);
  });

  it("each row links to #/runs/<id>", async () => {
    api();
    render(<RunsPage suite={SUITE_A} />);
    await waitFor(() => rowOf(RUN_2));
    const link = within(rowOf(RUN_2)).getByRole("link");
    expect(link.getAttribute("href")).toBe(`#/runs/${RUN_2}`);
  });

  it("a failed suite list shows role=alert, not an empty page", async () => {
    stubApi(() => ({ status: 500, body: { detail: "boom" } }));
    render(<RunsPage />);
    const alert = await screen.findByRole("alert");
    expect((alert.textContent ?? "").trim().length).toBeGreaterThan(0);
  });

  it("an invalid suite list body shows role=alert", async () => {
    stubApi(() => ({ body: { not: "a list" } }));
    render(<RunsPage />);
    expect(await screen.findByRole("alert")).toBeTruthy();
  });

  it("a failed runs fetch shows its error in the runs section and keeps the suite list", async () => {
    stubApi((url) => {
      if (url.pathname.endsWith("/suite-runs")) {
        return { body: [suiteRun(SUITE_A, "nightly-alpha"), suiteRun(SUITE_B, "nightly-beta")] };
      }
      return { status: 500, body: { detail: "boom" } };
    });
    render(<RunsPage suite={SUITE_A} />);
    const section = await screen.findByRole("region", { name: /runs in suite/i });
    const alert = await within(section).findByRole("alert");
    expect(alert.textContent).toMatch(/could not load runs/i);
    expect(screen.getByRole("link", { name: /nightly-alpha/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /nightly-beta/ })).toBeTruthy();
  });
});
