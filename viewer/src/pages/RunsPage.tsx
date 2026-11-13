import { useEffect, useState } from "react";
import { fetchRuns, fetchSuiteRuns, type LoadState } from "../api/client";
import type { RunListItem, SuiteRun } from "../api/schemas";
import { Loading } from "../components/Loading";
import { StatusBadge, VerdictBadge } from "../components/StatusBadge";
import { runHref, suiteHref } from "../router";
import { useTitle } from "../title";

const ROW_LIMIT = 50;
const SUITE_LIMIT = 50;

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** `suite` is the selected suite run id from the URL, if any. */
export function RunsPage({ suite: selected }: { suite?: string }) {
  const [suites, setSuites] = useState<SuiteRun[] | null>(null);
  const [moreSuites, setMoreSuites] = useState(false);
  const [suitesError, setSuitesError] = useState<string | null>(null);
  const [suitesLoad, setSuitesLoad] = useState<LoadState>("loading");
  const [rows, setRows] = useState<RunListItem[] | null>(null);
  const [moreRows, setMoreRows] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoad, setRunsLoad] = useState<LoadState>("loading");

  useEffect(() => {
    const ac = new AbortController();
    const live = () => !ac.signal.aborted;
    // One extra row tells us whether more exist than are shown.
    fetchSuiteRuns(
      { limit: SUITE_LIMIT + 1 },
      { onState: (s) => live() && setSuitesLoad(s), signal: ac.signal },
    )
      .then((list) => {
        if (!live()) return;
        setMoreSuites(list.length > SUITE_LIMIT);
        setSuites(list.slice(0, SUITE_LIMIT));
      })
      .catch((e: unknown) => live() && setSuitesError(msg(e)));
    return () => ac.abort();
  }, []);

  useEffect(() => {
    if (!selected) return;
    const ac = new AbortController();
    const live = () => !ac.signal.aborted;
    setRows(null);
    setMoreRows(false);
    setRunsError(null);
    fetchRuns(
      { suiteRunId: selected, limit: ROW_LIMIT + 1 },
      { onState: (s) => live() && setRunsLoad(s), signal: ac.signal },
    )
      .then((list) => {
        if (!live()) return;
        setMoreRows(list.length > ROW_LIMIT);
        setRows(list.slice(0, ROW_LIMIT));
      })
      .catch((e: unknown) => live() && setRunsError(msg(e)));
    return () => ac.abort();
  }, [selected]);

  const label = suites?.find((s) => s.id === selected)?.label;
  useTitle(label ? `Suite ${label}` : "Suite runs");

  return (
    <div>
      <h1>Suite runs</h1>
      {suitesError ? (
        <p role="alert">Could not load suite runs: {suitesError}</p>
      ) : !suites ? (
        <Loading load={suitesLoad} what="suite runs" />
      ) : (
        <>
          {suites.length === 0 && <p>No suite runs yet.</p>}
          <ul className="suite-list">
            {suites.map((s) => (
              <li key={s.id}>
                <a href={suiteHref(s.id)} aria-current={s.id === selected ? "page" : undefined}>
                  {s.label} ({s.variant}, {s.status})
                </a>
              </li>
            ))}
          </ul>
          {moreSuites && <p>Showing the latest {SUITE_LIMIT} suite runs; more exist.</p>}
        </>
      )}
      {selected && (
        <section aria-label="Runs in suite">
          <h2>Runs</h2>
          {runsError ? (
            <p role="alert">Could not load runs: {runsError}</p>
          ) : !rows ? (
            <Loading load={runsLoad} what="runs" />
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Task</th>
                    <th scope="col">Variant</th>
                    <th scope="col">Status</th>
                    <th scope="col">Pass/fail</th>
                    <th scope="col">Total tokens</th>
                    <th scope="col">Latency (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <a href={runHref(r.id)}>{r.task_id}</a>
                      </td>
                      <td>{r.variant}</td>
                      <td>
                        <StatusBadge status={r.status} />
                      </td>
                      <td>
                        <VerdictBadge passed={r.passed} />
                      </td>
                      <td>{r.input_tokens + r.output_tokens}</td>
                      <td>{r.latency_ms}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {rows && moreRows && <p>Showing the first {ROW_LIMIT} runs; more exist.</p>}
        </section>
      )}
    </div>
  );
}
