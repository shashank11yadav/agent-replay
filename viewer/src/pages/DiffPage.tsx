import { useEffect, useState } from "react";
import { fetchDiff, fetchRun, type LoadState } from "../api/client";
import type { Run, RunDiff, Span } from "../api/schemas";
import { DiffRow, signed } from "../components/DiffRow";
import { Loading } from "../components/Loading";
import { verdictLabel } from "../components/StatusBadge";
import { runHref } from "../router";
import { useTitle } from "../title";

const NUMERIC_ROWS: ReadonlyArray<[string, string]> = [
  ["input_tokens", "Input tokens"],
  ["output_tokens", "Output tokens"],
  ["latency_ms", "Latency (latency_ms)"],
  ["model_calls", "Model calls"],
  ["tool_calls", "Tool calls"],
];

function Deltas({ diff }: { diff: RunDiff }) {
  const pass = diff.deltas["task_pass"];
  return (
    <ul aria-label="Run-level deltas">
      {NUMERIC_ROWS.map(([key, title]) => {
        const d = diff.deltas[key];
        if (!d || !("delta" in d)) return null;
        return (
          <li key={key}>
            {title}: {d.left} → {d.right} ({signed(d.delta)})
          </li>
        );
      })}
      {pass && !("delta" in pass) && (
        <li>
          Pass/fail: {verdictLabel(pass.left)} → {verdictLabel(pass.right)}
        </li>
      )}
    </ul>
  );
}

export function DiffPage({ left, right }: { left: string; right: string }) {
  const [diff, setDiff] = useState<RunDiff | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<{ left?: Run; right?: Run }>({});
  const [load, setLoad] = useState<LoadState>("loading");

  useEffect(() => {
    const ac = new AbortController();
    const live = () => !ac.signal.aborted;
    setDiff(null);
    setError(null);
    fetchDiff(left, right, { onState: (s) => live() && setLoad(s), signal: ac.signal })
      .then((d) => live() && setDiff(d))
      .catch((e: unknown) => live() && setError(e instanceof Error ? e.message : String(e)));
    return () => ac.abort();
  }, [left, right]);

  useEffect(() => {
    const ac = new AbortController();
    const live = () => !ac.signal.aborted;
    setRuns({});
    // A failed run fetch only drops the per-side detail; the diff itself still renders.
    fetchRun(left, { signal: ac.signal })
      .then((r) => live() && setRuns((p) => ({ ...p, left: r })))
      .catch(() => undefined);
    fetchRun(right, { signal: ac.signal })
      .then((r) => live() && setRuns((p) => ({ ...p, right: r })))
      .catch(() => undefined);
    return () => ac.abort();
  }, [left, right]);

  useTitle("Diff");

  if (error || !diff) {
    return (
      <div>
        <h1>Diff</h1>
        {error ? (
          <p role="alert">Could not load diff: {error}</p>
        ) : (
          <Loading load={load} what="diff" />
        )}
      </div>
    );
  }

  const steps = [...diff.steps].sort((a, b) => a.seq - b.seq);
  const spanAt = (run: Run | undefined, seq: number): Span | undefined =>
    run?.spans.find((sp) => sp.seq === seq);
  return (
    <div>
      <h1>Diff</h1>
      <p>
        Left: <a href={runHref(left)}>{left}</a>
      </p>
      <p>
        Right: <a href={runHref(right)}>{right}</a>
      </p>
      {diff.first_divergence === null ? (
        <p>The runs match: no step differs.</p>
      ) : (
        <p>First divergence at step {diff.first_divergence}.</p>
      )}
      <Deltas diff={diff} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">Name</th>
              <th scope="col">Kind</th>
              <th scope="col">Left run</th>
              <th scope="col">Right run</th>
              <th scope="col">Input tokens</th>
              <th scope="col">Output tokens</th>
              <th scope="col">Latency (ms)</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s) => (
              <DiffRow
                key={s.seq}
                step={s}
                firstDivergence={s.seq === diff.first_divergence}
                leftSpan={runs.left && (spanAt(runs.left, s.seq) ?? null)}
                rightSpan={runs.right && (spanAt(runs.right, s.seq) ?? null)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
