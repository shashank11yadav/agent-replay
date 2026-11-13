import { useEffect, useState } from "react";
import { fetchRun, type LoadState } from "../api/client";
import type { Run } from "../api/schemas";
import { Loading } from "../components/Loading";
import { StatusBadge } from "../components/StatusBadge";
import { StepView } from "../components/StepView";
import { diffHref, runHref } from "../router";
import { useTitle } from "../title";

/** Keys typed into a form control belong to that control, not to step navigation. */
function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

// Replace rather than push, so Back leaves the run instead of walking back through its steps.
const showStep = (id: string, index: number) => window.location.replace(runHref(id, index + 1));

/** `step` counts from 1 and comes from the URL; it is clamped to the run's steps. */
export function RunPage({ id, step }: { id: string; step?: number }) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [load, setLoad] = useState<LoadState>("loading");

  useEffect(() => {
    const ac = new AbortController();
    const live = () => !ac.signal.aborted;
    setRun(null);
    setError(null);
    fetchRun(id, { onState: (s) => live() && setLoad(s), signal: ac.signal })
      .then((r) => live() && setRun(r))
      .catch((e: unknown) => live() && setError(e instanceof Error ? e.message : String(e)));
    return () => ac.abort();
  }, [id]);

  const count = run?.spans.length ?? 0;
  const index = Math.min(Math.max((step ?? 1) - 1, 0), Math.max(count - 1, 0));

  useEffect(() => {
    if (count === 0) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || isTyping(e.target)) return;
      if (e.key === "ArrowLeft" && index > 0) showStep(id, index - 1);
      else if (e.key === "ArrowRight" && index < count - 1) showStep(id, index + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [id, index, count]);

  useTitle(run ? `Run ${run.task_id}` : "Run");

  if (error || !run) {
    return (
      <div>
        <h1>Run</h1>
        {error ? (
          <p role="alert">Could not load run: {error}</p>
        ) : (
          <Loading load={load} what="run" />
        )}
      </div>
    );
  }

  const spans = [...run.spans].sort((a, b) => a.seq - b.seq);
  const span = spans[index];
  return (
    <div>
      <h1>
        Run {run.task_id} <StatusBadge status={run.status} />
      </h1>
      {run.source_run_id && <a href={diffHref(run.source_run_id, run.id)}>Diff with source</a>}
      {run.error && <p>Error: {run.error}</p>}
      {span ? (
        <>
          <nav className="step-nav" aria-label="Step navigation">
            <button
              type="button"
              disabled={index === 0}
              aria-keyshortcuts="ArrowLeft"
              onClick={() => showStep(id, index - 1)}
            >
              Previous
            </button>
            <span>
              Step {index + 1} of {spans.length}
            </span>
            <button
              type="button"
              disabled={index >= spans.length - 1}
              aria-keyshortcuts="ArrowRight"
              onClick={() => showStep(id, index + 1)}
            >
              Next
            </button>
          </nav>
          <StepView span={span} />
        </>
      ) : (
        <p>This run has no steps.</p>
      )}
    </div>
  );
}
