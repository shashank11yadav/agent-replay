import type { Span, StepDiff } from "../api/schemas";

/** Format a number with an explicit sign; zero has none and float noise is rounded away. */
export function signed(n: number): string {
  const r = Math.round(n * 1000) / 1000;
  if (r === 0) return "0";
  return r > 0 ? `+${r}` : `-${Math.abs(r)}`;
}

function Side({ span }: { span: Span | undefined | null }) {
  if (span === null) return <>absent</>;
  if (span === undefined) return <>{"\u2014"}</>;
  return (
    <>
      {span.kind}, in {span.input_tokens}, out {span.output_tokens}, {span.latency_ms} ms
    </>
  );
}

/**
 * One aligned step: the left and right run side by side. A side is `null` when the step is absent
 * there, and `undefined` when the run could not be loaded.
 */
export function DiffRow({
  step,
  firstDivergence,
  leftSpan,
  rightSpan,
}: {
  step: StepDiff;
  firstDivergence: boolean;
  leftSpan?: Span | null;
  rightSpan?: Span | null;
}) {
  const label = step.status === "same" ? null : step.status;
  const left = step.status === "added" ? null : leftSpan;
  const right = step.status === "removed" ? null : rightSpan;
  const cls = [label ? `diff-${label}` : "", firstDivergence ? "diff-first" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <tr className={cls || undefined}>
      <td>{step.seq}</td>
      <td>{step.name}</td>
      <td>{step.kind}</td>
      <td>
        <Side span={left} />
      </td>
      <td>
        <Side span={right} />
      </td>
      <td>{signed(step.input_tokens_delta)}</td>
      <td>{signed(step.output_tokens_delta)}</td>
      <td>{signed(step.latency_ms_delta)}</td>
      <td>
        {label && <strong>{label}</strong>}
        {firstDivergence && <em> first divergence</em>}
      </td>
    </tr>
  );
}
