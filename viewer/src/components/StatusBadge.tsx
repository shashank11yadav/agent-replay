// Badges are plain text: a role of "status" would turn every table row into a live region.
export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

/** The server's task verdict; null means the run has not been fully scored. */
export function verdictLabel(passed: boolean | null): string {
  return passed === null ? "not scored" : passed ? "pass" : "fail";
}

export function VerdictBadge({ passed }: { passed: boolean | null }) {
  if (passed === null) return <span className="muted">{verdictLabel(passed)}</span>;
  const label = verdictLabel(passed);
  return <span className={`badge badge-${label}`}>{label}</span>;
}
