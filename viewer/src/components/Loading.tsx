import type { LoadState } from "../api/client";

/** A live region, so screen readers hear when loading turns into a wait for a cold host. */
export function Loading({ load, what }: { load: LoadState; what: string }) {
  return (
    <p role="status" aria-live="polite">
      {load === "waking" ? "Waking up the server, retrying shortly..." : `Loading ${what}...`}
    </p>
  );
}
