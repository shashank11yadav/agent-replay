export type Route =
  | { page: "runs"; suite?: string }
  | { page: "run"; id: string; step?: number }
  | { page: "diff"; left: string; right: string };

const FALLBACK: Route = { page: "runs" };

/**
 * Parse a location hash such as "#/runs/abc/2". Anything unrecognised maps to the run list.
 * Steps in the URL count from 1, matching the "Step 1 of N" label on the run page.
 */
export function parseRoute(hash: string): Route {
  if (!hash.startsWith("#/")) return FALLBACK;
  const parts = hash.slice(2).split("/");
  const [head, a, b] = parts;
  if (head === "runs" && parts.length === 1) return { page: "runs" };
  if (head === "suites" && parts.length === 2 && a) {
    const suite = decode(a);
    return suite === null ? FALLBACK : { page: "runs", suite };
  }
  if (head === "runs" && (parts.length === 2 || parts.length === 3) && a) {
    const id = decode(a);
    if (id === null) return FALLBACK;
    if (parts.length === 2) return { page: "run", id };
    // A step that is not a positive whole number still opens the run, at its first step.
    return b && /^[1-9]\d{0,5}$/.test(b) ? { page: "run", id, step: Number(b) } : { page: "run", id };
  }
  if (head === "diff" && parts.length === 3 && a && b) {
    const left = decode(a);
    const right = decode(b);
    return left === null || right === null ? FALLBACK : { page: "diff", left, right };
  }
  return FALLBACK;
}

const enc = encodeURIComponent;

export const suiteHref = (id: string) => `#/suites/${enc(id)}`;

export const runHref = (id: string, step?: number) =>
  step === undefined ? `#/runs/${enc(id)}` : `#/runs/${enc(id)}/${step}`;

export const diffHref = (left: string, right: string) => `#/diff/${enc(left)}/${enc(right)}`;

/** Decode a path segment; a malformed escape (URIError) yields null so the caller can fall back. */
function decode(segment: string): string | null {
  try {
    return decodeURIComponent(segment);
  } catch (err) {
    if (err instanceof URIError) return null;
    throw err;
  }
}
