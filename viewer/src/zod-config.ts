import { z } from "zod";

// The API serves the viewer with a CSP that forbids eval. Without this, zod probes for
// eval while building its object schemas and the browser reports a CSP violation.
// Imported first in main.tsx so it runs before any schema module.
z.config({ jitless: true });
