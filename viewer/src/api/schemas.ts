import * as z from "zod";

// Money from the API may arrive as a JSON number (double columns) or a string (Decimal);
// it is kept as received and never parsed to a float here.
const Money = z.union([z.number(), z.string()]);
const Json = z.record(z.string(), z.unknown());

export const SpanSchema = z.object({
  id: z.number(),
  run_id: z.string(),
  seq: z.number(),
  trace_id: z.string(),
  span_id: z.string(),
  parent_span_id: z.string().nullable(),
  kind: z.string(),
  name: z.string(),
  status: z.string(),
  error_type: z.string().nullable(),
  attributes: Json,
  payload: Json,
  input_tokens: z.number(),
  output_tokens: z.number(),
  latency_ms: z.number(),
  cost_usd: Money,
  cache_status: z.string().nullable(),
  started_at: z.string(),
});

export const ScoreSchema = z.object({
  run_id: z.string(),
  check_name: z.string(),
  kind: z.string(),
  passed: z.boolean().nullable(),
  detail: Json,
  created_at: z.string(),
});

/** A row of GET /runs (no spans or scores). */
export const RunSummarySchema = z.object({
  id: z.string(),
  suite_run_id: z.string().nullable(),
  source_run_id: z.string().nullable(),
  task_id: z.string(),
  variant: z.string(),
  mode: z.string(),
  status: z.string(),
  error_type: z.string().nullable(),
  error: z.string().nullable(),
  final_output: z.string().nullable(),
  faults: z.array(z.unknown()),
  input_tokens: z.number(),
  output_tokens: z.number(),
  latency_ms: z.number(),
  model_calls: z.number(),
  tool_calls: z.number(),
  cache_hits: z.number(),
  cache_misses: z.number(),
  cost_usd: Money,
  started_at: z.string(),
  finished_at: z.string().nullable(),
});

/** GET /runs/{id}. */
export const RunSchema = RunSummarySchema.extend({
  spans: z.array(SpanSchema),
  scores: z.array(ScoreSchema),
});

/** A row of GET /runs also carries the task verdict; null means the run is not fully scored. */
export const RunListItemSchema = RunSummarySchema.extend({
  passed: z.boolean().nullable(),
});

export const RunListSchema = z.array(RunListItemSchema);

export const SuiteRunSchema = z.object({
  id: z.string(),
  label: z.string(),
  variant: z.string(),
  mode: z.string(),
  status: z.string(),
  tasks_path: z.string().nullable(),
  tasks_sha256: z.string().nullable(),
  task_count: z.number(),
  source_suite_run_id: z.string().nullable(),
  faults: z.array(z.unknown()),
  metrics: Json.nullable(),
  created_at: z.string(),
  finished_at: z.string().nullable(),
});

export const SuiteRunListSchema = z.array(SuiteRunSchema);

export const StepDiffSchema = z.object({
  seq: z.number(),
  kind: z.string(),
  name: z.string(),
  status: z.enum(["same", "changed", "added", "removed"]),
  changed: z.boolean(),
  input_tokens_delta: z.number(),
  output_tokens_delta: z.number(),
  latency_ms_delta: z.number(),
});

export const DeltaSchema = z.object({
  left: z.number(),
  right: z.number(),
  delta: z.number(),
});

export const PassDeltaSchema = z.object({
  left: z.boolean().nullable(),
  right: z.boolean().nullable(),
});

export const RunDiffSchema = z.object({
  first_divergence: z.number().nullable(),
  steps: z.array(StepDiffSchema),
  deltas: z.record(z.string(), z.union([DeltaSchema, PassDeltaSchema])),
});

export type Span = z.infer<typeof SpanSchema>;
export type RunListItem = z.infer<typeof RunListItemSchema>;
export type Run = z.infer<typeof RunSchema>;
export type SuiteRun = z.infer<typeof SuiteRunSchema>;
export type StepDiff = z.infer<typeof StepDiffSchema>;
export type RunDiff = z.infer<typeof RunDiffSchema>;
