import { describe, expect, it } from "vitest";
import { RunListSchema, RunSchema, type Run } from "./schemas";
import { RUN_1, runListRow, runSummary, validRun } from "../../test/helpers";

describe("response schemas (zod)", () => {
  it("parses a valid run fixture and infers a typed Run", () => {
    const parsed = RunSchema.safeParse(validRun());
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      const run: Run = parsed.data; // type inferred via z.infer
      expect(run.spans).toHaveLength(2);
      expect(run.spans[0]?.seq).toBe(0);
      expect(run.spans[1]?.name).toBe("get_order");
      expect(run.scores[0]?.check_name).toBe("contains");
      expect(run.task_id).toBe("t1");
    }
  });

  it("rejects wrongly typed fields", () => {
    const bad = { ...validRun(), input_tokens: "lots" };
    expect(RunSchema.safeParse(bad).success).toBe(false);
  });

  it("run list rows carry passed as true, false or null", () => {
    for (const passed of [true, false, null]) {
      const parsed = RunListSchema.safeParse([runListRow(RUN_1, null, passed)]);
      expect(parsed.success).toBe(true);
      if (parsed.success) expect(parsed.data[0]?.passed).toBe(passed);
    }
  });

  it("rejects a run list row without passed", () => {
    expect(RunListSchema.safeParse([runSummary(RUN_1, null)]).success).toBe(false);
  });
});
