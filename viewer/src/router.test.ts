import { describe, expect, it } from "vitest";
import { diffHref, parseRoute, runHref, suiteHref } from "./router";

describe("parseRoute", () => {
  it("#/runs -> runs", () => {
    expect(parseRoute("#/runs")).toEqual({ page: "runs" });
  });
  it("#/runs/abc -> run", () => {
    expect(parseRoute("#/runs/abc")).toEqual({ page: "run", id: "abc" });
  });
  it("#/runs/abc/3 -> run at step 3", () => {
    expect(parseRoute("#/runs/abc/3")).toEqual({ page: "run", id: "abc", step: 3 });
  });
  it.each(["0", "-1", "x", "1.5", "", "01"])("opens the run at its first step for step %j", (s) => {
    expect(parseRoute(`#/runs/abc/${s}`)).toEqual({ page: "run", id: "abc" });
  });
  it("#/suites/s1 -> runs with that suite selected", () => {
    expect(parseRoute("#/suites/s1")).toEqual({ page: "runs", suite: "s1" });
  });
  it("#/diff/a/b -> diff", () => {
    expect(parseRoute("#/diff/a/b")).toEqual({ page: "diff", left: "a", right: "b" });
  });
  it.each(["", "#", "#/", "#/nope", "#/diff/a", "#/diff", "#/diff//b", "#/x/runs/abc", "junk", "#/suites", "#/suites/", "#/suites/a/b", "#/runs/a/1/2"])(
    "falls back to runs for %j",
    (h) => {
      expect(parseRoute(h)).toEqual({ page: "runs" });
    },
  );
  it("falls back to runs for a malformed run id escape", () => {
    expect(parseRoute("#/runs/%E0")).toEqual({ page: "runs" });
  });
  it("falls back to runs when either diff id escape is malformed", () => {
    expect(parseRoute("#/diff/%E0/b")).toEqual({ page: "runs" });
    expect(parseRoute("#/diff/a/%E0")).toEqual({ page: "runs" });
  });
  it("falls back to runs for a malformed suite id escape", () => {
    expect(parseRoute("#/suites/%E0")).toEqual({ page: "runs" });
  });
  it("decodes a valid escaped run id", () => {
    expect(parseRoute("#/runs/a%20b")).toEqual({ page: "run", id: "a b" });
  });
});

describe("links", () => {
  it("round-trip through parseRoute, escaping odd ids", () => {
    const odd = "a/b c#d";
    expect(parseRoute(suiteHref(odd))).toEqual({ page: "runs", suite: odd });
    expect(parseRoute(runHref(odd))).toEqual({ page: "run", id: odd });
    expect(parseRoute(runHref(odd, 4))).toEqual({ page: "run", id: odd, step: 4 });
    expect(parseRoute(diffHref(odd, "x"))).toEqual({ page: "diff", left: odd, right: "x" });
  });
});
