// @vitest-environment node
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const root = join(__dirname, "..");
const read = (p: string) => readFileSync(join(root, p), "utf8");
const pkg = JSON.parse(read("package.json")) as {
  scripts: Record<string, string>;
};

describe("package.json", () => {
  it("has the required scripts", () => {
    expect(pkg.scripts.test).toMatch(/^vitest run/);
    expect(pkg.scripts.typecheck).toMatch(/^tsc --noEmit/);
    expect(pkg.scripts.dev).toBeTruthy();
    expect(pkg.scripts.build).toBeTruthy();
  });
});

describe("tsconfig", () => {
  it("has strict: true in a tsconfig*.json", () => {
    const files = readdirSync(root).filter((f) => /^tsconfig.*\.json$/.test(f));
    expect(files.length).toBeGreaterThan(0);
    const all = files.map(read).join("\n");
    expect(all).toMatch(/"strict"\s*:\s*true/);
    expect(all).not.toMatch(/"strict"\s*:\s*false/);
  });
});

describe("env typing", () => {
  it("declares VITE_API_URL in src/vite-env.d.ts", () => {
    expect(read("src/vite-env.d.ts")).toContain("VITE_API_URL");
  });
  it("no VITE_ variable name contains KEY, SECRET or TOKEN", () => {
    const names = new Set<string>();
    const walk = (dir: string) => {
      for (const e of readdirSync(join(root, dir), { withFileTypes: true })) {
        const rel = join(dir, e.name);
        if (e.isDirectory()) {
          if (e.name !== "node_modules" && e.name !== "dist") walk(rel);
        } else if (/\.(tsx?|ts|html|json|env.*)$/.test(e.name)) {
          for (const m of read(rel).matchAll(/VITE_[A-Z0-9_]+/g)) names.add(m[0]);
        }
      }
    };
    walk("src");
    for (const f of readdirSync(root)) {
      if (/^(vite\.config\.ts|index\.html|\.env.*)$/.test(f) && f !== ".env") {
        for (const m of read(f).matchAll(/VITE_[A-Z0-9_]+/g)) names.add(m[0]);
      }
    }
    expect(names.has("VITE_API_URL")).toBe(true);
    for (const n of names) expect(n).not.toMatch(/KEY|SECRET|TOKEN/);
  });
});
