import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The API itself is mounted under /api, so the dev proxy forwards the path unchanged.
const target = process.env.VIEWER_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
  },
});
