import { defineConfig } from "vite";
import { resolve } from "path";

// Locally the key comes from the repo root .env; on Modal from the
// process environment (a Modal Secret). Vite exposes both via envPrefix.
export default defineConfig({
  envDir: process.env.DUBLIN_ENV_DIR || resolve(__dirname, "../../.."),
  envPrefix: ["VITE_", "GOOGLE_TILES_"],
  server: { port: Number(process.env.RENDER_PORT || 5174), strictPort: true, host: "127.0.0.1" },
  logLevel: "warn",
});
