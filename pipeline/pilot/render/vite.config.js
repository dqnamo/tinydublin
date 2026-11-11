import { defineConfig } from "vite";
import { resolve } from "path";

// .env lives in the repo root (three levels up). Only GOOGLE_TILES_API_KEY is
// exposed to the page, via `envPrefix`.
export default defineConfig({
  envDir: resolve(__dirname, "../../.."),
  envPrefix: ["VITE_", "GOOGLE_TILES_"],
  server: { port: 5174, strictPort: true, host: "127.0.0.1" },
});
