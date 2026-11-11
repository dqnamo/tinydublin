// Renders the shot lattice for pilot/config.json into pilot/out/shots/ and
// writes pilot/out/manifest.json. Resumable: existing shots are skipped.
//
//   node render.mjs              plan + render every shot
//   node render.mjs --plan-only  just print/write the lattice (no API calls)
//   node render.mjs --only 0,0   render one shot (column,row) as a test
import { createServer } from "vite";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { launchBrowser } from "./browser.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const pilot = resolve(here, "..");
const cfg = JSON.parse(readFileSync(resolve(pilot, "config.json"), "utf8"));
const outDir = resolve(pilot, "out");
const shotDir = resolve(outDir, "shots");
mkdirSync(shotDir, { recursive: true });

const argv = process.argv.slice(2);
const planOnly = argv.includes("--plan-only");
const onlyArg = argv.includes("--only") ? argv[argv.indexOf("--only") + 1] : null;
const timeoutMs = 15 * 60 * 1000; // software WebGL is slow; be patient

const server = await createServer({ root: here, configFile: resolve(here, "vite.config.js"), logLevel: "warn" });
await server.listen();
const port = server.config.server.port;

const params = new URLSearchParams({
  lat: cfg.center.lat, lon: cfg.center.lon, target_height: cfg.target_height_m,
  azimuth: cfg.azimuth_deg, elevation: cfg.elevation_deg, mpp: cfg.mpp,
  shot_px: cfg.shot_px, error_target: cfg.error_target,
});

const browser = await launchBrowser();
const page = await browser.newPage({ viewport: { width: cfg.shot_px, height: cfg.shot_px }, deviceScaleFactor: 1 });
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) console.log(`[page ${m.type()}] ${m.text()}`); });
await page.goto(`http://127.0.0.1:${port}/?${params}`);
await page.waitForFunction(() => window.planLattice && window.RENDER, null, { timeout: 60000 });

const plan = await page.evaluate(
  ([e, n, h, m]) => window.planLattice(e, n, h, m),
  [cfg.half_east_m, cfg.half_north_m, cfg.max_building_height_m, cfg.margin_m],
);
const cam = await page.evaluate(() => window.cameraInfo());
const centrePx = await page.evaluate(([la, lo, h]) => window.projectLatLon(la, lo, h), [cfg.center.lat, cfg.center.lon, cfg.target_height_m]);
const cols = plan.c1 - plan.c0 + 1, rows = plan.r1 - plan.r0 + 1;
const manifest = {
  config: cfg, plan, camera: cam, cols, rows,
  canvas_px: { width: cols * cfg.shot_px, height: rows * cfg.shot_px },
  centre_view_m: centrePx,
  shots: [],
};
console.log(`lattice: ${cols} x ${rows} = ${cols * rows} shots, canvas ${manifest.canvas_px.width} x ${manifest.canvas_px.height} px`);

if (planOnly) {
  writeFileSync(resolve(outDir, "plan.json"), JSON.stringify(manifest, null, 1));
  await browser.close(); await server.close();
  process.exit(0);
}

const err = await page.evaluate(() => window.RENDER.error);
if (err) { console.error(err); process.exit(2); }
await page.waitForFunction(() => window.RENDER.ready, null, { timeout: 120000 });

let todo = [];
for (let r = plan.r0; r <= plan.r1; r++) for (let c = plan.c0; c <= plan.c1; c++) todo.push([c, r]);
if (onlyArg) todo = [onlyArg.split(",").map(Number)];

for (const [c, r] of todo) {
  const file = resolve(shotDir, `r${r}_c${c}.png`);
  manifest.shots.push({ c, r, file: `shots/r${r}_c${c}.png` });
  if (existsSync(file)) { console.log(`skip ${c},${r}`); continue; }
  const t0 = Date.now();
  await page.evaluate(([c, r]) => window.setShot(c, r), [c, r]);
  await page.waitForFunction(() => window.RENDER.stable, null, { timeout: timeoutMs, polling: 500 });
  const url = await page.evaluate(() => window.grabShot());
  writeFileSync(file + ".tmp", Buffer.from(url.split(",")[1], "base64"));
  const { renameSync } = await import("fs");
  renameSync(file + ".tmp", file); // atomic: no half-written shots
  const stats = await page.evaluate(() => window.RENDER.stats);
  console.log(`shot ${c},${r} in ${((Date.now() - t0) / 1000).toFixed(1)} s`, JSON.stringify(stats));
}

if (!onlyArg) writeFileSync(resolve(outDir, "manifest.json"), JSON.stringify(manifest, null, 1));
await browser.close();
await server.close();
