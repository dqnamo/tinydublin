// Render a batch of lattice shots with the pilot camera.
//   node render_batch.mjs <config.json> <outDir> "c,r;c,r;..."
// Writes <outDir>/r{r}_c{c}.png atomically, skips existing shots, and prints
// one JSON line per shot. Exit code 0 even if a shot times out (reported).
import { createServer } from "vite";
import { readFileSync, writeFileSync, mkdirSync, existsSync, renameSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { launchBrowser } from "./browser.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const [cfgPath, outDir, shotsArg] = process.argv.slice(2);
const cfg = JSON.parse(readFileSync(cfgPath, "utf8"));
mkdirSync(outDir, { recursive: true });
// "c,r" or "c,r,L": L=1 marks a mostly-land shot, which must reach full detail.
const shots = shotsArg.split(";").filter(Boolean).map((s) => s.split(",").map(Number));
const todo = shots.filter(([c, r]) => !existsSync(resolve(outDir, `r${r}_c${c}.png`)));
if (!todo.length) { console.log(JSON.stringify({ done: 0, skipped: shots.length })); process.exit(0); }

const server = await createServer({ root: here, configFile: resolve(here, "vite.config.js") });
await server.listen();
const port = server.config.server.port;
const params = new URLSearchParams({
  lat: cfg.center.lat, lon: cfg.center.lon, target_height: cfg.target_height_m,
  azimuth: cfg.azimuth_deg, elevation: cfg.elevation_deg, mpp: cfg.mpp,
  shot_px: cfg.shot_px, error_target: cfg.error_target, stable_ms: 4000,
});
const browser = await launchBrowser();
const page = await browser.newPage({ viewport: { width: cfg.shot_px, height: cfg.shot_px }, deviceScaleFactor: 1 });
let started = false;
for (let attempt = 0; attempt < 3 && !started; attempt++) {
  try {
    await page.goto(`http://127.0.0.1:${port}/?${params}`);
    await page.waitForFunction(() => window.RENDER, null, { timeout: 60000 });
    const err = await page.evaluate(() => window.RENDER.error);
    if (err) { console.error(err); process.exit(2); }
    await page.waitForFunction(() => window.RENDER.ready, null, { timeout: 120000 });
    started = true;
  } catch (e) {
    console.log(JSON.stringify({ startup_retry: attempt + 1, error: String(e).slice(0, 120) }));
    await new Promise((r) => setTimeout(r, 60000 * (attempt + 1) + Math.random() * 30000));
  }
}
if (!started) { console.error("could not load the root tileset"); process.exit(3); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let lastErr = "";
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") lastErr = m.text().slice(0, 200); });

async function settle(c, r) {
  await page.evaluate(([c, r]) => { window.tileErrors = 0; window.setShot(c, r); }, [c, r]);
  // Stable twice, 3 s apart: a pause in loading (token refresh, throttling)
  // must not be mistaken for the end of loading.
  for (let k = 0; k < 2; k++) {
    await page.waitForFunction(() => window.RENDER.stable, null, { timeout: 15 * 60 * 1000, polling: 500 });
    await sleep(3000);
  }
  await page.waitForFunction(() => window.RENDER.stable, null, { timeout: 15 * 60 * 1000, polling: 500 });
  return page.evaluate(() => ({ sky: window.skyFraction(), errs: window.tileErrors, stats: window.RENDER.stats, detail: window.detailMedian() }));
}

for (const [c, r, land] of todo) {
  const file = resolve(outDir, `r${r}_c${c}.png`);
  const t0 = Date.now();
  let ok = false, q = null;
  for (let attempt = 0; attempt < 6 && !ok; attempt++) {
    try {
      q = await settle(c, r);
      // A little sky with no failed tiles is a real hole in Google's mesh.
      ok = q.errs === 0 && q.stats.failed === 0 && q.sky < 0.01;
      // Land must reach full detail; after 3 tries accept and flag it.
      if (ok && land === 1 && q.detail < 11 && attempt < 3) ok = false;
    } catch (e) { q = { error: String(e).slice(0, 200) }; }
    if (!ok) {
      console.log(JSON.stringify({ c, r, retry: attempt + 1, q, lastErr }));
      await sleep(20000 * (attempt + 1) + Math.random() * 10000);
      // Re-request only the failed tiles. Reloading the page would cost a
      // root request, and Google caps those per day.
      await page.evaluate(() => window.resetFailed());
    }
  }
  if (!ok) { console.log(JSON.stringify({ c, r, error: "gave up", q, lastErr })); continue; }
  const url = await page.evaluate(() => window.grabShot());
  writeFileSync(file + ".tmp", Buffer.from(url.split(",")[1], "base64"));
  renameSync(file + ".tmp", file);
  console.log(JSON.stringify({ c, r, s: (Date.now() - t0) / 1000, detail: q && q.detail }));
}
await browser.close();
await server.close();
process.exit(0);
