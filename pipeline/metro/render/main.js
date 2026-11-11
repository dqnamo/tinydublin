// Orthographic lattice renderer for Google Photorealistic 3D Tiles.
//
// Adapted from isometric-nyc/src/web_render/main.js (MIT, Andy Coenen), with
// the tiny.london change: ONE fixed camera for the whole map. A shot is chosen
// by moving the orthographic frustum (left/right/top/bottom) by whole shot
// sizes, never by moving the camera, so all shots tile to the pixel.
//
// Driven by render.mjs through these globals:
//   window.RENDER.ready            true once the tileset root has loaded
//   window.RENDER.stable           true once no tiles are loading for STABLE_MS
//   window.setShot(c, r)           frame shot column c, row r (may be negative)
//   window.planLattice(halfE, halfN, maxHeight, marginM)
//                                  shot index range covering a ground box
//   window.cameraInfo()            camera pose for the manifest
//   window.grabShot()              PNG data URL of the current frame
//
// View-space convention: x to the right, y up, metres, origin at the camera
// axis. Shot (c, r) covers x in [c*S, (c+1)*S] and y in [-(r+1)*S, -r*S],
// where S = SHOT_PX * MPP. Canvas pixel = (x/MPP - c0*SHOT_PX, -y/MPP - r0*SHOT_PX).

import { TilesRenderer, WGS84_ELLIPSOID, CAMERA_FRAME } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import { Matrix4, Scene, WebGLRenderer, OrthographicCamera, MathUtils, Vector3 } from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

const API_KEY = import.meta.env.GOOGLE_TILES_API_KEY || import.meta.env.VITE_GOOGLE_TILES_API_KEY;
const q = new URLSearchParams(window.location.search);
const num = (k, d) => (q.has(k) ? parseFloat(q.get(k)) : d);

const CFG = {
  lat: num("lat", 53.3472),
  lon: num("lon", -6.2592),
  // Ellipsoidal height of the point the camera axis passes through. Dublin's
  // geoid sits about 57 m above the ellipsoid and the quays are a few metres
  // above sea level, so ~60 m puts the axis at street level.
  targetHeight: num("target_height", 60),
  azimuth: num("azimuth", -15), // same as isometric.nyc / tiny.london
  elevation: num("elevation", -45),
  mpp: num("mpp", 0.293), // metres per pixel (tiny.london)
  shotPx: num("shot_px", 2048),
  distance: num("distance", 20000), // camera pulled back along its axis
  errorTarget: num("error_target", 8),
  stableMs: num("stable_ms", 1500),
  cacheGB: num("cache_gb", 1.5),
};

window.RENDER = { ready: false, stable: false, frames: 0, error: null, cfg: CFG };

const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setClearColor(0x87ceeb); // sky blue, as in isometric-nyc
renderer.setPixelRatio(1);
renderer.setSize(CFG.shotPx, CFG.shotPx);
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const camera = new OrthographicCamera(-1, 1, 1, -1, 1, CFG.distance * 3);

const tiles = new TilesRenderer();
tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY, autoRefreshToken: true }));
tiles.registerPlugin(new TileCompressionPlugin());
tiles.registerPlugin(
  new GLTFExtensionsPlugin({ dracoLoader: new DRACOLoader().setDecoderPath("/draco/") }),
);
// The auth plugin sets errorTarget = 20 for Google tiles; we want sharper.
tiles.errorTarget = CFG.errorTarget;
tiles.lruCache.minBytesSize = 0.75 * CFG.cacheGB * 2 ** 30;
tiles.lruCache.maxBytesSize = CFG.cacheGB * 2 ** 30;
tiles.lruCache.maxSize = 20000;
tiles.lruCache.minSize = 15000;
tiles.group.rotation.x = -Math.PI / 2; // Z-up tiles into three.js Y-up
scene.add(tiles.group);

tiles.addEventListener("load-error", (e) => {
  window.tileErrors = (window.tileErrors || 0) + 1;
  console.warn("tile load error", e?.error?.message || e);
});

function placeCamera() {
  tiles.group.updateMatrixWorld(true);
  WGS84_ELLIPSOID.getObjectFrame(
    CFG.lat * MathUtils.DEG2RAD,
    CFG.lon * MathUtils.DEG2RAD,
    CFG.targetHeight,
    CFG.azimuth * MathUtils.DEG2RAD,
    CFG.elevation * MathUtils.DEG2RAD,
    0,
    camera.matrixWorld,
    CAMERA_FRAME,
  );
  camera.matrixWorld.multiply(new Matrix4().makeTranslation(0, 0, CFG.distance));
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(camera.position, camera.quaternion, camera.scale);
  camera.updateMatrixWorld(true);
}
placeCamera();

const S = CFG.shotPx * CFG.mpp; // shot size in metres
let lastBusy = performance.now();

window.setShot = (c, r) => {
  camera.left = c * S;
  camera.right = (c + 1) * S;
  camera.top = -r * S;
  camera.bottom = -(r + 1) * S;
  camera.near = 1;
  camera.far = CFG.distance * 3;
  camera.zoom = 1;
  camera.updateProjectionMatrix();
  window.RENDER.stable = false;
  window.RENDER.frames = 0;
  lastBusy = performance.now();
};
window.setShot(0, 0);

// World point (three.js world frame) -> view-space metres.
const toView = (v) => v.applyMatrix4(camera.matrixWorldInverse);

function enuPoint(e, n, u) {
  const enu = new Matrix4();
  WGS84_ELLIPSOID.getEastNorthUpFrame(
    CFG.lat * MathUtils.DEG2RAD,
    CFG.lon * MathUtils.DEG2RAD,
    CFG.targetHeight,
    enu,
  );
  return new Vector3(e, n, u).applyMatrix4(enu).applyMatrix4(tiles.group.matrixWorld);
}

window.projectLatLon = (lat, lon, h) => {
  const p = new Vector3();
  WGS84_ELLIPSOID.getCartographicToPosition(lat * MathUtils.DEG2RAD, lon * MathUtils.DEG2RAD, h, p);
  p.applyMatrix4(tiles.group.matrixWorld);
  const v = toView(p);
  return { x: v.x, y: v.y };
};

// Shot range covering a ground box of +-halfE east and +-halfN north metres
// round the centre, from the ground up to maxHeight, plus marginM all round.
window.planLattice = (halfE, halfN, maxHeight, marginM) => {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const e of [-halfE, halfE])
    for (const n of [-halfN, halfN])
      for (const u of [-10, maxHeight]) {
        const v = toView(enuPoint(e, n, u));
        x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x);
        y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y);
      }
  x0 -= marginM; x1 += marginM; y0 -= marginM; y1 += marginM;
  return {
    view_bounds_m: { x0, x1, y0, y1 },
    c0: Math.floor(x0 / S), c1: Math.ceil(x1 / S) - 1,
    r0: Math.floor(-y1 / S), r1: Math.ceil(-y0 / S) - 1,
  };
};

window.cameraInfo = () => {
  // Camera pose in the tileset's own (ECEF, Z-up) frame, so Python can
  // project WGS84 points without three.js: view = inverse(camera_ecef) * p.
  const groupInv = tiles.group.matrixWorld.clone().invert();
  const camEcef = camera.matrixWorld.clone().premultiply(groupInv);
  return {
    camera_ecef_colmajor: camEcef.elements.slice(),
    camera_world_colmajor: camera.matrixWorld.elements.slice(),
    group_world_colmajor: tiles.group.matrixWorld.elements.slice(),
    shot_m: S,
  };
};

window.grabShot = () => renderer.domElement.toDataURL("image/png");

// Share of pixels showing the clear colour (no tile drawn there). A complete
// shot has none: Google's tiles cover land and sea alike.
window.skyFraction = () => {
  const gl = renderer.getContext();
  const w = renderer.domElement.width, h = renderer.domElement.height;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let sky = 0, n = 0;
  for (let i = 0; i < px.length; i += 4 * 7) {
    n++;
    if (Math.abs(px[i] - 135) < 4 && Math.abs(px[i + 1] - 206) < 4 && Math.abs(px[i + 2] - 235) < 4) sky++;
  }
  return sky / n;
};
window.tileErrors = 0;
// Like tiles.resetFailedTiles(), but skips tiles whose internal state is not
// set up yet (the library version throws on those).
window.resetFailed = () => {
  if (tiles.rootLoadingState === -1) tiles.rootLoadingState = 0;
  tiles.traverse((tile) => {
    if (tile.internal && tile.internal.loadingState === -1) tile.internal.loadingState = 0;
  }, null, false);
  tiles.stats.failed = 0;
  window.tileErrors = 0;
  window.RENDER.stable = false;
};

// Median grey-level std over 16 px cells: full-detail land scores 13+, a
// coarse level of detail scores under ~8.
window.detailMedian = () => {
  const gl = renderer.getContext();
  const w = renderer.domElement.width, h = renderer.domElement.height;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const C = 16, vals = [];
  for (let cy = 0; cy < h; cy += C * 2) {
    for (let cx = 0; cx < w; cx += C * 2) {
      let s = 0, s2 = 0, n = 0;
      for (let y = cy; y < cy + C; y++) {
        for (let x = cx; x < cx + C; x++) {
          const i = (y * w + x) * 4;
          const g = (px[i] + px[i + 1] + px[i + 2]) / 3;
          s += g; s2 += g * g; n++;
        }
      }
      vals.push(Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2)));
    }
  }
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
};

tiles.addEventListener("load-tileset", () => {
  window.RENDER.ready = true;
});

function frame() {
  requestAnimationFrame(frame);
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolution(camera, CFG.shotPx, CFG.shotPx);
  tiles.update();
  renderer.render(scene, camera);
  window.RENDER.frames++;

  const s = tiles.stats;
  const busy = s.downloading > 0 || s.parsing > 0 || s.queued > 0 || !window.RENDER.ready;
  const now = performance.now();
  if (busy) lastBusy = now;
  window.RENDER.stable = !busy && now - lastBusy > CFG.stableMs && window.RENDER.frames > 5;
  window.RENDER.stats = { downloading: s.downloading, parsing: s.parsing, queued: s.queued, failed: s.failed };
}
if (!API_KEY) {
  window.RENDER.error = "GOOGLE_TILES_API_KEY is not set in the repo root .env";
  console.error(window.RENDER.error);
} else {
  frame();
}
