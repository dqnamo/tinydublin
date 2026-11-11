// Checks that headless Chromium can create a WebGL2 context and draw.
import { launchBrowser } from "./browser.mjs";

const browser = await launchBrowser();
const page = await browser.newPage();
await page.setContent("<canvas id=c width=256 height=256></canvas>");
const info = await page.evaluate(() => {
  const gl = document.getElementById("c").getContext("webgl2", { preserveDrawingBuffer: true });
  if (!gl) return { ok: false };
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  gl.clearColor(1, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const px = new Uint8Array(4);
  gl.readPixels(10, 10, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return {
    ok: true,
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxRenderbuffer: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
    maxViewport: Array.from(gl.getParameter(gl.MAX_VIEWPORT_DIMS)),
    pixel: Array.from(px),
  };
});
console.log(JSON.stringify(info, null, 2));
await browser.close();
process.exit(info.ok && info.pixel[0] === 255 ? 0 : 1);
