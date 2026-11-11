// Shared Chromium launcher. No GPU on this machine, so WebGL runs on
// SwiftShader (software). On a machine with a GPU, set RENDER_GPU=1.
import { chromium } from "playwright";

export async function launchBrowser() {
  const gpu = process.env.RENDER_GPU === "1";
  const args = gpu
    ? ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=default"]
    : ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"];
  return chromium.launch({ headless: true, args });
}
