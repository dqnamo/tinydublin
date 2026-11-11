"""Stitch the rendered shot lattice into one canvas and crop it to the pilot box.

Reads out/manifest.json and out/shots/*.png, writes out/render.png and adds a
"crop" entry to the manifest. The crop is aligned to 16 px (one latent token of
Qwen-Image-2.1's VAE) so the sampler can use it directly.

    uv run python stitch.py
"""

import json
import math
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).parent / "out"
ALIGN = 16


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    cfg = manifest["config"]
    plan = manifest["plan"]
    w = h = cfg["shot_px"]
    mpp = cfg["mpp"]
    canvas = Image.new("RGB", (manifest["canvas_px"]["width"], manifest["canvas_px"]["height"]))
    missing = []
    for shot in manifest["shots"]:
        path = OUT / shot["file"]
        if not path.exists():
            missing.append(shot["file"])
            continue
        img = Image.open(path).convert("RGB")
        assert img.size == (w, h), f"{path} is {img.size}, expected {(w, h)}"
        canvas.paste(img, ((shot["c"] - plan["c0"]) * w, (shot["r"] - plan["r0"]) * h))
    if missing:
        raise SystemExit(f"missing shots: {missing}")

    b = plan["view_bounds_m"]
    x0 = math.floor((b["x0"] / mpp - plan["c0"] * w) / ALIGN) * ALIGN
    x1 = math.ceil((b["x1"] / mpp - plan["c0"] * w) / ALIGN) * ALIGN
    y0 = math.floor((-b["y1"] / mpp - plan["r0"] * h) / ALIGN) * ALIGN
    y1 = math.ceil((-b["y0"] / mpp - plan["r0"] * h) / ALIGN) * ALIGN
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas.width, x1), min(canvas.height, y1)
    crop = canvas.crop((x0, y0, x1, y1))
    crop.save(OUT / "render.png", optimize=False, compress_level=1)
    manifest["crop"] = {"x0": x0, "y0": y0, "width": x1 - x0, "height": y1 - y0}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"render.png: {crop.width} x {crop.height} px (crop at {x0},{y0} of the lattice canvas)")


if __name__ == "__main__":
    main()
