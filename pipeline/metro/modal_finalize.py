"""Apply the clean-up to every drawn tile and write /main/final/{tx}_{ty}.png:

  1. repaint open water and lakes as flat sea colour (water_alpha.npy, cells)
  2. patch the few invented landmarks left (patch_alpha.npy + patch_rgb.npy)
  3. clip to the metro box (exact polygon, per pixel); outside is black

    ../pilot/.venv/bin/modal run modal_finalize.py
"""
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
vol = modal.Volume.from_name("dublin-metro")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pillow")
app = modal.App("dublin-metro-finalize", image=image)
LP, TILE, T = 16, 512, 32
SEA = (55, 81, 77)


@app.function(volumes={"/m": vol}, cpu=2, memory=6144, timeout=3600)
def run(names: list[str], box_px: list[list[float]]) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    wa = np.load("/m/clean/water_alpha.npy", mmap_mode="r")
    pa = np.load("/m/clean/patch_alpha.npy", mmap_mode="r")
    pr = np.load("/m/clean/patch_rgb.npy", mmap_mode="r")
    hl, wl = wa.shape
    out_dir = Path("/m/main/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(names))
    written = 0

    def up(arr, ty, tx, mode):
        # cells (T+2)^2 with a 1-cell border -> pixels, cropped to the tile
        y0, x0 = ty * T - 1, tx * T - 1
        sub = np.zeros((T + 2, T + 2) + arr.shape[2:], arr.dtype)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(hl, y0 + T + 2), min(wl, x0 + T + 2)
        if ye > ys and xe > xs:
            sub[ys - y0 : ye - y0, xs - x0 : xe - x0] = arr[ys:ye, xs:xe]
        if mode == "F":
            im = Image.fromarray(sub.astype(np.float32), "F").resize(((T + 2) * LP, (T + 2) * LP), Image.Resampling.BILINEAR)
            return np.asarray(im)[LP : LP + TILE, LP : LP + TILE]
        im = Image.fromarray(sub.astype(np.uint8)).resize(((T + 2) * LP, (T + 2) * LP), Image.Resampling.NEAREST)
        return np.asarray(im)[LP : LP + TILE, LP : LP + TILE]

    for n in names:
        tx, ty = (int(v) for v in n[:-4].split("_"))
        clip = Image.new("L", (TILE, TILE), 0)
        ImageDraw.Draw(clip).polygon([(x - tx * TILE, y - ty * TILE) for x, y in box_px], fill=255)
        inside = np.asarray(clip) > 127
        if not inside.any():
            continue
        img = np.asarray(Image.open(f"/m/main/out/{n}").convert("RGB")).astype(np.float32)
        a = up(wa, ty, tx, "F")[..., None]
        if a.max() > 0:
            sea = np.array(SEA, np.float32) + rng.normal(0, 1.2, (TILE, TILE, 1))
            img = img * (1 - a) + sea * a
        p = up(pa, ty, tx, "F")[..., None]
        if p.max() > 0:
            img = img * (1 - p) + up(pr, ty, tx, "N").astype(np.float32) * p
        img[~inside] = 0
        Image.fromarray(img.round().clip(0, 255).astype(np.uint8)).save(out_dir / n, compress_level=3)
        written += 1
    vol.commit()
    return written


@app.local_entrypoint()
def main() -> None:
    sys.path.insert(0, str(HERE.parent / "pilot"))
    from geo import Projector
    from shapely.geometry import box

    plan = json.loads((HERE / "plan.json").read_text())
    l, mpp, shot = plan["lattice"], plan["mpp"], plan["shot_px"]
    proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
    b = plan["box"]
    poly = box(b["lon0"], b["lat0"], b["lon1"], b["lat1"]).segmentize(0.0005)
    box_px = []
    for lon, lat in poly.exterior.coords:
        x, y = proj.view_m(lat, lon, 70.0)
        box_px.append([x / mpp - l["c0"] * shot, -y / mpp - l["r0"] * shot])
    with vol.batch_upload(force=True) as up:
        for f in ("water_alpha.npy", "patch_alpha.npy", "patch_rgb.npy"):
            up.put_file(str(HERE / f), f"/clean/{f}")
    names = [e.path.split("/")[-1] for e in vol.listdir("/main/out") if e.path.endswith(".png")]
    chunks = [names[i : i + 300] for i in range(0, len(names), 300)]
    total = sum(run.map(chunks, kwargs={"box_px": box_px}))
    print(f"{total} final tiles written from {len(names)} drawn tiles")
