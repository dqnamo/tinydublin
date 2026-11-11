"""Build a Deep Zoom (DZI) pyramid of 512 px WebP tiles from the drawn tiles.

Input:  dublin-metro Volume /{job}/out/{tx}_{ty}.png   (full-resolution, canvas grid)
Output: dublin-metro Volume /{job}/dzi/map_files/{level}/{col}_{row}.webp + map.dzi

Level max is the full resolution; each level below halves it. Missing tiles
stay missing (the viewer shows the background there).

    ../pilot/.venv/bin/modal run modal_pyramid.py --job main
"""

import json
import math
from pathlib import Path

import modal

TILE = 512
vol = modal.Volume.from_name("dublin-metro")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pillow")
app = modal.App("dublin-metro-pyramid", image=image)


@app.function(volumes={"/m": vol}, cpu=4, memory=8192, timeout=3600)
def full_level(names: list[str], job: str, level: int, quality: int, src: str = "out") -> int:
    from PIL import Image

    vol.reload()
    out = Path(f"/m/{job}/dzi/map_files/{level}")
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        tx, ty = n[:-4].split("_")
        Image.open(f"/m/{job}/{src}/{n}").convert("RGB").save(out / f"{tx}_{ty}.webp", quality=quality, method=4)
    vol.commit()
    return len(names)


@app.function(volumes={"/m": vol}, cpu=4, memory=8192, timeout=3600)
def reduce_level(parents: list[list[int]], job: str, level: int, quality: int, lw: int = 0, lh: int = 0) -> int:
    """Build tiles of `level` (size lw x lh px) from 2x2 children at level + 1.

    Tiles are always full 512 px except at the right/bottom edge of the level,
    as Deep Zoom expects; missing children are black.
    """
    from PIL import Image

    vol.reload()
    src = Path(f"/m/{job}/dzi/map_files/{level + 1}")
    out = Path(f"/m/{job}/dzi/map_files/{level}")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for px, py in parents:
        canvas = Image.new("RGB", (2 * TILE, 2 * TILE))
        any_child = False
        for dy in (0, 1):
            for dx in (0, 1):
                f = src / f"{2 * px + dx}_{2 * py + dy}.webp"
                if f.exists():
                    canvas.paste(Image.open(f).convert("RGB"), (dx * TILE, dy * TILE))
                    any_child = True
        if not any_child:
            continue
        small = canvas.resize((TILE, TILE), Image.Resampling.BOX)
        tw, th = min(TILE, lw - px * TILE), min(TILE, lh - py * TILE)
        if tw <= 0 or th <= 0:
            continue
        small.crop((0, 0, tw, th)).save(out / f"{px}_{py}.webp", quality=quality, method=4)
        n += 1
    vol.commit()
    return n


@app.local_entrypoint()
def main(job: str = "main", quality: int = 88, src: str = "final", from_level: int = -1) -> None:
    meta = json.loads(b"".join(vol.read_file("/canvas.json")))
    plan = json.loads((Path(__file__).parent / "plan.json").read_text())
    lat = plan["lattice"]
    W = (lat["c1"] - lat["c0"] + 1) * plan["shot_px"]
    H = (lat["r1"] - lat["r0"] + 1) * plan["shot_px"]
    max_level = math.ceil(math.log2(max(W, H)))
    names = [e.path.split("/")[-1] for e in vol.listdir(f"/{job}/{src}") if e.path.endswith(".png")]
    print(f"{len(names)} full-resolution tiles; canvas {W} x {H}; levels 0..{max_level}")
    if from_level < 0:
        chunks = [names[i : i + 200] for i in range(0, len(names), 200)]
        print("level", max_level, sum(full_level.map(chunks, kwargs={"job": job, "level": max_level, "quality": quality, "src": src})))
        present = {tuple(int(v) for v in n[:-4].split("_")) for n in names}
        start = max_level - 1
    else:
        lv = [e.path.split("/")[-1] for e in vol.listdir(f"/{job}/dzi/map_files/{from_level + 1}") if e.path.endswith(".webp")]
        present = {tuple(int(v) for v in n[:-5].split("_")) for n in lv}
        start = from_level
    for level in range(start, -1, -1):
        parents = sorted({(x // 2, y // 2) for x, y in present})
        chunks = [[list(p) for p in parents[i : i + 400]] for i in range(0, len(parents), 400)]
        lw = math.ceil(W / 2 ** (max_level - level))
        lh = math.ceil(H / 2 ** (max_level - level))
        made = sum(reduce_level.map(chunks, kwargs={"job": job, "level": level, "quality": quality, "lw": lw, "lh": lh}))
        print("level", level, made)
        present = set(parents)
    dzi = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="webp" Overlap="0" TileSize="{TILE}">\n'
        f'  <Size Width="{W}" Height="{H}"/>\n</Image>\n'
    )
    local = Path(__file__).parent / "map.dzi"
    local.write_text(dzi)
    with vol.batch_upload(force=True) as b:
        b.put_file(str(local), f"/{job}/dzi/map.dzi")
    print("done")
