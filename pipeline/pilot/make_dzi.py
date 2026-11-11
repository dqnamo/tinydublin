"""Cut a big PNG into a Deep Zoom (DZI) pyramid of 512 px WebP tiles for
OpenSeadragon, as tiny.london does.

    uv run python make_dzi.py out/pixel.png viewer/tiles/pixel
    uv run python make_dzi.py out/render.png viewer/tiles/render

Writes <dest>.dzi and <dest>_files/<level>/<col>_<row>.webp. Level N is full
size; each level below halves it, down to 1 x 1 px.
"""

import math
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TILE = 512
FMT = "webp"


def save_level(img: Image.Image, level_dir: Path, quality: int) -> int:
    level_dir.mkdir(parents=True, exist_ok=True)
    cols, rows = math.ceil(img.width / TILE), math.ceil(img.height / TILE)
    for c in range(cols):
        for r in range(rows):
            box = (c * TILE, r * TILE, min((c + 1) * TILE, img.width), min((r + 1) * TILE, img.height))
            img.crop(box).save(level_dir / f"{c}_{r}.{FMT}", quality=quality, method=4)
    return cols * rows


def make_dzi(src: Path, dest: Path, quality: int = 90) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    max_level = math.ceil(math.log2(max(w, h)))
    files = Path(f"{dest}_files")
    total = 0
    level_img = img
    for level in range(max_level, -1, -1):
        total += save_level(level_img, files / str(level), quality)
        nw, nh = max(1, math.ceil(level_img.width / 2)), max(1, math.ceil(level_img.height / 2))
        level_img = level_img.resize((nw, nh), Image.Resampling.BOX)
    Path(f"{dest}.dzi").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="{FMT}" Overlap="0" TileSize="{TILE}">\n'
        f'  <Size Width="{w}" Height="{h}"/>\n'
        f"</Image>\n"
    )
    print(f"{dest}.dzi: {w} x {h} px, {max_level + 1} levels, {total} tiles")


if __name__ == "__main__":
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    make_dzi(src, dest)
