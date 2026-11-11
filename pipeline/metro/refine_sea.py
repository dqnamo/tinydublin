"""Refine the sea-fill mask with the render: only far-sea cells that look like
water (blue/teal, not sand) and connect to open sea stay filled; the rest
(sand flats at low tide, rocks) are drawn by the model.

    ../pilot/.venv/bin/python refine_sea.py      # rewrites mask.npy (backup kept)
"""
import shutil
import numpy as np
from PIL import Image
from scipy import ndimage

mask = np.load("mask.npy")
shutil.copy("mask.npy", "mask_osm.npy") if not __import__("os").path.exists("mask_osm.npy") else None
mask = np.load("mask_osm.npy")
rgb = np.load("overview_rgb.npy").astype(np.int16)
have = np.load("overview_have.npy")
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
watery = (B > R + 12) & (B >= G - 25)
watery |= ~have  # no render yet: trust OSM
sea = (mask == 2) & watery
sea = ndimage.binary_opening(sea, iterations=3)
sea = ndimage.binary_closing(sea, iterations=3) & (mask == 2)
lab, n = ndimage.label(sea)
sizes = ndimage.sum(sea, lab, range(1, n + 1))
keep = np.zeros(n + 1, bool)
keep[1:] = sizes >= 20000  # only big connected open water
sea = keep[lab]
# Keep at least 12 cells (~190 px) of model-drawn water before the fill starts.
sea = ndimage.binary_erosion(sea, iterations=12, border_value=1) & sea
new = mask.copy()
new[(mask == 2) & ~sea] = 1
np.save("mask.npy", new)
print(f"sea fill cells: {int((mask == 2).sum())} -> {int((new == 2).sum())}; "
      f"now drawn: {int(((mask == 2) & (new == 1)).sum()) * 256 / 1e9:.2f} Gpx more")
y0, y1, x0, x1 = 2400, 4480, 4480, 7300
ov = np.load("overview_rgb.npy")[y0:y1, x0:x1].copy()
m = new[y0:y1, x0:x1]
ov[m == 2] = (ov[m == 2] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
Image.fromarray(ov).resize(((x1 - x0) // 3, (y1 - y0) // 3)).save("/tmp/tl/bay_mask2.png")
