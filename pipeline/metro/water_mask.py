"""Where to repaint the map as flat water (removes landmarks the model
invented on water, and evens out the sea).

Water = OpenStreetMap water polygons (each shifted vertically to fit the
render, since lakes in the hills sit higher than sea level) + the open-sea fill
+ big flat water-coloured areas of the render. Real structures in the water
(piers, walls, ships, islands) keep the drawing: cells whose render colour
differs clearly from that water body's own colour.

Writes water_alpha.npy (cells, 0..1) and a simulation sheet.
    ../pilot/.venv/bin/python water_mask.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from geo import Projector  # noqa: E402

LP = 16
plan = json.loads((HERE / "plan.json").read_text())
l, mpp, shot = plan["lattice"], plan["mpp"], plan["shot_px"]
proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
mask = np.load(HERE / "mask.npy")
hl, wl = mask.shape
r = np.load(HERE / "overview_rgb.npy").astype(np.int16)
rt = np.load(HERE / "overview_tex.npy").astype(np.float32)
have = np.load(HERE / "overview_have.npy")
m = np.load(HERE / "map_rgb.npy").astype(np.float32)


def cell(lat, lon, h=57.0):
    x, y = proj.view_m(lat, lon, h)
    return ((x / mpp - l["c0"] * shot) / LP, (-y / mpp - l["r0"] * shot) / LP)


# ---- OSM polygons ---------------------------------------------------------
polys = []
for e in json.loads((HERE / "osm" / "water.json").read_text())["elements"]:
    if e["type"] == "way" and "geometry" in e:
        pts = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            pg = Polygon(pts)
            if pg.is_valid and pg.area > 0:
                polys.append(pg)
    elif e["type"] == "relation":
        outer, inner = [], []
        for mem in e.get("members", []):
            if "geometry" not in mem:
                continue
            ls = LineString([(p["lon"], p["lat"]) for p in mem["geometry"]])
            (inner if mem.get("role") == "inner" else outer).append(ls)
        if outer:
            o = unary_union(list(polygonize(linemerge(outer))))
            if inner:
                o = o.difference(unary_union(list(polygonize(linemerge(inner)))))
            polys.extend(getattr(o, "geoms", [o]))
water_ll = unary_union([p.buffer(0) for p in polys])
parts = [p for p in getattr(water_ll, "geoms", [water_ll]) if p.area > 2.5e-7]  # ~2,000 m2
print(len(parts), "OSM water bodies")

R, G, B = r[..., 0], r[..., 1], r[..., 2]
flat = rt < 3.5
calm = rt < 6.0
osm = np.zeros((hl, wl), bool)
for p in parts:
    ext = [cell(y, x) for x, y in p.exterior.coords]
    xs, ys = zip(*ext)
    x0, x1 = int(max(0, min(xs) - 2)), int(min(wl, max(xs) + 3))
    y0, y1 = int(max(0, min(ys) - 45)), int(min(hl, max(ys) + 3))
    if x1 <= x0 or y1 <= y0:
        continue
    img = Image.new("L", (x1 - x0, y1 - y0 + 45), 0)
    d = ImageDraw.Draw(img)
    d.polygon([(x - x0, y - y0 + 45) for x, y in ext], fill=1)
    for hole in p.interiors:
        d.polygon([(cell(y, x)[0] - x0, cell(y, x)[1] - y0 + 45) for x, y in hole.coords], fill=0)
    full = np.asarray(img, bool)  # rows: y0-45 .. y1
    if full.sum() < 4:
        continue
    best, best_s = 0, -1
    for dy in range(0, 41, 1):  # higher ground appears higher up the map
        # shift the polygon up by dy cells
        rows = np.arange(y0, y1) - dy
        ok = (rows >= 0) & (rows < hl)
        if not ok.any():
            continue
        f = calm[rows[ok], x0:x1][full[45:, :][ok]]
        s = f.mean() if f.size else 0
        if s > best_s + 0.02:
            best, best_s = dy, s
    if best_s < 0.4:  # render does not agree this is calm water: leave it
        continue
    rows = np.arange(y0, y1) - best
    ok = (rows >= 0) & (rows < hl)
    osm[rows[ok], x0:x1] |= full[45:, :][ok]

# ---- colour water (sea, estuaries) ----------------------------------------
watery = (B > R + 8) & (B >= G - 30) & have & calm
watery = ndimage.binary_opening(watery, iterations=1)
lab, n = ndimage.label(watery)
sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
big = np.zeros(n + 1, bool)
big[1:] = sz >= 200
W = (osm | big[lab] | (mask == 2)) & (mask != 0)
# Close ripples and fill small holes (invented objects sit in such holes);
# big holes are real land (Bull Island, Dalkey Island, Ireland's Eye).
W = ndimage.binary_closing(W, iterations=3) & (mask != 0)
holes = ndimage.binary_fill_holes(W) & ~W
hl_lab, hn = ndimage.label(holes)
hsz = ndimage.sum(holes, hl_lab, range(1, hn + 1))
small = np.zeros(hn + 1, bool)
small[1:] = hsz <= 600
W |= small[hl_lab]

# ---- keep real structures ---------------------------------------------------
lab, n = ndimage.label(W)
med = np.zeros((n + 1, 3), np.float32)
for i, sl in enumerate(ndimage.find_objects(lab), start=1):
    if sl is None:
        continue
    sel = lab[sl] == i
    med[i] = np.median(r[sl][sel], axis=0)
diff = np.abs(r - med[lab]).sum(axis=2)
struct = W & (diff > 60) & ~calm & (mask != 2)
struct = ndimage.binary_opening(struct, iterations=1)
struct = ndimage.binary_dilation(struct, iterations=1)
Wr = W & ~struct
dist = ndimage.distance_transform_edt(Wr)
alpha = np.clip((dist - 2) / 5, 0, 1).astype(np.float32)
np.save(HERE / "water_alpha.npy", alpha.astype(np.float16))
print(f"repaint cells {int((alpha > 0).sum())}; OSM water cells {int(osm.sum())}; structures kept {int(struct.sum())}")

SEA = np.array([55, 81, 77], np.float32)
out = m * (1 - alpha[..., None]) + SEA * alpha[..., None]
crops = [(1080, 1330, 5700, 6150), (3060, 3200, 5180, 5360), (4330, 4430, 1600, 1720),
         (2700, 3400, 4800, 5700), (3000, 3200, 3500, 4200)]
tiles = []
for y0, y1, x0, x1 in crops:
    a = Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8))
    b = Image.fromarray(out[y0:y1, x0:x1].round().astype(np.uint8))
    s = 420 / max(a.width, a.height)
    a = a.resize((int(a.width * s), int(a.height * s)))
    b = b.resize(a.size)
    t = Image.new("RGB", (a.width * 2 + 6, a.height), (255, 255, 255))
    t.paste(a, (0, 0))
    t.paste(b, (a.width + 6, 0))
    tiles.append(t)
H = sum(t.height for t in tiles) + 6 * len(tiles)
c = Image.new("RGB", (max(t.width for t in tiles), H), (255, 255, 255))
y = 0
for t in tiles:
    c.paste(t, (0, y))
    y += t.height + 6
c.save("/tmp/tl/repaint_sim2.png")
