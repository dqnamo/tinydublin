"""Build the drawing mask for metro Dublin on the latent grid (16 px cells).

    0 = outside: not drawn (pinned black, never shown)
    1 = draw with the model
    2 = open sea more than SEA_KEEP_M from the coast: filled with flat water

Sea comes from OpenStreetMap coastline (land on the left of each way). Writes
metro/mask.npy and metro/mask_preview.png, plus osm/sea.geojson.

    ../pilot/.venv/bin/python masks.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from geo import Projector  # noqa: E402

LAT_PX = 16
SEA_KEEP_M = 450.0  # draw this much sea next to the shore with the model
AOI_PAD_PX = 1024  # draw this far beyond the box edge (the viewer crops)
SEA_H = 57.0  # sea level as ellipsoidal height in Dublin (geoid ~57 m)
GROUND_H = 70.0


def sea_polygons(bbox):
    d = json.loads((HERE / "osm" / "coastline.json").read_text())
    frame = box(*bbox)
    lines = []
    for e in d["elements"]:
        ls = LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
        lines.append(ls)
    merged = linemerge(lines)
    merged = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    clipped = unary_union([frame.boundary] + [ln.intersection(frame) for ln in merged])
    faces = list(polygonize(clipped))
    sea = []
    for f in faces:
        # Find a coastline segment on this face's boundary and test its right side.
        votes = 0
        for ln in merged:
            seg_parts = ln.intersection(f.boundary)
            if seg_parts.is_empty or seg_parts.length == 0:
                continue
            coords = list(ln.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                mid = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if f.boundary.distance(mid) > 1e-9:
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                n = math.hypot(dx, dy) or 1
                right = Point(mid.x + dy / n * 1e-5, mid.y - dx / n * 1e-5)
                votes += 1 if f.contains(right) else -1
                if abs(votes) >= 5:
                    break
            if abs(votes) >= 5:
                break
        if votes > 0:
            sea.append(f)
    return unary_union(sea)


def main() -> None:
    plan = json.loads((HERE / "plan.json").read_text())
    proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
    mpp, shot = plan["mpp"], plan["shot_px"]
    lat = plan["lattice"]
    W = (lat["c1"] - lat["c0"] + 1) * shot
    H = (lat["r1"] - lat["r0"] + 1) * shot
    wl, hl = W // LAT_PX, H // LAT_PX

    def to_cell(lat_, lon_, h):
        x, y = proj.view_m(lat_, lon_, h)
        return ((x / mpp - lat["c0"] * shot) / LAT_PX, (-y / mpp - lat["r0"] * shot) / LAT_PX)

    def raster(poly_or_multi, h) -> np.ndarray:
        img = Image.new("L", (wl, hl), 0)
        d = ImageDraw.Draw(img)
        polys = getattr(poly_or_multi, "geoms", [poly_or_multi])
        for p in polys:
            if p.is_empty:
                continue
            d.polygon([to_cell(y, x, h) for x, y in p.exterior.coords], fill=1)
            for hole in p.interiors:
                d.polygon([to_cell(y, x, h) for x, y in hole.coords], fill=0)
        return np.asarray(img, dtype=bool)

    b = plan["box"]
    aoi_poly = box(b["lon0"], b["lat0"], b["lon1"], b["lat1"])
    # Densify the box so its projected outline follows the camera exactly.
    aoi_poly = aoi_poly.segmentize(0.002)
    aoi = raster(aoi_poly, GROUND_H)
    aoi = ndimage.binary_dilation(aoi, iterations=AOI_PAD_PX // LAT_PX)

    sea = sea_polygons((b["lon0"] - 0.12, b["lat0"] - 0.08, b["lon1"] + 0.12, b["lat1"] + 0.08))
    (HERE / "osm" / "sea.geojson").write_text(json.dumps(mapping(sea)))
    sea_r = raster(sea, SEA_H)
    keep_cells = SEA_KEEP_M / mpp / LAT_PX
    dist = ndimage.distance_transform_edt(sea_r)
    far_sea = dist > keep_cells

    mask = np.zeros((hl, wl), np.uint8)
    mask[aoi] = 1
    mask[aoi & far_sea] = 2
    np.save(HERE / "mask.npy", mask)
    n1, n2 = int((mask == 1).sum()), int((mask == 2).sum())
    px_area = (LAT_PX * mpp) ** 2 / 1e6  # km^2 of view plane per cell
    print(f"mask {wl} x {hl} cells; draw {n1 * LAT_PX**2 / 1e9:.2f} Gpx, "
          f"sea fill {n2 * LAT_PX**2 / 1e9:.2f} Gpx, outside {(mask == 0).mean() * 100:.0f}%")
    prev = np.zeros((hl, wl, 3), np.uint8)
    prev[mask == 1] = (200, 190, 170)
    prev[mask == 2] = (55, 81, 77)
    Image.fromarray(prev).resize((wl // 8, hl // 8), Image.Resampling.NEAREST).save(HERE / "mask_preview.png")


if __name__ == "__main__":
    main()
