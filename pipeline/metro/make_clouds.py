"""Generate the cloud layer round the map: viewer/clouds.json.

Clouds are cartoon cumulus made of circles (drawn by the viewer as vector
shapes, so they stay sharp at any zoom):
  * edge clouds hug the map outline and are checked to cover it everywhere,
    so the straight clip line is never visible;
  * further out, bigger clouds; far out, a sea of large clouds.
Everything outside the outline is painted the cloud-deck colour first.

    ../pilot/.venv/bin/python make_clouds.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from geo import Projector  # noqa: E402

rng = np.random.default_rng(1916)
plan = json.loads((HERE / "plan.json").read_text())
l, mpp, shot = plan["lattice"], plan["mpp"], plan["shot_px"]
proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
b = plan["box"]
W = (l["c1"] - l["c0"] + 1) * shot
H = (l["r1"] - l["r0"] + 1) * shot

pts = []
for lon, lat in box(b["lon0"], b["lat0"], b["lon1"], b["lat1"]).segmentize(0.005).exterior.coords:
    x, y = proj.view_m(lat, lon, 70.0)
    pts.append((x / mpp - l["c0"] * shot, -y / mpp - l["r0"] * shot))
poly = Polygon(pts)
ppoly = prep(poly)
ring = poly.exterior
hull = list(poly.convex_hull.buffer(64, join_style=2).exterior.coords)


def make_cloud(R):
    """Circles (dx, dy, r) of one cumulus blob of rough radius R, flattish base."""
    m = int(rng.integers(6, 12))
    cs = []
    for i in range(m):
        x = rng.normal(0, 0.42 * R)
        y = rng.normal(-0.05 * R, 0.2 * R)
        k = min(1.0, math.hypot(x / R, y / (0.6 * R)))
        r = R * rng.uniform(0.28, 0.48) * (1.15 - 0.6 * k)
        r = min(r, 0.38 * R - y)  # flatten the base a little
        if r > 0.08 * R:
            cs.append((x, y, r))
    if not cs:
        cs = [(0.0, 0.0, 0.4 * R)]
    ext = max(math.hypot(x, y) + r for x, y, r in cs)
    return cs, ext


clouds = []  # (x, y, kind, circles, ext)   kind: 0 edge, 1 near, 2 far


def add(x, y, R, kind):
    cs, ext = make_cloud(R)
    clouds.append((float(x), float(y), kind, cs, ext))


def covered(px, py, shrink=6.0):
    for x, y, _, cs, ext in clouds:
        if abs(px - x) > ext or abs(py - y) > ext:
            continue
        for dx, dy, r in cs:
            if (px - x - dx) ** 2 + (py - y - dy) ** 2 < (r - shrink) ** 2:
                return True
    return False


# ---- edge clouds: an irregular front that reaches into the map -----------
L = ring.length


def front_noise(u):
    """Smooth noise in [0, 1] along the outline (u in px), a few octaves."""
    v = 0.0
    for wl, amp, ph in ((60000, 0.55, 0.3), (21000, 0.3, 1.7), (7500, 0.15, 4.1)):
        n = int(math.ceil(L / wl))
        g = np.random.default_rng(int(wl)).uniform(0, 1, n + 1)
        t = (u / wl) % n
        i = int(t)
        f = t - i
        f = f * f * (3 - 2 * f)
        v += amp * (g[i] * (1 - f) + g[(i + 1) % n] * f)
    return v


def frame(s_):
    p = ring.interpolate(s_)
    q = ring.interpolate(min(L, s_ + 5))
    tx, ty = q.x - p.x, q.y - p.y
    n = math.hypot(tx, ty) or 1
    nx, ny = ty / n, -tx / n
    if ppoly.contains(Point(p.x + nx * 20, p.y + ny * 20)):
        nx, ny = -nx, -ny
    return p, nx, ny  # (nx, ny) points outward


# Keep the front off the named places nearest the edge.
KEEP = [(p_["x"], p_["y"]) for p_ in json.loads((HERE / "viewer" / "places.json").read_text())
        if p_["kind"] in ("town", "airport", "village", "landmark", "station")]
KEEP = np.asarray(KEEP, float)
CLEAR = 2200.0


def depth_cap(px, py):
    d = np.hypot(KEEP[:, 0] - px, KEEP[:, 1] - py)
    near = d[d < 12000]
    return max(250.0, float(near.min()) - CLEAR) if near.size else 1e9


s_ = 0.0
while s_ < L:
    p, nx, ny = frame(s_)
    depth = min(300 + 6500 * front_noise(s_) ** 2.2, depth_cap(p.x, p.y))
    R = rng.uniform(0.8, 1.2) * (650 + 0.35 * depth)
    off = -(depth - 0.7 * R)  # centre offset along the outward normal (+ = out)
    add(p.x + nx * off, p.y + ny * off, R, 0)
    # rows behind, growing, until well outside the map
    back, Rb = off, R
    while back < 2500:
        Rb *= rng.uniform(1.25, 1.5)
        back += 0.75 * Rb
        add(p.x + nx * back + rng.normal(0, 0.3 * Rb), p.y + ny * back, Rb, 0)
    # now and then a small detached cloud drifting over the map
    if rng.uniform() < 0.12:
        d2 = depth + rng.uniform(600, 3500)
        qx, qy = p.x - nx * d2 + rng.normal(0, 500), p.y - ny * d2
        if depth_cap(qx, qy) > 1500:
            add(qx, qy, rng.uniform(280, 650), 0)
    s_ += rng.uniform(0.55, 0.85) * R

# Patch: the outline and a strip 60 px inside it must be under a cloud.
patched = 0
for _ in range(3):
    for i in range(int(L // 40)):
        p, nx, ny = frame(i * 40)
        for off in (0, -60):
            sx, sy = p.x + nx * off, p.y + ny * off
            if not covered(sx, sy):
                add(sx + nx * 100, sy + ny * 100, rng.uniform(500, 800), 0)
                patched += 1
print(f"edge clouds: {sum(1 for c in clouds if c[2] == 0)} ({patched} patches)")

# ---- near and far clouds (dart throwing) ---------------------------------
ex0, ey0, ex1, ey1 = -0.3 * W, -0.35 * H, 1.3 * W, 1.35 * H
centres, radii = [], []
tries = 0
while tries < 60000:
    tries += 1
    x, y = rng.uniform(ex0, ex1), rng.uniform(ey0, ey1)
    if ppoly.contains(Point(x, y)):
        continue
    dist = poly.exterior.distance(Point(x, y))
    if dist < 900:
        continue
    R = min(9000.0, 1400 + 0.25 * dist) * rng.uniform(0.7, 1.3)
    if centres:
        c = np.asarray(centres)
        dd = np.hypot(c[:, 0] - x, c[:, 1] - y)
        if (dd < 0.45 * (np.asarray(radii) + R)).any():
            continue
    centres.append((x, y))
    radii.append(R)
    add(x, y, R, 1 if dist < 6000 else 2)
print(f"near clouds: {sum(1 for c in clouds if c[2] == 1)}, far clouds: {sum(1 for c in clouds if c[2] == 2)}")

# Draw far ones first, then near, then edge; within a group, top to bottom.
clouds.sort(key=lambda c: (-c[2], c[1]))
out = {
    "size": [W, H],
    "poly": [[round(x), round(y)] for x, y in pts],
    "hull": [[round(x), round(y)] for x, y in hull],
    "extent": [round(ex0), round(ey0), round(ex1), round(ey1)],
    "clouds": [
        [round(x), round(y), k, round(e)] + [round(v) for c in cs for v in c]
        for x, y, k, cs, e in clouds
    ],
}
path = HERE / "viewer" / "clouds.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"{len(clouds)} clouds, {path.stat().st_size / 1e3:.0f} KB")
