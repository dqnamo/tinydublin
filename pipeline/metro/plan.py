"""Plan the metro Dublin shot lattice with the pilot's camera.

The camera is the pilot's (same centre, angles and scale), so pilot shots keep
their (c, r) indices and are reused. Writes metro/plan.json with the list of
shots whose square touches the projected footprint of the metro box.

    ../pilot/.venv/bin/python plan.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from geo import Projector  # noqa: E402
from shapely.geometry import MultiPoint, box  # noqa: E402

BOX = {"lat0": 53.235, "lat1": 53.475, "lon0": -6.49, "lon1": -6.04}
# Ellipsoidal heights: Dublin's sea level is ~57 m above the ellipsoid; the
# foothills inside the box reach ~350 m, plus buildings.
H_LO, H_HI = 50.0, 420.0
MARGIN_M = 60.0


def main() -> None:
    proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
    m = json.loads((HERE.parent / "pilot" / "out" / "plan.json").read_text())
    S = m["camera"]["shot_m"]
    pts = []
    # Sample the box edges densely: the projection is not exactly affine.
    n = 40
    for i in range(n + 1):
        f = i / n
        for lat, lon in [
            (BOX["lat0"] + f * (BOX["lat1"] - BOX["lat0"]), BOX["lon0"]),
            (BOX["lat0"] + f * (BOX["lat1"] - BOX["lat0"]), BOX["lon1"]),
            (BOX["lat0"], BOX["lon0"] + f * (BOX["lon1"] - BOX["lon0"])),
            (BOX["lat1"], BOX["lon0"] + f * (BOX["lon1"] - BOX["lon0"])),
        ]:
            for h in (H_LO, H_HI):
                pts.append(proj.view_m(lat, lon, h))
    hull = MultiPoint(pts).convex_hull.buffer(MARGIN_M)
    x0, y0, x1, y1 = hull.bounds
    c0, c1 = math.floor(x0 / S), math.ceil(x1 / S) - 1
    r0, r1 = math.floor(-y1 / S), math.ceil(-y0 / S) - 1
    shots = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            sq = box(c * S, -(r + 1) * S, (c + 1) * S, -r * S)
            if sq.intersects(hull):
                shots.append([c, r])
    plan = {
        "box": BOX,
        "heights": [H_LO, H_HI],
        "shot_m": S,
        "shot_px": m["config"]["shot_px"],
        "mpp": m["config"]["mpp"],
        "camera": m["camera"],
        "config": m["config"],
        "lattice": {"c0": c0, "c1": c1, "r0": r0, "r1": r1},
        "footprint_view_m": [list(p) for p in hull.exterior.coords],
        "shots": shots,
    }
    (HERE / "plan.json").write_text(json.dumps(plan))
    cols, rows = c1 - c0 + 1, r1 - r0 + 1
    print(f"lattice {cols} x {rows} = {cols * rows}; shots touching the box: {len(shots)}")
    print(f"canvas {cols * plan['shot_px']} x {rows * plan['shot_px']} px; "
          f"footprint {hull.area / 1e6:.0f} km^2 of view plane")


if __name__ == "__main__":
    main()
