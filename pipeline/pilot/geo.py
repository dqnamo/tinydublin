"""WGS84 lat/lon/height -> pixel on the pilot canvas, using the camera saved
by the renderer. Needed later to overlay OpenStreetMap (water, landmarks,
search) on the map; the maths matches render/main.js exactly.

    uv run python geo.py 53.3472 -6.2592 60     # O'Connell Bridge
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def ecef(lat: float, lon: float, h: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return np.array(
        [
            (n + h) * math.cos(la) * math.cos(lo),
            (n + h) * math.cos(la) * math.sin(lo),
            (n * (1 - E2) + h) * math.sin(la),
            1.0,
        ]
    )


class Projector:
    def __init__(self, manifest_path: Path = OUT / "manifest.json") -> None:
        m = json.loads(manifest_path.read_text())
        cam = np.array(m["camera"]["camera_ecef_colmajor"]).reshape(4, 4).T  # column-major
        self.view_from_ecef = np.linalg.inv(cam)
        self.mpp = m["config"]["mpp"]
        self.shot = m["config"]["shot_px"]
        self.c0, self.r0 = m["plan"]["c0"], m["plan"]["r0"]
        crop = m.get("crop", {"x0": 0, "y0": 0})
        self.crop_x0, self.crop_y0 = crop["x0"], crop["y0"]

    def view_m(self, lat: float, lon: float, h: float) -> tuple[float, float]:
        v = self.view_from_ecef @ ecef(lat, lon, h)
        return float(v[0]), float(v[1])

    def pixel(self, lat: float, lon: float, h: float) -> tuple[float, float]:
        """Pixel (x, y) on out/render.png (the cropped canvas)."""
        x, y = self.view_m(lat, lon, h)
        px = x / self.mpp - self.c0 * self.shot - self.crop_x0
        py = -y / self.mpp - self.r0 * self.shot - self.crop_y0
        return px, py


if __name__ == "__main__":
    path = OUT / "manifest.json"
    if not path.exists():
        path = OUT / "plan.json"
    p = Projector(path)
    lat, lon, h = (float(a) for a in sys.argv[1:4])
    print("view metres:", p.view_m(lat, lon, h))
    print("pixel:", p.pixel(lat, lon, h))
