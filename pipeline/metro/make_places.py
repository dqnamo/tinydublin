"""OpenStreetMap places (osm/places.json) -> viewer/places.json in canvas pixels.

    ../pilot/.venv/bin/python make_places.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from geo import Projector  # noqa: E402

KIND = {
    "city": ("city", 0, 0.08), "town": ("town", 1, 0.2), "suburb": ("area", 2, 0.35),
    "quarter": ("area", 3, 0.5), "village": ("village", 3, 0.4), "neighbourhood": ("area", 4, 0.6),
    "station": ("station", 5, 0.8), "aerodrome": ("airport", 1, 0.25), "stadium": ("stadium", 3, 0.7),
    "park": ("park", 4, 0.5), "attraction": ("landmark", 3, 0.9),
}

plan = json.loads((HERE / "plan.json").read_text())
l, mpp, shot = plan["lattice"], plan["mpp"], plan["shot_px"]
proj = Projector(HERE.parent / "pilot" / "out" / "plan.json")
b = plan["box"]
out, seen = [], set()
for e in json.loads((HERE / "osm" / "places.json").read_text())["elements"]:
    t = e.get("tags", {})
    name = t.get("name:en") or t.get("name")
    if not name:
        continue
    lat = e.get("lat", e.get("center", {}).get("lat"))
    lon = e.get("lon", e.get("center", {}).get("lon"))
    if lat is None or not (b["lat0"] <= lat <= b["lat1"] and b["lon0"] <= lon <= b["lon1"]):
        continue
    key = t.get("place") or t.get("railway") or t.get("aeroway") or t.get("leisure") or t.get("tourism")
    if key not in KIND:
        continue
    kind, rank, zoom = KIND[key]
    if (name, kind) in seen:
        continue
    seen.add((name, kind))
    vx, vy = proj.view_m(lat, lon, 70.0)
    x = vx / mpp - l["c0"] * shot
    y = -vy / mpp - l["r0"] * shot
    out.append({"name": name, "kind": kind, "rank": rank, "zoom": zoom, "x": round(x), "y": round(y)})
EXTRA = [
    ("Phoenix Park", 53.3559, -6.3298, "park"), ("Dublin Zoo", 53.3562, -6.3053, "landmark"),
    ("Trinity College", 53.3438, -6.2546, "landmark"), ("St Stephen's Green", 53.3382, -6.2591, "park"),
    ("Aviva Stadium", 53.3352, -6.2285, "stadium"), ("Guinness Storehouse", 53.3419, -6.2867, "landmark"),
    ("O'Connell Bridge", 53.3472, -6.2592, "landmark"), ("The Spire", 53.3498, -6.2603, "landmark"),
    ("North Bull Island", 53.3700, -6.1500, "landmark"), ("Poolbeg Chimneys", 53.3409, -6.1898, "landmark"),
    ("Dublin Port", 53.3490, -6.2010, "landmark"), ("Killiney Hill", 53.2651, -6.1131, "park"),
    ("Malahide Castle", 53.4446, -6.1650, "landmark"),
]
for name, lat, lon, kind in EXTRA:
    if any(o["name"] == name for o in out):
        continue
    vx, vy = proj.view_m(lat, lon, 70.0)
    out.append({"name": name, "kind": kind, "rank": 2, "zoom": 0.6,
                "x": round(vx / mpp - l["c0"] * shot), "y": round(-vy / mpp - l["r0"] * shot)})
out.sort(key=lambda p: (p["rank"], p["name"]))
(HERE / "viewer").mkdir(exist_ok=True)
(HERE / "viewer" / "places.json").write_text(json.dumps(out, ensure_ascii=False))
print(len(out), "places;", ", ".join(p["name"] for p in out[:12]))
