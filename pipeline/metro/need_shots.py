"""Print which lattice shots blocks need and how many are on the volume.
    ../pilot/.venv/bin/python need_shots.py 6_0 7_0"""
import json, math, subprocess, sys
plan = json.load(open("plan.json")); l = plan["lattice"]; S = 2048; CORE, M = 12288, 1024
planned = {tuple(s) for s in plan["shots"]}
need = set()
for b in sys.argv[1:]:
    bx, by = map(int, b.split("_"))
    x0, y0 = bx * CORE - M, by * CORE - M
    for gy in range(math.floor(y0 / S), math.ceil((y0 + CORE + 2 * M) / S)):
        for gx in range(math.floor(x0 / S), math.ceil((x0 + CORE + 2 * M) / S)):
            if (gx + l["c0"], gy + l["r0"]) in planned:
                need.add((gx + l["c0"], gy + l["r0"]))
out = subprocess.run(["../pilot/.venv/bin/modal", "volume", "ls", "dublin-metro", "/shots"], capture_output=True, text=True).stdout
have = {ln.strip().split("/")[-1] for ln in out.splitlines()}
missing = [s for s in need if f"r{s[1]}_c{s[0]}.png" not in have]
print(f"need {len(need)}, missing {len(missing)}")
