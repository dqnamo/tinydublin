"""List blocks (bx_by) of given phases whose shots (core + margin) are all on the volume.
    ../pilot/.venv/bin/python ready_blocks.py 0"""
import json, math, subprocess, sys
import numpy as np
sys.path.insert(0, "."); import modal_draw as m
plan = json.load(open("plan.json")); l = plan["lattice"]; S = 2048
planned = {tuple(s) for s in plan["shots"]}
out = subprocess.run(["../pilot/.venv/bin/modal", "volume", "ls", "dublin-metro", "/shots"], capture_output=True, text=True).stdout
have = {ln.strip().split("/")[-1] for ln in out.splitlines()}
phases = [int(p) for p in sys.argv[1].split(",")]
ready, waiting = [], []
for t in m.block_table(np.load("mask.npy")):
    if m.phase_of(t["bx"], t["by"]) not in phases or t["draw"] == 0:
        continue
    x0, y0 = m.block_origin(t["bx"], t["by"])
    miss = 0
    for gy in range(math.floor(y0 / S), math.ceil((y0 + m.SIDE) / S)):
        for gx in range(math.floor(x0 / S), math.ceil((x0 + m.SIDE) / S)):
            c, r = gx + l["c0"], gy + l["r0"]
            if (c, r) in planned and f"r{r}_c{c}.png" not in have:
                miss += 1
    (ready if miss == 0 else waiting).append(f"{t['bx']}_{t['by']}")
print(",".join(ready)); print("waiting:", len(waiting), file=sys.stderr)
