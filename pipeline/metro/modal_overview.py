"""Low-resolution overview of the render: one RGB mean and one texture value
per 16 px latent cell, for every shot. Saved locally as overview_rgb.npy
(H/16, W/16, 3) uint8 and overview_tex.npy (H/16, W/16) float16, plus a
small PNG preview.

    ../pilot/.venv/bin/modal run modal_overview.py
"""

import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
vol = modal.Volume.from_name("dublin-metro")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pillow")
app = modal.App("dublin-metro-overview", image=image)
LP = 16


@app.function(volumes={"/m": vol}, cpu=2, memory=4096, timeout=1800)
def cells(names: list[str]) -> list[tuple[str, bytes]]:
    import numpy as np
    from PIL import Image

    out = []
    for n in names:
        p = Path("/m/shots") / n
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        h, w = a.shape[0] // LP, a.shape[1] // LP
        blk = a.reshape(h, LP, w, LP, 3)
        mean = blk.mean(axis=(1, 3))
        g = blk.mean(axis=4)
        tex = g.std(axis=(1, 3))
        buf = io.BytesIO()
        np.savez_compressed(buf, mean=mean.round().astype(np.uint8), tex=tex.astype(np.float16))
        out.append((n, buf.getvalue()))
    return out


@app.local_entrypoint()
def main() -> None:
    import numpy as np
    from PIL import Image

    plan = json.loads((HERE / "plan.json").read_text())
    l = plan["lattice"]
    S = plan["shot_px"] // LP
    W, H = (l["c1"] - l["c0"] + 1) * S, (l["r1"] - l["r0"] + 1) * S
    rgb = np.zeros((H, W, 3), np.uint8)
    tex = np.zeros((H, W), np.float16)
    have = np.zeros((H, W), bool)
    names = [f"r{r}_c{c}.png" for c, r in plan["shots"]]
    chunks = [names[i : i + 40] for i in range(0, len(names), 40)]
    got = 0
    for res in cells.map(chunks):
        for n, data in res:
            d = np.load(io.BytesIO(data))
            r, c = (int(v[1:]) for v in n[:-4].split("_"))
            y, x = (r - l["r0"]) * S, (c - l["c0"]) * S
            rgb[y : y + S, x : x + S] = d["mean"]
            tex[y : y + S, x : x + S] = d["tex"]
            have[y : y + S, x : x + S] = True
            got += 1
    np.save(HERE / "overview_rgb.npy", rgb)
    np.save(HERE / "overview_tex.npy", tex)
    np.save(HERE / "overview_have.npy", have)
    Image.fromarray(rgb).resize((W // 8, H // 8), Image.Resampling.BOX).save(HERE / "overview.png")
    print(f"{got} shots summarised; overview {W} x {H} cells")
