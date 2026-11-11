"""Per-cell (16 px) mean colour and texture of the *drawn* map, like
modal_overview.py does for the render. Saved as map_rgb.npy / map_tex.npy.
Also writes map_preview.png at 1/64 scale.

    ../pilot/.venv/bin/modal run modal_mapstats.py --job main
"""
import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
vol = modal.Volume.from_name("dublin-metro")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pillow")
app = modal.App("dublin-metro-mapstats", image=image)
LP, TILE = 16, 512


@app.function(volumes={"/m": vol}, cpu=2, memory=4096, timeout=1800)
def cells(names: list[str], job: str) -> list[tuple[str, bytes]]:
    import numpy as np
    from PIL import Image

    out = []
    for n in names:
        a = np.asarray(Image.open(f"/m/{job}/out/{n}").convert("RGB")).astype(np.float32)
        h, w = a.shape[0] // LP, a.shape[1] // LP
        blk = a[: h * LP, : w * LP].reshape(h, LP, w, LP, 3)
        mean = blk.mean(axis=(1, 3))
        tex = blk.mean(axis=4).std(axis=(1, 3))
        buf = io.BytesIO()
        np.savez_compressed(buf, mean=mean.round().astype(np.uint8), tex=tex.astype(np.float16))
        out.append((n, buf.getvalue()))
    return out


@app.local_entrypoint()
def main(job: str = "main") -> None:
    import numpy as np
    from PIL import Image

    plan = json.loads((HERE / "plan.json").read_text())
    l = plan["lattice"]
    S = plan["shot_px"] // LP
    W, H = (l["c1"] - l["c0"] + 1) * S, (l["r1"] - l["r0"] + 1) * S
    rgb = np.zeros((H, W, 3), np.uint8)
    tex = np.zeros((H, W), np.float16)
    names = [e.path.split("/")[-1] for e in vol.listdir(f"/{job}/out") if e.path.endswith(".png")]
    chunks = [names[i : i + 150] for i in range(0, len(names), 150)]
    T = TILE // LP
    for res in cells.map(chunks, kwargs={"job": job}):
        for n, data in res:
            d = np.load(io.BytesIO(data))
            tx, ty = (int(v) for v in n[:-4].split("_"))
            m = d["mean"]
            rgb[ty * T : ty * T + m.shape[0], tx * T : tx * T + m.shape[1]] = m
            tex[ty * T : ty * T + m.shape[0], tx * T : tx * T + m.shape[1]] = d["tex"]
    np.save(HERE / "map_rgb.npy", rgb)
    np.save(HERE / "map_tex.npy", tex)
    Image.fromarray(rgb).resize((W // 4, H // 4), Image.Resampling.BOX).save(HERE / "map_preview.png")
    print(f"{len(names)} tiles summarised")
