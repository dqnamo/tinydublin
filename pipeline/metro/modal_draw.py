"""Draw metro Dublin block by block on Modal H100s (the tiny.london method).

The canvas is cut into blocks of CORE px. Each block is drawn as one shared
latent (MultiDiffusion: 1024 px windows on a 768 px stride, odd steps on a grid
shifted 384 px, tent weights) over its core plus a MARGIN of context. Finished
neighbours and the open-sea fill are *pinned* (RePaint): at every step their
latent is replaced by the known latent noised to the current level, so the new
block grows out of them. Blocks run in four phases by the parity of their row
and column; blocks in one phase never touch, so they run on parallel GPUs.

Output: lossless 512 px PNG tiles in the `dublin-metro` Volume, /out/{tx}_{ty}.png,
in canvas coordinates. Each block writes its core and blends a STRIP px band
over finished neighbours. /blocks/{bx}_{by}.json marks a block done.

    ../pilot/.venv/bin/modal run modal_draw.py::draw --phases 0,1,2,3
    ../pilot/.venv/bin/modal run modal_draw.py::draw --only 3_2       # one block

Licences: Qwen-Image-2.1 and the tiny.london LoRA are under the Qwen Research
License (non-commercial).
"""

import io
import json
import math
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent

BASE_REPO = "Qwen/Qwen-Image-2.1-Turbo"
LORA_FILE = "tiny-london-v3.safetensors"
PROMPT = (
    "Fill in the outlined section with the missing pixels corresponding to the "
    "<isometric london pixel art> style, removing the border and exactly following "
    "the shape/style/structure of the surrounding image (if present)."
)
SEED = 1916

LP = 16  # px per latent cell
WIN = 64  # 1024 px window (latent cells)
STRIDE = 48  # 768 px
SHIFT = 24  # 384 px
RAMP = 16  # 256 px ramps -> flat middle 512 px
TILE = 512
CORE = 12288  # block core, px (24 tiles, 6 shots)
MARGIN = 1024  # context round the core, px
STRIP = 256  # blend band written over finished neighbours, px
UNPIN = 384  # redraw this much of finished neighbours (pinning starts beyond it)
SEA_RGB = (55, 81, 77)

weights_vol = modal.Volume.from_name("dublin-pilot")
metro_vol = modal.Volume.from_name("dublin-metro", create_if_missing=True)
W_PATH, M_PATH = "/w", "/m"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.15.0",
        "torchvision",
        "diffusers==0.42.1",
        "transformers==5.22.2",
        "accelerate==1.15.0",
        "huggingface_hub[hf_xet]>=1.32,<3",
        "safetensors>=0.8",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": f"{W_PATH}/hf", "HF_HUB_OFFLINE": "1"})
)
app = modal.App("dublin-metro-draw", image=image)


# --------------------------------------------------------------------------- #
# Geometry helpers (shared by the local driver and the container)
# --------------------------------------------------------------------------- #
def axis_positions(n: int, offset: int) -> list[int]:
    if n <= WIN:
        return [0]
    pos = list(range(offset, n - WIN + 1, STRIDE))
    if offset:
        pos = [0] + pos
    if pos[-1] != n - WIN:
        pos.append(n - WIN)
    return sorted(set(pos))


def axis_weight(start: int, n: int):
    import torch

    w = torch.ones(WIN)
    ramp = (torch.arange(RAMP) + 0.5) / RAMP
    if start > 0:
        w[:RAMP] = ramp
    if start + WIN < n:
        w[-RAMP:] = ramp.flip(0)
    return w


def block_origin(bx: int, by: int) -> tuple[int, int]:
    """Top-left canvas px of the block's padded region (core minus margin)."""
    return bx * CORE - MARGIN, by * CORE - MARGIN


SIDE = CORE + 2 * MARGIN  # padded block side, px
SIDE_L = SIDE // LP


def phase_of(bx: int, by: int) -> int:
    return (by % 2) * 2 + (bx % 2)


# --------------------------------------------------------------------------- #
# Container helpers
# --------------------------------------------------------------------------- #
def merge_lora(transformer, path: str, scale: float = 1.0) -> int:
    import torch
    from safetensors.torch import load_file

    sd = load_file(path)
    groups: dict[str, dict] = {}
    for key, tensor in sd.items():
        base, kind = key[len("diffusion_model.") :].rsplit(".lora_", 1)
        groups.setdefault(base, {})[kind[0]] = tensor
    modules = dict(transformer.named_modules())
    merged = 0
    for base, ab in groups.items():
        delta = (ab["B"].to("cuda", torch.float32) @ ab["A"].to("cuda", torch.float32)) * scale
        if base.endswith("img_mlp.gate_up"):
            gate, up = delta.chunk(2, dim=0)
            targets = [(base.replace("gate_up", "gate_layer"), gate), (base.replace("gate_up", "proj"), up)]
        else:
            targets = [(base, delta)]
        for name, d in targets:
            lin = modules[name]
            assert tuple(lin.weight.shape) == tuple(d.shape), (name, lin.weight.shape, d.shape)
            lin.weight.data += d.to(lin.weight.dtype)
            merged += 1
    return merged


def red_border(img, px: int = 2):
    from PIL import ImageDraw

    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(px):
        d.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(255, 0, 0))
    return img


def load_region(x0: int, y0: int, w: int, h: int, folder: Path, name_fn, unit: int):
    """Assemble an RGB region from a grid of square images of side `unit`.

    Returns (image, have) where have is a bool array (h, w) of pixels covered.
    """
    import numpy as np
    from PIL import Image

    out = np.zeros((h, w, 3), np.uint8)
    have = np.zeros((h, w), bool)
    for gy in range(math.floor(y0 / unit), math.ceil((y0 + h) / unit)):
        for gx in range(math.floor(x0 / unit), math.ceil((x0 + w) / unit)):
            f = folder / name_fn(gx, gy)
            if not f.exists():
                continue
            im = np.asarray(Image.open(f).convert("RGB"))
            sx0, sy0 = gx * unit, gy * unit
            ax0, ay0 = max(x0, sx0), max(y0, sy0)
            ax1, ay1 = min(x0 + w, sx0 + unit), min(y0 + h, sy0 + unit)
            out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = im[ay0 - sy0 : ay1 - sy0, ax0 - sx0 : ax1 - sx0]
            have[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = True
    return out, have


@app.cls(
    gpu="H100",
    volumes={W_PATH: weights_vol, M_PATH: metro_vol},
    timeout=2 * 3600,
    memory=65536,
    max_containers=4,
    scaledown_window=60,
)
class BlockPainter:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        t0 = time.time()
        self.pipe = QwenImage21Pipeline.from_pretrained(BASE_REPO, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        n = merge_lora(self.pipe.transformer, f"{W_PATH}/lora/{LORA_FILE}")
        assert n == 224, n
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import retrieve_timesteps

        retrieve_timesteps(self.pipe.scheduler, None, "cuda", sigmas=self.pipe.config.sample_sigmas, mu=0.0)
        self.sigmas = [float(s) for s in self.pipe.scheduler.sigmas]
        vae = self.pipe.vae
        self.lat_mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
        self.lat_std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
        print(f"model loaded in {time.time() - t0:.0f} s; sigmas {self.sigmas}")

    # ---- model plumbing (same as the pilot, which matched the stock pipeline) ----
    def _encode_window(self, crop) -> dict:
        import torch

        pipe = self.pipe
        img = red_border(crop).convert("RGBA")
        size = WIN * LP
        resized = pipe.image_processor.resize(img, width=size, height=size)
        pe, pm, ipm = pipe.encode_prompt(prompt=PROMPT, image=[resized], device="cuda")
        vae_in = pipe.image_processor.preprocess(img, width=size, height=size).unsqueeze(2)
        cond = pipe._encode_vae_image(vae_in.to("cuda", pipe.vae.dtype), generator=None)
        cond = pipe._pack_latents(cond.to(torch.bfloat16), 1, cond.shape[1], WIN, WIN)
        ipm = torch.cat([ipm, ipm.new_ones(1, WIN * WIN // 4)], dim=1)
        return {"pe": pe, "pm": pm, "ipm": ipm, "cond": cond}

    def _predict(self, c: dict, lat_win, sigma: float):
        import torch

        tgt = lat_win.to(torch.bfloat16).reshape(1, lat_win.shape[1], WIN * WIN).transpose(1, 2)
        hidden = torch.cat([c["cond"], tgt], dim=1)
        t = torch.full((1,), sigma, device="cuda", dtype=torch.bfloat16)
        out = self.pipe.transformer(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=c["pe"],
            encoder_hidden_states_mask=c["pm"],
            img_shapes=[[(1, WIN, WIN), (1, WIN, WIN)]],
            img_mask=c["ipm"],
            return_dict=False,
        )[0]
        return out[:, -WIN * WIN :].transpose(1, 2).reshape(1, -1, WIN, WIN).float()

    def _encode_image(self, rgb):
        """Tiled VAE encode of an (H, W, 3) uint8 array -> normalized latent (1, C, H/16, W/16)."""
        import numpy as np
        import torch
        from PIL import Image

        pipe = self.pipe
        H, W = rgb.shape[:2]
        hl, wl = H // LP, W // LP
        acc = wsum = None
        for y in axis_positions(hl, 0):
            for x in axis_positions(wl, 0):
                crop = Image.fromarray(rgb[y * LP : (y + WIN) * LP, x * LP : (x + WIN) * LP]).convert("RGBA")
                t = pipe.image_processor.preprocess(crop, width=WIN * LP, height=WIN * LP).unsqueeze(2)
                with torch.no_grad():
                    z = pipe._encode_vae_image(t.to("cuda", pipe.vae.dtype), generator=None)[:, :, 0].float()
                if acc is None:
                    acc = torch.zeros((1, z.shape[1], hl, wl), device="cuda")
                    wsum = torch.zeros((1, 1, hl, wl), device="cuda")
                w = (axis_weight(y, hl)[:, None] * axis_weight(x, wl)[None, :]).to("cuda")
                acc[:, :, y : y + WIN, x : x + WIN] += z * w
                wsum[:, :, y : y + WIN, x : x + WIN] += w
        return acc / wsum

    def _decode(self, lat):
        import numpy as np
        import torch

        vae = self.pipe.vae
        _, _, hl, wl = lat.shape
        H, W = hl * LP, wl * LP
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for y in axis_positions(hl, 0):
            for x in axis_positions(wl, 0):
                z = lat[:, :, y : y + WIN, x : x + WIN].unsqueeze(2)
                z = (z * self.lat_std + self.lat_mean).to(vae.dtype)
                with torch.no_grad():
                    px = vae.decode(z, return_dict=False)[0][:, :3, 0]
                px = ((px[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
                wy = axis_weight(y, hl).repeat_interleave(LP).numpy()
                wx = axis_weight(x, wl).repeat_interleave(LP).numpy()
                w = (wy[:, None] * wx[None, :])[..., None]
                ys, xs = y * LP, x * LP
                acc[ys : ys + WIN * LP, xs : xs + WIN * LP] += px * w
                wsum[ys : ys + WIN * LP, xs : xs + WIN * LP] += w
        return (acc / wsum).round().clip(0, 255).astype(np.uint8)

    # ---- one block -----------------------------------------------------------
    @modal.method()
    def draw_block(self, bx: int, by: int, job: str = "main") -> dict:
        import numpy as np
        import torch
        from PIL import Image

        t_start = time.time()
        metro_vol.reload()
        root = Path(M_PATH)
        meta = json.loads((root / "canvas.json").read_text())
        c0, r0, shot = meta["c0"], meta["r0"], meta["shot_px"]
        x0, y0 = block_origin(bx, by)
        outdir = root / job / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        # Mask cells for this padded block (0 outside, 1 draw, 2 sea fill).
        full_mask = np.load(root / "mask.npy", mmap_mode="r")
        mhl, mwl = full_mask.shape
        mask = np.zeros((SIDE_L, SIDE_L), np.uint8)
        lx0, ly0 = x0 // LP, y0 // LP
        sx0, sy0 = max(0, lx0), max(0, ly0)
        sx1, sy1 = min(mwl, lx0 + SIDE_L), min(mhl, ly0 + SIDE_L)
        if sx1 > sx0 and sy1 > sy0:
            mask[sy0 - ly0 : sy1 - ly0, sx0 - lx0 : sx1 - lx0] = full_mask[sy0:sy1, sx0:sx1]

        # Render for this block (canvas px -> shots named by lattice index).
        render, _ = load_region(
            x0, y0, SIDE, SIDE, root / "shots",
            lambda gx, gy: f"r{gy + r0}_c{gx + c0}.png", shot,
        )

        # Known image: sea fill, then finished neighbour tiles on top.
        known = np.zeros((SIDE, SIDE, 3), np.uint8)
        rng = np.random.default_rng(bx * 1000 + by)
        sea_px = np.kron(mask == 2, np.ones((LP, LP), bool))
        noise = rng.normal(0, 1.5, (SIDE, SIDE, 1))
        sea_img = np.clip(np.array(SEA_RGB)[None, None, :] + noise, 0, 255).astype(np.uint8)
        known[sea_px] = sea_img[sea_px]
        done_img, done_have = load_region(
            x0, y0, SIDE, SIDE, outdir, lambda gx, gy: f"{gx}_{gy}.png", TILE,
        )
        # Never pin our own core (re-runs redraw it from scratch).
        core = np.zeros((SIDE, SIDE), bool)
        core[MARGIN : MARGIN + CORE, MARGIN : MARGIN + CORE] = True
        done_have &= ~core
        known[done_have] = done_img[done_have]
        # Distance (px) outside the core. Finished neighbours are redrawn up to
        # UNPIN px out and blended back over STRIP px, so the step between
        # pinned and generated latents lands where the old pixels are kept.
        yy = np.arange(SIDE)
        dy = np.maximum(np.maximum(MARGIN - yy, yy - (MARGIN + CORE - 1)), 0)
        dist = np.maximum(dy[:, None], dy[None, :]).astype(np.float32)
        pin_done = done_have & (dist > UNPIN)
        done_l = pin_done.reshape(SIDE_L, LP, SIDE_L, LP).all(axis=(1, 3))
        pinned_l = (mask != 1) | done_l  # outside, sea fill, or finished
        free_core = (~pinned_l)[MARGIN // LP : (MARGIN + CORE) // LP, MARGIN // LP : (MARGIN + CORE) // LP]
        if not free_core.any():
            return {"block": [bx, by], "skipped": "nothing to draw in core"}

        known_lat = self._encode_image(known)
        pin = torch.from_numpy(pinned_l.astype(np.float32)).to("cuda")[None, None]

        grids = []
        for off in (0, SHIFT):
            g = []
            for y in axis_positions(SIDE_L, off):
                for x in axis_positions(SIDE_L, off):
                    if (~pinned_l[y : y + WIN, x : x + WIN]).any():
                        g.append((y, x))
            grids.append(g)
        t0 = time.time()
        conds = {}
        with torch.no_grad():
            for y, x in sorted(set(grids[0]) | set(grids[1])):
                crop = Image.fromarray(render[y * LP : (y + WIN) * LP, x * LP : (x + WIN) * LP])
                conds[(y, x)] = self._encode_window(crop)
        t_enc = time.time() - t0
        wts = {k: (axis_weight(k[0], SIDE_L)[:, None] * axis_weight(k[1], SIDE_L)[None, :]).to("cuda") for k in conds}

        sig = self.sigmas
        g = torch.Generator("cuda").manual_seed(SEED)
        # Noise is drawn per block but seeded by global position for repeatability.
        g.manual_seed(SEED * 100003 + bx * 1009 + by)
        chans = known_lat.shape[1]
        noise_t = torch.randn((1, chans, SIDE_L, SIDE_L), generator=g, device="cuda")
        lat = pin * ((1 - sig[0]) * known_lat + sig[0] * noise_t) + (1 - pin) * noise_t
        t1 = time.time()
        with torch.no_grad():
            for i in range(len(sig) - 1):
                vel = torch.zeros_like(lat)
                wsum = torch.zeros((1, 1, SIDE_L, SIDE_L), device="cuda")
                for y, x in grids[i % 2]:
                    v = self._predict(conds[(y, x)], lat[:, :, y : y + WIN, x : x + WIN], sig[i])
                    w = wts[(y, x)]
                    vel[:, :, y : y + WIN, x : x + WIN] += v * w
                    wsum[:, :, y : y + WIN, x : x + WIN] += w
                lat = lat + (sig[i + 1] - sig[i]) * vel / wsum.clamp_min(1e-6)
                pinned_now = (1 - sig[i + 1]) * known_lat + sig[i + 1] * noise_t
                lat = pin * pinned_now + (1 - pin) * lat
                print(f"block {bx},{by} step {i + 1}/{len(sig) - 1}: {len(grids[i % 2])} windows, {time.time() - t1:.0f} s", flush=True)
        t_sample = time.time() - t1
        del conds
        torch.cuda.empty_cache()

        t2 = time.time()
        rgb = self._decode(lat)
        # Where a pixel is outside the drawing area keep black (never shown).
        outside_px = np.kron(mask == 0, np.ones((LP, LP), bool))

        # Write core tiles, and blend a STRIP band over finished neighbours.
        alpha = np.zeros((SIDE, SIDE), np.float32)
        alpha[MARGIN : MARGIN + CORE, MARGIN : MARGIN + CORE] = 1.0
        band = (dist > 0) & (dist <= STRIP) & done_have
        t = dist[band] / (STRIP + 1)
        alpha[band] = 0.5 * (1 + np.cos(np.pi * t))  # smooth 1 -> 0
        written = 0
        tiles_side = SIDE // TILE
        for ty in range(tiles_side):
            for tx in range(tiles_side):
                sl = (slice(ty * TILE, (ty + 1) * TILE), slice(tx * TILE, (tx + 1) * TILE))
                a = alpha[sl]
                if a.max() == 0 or outside_px[sl].all():
                    continue
                gtx, gty = (x0 // TILE) + tx, (y0 // TILE) + ty
                if gtx < 0 or gty < 0:
                    continue
                new = rgb[sl].astype(np.float32)
                old = done_img[sl].astype(np.float32)
                out = new * a[..., None] + old * (1 - a[..., None])
                out[outside_px[sl]] = 0
                path = outdir / f"{gtx}_{gty}.png"
                tmp = path.with_suffix(".tmp.png")
                Image.fromarray(out.round().astype(np.uint8)).save(tmp, compress_level=3)
                tmp.rename(path)
                written += 1
        stats = {
            "block": [bx, by],
            "windows": [len(grids[0]), len(grids[1])],
            "tiles_written": written,
            "seconds": {
                "encode": round(t_enc), "sample": round(t_sample),
                "decode_write": round(time.time() - t2), "total": round(time.time() - t_start),
            },
        }
        bdir = root / job / "blocks"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{bx}_{by}.json").write_text(json.dumps(stats))
        metro_vol.commit()
        return stats


# --------------------------------------------------------------------------- #
# Sea-only / outside blocks: written on CPU, no model.
# --------------------------------------------------------------------------- #
@app.function(volumes={M_PATH: metro_vol}, cpu=2, memory=8192, timeout=3600)
def fill_block(bx: int, by: int, job: str = "main") -> dict:
    import numpy as np
    from PIL import Image

    metro_vol.reload()
    root = Path(M_PATH)
    full_mask = np.load(root / "mask.npy", mmap_mode="r")
    outdir = root / job / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(bx * 1000 + by)
    written = 0
    for ty in range(CORE // TILE):
        for tx in range(CORE // TILE):
            gx, gy = bx * CORE + tx * TILE, by * CORE + ty * TILE
            m = np.asarray(full_mask[gy // LP : (gy + TILE) // LP, gx // LP : (gx + TILE) // LP])
            if m.size == 0 or not (m == 2).any():
                continue
            mpx = np.kron(m, np.ones((LP, LP), np.uint8))
            h, w = mpx.shape
            img = np.zeros((TILE, TILE, 3), np.uint8)
            sea = np.clip(np.array(SEA_RGB)[None, None, :] + rng.normal(0, 1.5, (h, w, 1)), 0, 255)
            img[:h, :w][mpx == 2] = sea[mpx == 2].astype(np.uint8)
            Image.fromarray(img).save(outdir / f"{gx // TILE}_{gy // TILE}.png", compress_level=3)
            written += 1
    bdir = root / job / "blocks"
    bdir.mkdir(parents=True, exist_ok=True)
    stats = {"block": [bx, by], "filled_tiles": written}
    (bdir / f"{bx}_{by}.json").write_text(json.dumps(stats))
    metro_vol.commit()
    return stats


# --------------------------------------------------------------------------- #
# Local driver
# --------------------------------------------------------------------------- #
def block_table(mask) -> list[dict]:
    """Per block: how many draw / sea cells its core has."""
    import numpy as np

    hl, wl = mask.shape
    nbx, nby = math.ceil(wl * LP / CORE), math.ceil(hl * LP / CORE)
    cl = CORE // LP
    rows = []
    for by in range(nby):
        for bx in range(nbx):
            m = mask[by * cl : (by + 1) * cl, bx * cl : (bx + 1) * cl]
            rows.append({"bx": bx, "by": by, "draw": int((m == 1).sum()), "sea": int((m == 2).sum())})
    return rows


@app.function(volumes={M_PATH: metro_vol}, cpu=1, memory=4096, timeout=24 * 3600)
def orchestrate(phases: str, only: str, job: str, dry: bool = False) -> None:
    """Runs on Modal so the multi-hour run survives the local client going away."""
    import numpy as np

    metro_vol.reload()
    mask = np.load(f"{M_PATH}/mask.npy")
    table = block_table(mask)
    if only:
        keep = {tuple(int(v) for v in b.split("_")) for b in only.split(",")}
        table = [t for t in table if (t["bx"], t["by"]) in keep]
    painter = BlockPainter()
    for ph in [int(p) for p in phases.split(",")]:
        metro_vol.reload()
        try:
            done = {e.path.split("/")[-1][:-5] for e in metro_vol.listdir(f"/{job}/blocks")}
        except Exception:
            done = set()
        rows = [t for t in table if phase_of(t["bx"], t["by"]) == ph and f"{t['bx']}_{t['by']}" not in done]
        gpu = [t for t in rows if t["draw"] > 0]
        cpu = [t for t in rows if t["draw"] == 0 and t["sea"] > 0]
        print(f"phase {ph}: {len(gpu)} GPU blocks, {len(cpu)} sea-fill blocks "
              f"({sum(t['draw'] for t in gpu) * LP * LP / 1e9:.2f} Gpx to draw)", flush=True)
        if dry:
            continue
        if cpu:
            for res in fill_block.starmap([(t["bx"], t["by"], job) for t in cpu], return_exceptions=True):
                if isinstance(res, Exception):
                    print("fill failed:", repr(res)[:300], flush=True)
        if not gpu:
            continue
        t0 = time.time()
        # Biggest blocks first so the phase does not end on one long straggler.
        gpu.sort(key=lambda t: -t["draw"])
        for res in painter.draw_block.starmap(
            [(t["bx"], t["by"], job) for t in gpu], return_exceptions=True, order_outputs=False
        ):
            if isinstance(res, Exception):
                print("BLOCK FAILED:", repr(res)[:500], flush=True)
            else:
                print(f"[{time.time() - t0:6.0f} s] {json.dumps(res)}", flush=True)
        # A failed block stays unmarked; retry it once before moving on, since
        # later phases grow out of it.
        metro_vol.reload()
        done = {e.path.split("/")[-1][:-5] for e in metro_vol.listdir(f"/{job}/blocks")}
        again = [t for t in gpu if f"{t['bx']}_{t['by']}" not in done]
        if again:
            print(f"phase {ph}: retrying {len(again)} blocks", flush=True)
            for res in painter.draw_block.starmap([(t["bx"], t["by"], job) for t in again], return_exceptions=True):
                print("retry:", repr(res)[:300] if isinstance(res, Exception) else json.dumps(res), flush=True)
    print("orchestrator finished", flush=True)


@app.local_entrypoint()
def draw(phases: str = "0,1,2,3", only: str = "", job: str = "main", upload: bool = True, dry: bool = False) -> None:
    plan = json.loads((HERE / "plan.json").read_text())
    if upload:
        meta = {
            "c0": plan["lattice"]["c0"], "r0": plan["lattice"]["r0"], "shot_px": plan["shot_px"],
            "core": CORE, "margin": MARGIN, "tile": TILE,
        }
        (HERE / "canvas.json").write_text(json.dumps(meta))
        with metro_vol.batch_upload(force=True) as b:
            b.put_file(str(HERE / "mask.npy"), "/mask.npy")
            b.put_file(str(HERE / "canvas.json"), "/canvas.json")
    orchestrate.remote(phases, only, job, dry)
