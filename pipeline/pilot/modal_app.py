"""Draw the Dublin pilot with the tiny.london LoRA on Modal.

Model loading path (chosen as the simplest reliable one):
  * Base: Qwen/Qwen-Image-2.1-Turbo in diffusers format, bf16, loaded with
    diffusers' QwenImage21Pipeline (diffusers 0.42.1). Fits on one H100 80 GB:
    transformer ~20 GB + Qwen3-VL text encoder ~17.5 GB.
  * The LoRA was trained by ai-toolkit on the ComfyUI int8_convrot copy of the
    same weights. diffusers cannot read that int8 format, so we use the bf16
    original; int8 is a quantisation of it, so the LoRA should transfer.
  * LoRA: ktoyame/tiny-london-v3 (rank 32, alpha 32 -> scale 1.0). Its keys
    use ComfyUI's fused `img_mlp.gate_up` ([gate; up] rows), which diffusers'
    LoRA loader does not split, so we merge it into the weights ourselves:
    W += B @ A, with the gate_up delta split into gate_layer / proj.

Sampler: MultiDiffusion over one latent for the whole canvas (tiny.london):
  1024 px windows (64 latent tokens) on a 768 px stride, odd steps on a grid
  shifted by 384 px, tent weights flat in the middle 512 px, 8 Euler steps on
  the model's own sigma schedule, no CFG. Each window is conditioned on its own
  render crop with a 2 px red border, exactly as the LoRA was trained.

Usage (from pipeline/pilot, with Modal credentials configured):
  uv run modal run modal_app.py::download        # once: weights -> Volume (CPU only)
  uv run modal run modal_app.py::single          # 1 window via the stock pipeline (sanity check)
  uv run modal run modal_app.py::draw            # the whole out/render.png -> out/pixel.png

Licences: the LoRA and Qwen-Image-2.1 weights are under the Qwen Research
License (non-commercial research/evaluation only).
"""

import io
import json
import time
from pathlib import Path

import modal

APP_NAME = "dublin-pilot"
BASE_REPO = "Qwen/Qwen-Image-2.1-Turbo"
LORA_REPO = "ktoyame/tiny-london-lora"
LORA_FILE = "tiny-london-v3.safetensors"
LORA_SCALE = 1.0  # rank 32, alpha 32
VOL_PATH = "/vol"
HF_CACHE = f"{VOL_PATH}/hf"

LATENT_PX = 16  # one latent token = 16 x 16 px
WIN = 64  # 1024 px window
STRIDE = 48  # 768 px stride
SHIFT = 24  # 384 px shift on odd steps
RAMP = 16  # 256 px tent ramp each side -> flat middle 512 px

DEFAULT_PROMPT = (
    "Fill in the outlined section with the missing pixels corresponding to the "
    "<isometric london pixel art> style, removing the border and exactly following "
    "the shape/style/structure of the surrounding image (if present)."
)

volume = modal.Volume.from_name("dublin-pilot", create_if_missing=True)

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
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App(APP_NAME, image=image)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL_PATH: volume}, timeout=3 * 3600, cpu=4, memory=8192)
def download_weights() -> str:
    from huggingface_hub import hf_hub_download, snapshot_download

    t0 = time.time()
    base = snapshot_download(BASE_REPO, ignore_patterns=["assets/*"])
    lora = hf_hub_download(LORA_REPO, LORA_FILE, local_dir=f"{VOL_PATH}/lora")
    volume.commit()
    return f"base at {base}\nlora at {lora}\n{time.time() - t0:.0f} s"


# --------------------------------------------------------------------------- #
# Helpers (run inside the container)
# --------------------------------------------------------------------------- #
def merge_lora(transformer, path: str, scale: float = 1.0) -> int:
    import torch
    from safetensors.torch import load_file

    sd = load_file(path)
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        assert key.startswith("diffusion_model."), key
        base, kind = key[len("diffusion_model.") :].rsplit(".lora_", 1)
        groups.setdefault(base, {})[kind[0]] = tensor  # "A" or "B"
    modules = dict(transformer.named_modules())
    merged = 0
    for base, ab in groups.items():
        a = ab["A"].to("cuda", torch.float32)
        b = ab["B"].to("cuda", torch.float32)
        delta = (b @ a) * scale
        if base.endswith("img_mlp.gate_up"):
            gate, up = delta.chunk(2, dim=0)  # ComfyUI order: gate rows first
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
    if start > 0:  # interior edge: taper; canvas edge: keep full weight
        w[:RAMP] = ramp
    if start + WIN < n:
        w[-RAMP:] = ramp.flip(0)
    return w


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@app.cls(gpu="H100", volumes={VOL_PATH: volume}, timeout=3 * 3600, memory=65536, scaledown_window=120)
class Painter:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        t0 = time.time()
        self.pipe = QwenImage21Pipeline.from_pretrained(BASE_REPO, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        n = merge_lora(self.pipe.transformer, f"{VOL_PATH}/lora/{LORA_FILE}", LORA_SCALE)
        print(f"loaded in {time.time() - t0:.0f} s, merged LoRA into {n} linear layers")
        assert n == 32 * 7, f"expected 224 merged layers, got {n}"

    # ---- stock pipeline, one 1024 px frame: the reference for the sampler ----
    @modal.method()
    def single(self, png: bytes, prompt: str, seed: int) -> bytes:
        import torch
        from PIL import Image

        img = red_border(Image.open(io.BytesIO(png)).convert("RGB")).convert("RGBA")
        out = self.pipe(
            prompt=prompt,
            image=[img],
            width=img.width,
            height=img.height,
            true_cfg_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_resolution=img.width,
        ).images[0]
        buf = io.BytesIO()
        out.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    # ---- per-window conditioning ------------------------------------------
    def _encode_window(self, crop, prompt: str) -> dict:
        import torch

        pipe = self.pipe
        img = red_border(crop).convert("RGBA")
        size = WIN * LATENT_PX
        resized = pipe.image_processor.resize(img, width=size, height=size)
        pe, pm, ipm = pipe.encode_prompt(prompt=prompt, image=[resized], device="cuda")
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
        v = out[:, -WIN * WIN :].transpose(1, 2).reshape(1, -1, WIN, WIN)
        return v.float()

    def _sigmas(self) -> list[float]:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import retrieve_timesteps

        sample = self.pipe.config.sample_sigmas
        retrieve_timesteps(self.pipe.scheduler, None, "cuda", sigmas=sample, mu=0.0)
        return [float(s) for s in self.pipe.scheduler.sigmas]  # includes the terminal 0

    def _decode(self, lat):
        """Tiled VAE decode into CPU memory, feathered over a 256 px overlap."""
        import numpy as np
        import torch

        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
        _, _, hl, wl = lat.shape
        H, W = hl * LATENT_PX, wl * LATENT_PX
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for y in axis_positions(hl, 0):
            for x in axis_positions(wl, 0):
                z = lat[:, :, y : y + WIN, x : x + WIN].unsqueeze(2)
                z = (z * std + mean).to(vae.dtype)
                with torch.no_grad():
                    px = vae.decode(z, return_dict=False)[0][:, :3, 0]  # RGB, [-1, 1]
                px = ((px[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
                wy = axis_weight(y, hl).repeat_interleave(LATENT_PX).numpy()
                wx = axis_weight(x, wl).repeat_interleave(LATENT_PX).numpy()
                w = (wy[:, None] * wx[None, :])[..., None]
                ys, xs = y * LATENT_PX, x * LATENT_PX
                acc[ys : ys + WIN * LATENT_PX, xs : xs + WIN * LATENT_PX] += px * w
                wsum[ys : ys + WIN * LATENT_PX, xs : xs + WIN * LATENT_PX] += w
        return (acc / wsum).round().clip(0, 255).astype(np.uint8)

    # ---- the joint sampler -------------------------------------------------
    @modal.method()
    def draw(self, job: str, prompt: str, seed: int) -> dict:
        import torch
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        volume.reload()
        jobdir = Path(VOL_PATH) / "jobs" / job
        render = Image.open(jobdir / "render.png").convert("RGB")
        W, H = render.size
        assert W % LATENT_PX == 0 and H % LATENT_PX == 0, "canvas must be a multiple of 16 px"
        hl, wl = H // LATENT_PX, W // LATENT_PX
        assert hl >= WIN and wl >= WIN, "canvas must be at least 1024 px each way"

        grids = [
            [(y, x) for y in axis_positions(hl, 0) for x in axis_positions(wl, 0)],
            [(y, x) for y in axis_positions(hl, SHIFT) for x in axis_positions(wl, SHIFT)],
        ]
        t0 = time.time()
        conds = {}
        with torch.no_grad():
            for y, x in sorted(set(grids[0]) | set(grids[1])):
                box = (x * LATENT_PX, y * LATENT_PX, (x + WIN) * LATENT_PX, (y + WIN) * LATENT_PX)
                conds[(y, x)] = self._encode_window(render.crop(box), prompt)
        t_enc = time.time() - t0
        print(f"canvas {W}x{H}, {len(conds)} windows encoded in {t_enc:.0f} s")

        weights = {}
        for y, x in conds:
            weights[(y, x)] = (axis_weight(y, hl)[:, None] * axis_weight(x, wl)[None, :]).to("cuda")

        sigmas = self._sigmas()
        channels = self.pipe.transformer.config.in_channels
        g = torch.Generator("cuda").manual_seed(seed)
        noise = torch.randn((1, channels, hl, wl), generator=g, device="cuda", dtype=torch.float32)
        lat = noise.clone()
        t1 = time.time()
        with torch.no_grad():
            for i in range(len(sigmas) - 1):
                vel = torch.zeros_like(lat)
                wsum = torch.zeros((1, 1, hl, wl), device="cuda")
                for y, x in grids[i % 2]:
                    v = self._predict(conds[(y, x)], lat[:, :, y : y + WIN, x : x + WIN], sigmas[i])
                    wt = weights[(y, x)]
                    vel[:, :, y : y + WIN, x : x + WIN] += v * wt
                    wsum[:, :, y : y + WIN, x : x + WIN] += wt
                lat += (sigmas[i + 1] - sigmas[i]) * vel / wsum
                torch.save(lat.cpu(), jobdir / "latent.pt")  # checkpoint every step
                print(f"step {i + 1}/{len(sigmas) - 1} sigma {sigmas[i]:.3f}, {time.time() - t1:.0f} s")
        t_sample = time.time() - t1

        t2 = time.time()
        rgb = self._decode(lat)
        Image.fromarray(rgb).save(jobdir / "pixel.png", compress_level=1)
        volume.commit()
        stats = {
            "canvas": [W, H],
            "windows": len(conds),
            "windows_per_step": [len(grids[0]), len(grids[1])],
            "sigmas": sigmas,
            "seconds": {"encode": t_enc, "sample": t_sample, "decode": time.time() - t2},
        }
        (jobdir / "stats.json").write_text(json.dumps(stats, indent=1))
        volume.commit()
        return stats


# --------------------------------------------------------------------------- #
# Local entrypoints
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
OUT = HERE / "out"


def _cfg() -> dict:
    return json.loads((HERE / "config.json").read_text())


@app.local_entrypoint()
def download() -> None:
    print(download_weights.remote())


@app.local_entrypoint()
def single(size: int = 1024, src: str = "", tag: str = "single") -> None:
    """Draw one window from the centre of out/render.png (or --src) with the stock pipeline."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    cfg = _cfg()
    render = Image.open(Path(src) if src else OUT / "render.png").convert("RGB")
    cx, cy = render.width // 2, render.height // 2
    crop = render.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    out = Painter().single.remote(buf.getvalue(), cfg["lora_prompt"], cfg["seed"])
    crop.save(OUT / f"{tag}_render.png")
    (OUT / f"{tag}_pixel.png").write_bytes(out)
    print(f"wrote out/{tag}_render.png and out/{tag}_pixel.png")


@app.local_entrypoint()
def draw(job: str = "pilot-v1", src: str = "", dest: str = "") -> None:
    cfg = _cfg()
    src_path = Path(src) if src else OUT / "render.png"
    dest_path = Path(dest) if dest else OUT / "pixel.png"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(src_path), f"/jobs/{job}/render.png")
    stats = Painter().draw.remote(job, cfg["lora_prompt"], cfg["seed"])
    print(json.dumps(stats, indent=1))
    with open(dest_path, "wb") as f:
        for chunk in volume.read_file(f"/jobs/{job}/pixel.png"):
            f.write(chunk)
    print(f"wrote {dest_path}")
