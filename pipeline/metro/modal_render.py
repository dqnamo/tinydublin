"""Render the metro Dublin shot lattice on many Modal CPU containers.

Each container runs headless Chromium with software WebGL (SwiftShader) and
renders a batch of neighbouring shots into the `dublin-metro` Volume under
/shots. Resumable: existing shots are skipped, so just run it again.

    ../pilot/.venv/bin/modal run modal_render.py::render --limit 2     # test
    ../pilot/.venv/bin/modal run modal_render.py::render               # all
"""

import json
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
BATCH = 12

volume = modal.Volume.from_name("dublin-metro", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "gnupg")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
    )
    .add_local_dir(
        HERE / "render", "/app/render", copy=True, ignore=["node_modules", "*.log"]
    )
    .run_commands(
        "cd /app/render && npm ci --no-audit --no-fund",
        "cd /app/render && npx playwright install --with-deps chromium",
    )
    .env({"DUBLIN_ENV_DIR": "/app/render"})
    .pip_install("numpy", "pillow")
)

app = modal.App("dublin-metro-render", image=image)


@app.function(
    cpu=4,
    memory=6144,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("dublin-tiles")],
    timeout=4 * 3600,
    max_containers=12,
    retries=1,
)
def render_batch(shots: list[list[int]], cfg: dict) -> list[dict]:
    Path("/tmp/cfg.json").write_text(json.dumps(cfg))
    arg = ";".join(",".join(str(v) for v in s) for s in shots)
    t0 = time.time()
    proc = subprocess.run(
        ["node", "render_batch.mjs", "/tmp/cfg.json", "/vol/shots", arg],
        cwd="/app/render", capture_output=True, text=True, timeout=4 * 3600 - 300,
    )
    volume.commit()
    out = []
    for line in proc.stdout.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0:
        out.append({"error": f"exit {proc.returncode}", "stderr": proc.stderr[-2000:]})
    out.append({"batch_s": round(time.time() - t0, 1)})
    return out


@app.function(volumes={"/vol": volume}, cpu=2, memory=4096, timeout=3600)
def scan_chunk(names: list[str]) -> list[list]:
    import numpy as np
    from PIL import Image

    bad = []
    for n in names:
        p = Path("/vol/shots") / n
        try:
            a = np.asarray(Image.open(p).convert("RGB"))[::7, ::7].astype(int)
        except Exception as e:  # unreadable -> bad
            bad.append([n, -1.0])
            continue
        sky = float(((np.abs(a - [135, 206, 235]).sum(2)) < 10).mean())
        if sky > 0.01:
            bad.append([n, sky])
    for n, _ in bad:
        (Path("/vol/shots") / n).unlink(missing_ok=True)
    volume.commit()
    return bad


@app.function(volumes={"/vol": volume}, cpu=1, timeout=600)
def delete_shots(names: list[str]) -> int:
    n = 0
    for name in names:
        p = Path("/vol/shots") / name
        if p.exists():
            p.unlink()
            n += 1
    volume.commit()
    return n


@app.local_entrypoint()
def delete(listfile: str) -> None:
    pairs = json.loads(Path(listfile).read_text())
    print("deleted", delete_shots.remote([f"r{r}_c{c}.png" for c, r in pairs]))


@app.local_entrypoint()
def scan() -> None:
    names = [e.path.split("/")[-1] for e in volume.listdir("/shots") if e.path.endswith(".png")]
    chunks = [names[i : i + 60] for i in range(0, len(names), 60)]
    bad = []
    for res in scan_chunk.map(chunks):
        bad += res
    print(f"scanned {len(names)} shots, deleted {len(bad)} with missing tiles")
    (HERE / "scan_bad.json").write_text(json.dumps(bad))


@app.local_entrypoint()
def render(limit: int = 0, upload_pilot: bool = True) -> None:
    plan = json.loads((HERE / "plan.json").read_text())
    cfg = plan["config"]
    have = set()
    try:
        have = {e.path.split("/")[-1] for e in volume.listdir("/shots")}
    except Exception:
        pass
    if upload_pilot:
        pilot = HERE.parent / "pilot" / "out" / "shots"
        new = [p for p in pilot.glob("r*_c*.png") if p.name not in have]
        if new:
            with volume.batch_upload(force=True) as b:
                for p in new:
                    b.put_file(str(p), f"/shots/{p.name}")
            have |= {p.name for p in new}
            print(f"uploaded {len(new)} pilot shots")
    land = json.loads((HERE / "landfrac.json").read_text())
    todo = [s + [1 if land.get(f"{s[0]},{s[1]}", 0) > 0.5 else 0]
            for s in plan["shots"] if f"r{s[1]}_c{s[0]}.png" not in have]
    todo.sort(key=lambda s: (s[1], s[0]))
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    if limit:
        batches = batches[:limit]
    print(f"{len(plan['shots'])} shots planned, {len(have)} on the volume, "
          f"{len(todo)} to render in {len(batches)} batches")
    done = errors = 0
    t0 = time.time()
    for res in render_batch.map(batches, kwargs={"cfg": cfg}, return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            errors += 1
            print("batch failed:", repr(res)[:300])
            continue
        for line in res:
            if "retry" in line:
                print("retry:", json.dumps(line)[:300])
            elif "s" in line:
                if line.get("detail") is not None and line["detail"] < 11:
                    print("low detail accepted:", json.dumps(line))
                done += 1
            elif "error" in line:
                errors += 1
                print("error:", json.dumps(line)[:500])
        secs = [l["s"] for l in res if "s" in l]
        print(f"[{time.time() - t0:6.0f} s] {done} shots done, {errors} errors; "
              f"batch avg {sum(secs) / max(1, len(secs)):.0f} s/shot", flush=True)
