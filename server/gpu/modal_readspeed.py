"""How long each model takes to read from the models Volume: the disk part of a GPU container's cold start, which is
billed at the GPU's price. Runs on CPU containers only (a few cents), one fresh container per model so nothing is
cached, one at a time so they don't share the Volume's bandwidth.

    modal run server/gpu/modal_readspeed.py                       # every installed model folder
    modal run server/gpu/modal_readspeed.py --models wan22-t2v,ltx23

Prints, per model folder: weight GB, read seconds, GB/s, the GPU of the task that loads it, and what the read costs on
that GPU per cold start. A CPU container's network may differ from a GPU container's, so compare with a real cold
start (``modal_check.py``'s ``loadSeconds``) before trusting the dollars.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = Path("/models")
WEIGHTS = {".safetensors", ".pth", ".pt", ".bin", ".ckpt", ".gguf"}
CHUNK = 64 * 2**20

app = modal.App("aistudio-gpu-readspeed")
volume = modal.Volume.from_name("aistudio-gpu-models")
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={MODELS: volume}, single_use_containers=True, max_containers=1, timeout=60 * 60)
def installed() -> list[str]:
    return sorted(p.name for p in MODELS.iterdir() if p.is_dir() and (p / ".complete").exists())


@app.function(image=image, volumes={MODELS: volume}, cpu=2, memory=2048, single_use_containers=True, max_containers=1,
              timeout=60 * 60)
def read(name: str) -> dict:
    """Reads every weight file of one model folder once (a fresh container: single_use_containers=True)."""
    root = MODELS / name
    files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix in WEIGHTS)
    total, start = 0, time.monotonic()
    buf = bytearray(CHUNK)
    for f in files:
        with open(f, "rb", buffering=0) as fh:
            while n := fh.readinto(buf):
                total += n
    return {"name": name, "files": len(files), "bytes": total, "seconds": time.monotonic() - start}


def gpu_of(name: str) -> tuple[str, float | None]:
    """The GPU (and USD per second) of the costliest task that loads this model folder."""
    from aistudio_gpu.tasks import TASKS

    best: tuple[str, float | None] = ("-", None)
    for spec in TASKS.values():
        if name in spec.models and not spec.cpu_only:
            price = spec.price_per_second
            if price is not None and (best[1] is None or price > best[1]):
                best = (spec.gpu, price)
    return best


@app.local_entrypoint()
def main(models: str = ""):
    names = [m.strip() for m in models.split(",") if m.strip()] or installed.remote()
    rows = []
    print(f"{'model':28} {'GB':>7} {'read s':>8} {'GB/s':>6} {'GPU':>6} {'$ / cold start':>15}")
    for name in names:
        r = read.remote(name)
        gb, secs = r["bytes"] / 1e9, r["seconds"]
        gpu, price = gpu_of(name)
        cost = f"{secs * price:.3f}" if price else "-"
        print(f"{name:28} {gb:7.1f} {secs:8.1f} {gb / max(secs, 1e-6):6.2f} {gpu:>6} {cost:>15}", flush=True)
        rows.append((name, gb, secs, gpu, price))
    total_gb = sum(r[1] for r in rows)
    total_s = sum(r[2] for r in rows)
    print(f"{'all':28} {total_gb:7.1f} {total_s:8.1f} {total_gb / max(total_s, 1e-6):6.2f}")
