"""Runs one job on the deployed GPU workers straight through the Modal SDK (your `modal setup` login, no proxy token):
prints the status lines, steps and previews as they arrive with timestamps, saves the result, and reports the
measured times and the cost. For checking a deployment and measuring cold and warm starts.

    python server/gpu/modal_check.py wan22-t2v [--prompt "…"] [--width 832 --height 480 --length 81] [--full]
    python server/gpu/modal_check.py wan22-i2v --image start.png
    python server/gpu/modal_check.py flux1-dev --width 1024 --height 1024
    python server/gpu/modal_check.py qwen-image --prompt "a sign reading OPEN" [--full] [--set model=<fine-tune id>]
    python server/gpu/modal_check.py stable-audio-3 --prompt "rain on a tin roof" --set seconds=20
    python server/gpu/modal_check.py ace-step-xl --prompt "indie pop, female vocals" --set lyrics=@lyrics.txt
    python server/gpu/modal_check.py qwen3-tts --prompt "Hello!" --set voice=designed --set description="…"
    python server/gpu/modal_check.py qwen3-tts --prompt "Hello!" --input reference=voice.wav

``--prompt`` fills the task's text (prompt, tags or text); ``--set name=value`` any other parameter (``@file`` reads
the value from a file); ``--input name=path`` uploads a file input.
"""

from __future__ import annotations

import argparse
import io
import queue
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal  # noqa: E402

from aistudio_gpu.tasks import GPU_PRICES, TASKS  # noqa: E402

CLASSES = {"wan22-t2v": "WanT2V", "wan22-i2v": "WanI2V", "flux1-dev": "Flux", "qwen-image": "QwenImage",
           "ace-step-xl": "Music",
           "stable-audio-3": "Sound", "qwen3-tts": "Speech"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=list(CLASSES))
    p.add_argument("--prompt", default="A red fox trotting through fresh snow at sunrise, its breath steaming")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--length", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--full", action="store_true", help="Wan / Qwen-Image without the Lightning LoRAs")
    p.add_argument("--image", type=Path, help="wan22-i2v: the start image")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="any task parameter")
    p.add_argument("--input", action="append", default=[], metavar="NAME=PATH", help="a file input")
    p.add_argument("--via-volume", action="store_true",
                   help="send file inputs through the uploads Volume, as the deployed front desk does")
    p.add_argument("--out", type=Path, default=Path("."))
    a = p.parse_args()

    spec = TASKS[a.task]
    text = next((k for k in ("prompt", "tags", "text") if k in spec.params), "prompt")
    params = {text: a.prompt, "seed": a.seed}
    for k in ("width", "height", "length", "steps"):
        if getattr(a, k) is not None and k in spec.params:
            params[k] = getattr(a, k)
    if a.full:
        params["fast"] = False
    for item in a.set:
        k, _, v = item.partition("=")
        params[k] = Path(v[1:]).read_text() if v.startswith("@") else v
    params = spec.normalize(params)
    inputs = {}
    if "image" in spec.inputs:
        if not a.image:
            p.error("this task needs --image")
        inputs["image"] = (a.image.name, a.image.read_bytes())
    for item in a.input:
        k, _, v = item.partition("=")
        inputs[k] = (Path(v).name, Path(v).read_bytes())

    if a.via_volume and inputs:
        import hashlib

        volume = modal.Volume.from_name("aistudio-gpu-uploads", create_if_missing=True)
        with volume.batch_upload(force=True) as batch:
            for k, (fname, data) in list(inputs.items()):
                ext = Path(fname).suffix.lower()
                name = hashlib.sha256(data).hexdigest() + (ext if 1 < len(ext) <= 9 and ext[1:].isalnum() else "")
                batch.put_file(io.BytesIO(data), f"/{name}")
                inputs[k] = (name, None)

    events = modal.Queue.from_name("aistudio-gpu-events", create_if_missing=True)
    worker = modal.Cls.from_name("aistudio-gpu", CLASSES[a.task])()
    job = uuid.uuid4().hex
    start = time.monotonic()

    def say(text: str) -> None:
        print(f"[{time.monotonic() - start:7.1f}s] {text}", flush=True)

    say(f"{a.task} {params}")
    call = worker.run.spawn(job, a.task, params, inputs)
    first_event = None
    previews = 0
    while True:
        try:
            items = events.get_many(64, partition=job, timeout=1.0)
        except queue.Empty:  # Modal raises on a timeout
            items = []
        for kind, step, total, payload in items:
            first_event = first_event or time.monotonic() - start
            if kind == "status":
                say(f"status: {payload}")
            elif kind == "progress":
                say(f"step {step}/{total}")
            elif kind == "preview":
                previews += 1
        try:
            result = call.get(timeout=0)
            break
        except TimeoutError:
            continue
    total = time.monotonic() - start
    events.clear(partition=job)
    if "error" in result:
        say(f"FAILED: {result['error']}")
        return 1
    out = a.out / f"{a.task}-{job[:8]}{result['suffix']}"
    out.write_bytes(result["data"])
    price = GPU_PRICES.get(spec.gpu, 0)
    gpu, load = result["gpuSeconds"], result["loadSeconds"]
    say(f"saved {out} ({len(result['data']) / 1e6:.1f} MB), {previews} previews, {result.get('extra')}")
    print(f"\n  total wall time      {total:7.1f} s   (first worker event after {first_event or 0:.1f} s)")
    print(f"  job on the GPU       {gpu:7.1f} s")
    print(f"  model loading        {load:7.1f} s   ({'cold container' if load else 'warm container'})")
    print(f"  GPU                  {spec.gpu}, ${price * 3600:.2f}/h")
    print(f"  cost (load + job)    ${(gpu + load) * price:.4f}   (+ the idle window until scale-down)")
    print(f"  cost incl. start-up  ${(first_event or 0) * price + gpu * price:.4f}   (wall time to first event × price)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
