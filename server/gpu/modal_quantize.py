"""Makes the NVFP4 version of an installed LLM on the deployed app (llm/quantize.py): a one-time paid GPU job that
installs ``<id>-nvfp4`` next to the original. ``--plan`` only checks it on CPUs (a few cents): the recipe, and which
layers go to NVFP4, fp8 or stay 16-bit.

    python server/gpu/modal_quantize.py qwen3.8-27b-abliterated --plan
    python server/gpu/modal_quantize.py qwen3.8-27b-abliterated [--gpu H100] [--samples 256 --seq-len 2048]

The model in bf16 has to fit in the container's host memory (asked for from the plan's parameter count); llm-compressor
moves one layer at a time to the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import modal

APP = "aistudio-gpu"
MAX_MEMORY = 1_000_000  # MiB: about Modal's largest container


def memory_for(params: int) -> tuple[int, int]:
    """(request, limit) MiB: the bf16 model, the quantized copy being written and room for calibration."""
    need = int(params * 2 * 1.35 / (1 << 20)) + 32768
    return min(need, MAX_MEMORY), min(max(need * 5 // 4, 131072), MAX_MEMORY)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--plan", action="store_true", help="only check the recipe on CPUs")
    p.add_argument("--gpu", default="H100")
    p.add_argument("--samples", type=int, default=0, help="calibration samples (default: the recipe's)")
    p.add_argument("--seq-len", type=int, default=0, help="tokens per calibration sample (default: the recipe's)")
    a = p.parse_args()
    plan = modal.Function.from_name(APP, "quantize_llm_plan").remote(a.model)
    print(json.dumps(plan, indent=1))
    if a.plan or plan.get("error"):
        return 1 if plan.get("error") else 0
    memory = memory_for(int(plan["params"]))
    print(f"quantize: {a.model} on {a.gpu}, host memory {memory[0] // 1024}–{memory[1] // 1024} GiB", flush=True)
    start = time.monotonic()
    fn = modal.Function.from_name(APP, "quantize_llm").with_options(gpu=a.gpu, memory=memory)
    result = fn.remote(a.model, a.samples, a.seq_len)
    print(json.dumps(result, indent=1))
    print(f"quantize: done in {time.monotonic() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
