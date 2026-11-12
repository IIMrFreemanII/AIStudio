"""Which GPUs a model fits on, and the cheapest.

Per GPU it needs its share of the weights (tensor parallel splits them, expert parallel too; n-gram embedding
tables that vLLM keeps in CPU memory, ``Facts.cpu_bytes``, don't count), a fixed overhead
(CUDA context, activations, CUDA graphs, a vision encoder) and the KV cache for the context. vLLM is given 90% of the
memory; what's left after the weights and overhead is KV cache, which sets the longest context that fits.

- KV per token: 2 (K and V) × the layers with full attention × KV heads × head dim × bytes, split across GPUs by
  KV head (heads are copied when there are more GPUs than KV heads). Linear-attention layers (Gated DeltaNet,
  Mamba) keep a small fixed state, counted in the overhead. MLA models (DeepSeek) cache one latent per layer, which
  every GPU keeps whole.
- Options: L40S (at most 2: no NVLink), A100 80 GB, H100, H200 and B200, on 1, 2, 4 or 8 GPUs (the attention heads
  must split evenly). FP8 weights need Ada or newer to run at full speed; FP4 needs Hopper (MXFP4) or Blackwell
  (NVFP4).
- GGUF on llama.cpp splits the model by layers instead (no head constraint); its KV cache is split the same way.
- The pick: the cheapest option whose longest context reaches the one asked for; failing that, the longest.
  Options with a caveat (FP8 on A100) are picked only when nothing else fits.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from ..tasks import GPU_PRICES
from .facts import Facts

GIB = 1 << 30
# Usable memory per GPU as the driver reports it (MiB).
GPUS: dict[str, int] = {"L40S": 46068, "A100-80GB": 81920, "H100": 81559, "H200": 143771, "B200": 183359}
COUNTS = (1, 2, 4, 8)
MAX_COUNT = {"L40S": 2}
UTILIZATION = 0.90
OVERHEAD = 4 * GIB
VISION_OVERHEAD = 2 * GIB
MIN_CONTEXT = 8192
DEFAULT_CONTEXT = 131072
MAX_CONTEXT = 1_048_576  # 1M tokens: the most the app offers, whatever the model's native window
ADA_OR_NEWER = {"L40S", "H100", "H200", "B200"}
HOPPER_OR_NEWER = {"H100", "H200", "B200"}


def max_gpus() -> int:
    return int(os.environ.get("AISTUDIO_LLM_MAX_GPUS", "8"))


@dataclass
class Option:
    gpu: str
    count: int
    fits: bool
    usd_per_hour: float
    max_context: int
    weights_per_gpu_gb: float
    kv_per_token_kb: float
    note: str = ""
    kv_tokens: int = 0  # tokens of KV cache the memory left holds (not capped at the context window); 0: no fit

    def public(self) -> dict:
        d = asdict(self)
        return {"gpu": d["gpu"], "count": d["count"], "fits": d["fits"], "usdPerHour": round(d["usd_per_hour"], 2),
                "maxContext": d["max_context"], "weightsPerGPUGB": round(d["weights_per_gpu_gb"], 1),
                "kvPerTokenKB": round(d["kv_per_token_kb"], 1), "note": d["note"], "kvTokens": d["kv_tokens"]}


def kv_bytes_per_token(f: Facts, count: int, kv_dtype_bytes: int = 2) -> float:
    """KV cache bytes per token on one of ``count`` GPUs."""
    if f.engine == "llamacpp":  # layers split across GPUs, and their caches with them
        per_layer = (f.kv_lora_rank + f.rope_head_dim) if f.kv_lora_rank else 2 * f.kv_heads * f.head_dim
        return f.attention_layers * per_layer * kv_dtype_bytes / count
    if f.kv_lora_rank:
        return f.attention_layers * (f.kv_lora_rank + f.rope_head_dim) * kv_dtype_bytes
    heads_per_gpu = max(1, -(-f.kv_heads // count))  # ceil: copied when count > KV heads
    return 2 * f.attention_layers * heads_per_gpu * f.head_dim * kv_dtype_bytes


def unsupported(f: Facts, gpu: str) -> str | None:
    method, bits = (f.quant.get("method") or ""), f.quant.get("bits")
    if "nvfp4" in method or (method in ("modelopt", "modelopt_fp4") and bits == 4):
        return None if gpu == "B200" else "NVFP4 needs B200"
    if "mxfp4" in method and gpu not in HOPPER_OR_NEWER:
        return "MXFP4 needs H100 or newer"
    return None


def options(f: Facts, kv_dtype_bytes: int = 2, replicated_bytes: int = 0) -> list[Option]:
    out = []
    for gpu, mib in GPUS.items():
        price = GPU_PRICES.get(gpu)
        if price is None:
            continue
        for n in COUNTS:
            if n > max_gpus() or n > MAX_COUNT.get(gpu, 8):
                continue
            if n > 1 and f.engine != "llamacpp" and f.attention_heads and f.attention_heads % n:
                continue
            why = unsupported(f, gpu)
            on_gpus = f.weight_bytes - (f.cpu_bytes if f.engine != "llamacpp" else 0)  # less what stays in CPU memory
            weights = on_gpus / n + replicated_bytes
            budget = mib * (1 << 20) * UTILIZATION - weights - OVERHEAD - (VISION_OVERHEAD if f.vision else 0)
            per_token = kv_bytes_per_token(f, n, kv_dtype_bytes) or 1
            kv_tokens = int(budget / per_token) if budget > 0 else 0
            context = min(f.native_context, MAX_CONTEXT, kv_tokens)
            note = why or ""
            if not note and f.quant.get("method") == "fp8" and gpu not in ADA_OR_NEWER:
                note = "FP8 runs slower on A100"
            out.append(Option(gpu, n, why is None and context >= min(MIN_CONTEXT, f.native_context),
                              price * n * 3600, context, weights / 1e9, per_token / 1024, note, kv_tokens))
    return out


def recommend(opts: list[Option], context: int) -> Option | None:
    fitting = [o for o in opts if o.fits]
    if not fitting:
        return None
    # Options with a caveat ("FP8 runs slower on A100") stay in the list, but aren't picked when another fits.
    fitting = [o for o in fitting if not o.note] or fitting
    enough = [o for o in fitting if o.max_context >= context]
    if enough:
        return min(enough, key=lambda o: (o.usd_per_hour, o.count))
    return max(fitting, key=lambda o: (o.max_context, -o.usd_per_hour))


def default_context(f: Facts) -> int:
    return min(f.native_context, DEFAULT_CONTEXT)


def plan(f: Facts, context: int | None = None, kv_dtype_bytes: int = 2, replicated_bytes: int = 0) -> dict:
    """The options (cheapest first, those that fit first) and the recommended one, as the API returns them."""
    want = min(context or default_context(f), f.native_context, MAX_CONTEXT)
    opts = options(f, kv_dtype_bytes, replicated_bytes)
    best = recommend(opts, want)
    ranked = sorted(opts, key=lambda o: (not o.fits, o.usd_per_hour, o.count))
    return {"context": want, "nativeContext": min(f.native_context, MAX_CONTEXT), "recommended": best.public() if best else None,
            "options": [o.public() for o in ranked]}
