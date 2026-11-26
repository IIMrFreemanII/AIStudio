"""Keeping a model's parts on the GPU together.

The user's rule: every part of a model (transformer, text encoder, VAEs) stays in GPU memory at once, on a GPU big
enough for it, instead of taking turns per job through CPU offload. Swapping 60 GB components through host memory cost
MiniMax H3 Image ≈ 80 s and $0.12 a picture on an H200 against ≈ 5 s and $0.011 on a B200 with everything resident.
Offload stays only as the fallback for a smaller card or a job too big to fit beside the weights.
"""

from __future__ import annotations


def weight_bytes(module) -> int:
    """Bytes of a torch module's parameters and buffers (0 for tokenizers, processors and schedulers)."""
    if not hasattr(module, "parameters") or not hasattr(module, "buffers"):
        return 0
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


def fits(weights: int, gpu_total: int, margin_gib: float) -> bool:
    """Whether `weights` fit on a GPU of `gpu_total` bytes with `margin_gib` left for activations."""
    return gpu_total - weights >= margin_gib * 2**30


def modules_of(components: dict) -> dict:
    """The torch modules among a pipeline's components (name → module)."""
    return {n: m for n, m in components.items() if hasattr(m, "parameters") and hasattr(m, "to")}
