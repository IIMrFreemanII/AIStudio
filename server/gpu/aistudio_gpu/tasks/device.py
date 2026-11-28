"""Where the models run: an NVIDIA GPU (Modal, a GPU box), Apple's GPU through Metal (MPS) on a Mac, or the CPU."""

from __future__ import annotations

import os


def pick_device():
    """(device, dtype). `AISTUDIO_DEVICE=cuda|mps|cpu` overrides the choice."""
    import torch

    forced = os.environ.get("AISTUDIO_DEVICE")
    if forced:
        device = forced
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # bf16 on GPUs; float32 on the CPU (bf16 matmuls there are slow, and the CPU is only for tests).
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return device, dtype


def generator(seed: int):
    """A CPU generator, so the same seed gives the same noise on every device."""
    import torch

    return torch.Generator(device="cpu").manual_seed(int(seed))


def fp8_cast(module, compute_dtype) -> None:
    """fp8 weight storage (e4m3, a plain cast with no scales), cast to `compute_dtype` per layer while computing.
    Norms, embeddings and LoRA weights stay as they are. The same call makes the fp8 copies on disk
    (`download.fp8_copy`), so loading one gives exactly the weights this would make from the original."""
    import torch
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype,
                                    skip_modules_pattern=tuple(DEFAULT_SKIP_MODULES_PATTERN) + ("lora_",))
    for m in module.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dtype == torch.float8_e4m3fn:
            # PEFT gives a LoRA its base layer's dtype before loading its values (fp8 would round them) unless the
            # layer names a compute dtype, as bitsandbytes' do.
            m.compute_dtype = compute_dtype


def fp8_storage_supported(device: str) -> bool:
    """fp8 weights (cast to bf16 per layer while computing) need CUDA; `AISTUDIO_FP8=force` allows the CPU too
    (tests), `AISTUDIO_FP8=0` turns it off."""
    setting = os.environ.get("AISTUDIO_FP8", "1")
    if setting == "0":
        return False
    return device == "cuda" or (setting == "force" and device == "cpu")
