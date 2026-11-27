"""fp8 weights for the biggest models (MiniMax H3, H3 Image, LTX-2.3, LongCat Avatar): half the GPU memory for
their transformers and text encoders on the same GPU, and fp8 tensor-core matmuls for their Linear layers.

A converted ``nn.Linear`` keeps its weight as fp8 e4m3 with one float32 scale per output channel (``w_scale``, a second
parameter beside ``weight``): the format of unsloth's and Lightricks' fp8 files, made here from our own bf16 weights at
load time (on the GPU: a moment), so nothing new is downloaded. Layers that steer the whole model stay bf16: AdaLN
modulation, timestep and embedding projections, norms, output heads, and (in text encoders) the vision tower and LM
head. ``AISTUDIO_PRECISION_<FAMILY>=bf16`` keeps a family in bf16 (``wanted``).

Two ways to multiply (``AISTUDIO_FP8_MATMUL``):
- ``scaled`` (the default): fp8 tensor-core matmuls (``torch._scaled_mm``) with each activation row scaled to fp8 on
  the fly; needs an H100 or newer. Falls back to ``dequant`` for a layer it can't take.
- ``dequant``: the weight turned back into bf16 for each multiply: weight memory halves, speed doesn't change.

Because ``weight`` and ``w_scale`` are both parameters, PEFT LoRAs wrap a converted layer as any ``nn.Linear`` (its
``compute_dtype`` keeps their factors in bf16), and ``adapters.WeightPark`` swaps both pointers on a mode change.
Fine-tune weights read from disk are quantized the same way (``quantize_into``)."""

from __future__ import annotations

import os

FP8_MAX = 448.0  # the largest e4m3 value
MATMUL = os.environ.get("AISTUDIO_FP8_MATMUL", "scaled").lower()
# Substrings of module names that stay bf16 (lower case).
TRANSFORMER_KEEP = ("adaln", "time", "norm", "embed", "modulation", "lora_", "scale_shift", "proj_out", "proj_in")
# T5's feed-forward output (``…DenseReluDense.wo``) stays: T5 runs it in fp32 where it can, and casts its input to
# that layer's weight dtype, which would be fp8.
TEXT_KEEP = ("embed", "lm_head", "visual", "vision", "multi_modal_projector", "norm", "lora_", "densereludense.wo")
MIN_FEATURES = 256  # smaller layers aren't worth it

_linear = None
_capability_ok = None


def wanted(env_name: str) -> bool:
    """Whether a family runs in fp8: its ``AISTUDIO_PRECISION_*`` (H3, H3_IMAGE, LTX, LONGCAT) is ``fp8`` (the
    default) rather than ``bf16``."""
    return os.environ.get(env_name, "fp8").strip().lower() == "fp8"


def is_text_encoder(name: str) -> bool:
    return any(k in name.lower() for k in ("text_encoder", "conditioner", "encoder_model"))


def convert_parts(parts: dict, device: str = "cuda", skip: tuple[str, ...] = ("vae", "vocoder", "audio"),
                  label: str = "") -> dict:
    """Converts a pipeline's transformers and text encoders (name → module) in place, the text encoders with
    ``TEXT_KEEP``; VAEs, vocoders and audio encoders stay as they are. Logs and returns each part's sizes."""
    import time

    start = time.monotonic()
    out = {}
    for name, module in parts.items():
        if module is None or any(k in name.lower() for k in skip) or not hasattr(module, "named_modules"):
            continue
        out[name] = convert(module, TEXT_KEEP if is_text_encoder(name) else TRANSFORMER_KEEP, device)
    print(f"{label or 'fp8'}: {out} in {time.monotonic() - start:.1f} s", flush=True)
    return out


def quantize_weight(w):
    """(fp8 weight, float32 scale per output row) for a 2-D weight, on the weight's device."""
    import torch

    w = w.float()
    scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    return (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def is_fp8(t) -> bool:
    import torch

    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def scale_name(weight_name: str) -> str | None:
    """``…to_q.weight`` → ``…to_q.w_scale`` (None for a name that isn't a weight)."""
    return weight_name[: -len("weight")] + "w_scale" if weight_name.endswith("weight") else None


def quantize_into(weight, scale, tensor) -> None:
    """Copies a bf16/fp32 tensor read from disk into a converted layer's fp8 ``weight`` and its ``scale``, in place."""
    q, s = quantize_weight(tensor.to(weight.device))
    weight.data.copy_(q)
    scale.data.copy_(s)


def _scaled_ok() -> bool:
    global _capability_ok
    if _capability_ok is None:
        import torch

        _capability_ok = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    return _capability_ok


def linear_class():
    """The ``FP8Linear`` class (made on first use, so this module imports without torch on the front desk)."""
    global _linear
    if _linear is not None:
        return _linear
    import torch
    import torch.nn.functional as F
    from torch import nn

    class FP8Linear(nn.Linear):
        def forward(self, x):
            global MATMUL
            if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):  # a caller cast its input to our weight's dtype
                x = x.to(getattr(self, "compute_dtype", torch.bfloat16))
            if MATMUL == "scaled" and x.is_cuda and x.dtype == torch.bfloat16 and _scaled_ok():
                try:
                    return self._scaled(x)
                except RuntimeError as e:  # an unsupported shape or GPU: multiply the slow way from now on
                    print(f"fp8: fast matmul unavailable ({e}); dequantizing weights instead", flush=True)
                    MATMUL = "dequant"
            w = self.weight.to(x.dtype) * self.w_scale.to(x.dtype)[:, None]
            return F.linear(x, w, self.bias)

        def _scaled(self, x):
            shape = x.shape
            a = x.reshape(-1, shape[-1])
            if not a.is_contiguous():
                a = a.contiguous()
            sa = (a.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
            qa = (a / sa.to(a.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            sa = sa.to(a.dtype).float()  # the scale exactly as applied above
            bias = self.bias if self.bias is not None and self.bias.dtype == x.dtype else None
            y = torch._scaled_mm(qa, self.weight.t(), scale_a=sa, scale_b=self.w_scale[None, :], bias=bias,
                                 out_dtype=x.dtype)
            if self.bias is not None and bias is None:
                y = y + self.bias.to(y.dtype)
            return y.reshape(*shape[:-1], y.shape[-1])

    _linear = FP8Linear
    return _linear


def convert(model, keep: tuple[str, ...] = TRANSFORMER_KEEP, device: str = "cuda") -> dict:
    """Turns ``model``'s bf16 ``nn.Linear`` layers into fp8 in place, each moved to ``device`` first (so a model
    loaded on the CPU never needs its bf16 copy on the GPU); the rest of the model stays where it is. PEFT-wrapped
    layers are converted through their ``base_layer``. Returns sizes for the log."""
    import torch
    from torch import nn

    cls = linear_class()
    done = kept = before = after = 0
    for name, m in list(model.named_modules()):
        if type(m) is not nn.Linear:
            continue
        nbytes = m.weight.numel() * m.weight.element_size()
        small = min(m.in_features, m.out_features) < MIN_FEATURES or m.in_features % 16 or m.out_features % 16
        if m.weight.dtype != torch.bfloat16 or small or any(k in name.lower() for k in keep):
            kept += nbytes
            continue
        q, s = quantize_weight(m.weight.data.to(device))
        m.weight = nn.Parameter(q, requires_grad=False)
        m.w_scale = nn.Parameter(s, requires_grad=False)
        if m.bias is not None:
            m.bias.data = m.bias.data.to(device)
        m.__class__ = cls
        m.compute_dtype = torch.bfloat16  # what PEFT gives a LoRA on this layer (else fp8: rounded factors)
        done += 1
        before += nbytes
        after += q.numel() + s.numel() * 4
    if done:
        pin_dtype(model, torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"layers": done, "bf16_gb": round(before / 1e9, 2), "fp8_gb": round(after / 1e9, 2),
            "kept_gb": round(kept / 1e9, 2)}


def pin_dtype(model, dtype) -> None:
    """diffusers' and transformers' ``model.dtype`` is the first floating parameter's dtype, which may now be fp8 (and
    would cast the inputs to it): report the compute dtype instead."""
    cls = type(model)
    if getattr(cls, "_fp8_pinned", False):
        return
    model.__class__ = type(cls.__name__, (cls,), {"dtype": property(lambda self: dtype), "_fp8_pinned": True})
