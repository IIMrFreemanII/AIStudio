"""fp4 weights (experimental): NVFP4, the format Blackwell's fp4 tensor cores take. Each Linear weight is stored as
fp4 e2m1 values (two per byte), one e4m3 scale per 16 values along the input, and one float32 scale per tensor.

Turned on per family with ``AISTUDIO_PRECISION_<FAMILY>=fp4`` or ``fp4w`` (``fp8.precision``); only the transformers
change, text encoders stay fp8. Two ways to multiply:
- ``fp4``: each input row is quantized to NVFP4 the same way and multiplied on the fp4 tensor cores
  (``torch._scaled_mm`` with block scales; a B200, torch 2.8 or newer). Falls back to ``fp4w`` if that fails.
- ``fp4w``: the weight turned back into bf16 for each multiply (weight memory a quarter of bf16, bf16 inputs).

Not yet: fp4 copies on disk (fp4 parts load from the bf16 files and are converted at load), and LoRA mode switches or
fine-tunes swapped into fp4 layers (``adapters`` refuses them). PEFT LoRAs on top of an fp4 layer work as on fp8."""

from __future__ import annotations

import os

F8_MAX, F4_MAX = 448.0, 6.0
BLOCK = 16
_linear = None
_quantize = None
_tensor_cores = True  # until an fp4 matmul fails


def codes(x):
    """float32 values already within ±6 → e2m1 codes 0–15 (sign in bit 3), rounded to nearest, ties to even."""
    import torch

    a = x.abs()
    c = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5))
    return c | ((x < 0).to(torch.uint8) << 3)


def blocked(s):
    """(rows, cols) scales → the 128×4-tile layout cuBLAS reads for block-scaled matmuls, flat, zero-padded."""
    import torch

    rows, cols = s.shape
    r, c = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(r * 128, c * 4, dtype=s.dtype, device=s.device)
    p[:rows, :cols] = s
    return (p.view(r, 128, c, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
            .flatten())


def quantize(x):
    """(M, K) → (packed fp4 (M, K/2), e4m3 block scales (M, K/16), float32 tensor scale)."""
    import torch

    m, k = x.shape
    xf = x.float()
    ts = (xf.abs().amax() / (F8_MAX * F4_MAX)).clamp(min=1e-12)
    b = xf.view(m, k // BLOCK, BLOCK)
    bs = (b.abs().amax(dim=-1) / F4_MAX / ts).clamp(max=F8_MAX).to(torch.float8_e4m3fn)
    q = codes((b / (bs.float()[..., None] * ts)).clamp(-F4_MAX, F4_MAX)).view(m, k)
    return ((q[:, 1::2] << 4) | q[:, ::2]).view(torch.float4_e2m1fn_x2), bs, ts


def dequantize(packed, bs, ts, dtype):
    """The inverse of ``quantize``, in ``dtype``."""
    import torch

    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=packed.device)
    p = packed.view(torch.uint8)
    q = torch.stack([p & 15, p >> 4], dim=-1).view(p.shape[0], -1)
    m, k = q.shape
    return (grid[q.long()].view(m, k // BLOCK, BLOCK) * bs.float()[..., None] * ts).view(m, k).to(dtype)


def _input_quantizer():
    """``quantize`` plus the tile layout, compiled into few kernels on first use (else the plain steps)."""
    global _quantize
    if _quantize is None:
        import torch

        def plain(a):
            q, bs, ts = quantize(a)
            return q, blocked(bs), ts

        _quantize = plain
        if os.environ.get("AISTUDIO_FP8_COMPILE", "1") != "0":
            try:
                _quantize = torch.compile(plain, dynamic=True)
            except Exception as e:  # noqa: BLE001
                print(f"fp4: compiling the input quantization failed ({e}); unfused", flush=True)
    return _quantize


def linear_class():
    global _linear
    if _linear is not None:
        return _linear
    import torch
    import torch.nn.functional as F
    from torch import nn

    class FP4Linear(nn.Linear):
        """``weight`` packed fp4 (out, in/2); ``w_bs`` e4m3 block scales (out, in/16), ``w_blocked`` the same in tile
        layout, ``w_ts`` the tensor scale. ``fp4_mode`` is ``fp4`` (fp4 tensor cores) or ``fp4w`` (dequantized)."""

        def forward(self, x):
            if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                x = x.to(self.compute_dtype)
            global _tensor_cores
            if self.fp4_mode == "fp4" and _tensor_cores and x.is_cuda and x.dtype == torch.bfloat16:
                try:
                    return self._fp4(x)
                except RuntimeError as e:
                    print(f"fp4: fp4 matmul unavailable ({e}); dequantizing weights instead", flush=True)
                    _tensor_cores = False
            w = dequantize(self.weight, self.w_bs, self.w_ts, x.dtype)
            return F.linear(x, w, self.bias)

        def _fp4(self, x):
            shape = x.shape
            a = x.reshape(-1, shape[-1])
            if not a.is_contiguous():
                a = a.contiguous()
            qa, bsa, tsa = _input_quantizer()(a)
            y = torch._scaled_mm(qa, self.weight.t(), scale_a=bsa, scale_b=self.w_blocked, out_dtype=x.dtype)
            y = y * (tsa * self.w_ts).to(y.dtype)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
            return y.reshape(*shape[:-1], y.shape[-1])

    _linear = FP4Linear
    return _linear


def convert(model, keep: tuple[str, ...], device: str = "cuda", mode: str = "fp4") -> dict:
    """Turns ``model``'s bf16 ``nn.Linear`` layers into fp4 in place (``fp8.convert``'s rules for which stay)."""
    import torch
    from torch import nn

    from . import fp8

    cls = linear_class()
    done = kept = before = after = 0
    for name, m in list(model.named_modules()):
        if type(m) is not nn.Linear:
            continue
        nbytes = m.weight.numel() * m.weight.element_size()
        small = min(m.in_features, m.out_features) < fp8.MIN_FEATURES or m.in_features % 32 or m.out_features % 16
        if m.weight.dtype != torch.bfloat16 or small or any(k in name.lower() for k in keep):
            kept += nbytes
            continue
        q, bs, ts = quantize(m.weight.data.to(device))
        m.weight = nn.Parameter(q, requires_grad=False)
        m.w_bs = nn.Parameter(bs, requires_grad=False)
        m.w_blocked = nn.Parameter(blocked(bs), requires_grad=False)
        m.w_ts = nn.Parameter(ts.reshape(()), requires_grad=False)
        if m.bias is not None:
            m.bias.data = m.bias.data.to(device)
        m.__class__ = cls
        if getattr(m.__dict__.get("forward"), "__func__", None) is nn.Linear.forward:
            del m.__dict__["forward"]
        m.compute_dtype = torch.bfloat16
        m.fp4_mode = mode
        done += 1
        before += nbytes
        after += q.numel() + bs.numel() * 2 + 4
    if done:
        fp8.pin_dtype(model, torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"layers": done, "bf16_gb": round(before / 1e9, 2), "fp4_gb": round(after / 1e9, 2),
            "kept_gb": round(kept / 1e9, 2)}


def has_fp4(module) -> bool:
    return any(type(m).__name__ == "FP4Linear" for m in module.modules())
