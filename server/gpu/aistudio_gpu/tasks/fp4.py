"""fp4 weights (experimental): NVFP4, the format Blackwell's fp4 tensor cores take. Each Linear weight is stored as
fp4 e2m1 values (two per byte), one e4m3 scale per 16 values along the input, and one float32 scale per tensor.

Turned on per family with ``AISTUDIO_PRECISION_<FAMILY>=fp4`` or ``fp4w`` (``fp8.precision``); only the transformers
change, text encoders stay fp8. Two ways to multiply:
- ``fp4``: each input row is quantized to NVFP4 the same way (one Triton kernel, ``quantize_rows_fast``) and
  multiplied on the fp4 tensor cores with the scales and bias applied inside (torch's ``scaled_mm``; a B200).
  Falls back to ``fp4w`` if that fails.
- ``fp4w``: the weight turned back into bf16 for each multiply (weight memory a quarter of bf16, bf16 inputs).

Converted transformers are stored as fp4 copies (``<folder>-fp4``, ``fp8.save_copy``) for the next cold start, which
reads them instead of the bf16 files. Not yet: LoRA mode switches or fine-tunes swapped into fp4 layers (``adapters``
refuses them). PEFT LoRAs on top of an fp4 layer work as on fp8."""

from __future__ import annotations

import os

F8_MAX, F4_MAX = 448.0, 6.0
BLOCK = 16
COPY_FORMAT = 1  # fp4 copies on disk (fp8.save_copy): bump when the stored layout or quantization changes
_linear = None
_quantize = None
_tensor_cores = True  # until an fp4 matmul fails
_two_level = True  # scales and bias inside the matmul (torch's scaled_mm), until that fails
_RECIPE = _SWIZZLE = None
_V2 = None  # the same as ints, for torch._scaled_mm_v2 (F.scaled_mm minus its Python argument handling)


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
    # Multiplying by the reciprocals, not dividing by the constants: torch divides by a constant that way on the GPU
    # but not on the CPU, and the Triton kernel does the same, so every device gives the same bits.
    ts = (xf.abs().amax() * (1.0 / (F8_MAX * F4_MAX))).clamp(min=1e-12)
    b = xf.view(m, k // BLOCK, BLOCK)
    bs = (b.abs().amax(dim=-1) * (1.0 / F4_MAX) / ts).clamp(max=F8_MAX).to(torch.float8_e4m3fn)
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


_kernel = None


def _triton_kernel():
    """One pass over the input: each 16-value block's scale, the fp4 codes packed two per byte, and the scale written
    straight into its place in ``blocked``'s tile layout. Bit for bit what ``quantize`` + ``blocked`` give (no clamp
    is needed: codes saturate at 6, and a zero block's 0/0 codes to 0 either way)."""
    global _kernel
    if _kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def nvfp4_rows(x_ptr, q_ptr, s_ptr, amax_ptr, ts_ptr, M, K, stride_xm, col_tiles, ROWS_P, KB_P,
                       INV_TS: tl.constexpr, INV_F4: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_KB: tl.constexpr):
            rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
            kb = tl.program_id(1) * BLOCK_KB + tl.arange(0, BLOCK_KB)
            cols = tl.program_id(1) * BLOCK_KB * 16 + tl.arange(0, BLOCK_KB * 16)
            ok_rows = rows < M
            x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :],
                        mask=ok_rows[:, None] & (cols[None, :] < K), other=0.0).to(tl.float32)
            # The tensor scale from the input's largest |value|; the first program stores it for the matmul.
            ts = tl.maximum(tl.load(amax_ptr).to(tl.float32) * INV_TS, 1e-12)
            if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
                tl.store(ts_ptr, ts)
            xb = tl.reshape(x, (BLOCK_M, BLOCK_KB, 16))
            amax = tl.max(tl.abs(xb), axis=2)
            bs = tl.minimum(tl.math.div_rn(amax * INV_F4, ts), 448.0).to(tl.float8e4nv)
            v = tl.math.div_rn(xb, (bs.to(tl.float32) * ts)[:, :, None])
            a = tl.abs(v)
            c = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
                 + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
                 + (a > 5.0).to(tl.int32))
            c = c | tl.where(v < 0, 8, 0)
            lo, hi = tl.split(tl.reshape(c, (BLOCK_M, BLOCK_KB * 8, 2)))
            packed = (lo | (hi << 4)).to(tl.uint8)
            qcols = tl.program_id(1) * BLOCK_KB * 8 + tl.arange(0, BLOCK_KB * 8)
            tl.store(q_ptr + rows[:, None] * (K // 2) + qcols[None, :], packed,
                     mask=ok_rows[:, None] & (qcols[None, :] < K // 2))
            r = rows[:, None]
            k = kb[None, :]
            offset = ((r // 128) * col_tiles + k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
            # Padded rows and columns of the tile layout load as zeros, so their scales are written as zeros too.
            tl.store(s_ptr + offset, bs.to(tl.uint8, bitcast=True), mask=(r < ROWS_P) & (k < KB_P))

        _kernel = nvfp4_rows
    return _kernel


def quantize_rows_fast(a, block_m: int = 32, block_kb: int = 8):
    """``quantize`` + ``blocked`` for a contiguous (M, K) CUDA tensor in one Triton kernel (plus one reduction for
    the tensor scale, kept on the GPU): (packed fp4, scales in tile layout, tensor scale)."""
    import torch

    m, k = a.shape
    # Two launches in all: the largest |value| (exact in bf16), then the kernel, which derives the tensor scale from
    # it and fills the tile layout's padding (short layers' launch overhead is most of their cost).
    amax = torch.linalg.vector_norm(a, ord=float("inf"))
    ts = torch.empty(1, dtype=torch.float32, device=a.device)
    q = torch.empty(m, k // 2, dtype=torch.uint8, device=a.device)
    kb = k // BLOCK
    rows, col_tiles = -(-m // 128) * 128, -(-kb // 4)
    s = torch.empty(rows * col_tiles * 4, dtype=torch.uint8, device=a.device)
    grid = (rows // block_m, -(-col_tiles * 4 // block_kb))
    _triton_kernel()[grid](a, q, s, amax, ts, m, k, a.stride(0), col_tiles, rows, col_tiles * 4,
                           INV_TS=1.0 / (F8_MAX * F4_MAX), INV_F4=1.0 / F4_MAX, BLOCK_M=block_m, BLOCK_KB=block_kb)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn), ts


def _input_quantizer():
    """The input quantizer ``FP4Linear`` uses: the Triton kernel where it works, else ``quantize`` + ``blocked``
    compiled (``AISTUDIO_FP8_COMPILE=0``: the plain steps). ``AISTUDIO_FP4_TRITON=0`` skips the kernel."""
    global _quantize
    if _quantize is None:
        import torch

        def plain(a):
            q, bs, ts = quantize(a)
            return q, blocked(bs).view(torch.float8_e4m3fn), ts.reshape(1)

        _quantize = plain
        if os.environ.get("AISTUDIO_FP8_COMPILE", "1") != "0":
            try:
                _quantize = torch.compile(plain, dynamic=True)
            except Exception as e:  # noqa: BLE001
                print(f"fp4: compiling the input quantization failed ({e}); unfused", flush=True)
        if os.environ.get("AISTUDIO_FP4_TRITON", "1") != "0" and torch.cuda.is_available():
            fallback = _quantize
            try:
                quantize_rows_fast(torch.ones(128, 64, dtype=torch.bfloat16, device="cuda"))
                _quantize = quantize_rows_fast
            except Exception as e:  # noqa: BLE001 (no Triton, an old GPU, …)
                print(f"fp4: Triton input kernel unavailable ({e}); compiled steps instead", flush=True)
                _quantize = fallback
    return _quantize


def linear_class():
    global _linear
    if _linear is not None:
        return _linear
    import torch
    import torch.nn.functional as F
    from torch import nn

    global _RECIPE, _SWIZZLE, _two_level
    try:  # NVFP4 as torch's scaled_mm names it: e4m3 scales per 16 values in tile layout, then one per tensor
        _RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        _SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        global _V2
        if hasattr(torch, "_scaled_mm_v2") and hasattr(F, "_enum_list_as_int_list"):
            _V2 = (F._enum_list_as_int_list(_RECIPE), F._enum_list_as_int_list(_SWIZZLE))
    except AttributeError:  # an older torch: scale after the matmul
        _two_level = False

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
            global _two_level
            bias = self.bias if self.bias is not None and self.bias.dtype == x.dtype else None
            if _two_level:
                # Block and tensor scales and the bias applied inside the matmul: two passes over the output fewer
                # (B200, LTX's layers at 3,840 tokens: 0.66 ms against 1.11 with them applied after).
                try:
                    if _V2 is not None:
                        y = torch._scaled_mm_v2(qa, self.weight.t(), [bsa, tsa], _V2[0], _V2[1],
                                                [self.w_blocked, self.w_ts], _V2[0], _V2[1], bias, x.dtype, [], False)
                    else:
                        y = F.scaled_mm(qa, self.weight.t(), scale_a=[bsa, tsa], scale_recipe_a=_RECIPE,
                                        scale_b=[self.w_blocked, self.w_ts], scale_recipe_b=_RECIPE,
                                        swizzle_a=_SWIZZLE, swizzle_b=_SWIZZLE, bias=bias, output_dtype=x.dtype)
                except (RuntimeError, TypeError, AttributeError) as e:
                    print(f"fp4: two-level scaled matmul unavailable ({e}); scaling after it", flush=True)
                    _two_level = False
            if not _two_level:
                y = torch._scaled_mm(qa, self.weight.t(), scale_a=bsa, scale_b=self.w_blocked, out_dtype=x.dtype)
                y = y * (tsa * self.w_ts).to(y.dtype)
                if bias is not None:
                    y = y + bias
            if self.bias is not None and bias is None:
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
        m.w_ts = nn.Parameter(ts.reshape(1), requires_grad=False)
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
