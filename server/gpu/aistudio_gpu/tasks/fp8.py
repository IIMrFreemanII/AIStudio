"""fp8 weights for the biggest models (MiniMax H3, H3 Image, LTX-2.3, LongCat Avatar): half the GPU memory for
their transformers and text encoders on the same GPU, and fp8 tensor-core matmuls for their Linear layers.

A converted ``nn.Linear`` keeps its weight as fp8 e4m3 with one float32 scale per output channel (``w_scale``, a second
parameter beside ``weight``): the format of unsloth's and Lightricks' fp8 files, made here from our own bf16 weights at
load time (on the GPU: a moment), so nothing new is downloaded. Layers that steer the whole model stay bf16: AdaLN
modulation, timestep and embedding projections, norms, output heads, and (in text encoders) the vision tower and LM
head. ``AISTUDIO_PRECISION_<FAMILY>=bf16`` keeps a family in bf16 (``wanted``).

Two ways to multiply (``AISTUDIO_FP8_MATMUL``):
- ``scaled`` (the default): fp8 tensor-core matmuls (``torch._scaled_mm``) with each activation row scaled to fp8 on
  the fly; needs an H100 or newer and torch 2.7 or newer (``_scaled_ok``). Falls back to ``dequant`` otherwise.
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
_quantize = None  # the row quantizer FP8Linear uses: compiled (fused) on first use, else the plain one


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
    """fp8 tensor cores (an H100 or newer) and a torch whose row-scaled ``_scaled_mm`` is fast: on torch 2.6
    (LongCat's image) it took about twice as long as the bf16 matmul on an H100, so the weight is dequantized there
    (bf16 speed, fp8 memory); on torch 2.15 (B200) it beats bf16."""
    global _capability_ok
    if _capability_ok is None:
        import torch

        version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        _capability_ok = (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
                          and version >= (2, 7))
        if torch.cuda.is_available() and not _capability_ok:
            print(f"fp8: torch {torch.__version__} on {torch.cuda.get_device_name()}: dequantized bf16 matmuls",
                  flush=True)
    return _capability_ok


def linear_class():
    """The ``FP8Linear`` class (made on first use, so this module imports without torch on the front desk)."""
    global _linear
    if _linear is not None:
        return _linear
    import torch
    import torch.nn.functional as F
    from torch import nn

    def quantize_rows(a):
        """Each row of ``a`` scaled into fp8: (fp8 rows, float32 scale per row, as applied)."""
        sa = (a.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
        qa = (a / sa.to(a.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return qa, sa.to(a.dtype).float()

    def quantize(a):
        # Unfused, these five element-wise steps cost more than the fp8 matmul saves at LTX's and H3's sizes (B200,
        # torch 2.15: 1.11 ms against bf16's 0.79 for LTX's layers at 3,840 tokens); one compiled kernel makes it
        # 0.70. ``AISTUDIO_FP8_COMPILE=0`` keeps the plain steps; a compile that fails falls back to them.
        global _quantize
        if _quantize is None:
            _quantize = quantize_rows
            if os.environ.get("AISTUDIO_FP8_COMPILE", "1") != "0":
                compiled = torch.compile(quantize_rows, dynamic=True)
                try:
                    out = compiled(a)
                    _quantize = compiled
                    return out
                except Exception as e:  # noqa: BLE001 (no Triton, an old compiler, …)
                    print(f"fp8: compiling the input quantization failed ({e}); unfused", flush=True)
        return _quantize(a)

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
            qa, sa = quantize(a)
            bias = self.bias if self.bias is not None and self.bias.dtype == x.dtype else None
            y = torch._scaled_mm(qa, self.weight.t(), scale_a=sa, scale_b=self.w_scale[None, :], bias=bias,
                                 out_dtype=x.dtype)
            if self.bias is not None and bias is None:
                y = y + self.bias.to(y.dtype)
            return y.reshape(*shape[:-1], y.shape[-1])

    FP8Linear.quantize = staticmethod(quantize)
    FP8Linear.quantize_rows = staticmethod(quantize_rows)
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
        # A library that hooked the layer's forward and unhooked it again (LongCat's LoRAs) may leave the plain
        # Linear forward behind as an instance attribute: it would run bf16 math on the fp8 weight.
        if getattr(m.__dict__.get("forward"), "__func__", None) is nn.Linear.forward:
            del m.__dict__["forward"]
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


# MARK: fp8 copies on disk

# Converted modules stored beside their bf16 folder (``<folder>-fp8``), so a cold start reads half the bytes: the
# module's own tensors (fp8 weights with their scales, the rest as they were), its config, the buffers it doesn't save
# itself, and a marker written last. Made by the worker from the module it just converted, so loading one gives exactly
# what converting at load gives. A marker from another format or keep list is stale: the copy is made again.
COPY_FORMAT = 1
COPY_MARKER = ".fp8-copy.json"
SHARD_BYTES = 5 * 2**30


def copy_dir(src) -> "Path":
    from pathlib import Path

    src = Path(src)
    return src.with_name(src.name + "-fp8")


def _copy_meta(text_encoder: bool) -> dict:
    keep = TEXT_KEEP if text_encoder else TRANSFORMER_KEEP
    return {"format": COPY_FORMAT, "keep": list(keep), "min_features": MIN_FEATURES}


def copy_ready(src, text_encoder: bool = False) -> bool:
    """Whether ``src`` has a current fp8 copy (this format and keep list)."""
    import json

    marker = copy_dir(src) / COPY_MARKER
    try:
        meta = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return all(meta.get(k) == v for k, v in _copy_meta(text_encoder).items())


def _base_class(module):
    """The model's own class (``pin_dtype`` puts a subclass of the same name in front of it)."""
    for cls in type(module).__mro__:
        if "_fp8_pinned" not in cls.__dict__:
            return cls
    return type(module)


def save_copy(module, src, text_encoder: bool = False) -> bool:
    """Writes ``module`` (converted by ``convert``) as ``src``'s fp8 copy, unless a current one is there. PEFT's
    wrappers are unwrapped and LoRA factors left out (LoRAs load on top as usual); a tensor shared by two names is
    stored once. Returns whether it wrote one."""
    import json
    import shutil
    import time
    import uuid

    import torch
    from safetensors.torch import save_file

    if copy_ready(src, text_encoder):
        return False
    start = time.monotonic()
    out = copy_dir(src)
    tmp = out.with_name(f"{out.name}.partial-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        state = {}
        for k, v in module.state_dict(keep_vars=False).items():
            if ".lora_" in k:
                continue
            state[k.replace(".base_layer.", ".")] = v
        seen, aliases, tensors = {}, {}, {}
        for k, v in state.items():
            key = (v.device, v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), v.dtype)
            if key in seen:
                aliases[k] = seen[key]
            else:
                seen[key] = k
                tensors[k] = v
        shard, size, n = {}, 0, 0
        for k, v in tensors.items():
            shard[k] = v.detach().to("cpu").contiguous()
            size += v.numel() * v.element_size()
            if size >= SHARD_BYTES:
                save_file(shard, str(tmp / f"model-{n:05d}.safetensors"))
                shard, size, n = {}, 0, n + 1
        if shard:
            save_file(shard, str(tmp / f"model-{n:05d}.safetensors"))
        saved = set(state) | {k.replace(".base_layer.", ".") for k in module.state_dict()}
        extra = {}
        for k, b in module.named_buffers():
            k = k.replace(".base_layer.", ".")
            if k not in saved and ".lora_" not in k and b.dtype in (torch.float32, torch.bfloat16, torch.float16,
                                                                      torch.int64, torch.int32, torch.bool):
                extra[k] = b.detach().to("cpu").contiguous()
        if extra:
            save_file(extra, str(tmp / "buffers.safetensors"))
        if hasattr(module, "save_config"):  # diffusers
            module.save_config(str(tmp))
        elif hasattr(getattr(module, "config", None), "save_pretrained"):  # transformers
            module.config.save_pretrained(str(tmp))
        cls = _base_class(module)
        meta = {**_copy_meta(text_encoder), "class": f"{cls.__module__}:{cls.__qualname__}", "aliases": aliases}
        (tmp / COPY_MARKER).write_text(json.dumps(meta))
        if out.exists():  # stale (another format), or another container finished first
            if copy_ready(src, text_encoder):
                shutil.rmtree(tmp, ignore_errors=True)
                return False
            shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"fp8: stored {out.name} ({gb:.1f} GB) in {time.monotonic() - start:.0f} s", flush=True)
    return True


def _import(path: str):
    import importlib

    module, _, qualname = path.partition(":")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _skeleton(cls, folder, init_kwargs: dict):
    """The model built with its parameters on the meta device (its buffers real), from the copy's config."""
    from accelerate import init_empty_weights

    if hasattr(cls, "load_config") and hasattr(cls, "from_config"):  # diffusers
        config = cls.load_config(str(folder))
        with init_empty_weights():
            return cls.from_config(config, **init_kwargs)
    config = cls.config_class.from_pretrained(str(folder))  # transformers
    with init_empty_weights():
        return cls._from_config(config, **init_kwargs)


def load_copy(src, device: str = "cuda", init_kwargs: dict | None = None, threads: int = 8):
    """``src``'s fp8 copy as a ready module on ``device``: the shards read in parallel straight onto it, the fp8
    layers made ``FP8Linear`` before the tensors are assigned, the saved buffers put back."""
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from safetensors.torch import load_file
    from torch import nn

    start = time.monotonic()
    folder = copy_dir(src)
    meta = json.loads((folder / COPY_MARKER).read_text())
    model = _skeleton(_import(meta["class"]), folder, init_kwargs or {})
    files = sorted(folder.glob("model-*.safetensors"))
    with ThreadPoolExecutor(max(1, threads)) as pool:
        parts = list(pool.map(lambda f: load_file(str(f), device=device), files))
    state = {k: v for p in parts for k, v in p.items()}
    for alias, name in meta.get("aliases", {}).items():
        state[alias] = state[name]
    cls = linear_class()
    for name in [k[: -len(".w_scale")] for k in state if k.endswith(".w_scale")]:
        m = model.get_submodule(name)
        m.__class__ = cls
        m.w_scale = nn.Parameter(torch.empty(m.out_features, device="meta"), requires_grad=False)
        m.compute_dtype = torch.bfloat16
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"fp8 copy {folder.name}: {len(missing)} missing (e.g. {missing[:2]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:2]})")
    buffers = folder / "buffers.safetensors"
    if buffers.exists():
        for k, v in load_file(str(buffers), device=device).items():
            owner, _, leaf = k.rpartition(".")
            model.get_submodule(owner)._buffers[leaf] = v
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pin_dtype(model, torch.bfloat16)
    print(f"fp8: loaded {folder.name} in {time.monotonic() - start:.1f} s", flush=True)
    return model


def copies_usable(sources: dict, env_name: str, margin_gib: float = 40) -> bool:
    """Whether a worker loads its parts (name → bf16 folder) from their fp8 copies: fp8 wanted, every copy current,
    and a CUDA GPU with room for them beside ``margin_gib`` (else the bf16 path with its own offload fallback)."""
    import torch

    if not wanted(env_name) or not torch.cuda.is_available():
        return False
    if not all(copy_ready(src, is_text_encoder(name)) for name, src in sources.items()):
        return False
    size = sum(f.stat().st_size for src in sources.values() for f in copy_dir(src).glob("*.safetensors"))
    return torch.cuda.get_device_properties(0).total_memory - size >= margin_gib * 2**30


def load_copies(sources: dict, init_kwargs: dict | None = None) -> dict:
    """Every part (name → bf16 folder) from its fp8 copy, on the GPU."""
    return {name: load_copy(src, "cuda", (init_kwargs or {}).get(name)) for name, src in sources.items()}


def save_copies(parts: dict, sources: dict) -> bool:
    """Stores the converted parts (name → module) as their folders' fp8 copies where a current one is missing.
    Returns whether it wrote any (the worker's first job then commits the Volume)."""
    wrote = False
    for name, src in sources.items():
        module = parts.get(name)
        if module is not None:
            wrote |= save_copy(module, src, is_text_encoder(name))
    return wrote


class LinearNamed:
    """While open, ``FP8Linear`` calls itself "Linear": for LongCat's LoRA builder, which only wraps modules whose class
    has that name (its forward wrapper works with any)."""

    def __enter__(self):
        self.cls = linear_class()
        self.name = self.cls.__name__
        self.cls.__name__ = "Linear"
        return self

    def __exit__(self, *exc):
        self.cls.__name__ = self.name
        return False
