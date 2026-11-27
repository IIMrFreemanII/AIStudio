"""What the image workers (FLUX.1 dev, Qwen-Image, SD 3.5 Large, SDXL) share: up to four LoRAs per job, kept loaded in a warm container,
and fine-tunes ("variants", ``variants.py``) swapped into the loaded transformer.

LoRAs arrive as uploaded files (``lora1`` … ``lora4``, named by their SHA-256) and load through diffusers, which reads
kohya, diffusers/PEFT, ai-toolkit and XLabs files. A loaded LoRA stays as an inactive adapter (up to ``KEEP_LOADED``),
so the next job with it only switches adapters. A worker's own adapters (Qwen-Image's Lightning LoRA) are
switched on by passing them as ``pinned``, and never dropped.

A variant swap copies the fine-tune's weights into the transformer's parameters in place: the text encoders, the
VAE and every loaded LoRA stay (a LoRA adds to the layer it wraps, whatever its weights are). An SDXL fine-tune
replaces the UNet and both text encoders (``variant_components``); LoRAs load into the UNet (the pipeline loads a
LoRA's text-encoder parts by itself).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from . import FLUX_LORAS

KEEP_LOADED = 8  # user LoRAs kept in a warm container


class AdapterSlots:
    """Mixed into a worker that sets ``pipe`` and ``models_root``; ``family`` is the library family (variants.py)
    and ``title`` the base model's name, for messages."""

    family = ""
    title = ""

    components: tuple[str, ...] = ("transformer",)  # what LoRAs load into; Wan: ("transformer", "transformer_2")
    variant_components: tuple[str, ...] = ()  # what a fine-tune replaces, when more than ``components`` (SDXL)
    # The built-in modes (base and a fast transformer such as Chroma's "@flash"): once loaded, their weights stay on
    # the GPU and a mode change swaps tensors instead of reading the other's from disk (``WeightPark``).
    parked_variants: tuple[str, ...] = ()

    def _init_slots(self) -> None:
        self.variant = "base"  # the fine-tune whose transformer weights are loaded
        self.loaded: dict[str, Path] = {}  # component → the folder its weights came from (unset: the base's)
        self.loras: OrderedDict[str, None] = OrderedDict()  # user adapters, least recently used first
        self.park = WeightPark()

    # MARK: fine-tunes

    def _use_variant(self, vid: str, sink) -> None:
        """Loads a fine-tune's transformer weights in place (or the base model's again); a re-run with the same model
        costs nothing. A Wan fine-tune of one expert leaves the other transformer the base's."""
        vid = vid or "base"
        if vid == self.variant:
            return
        if vid != "base" and not self._variant_installed(vid):
            raise ValueError(f"the model {vid!r} isn't installed on this server: install it from the app's Models window")
        parts = self.variant_components or self.components
        base = {c: self._variant_dir("base", c) for c in parts}
        wanted = {c: self._variant_dir(vid, c) for c in parts}
        changed = [c for c in parts if self.loaded.get(c, base[c]) != wanted[c]]
        current = self.variant
        self.variant = None  # half-loaded if this fails: the next job reloads
        if changed and vid in self.parked_variants and len(parts) == 1:
            c = parts[0]
            # A built-in mode: swap in its parked weights (or load them beside the current ones, when they fit), and
            # park the current ones if they're a built-in mode too.
            if self.park.switch(getattr(self.pipe, c), current if current in self.parked_variants else None, vid,
                                wanted[c], sink):
                self.loaded[c] = wanted[c]
                self.variant = vid
                return
        if changed:
            sink.status("Loading the base model" if vid == "base" else f"Loading {vid}")
        for c in changed:
            # Unknown until the copy finishes: if it fails, the next job copies this component again (absent would
            # read as "the base's" and silently keep whatever weights it holds).
            self.loaded[c] = None
            _copy_weights(getattr(self.pipe, c), wanted[c], vid)
            self.loaded[c] = wanted[c]
        self.variant = vid

    def _variant_installed(self, vid: str) -> bool:
        from .. import variants

        return variants.installed(self.models_root, self.family, vid)

    def _variant_dir(self, vid: str, component: str) -> Path:
        """Where a variant's weights for ``component`` are; a worker may add its own (Chroma's Flash transformer)."""
        from .. import variants

        return variants.transformer_dir(self.models_root, self.family, vid, component)

    # MARK: LoRAs

    def _use_loras(self, params: dict, inputs: dict[str, Path], sink, pinned: dict[str, float] | None = None) -> int:
        """Loads this job's LoRAs (unless a previous job did) and activates exactly them (plus ``pinned``), at their
        strengths. Returns the number of user LoRAs.

        Wan (two transformers): slot N's file goes to both experts, or only to one (``loraN_expert`` high / low);
        with a ``loraN_low`` file it's a pair, ``loraN`` for the high-noise expert and ``loraN_low`` for the other."""
        pipe = self.pipe
        weights: dict[str, float] = {}
        for i in range(1, FLUX_LORAS + 1):
            path, strength = inputs.get(f"lora{i}"), float(params[f"lora{i}_strength"])
            low = inputs.get(f"lora{i}_low")
            if path is None or strength == 0:
                continue
            expert = "pair" if low is not None else params.get(f"lora{i}_expert", "both")
            if len(self.components) == 1:
                expert = "both"
            # The upload's SHA-256 (and how it's applied): the same file, the same adapter.
            name = "lora_" + path.name.split(".")[0][:16]
            if expert == "pair":
                name += "_" + low.name.split(".")[0][:8]
            elif expert != "both":
                name += "_" + expert[0]
            if name not in self.loras:
                sink.status(f"Loading LoRA {i}")
                parts = {"both": [(path, c) for c in self.components], "high": [(path, self.components[0])],
                         "low": [(path, self.components[-1])],
                         "pair": [(path, self.components[0]), (low, self.components[-1])]}[expert]
                self._load_lora(name, parts, i)
            self.loras[name] = None
            self.loras.move_to_end(name)
            weights[name] = weights.get(name, 0.0) + strength  # the same file twice: the strengths add up
        active = {**(pinned or {}), **weights}
        if active:
            pipe.enable_lora()
            pipe.set_adapters(list(active), list(active.values()))
        elif any(pipe.get_list_adapters().values()):
            pipe.disable_lora()
        for name in [n for n in self.loras if n not in weights][:max(0, len(self.loras) - KEEP_LOADED)]:
            pipe.delete_adapters(name)
            del self.loras[name]
        return len(weights)

    def _prepare_lora(self, state: dict) -> dict:
        """A worker's chance to rename or rescale a LoRA file's tensors before diffusers reads them (Anima's kohya
        files); raise ``ValueError`` for a file that can't fit."""
        return state

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """Loads each (file, transformer) part under one adapter name."""
        from safetensors.torch import load_file

        try:
            for path, component in parts:
                try:
                    state = load_file(str(path))
                except Exception as e:  # noqa: BLE001
                    raise NotSafetensors(f"LoRA {slot} isn't a .safetensors file ({e})") from e
                state = self._prepare_lora(state)
                extra = {"load_into_transformer_2": True} if component == "transformer_2" else {}
                self.pipe.load_lora_weights(state, adapter_name=name, **extra)
                _lora_to_compute_dtype(getattr(self.pipe, component), getattr(self, "dtype", None))
        except NotSafetensors:
            self._drop_adapter(name)
            raise
        except Exception as e:  # noqa: BLE001
            self._drop_adapter(name)
            raise ValueError(f"LoRA {slot} doesn't fit {self.title}: {e}") from e

    def _drop_adapter(self, name: str) -> None:
        try:  # a half-loaded adapter would stay in the model
            if any(name in names for names in self.pipe.get_list_adapters().values()):
                self.pipe.delete_adapters(name)
        except Exception:  # noqa: BLE001
            pass


class NotSafetensors(ValueError):
    pass


def _lora_to_compute_dtype(module, dtype) -> None:
    """With fp8 weight storage (layer-wise casting), a LoRA loaded afterwards takes its layer's fp8 dtype but gets no
    casting hook: keep LoRA weights in the compute dtype instead (they're small)."""
    import torch

    for n, p in module.named_parameters():
        if ".lora_" in n and p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            p.data = p.data.to(dtype or torch.bfloat16)


PARK_MARGIN_GIB = 16  # GPU memory left free for a job after parking another mode's weights


class WeightPark:
    """Other modes' transformer weights kept on the GPU (the user's rule: no swapping weights through the disk or the
    CPU). A switch points the module's parameters at the wanted mode's tensors and keeps the current ones: no copy.
    Weights load once, beside the current ones, only when the GPU has room for them and a job after; otherwise the
    caller falls back to copying the other's weights from disk in place."""

    def __init__(self):
        self.parked: dict[str, dict[str, object]] = {}  # mode → parameter name → tensor

    def switch(self, module, current: str | None, wanted: str, src: Path, sink) -> bool:
        params = _base_params(module)
        tensors = self.parked.pop(wanted, None)
        if tensors is None:
            if not _room_for(params):
                return False
            sink.status(f"Loading {wanted} beside the loaded weights")
            tensors = _read_weights(module, params, src, wanted)
        if current is not None:
            self.parked[current] = {n: p.data for n, p in params.items()}
        for n, p in params.items():
            p.data = tensors[n]
        return True

    def drop(self) -> None:
        self.parked.clear()


def _base_params(module) -> dict:
    """A module's own weights by their plain names (PEFT's ``base_layer`` unwrapped; LoRA factors left out)."""
    return {_plain(n): p for n, p in module.named_parameters() if ".lora_" not in n}


def _room_for(params: dict) -> bool:
    """Whether another copy of these weights fits on their GPU with ``PARK_MARGIN_GIB`` left."""
    import torch

    first = next(iter(params.values()), None)
    if first is None or first.device.type != "cuda":
        return False
    size = sum(p.numel() * p.element_size() for p in params.values())
    free, _ = torch.cuda.mem_get_info(first.device)
    # PyTorch keeps freed activation memory reserved for reuse: the driver counts it as used, but it is free for this
    free += torch.cuda.memory_reserved(first.device) - torch.cuda.memory_allocated(first.device)
    return free - size >= PARK_MARGIN_GIB * 2**30


def _read_weights(module, params: dict, src: Path, vid: str) -> dict:
    """A diffusers folder's tensors as new tensors on the parameters' devices and dtypes, by parameter name; every
    parameter must be covered (saved buffers are skipped: modes share them)."""
    from safetensors import safe_open

    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{vid}: no weights in {src.name}")
    buffers = {_plain(n) for n, _ in module.named_buffers()}
    out = {}
    for f in files:
        with safe_open(str(f), "pt", device="cpu") as sf:
            for key in sf.keys():
                name = _name_for(params, key)
                if name is None and key in buffers:
                    continue
                if name is None or name not in params:
                    raise ValueError(f"{vid}: unexpected weight {key}")
                p = params[name]
                scale = _fp8_scale_of(params, name)
                if scale is not None and not key.endswith(".w_scale"):
                    from . import fp8

                    # An fp8 layer: the file's bf16/fp32 weight quantized as the loaded one was, with its scale.
                    q, sc = fp8.quantize_weight(sf.get_tensor(key).to(p.device))
                    out[name], out[fp8.scale_name(name)] = q, sc.to(scale.dtype)
                    continue
                out[name] = sf.get_tensor(key).to(device=p.device, dtype=p.dtype)
    missing = set(params) - set(out)
    if missing:
        raise ValueError(f"{vid}: {len(missing)} weights missing, e.g. {sorted(missing)[0]}")
    return out


def _copy_weights(module, src: Path, vid: str) -> None:
    """Copies every tensor of a diffusers transformer folder into ``module``'s parameters, in place (cast to their
    dtype: fp8 storage stays fp8). PEFT's wrapped layers (``…base_layer.weight``) are found too."""
    from safetensors import safe_open

    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{vid}: no weights in {src.name}")
    params = dict(module.named_parameters())
    params.update(dict(module.named_buffers()))
    wanted = {_plain(k) for k, _ in module.named_parameters() if ".lora_" not in k}
    # Every key is matched before anything is copied, so a file that doesn't fit leaves the module as it was.
    plan = []
    for f in files:
        with safe_open(str(f), "pt", device="cpu") as sf:
            for key in sf.keys():
                name = _name_for(params, key)
                if name is None:
                    raise ValueError(f"{vid}: unexpected weight {key}")
                plan.append((f, key, name))
    seen = set()
    for f in files:
        with safe_open(str(f), "pt", device="cpu") as sf:
            for key, name in [(k, n) for g, k, n in plan if g == f]:
                target = _param_for(params, name)
                scale = _fp8_scale_of(params, _plain(name))
                if scale is not None and not key.endswith(".w_scale"):
                    from . import fp8

                    fp8.quantize_into(target, scale, sf.get_tensor(key))  # quantized as at load, with its scale
                    seen.add(fp8.scale_name(_plain(name)))
                else:
                    target.data.copy_(sf.get_tensor(key).to(device=target.device, dtype=target.dtype))
                seen.add(_plain(name))
    missing = wanted - seen
    if missing:
        raise ValueError(f"{vid}: {len(missing)} weights missing, e.g. {sorted(missing)[0]}")


def _fp8_scale_of(params: dict, name: str):
    """The per-channel scale beside ``name`` when it is a converted fp8 layer's weight (``fp8.convert``), else None."""
    if not name.endswith(".weight"):
        return None
    scale = _param_for(params, name[: -len("weight")] + "w_scale")
    weight = _param_for(params, name)
    if scale is None or weight is None or weight.dtype not in _fp8_dtypes():
        return None
    return scale


def _fp8_dtypes():
    import torch

    return (torch.float8_e4m3fn, torch.float8_e5m2)


def _plain(name: str) -> str:
    """A parameter's name without PEFT's wrapper (``to_q.base_layer.weight`` → ``to_q.weight``)."""
    return name.replace(".base_layer.", ".")


# Names a module's own loader renames: transformers 5's `CLIPTextModel` dropped the `text_model.` prefix that SDXL's
# Hugging Face files (and `CLIPTextModelWithProjection`) still use.
RENAMED_PREFIXES = ("text_model.",)


def _name_for(params: dict, key: str) -> str | None:
    """The module's name for a file's weight: as is, or with a prefix its loader would strip or add."""
    candidates = [key]
    for p in RENAMED_PREFIXES:
        candidates.append(key[len(p):] if key.startswith(p) else p + key)
    return next((c for c in candidates if _param_for(params, c) is not None), None)


def _param_for(params: dict, key: str):
    if key in params:
        return params[key]
    module, _, leaf = key.rpartition(".")
    return params.get(f"{module}.base_layer.{leaf}")
