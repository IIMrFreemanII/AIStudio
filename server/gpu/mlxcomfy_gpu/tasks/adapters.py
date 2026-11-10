"""What the image workers (FLUX.1 dev, Qwen-Image) share: up to four LoRAs per job, kept loaded in a warm container,
and fine-tunes ("variants", ``variants.py``) swapped into the loaded transformer.

LoRAs arrive as uploaded files (``lora1`` … ``lora4``, named by their SHA-256) and load through diffusers, which reads
kohya, diffusers/PEFT, ai-toolkit and XLabs files. A loaded LoRA stays as an inactive adapter (up to ``KEEP_LOADED``),
so the next job with it only switches adapters. A worker's own adapters (Qwen-Image's Lightning LoRA) are
switched on by passing them as ``pinned``, and never dropped.

A variant swap copies the fine-tune's weights into the transformer's parameters in place: the text encoders, the
VAE and every loaded LoRA stay (a LoRA adds to the layer it wraps, whatever its weights are).
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

    components: tuple[str, ...] = ("transformer",)  # Wan: ("transformer", "transformer_2"), high / low noise

    def _init_slots(self) -> None:
        self.variant = "base"  # the fine-tune whose transformer weights are loaded
        self.loaded: dict[str, Path] = {}  # component → the folder its weights came from (unset: the base's)
        self.loras: OrderedDict[str, None] = OrderedDict()  # user adapters, least recently used first

    # MARK: fine-tunes

    def _use_variant(self, vid: str, sink) -> None:
        """Loads a fine-tune's transformer weights in place (or the base model's again); a re-run with the same model
        costs nothing. A Wan fine-tune of one expert leaves the other transformer the base's."""
        from .. import variants

        vid = vid or "base"
        if vid == self.variant:
            return
        if vid != "base" and not variants.installed(self.models_root, self.family, vid):
            raise ValueError(f"the model {vid!r} isn't installed on this server: install it from the app's Models window")
        base = {c: variants.transformer_dir(self.models_root, self.family, "base", c) for c in self.components}
        wanted = {c: variants.transformer_dir(self.models_root, self.family, vid, c) for c in self.components}
        changed = [c for c in self.components if self.loaded.get(c, base[c]) != wanted[c]]
        self.variant = None  # half-loaded if this fails: the next job reloads
        if changed:
            sink.status("Loading the base model" if vid == "base" else f"Loading {vid}")
        for c in changed:
            self.loaded.pop(c, None)
            _copy_weights(getattr(self.pipe, c), wanted[c], vid)
            self.loaded[c] = wanted[c]
        self.variant = vid

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

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """Loads each (file, transformer) part under one adapter name."""
        from safetensors.torch import load_file

        try:
            for path, component in parts:
                try:
                    state = load_file(str(path))
                except Exception as e:  # noqa: BLE001
                    raise NotSafetensors(f"LoRA {slot} isn't a .safetensors file ({e})") from e
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
    seen = set()
    for f in files:
        with safe_open(str(f), "pt", device="cpu") as sf:
            for key in sf.keys():
                target = _param_for(params, key)
                if target is None:
                    raise ValueError(f"{vid}: unexpected weight {key}")
                target.data.copy_(sf.get_tensor(key).to(device=target.device, dtype=target.dtype))
                seen.add(key)
    missing = wanted - seen
    if missing:
        raise ValueError(f"{vid}: {len(missing)} weights missing, e.g. {sorted(missing)[0]}")


def _plain(name: str) -> str:
    """A parameter's name without PEFT's wrapper (``to_q.base_layer.weight`` → ``to_q.weight``)."""
    return name.replace(".base_layer.", ".")


def _param_for(params: dict, key: str):
    if key in params:
        return params[key]
    module, _, leaf = key.rpartition(".")
    return params.get(f"{module}.base_layer.{leaf}")
