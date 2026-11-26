"""Anima text-to-image (non-commercial licence): a 2B anime model, a Cosmos DiT with a Qwen3-0.6B text encoder (through
a small text conditioner) and the Qwen-Image VAE. diffusers 0.42.1 runs it as ``AnimaModularPipeline`` (Modular
Diffusers: the repo's ``modular_model_index.json`` names the Hub repo, so components load from the volume instead).

Up to four LoRAs per job and fine-tunes from links (``adapters.py``). Civitai's Anima LoRAs are kohya-style
(``lora_unet_blocks_0_self_attn_q_proj.lora_down.weight`` + ``.alpha``); diffusers only converts ``diffusion_model.*``
names, so ``_prepare_lora`` renames them and folds the alpha into ``lora_A`` first. A fine-tune is a ComfyUI
single-file checkpoint (``variants.convert_anima``): its transformer, and its text conditioner when it ships one.

Classifier-free guidance is the pipeline's own guider (``guidance_scale`` is read every step; 1.0 skips the negative
pass). The denoise loop's progress bar is replaced for the call, so each step reports progress and honours a cancel.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from ..media import write_png
from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import pick_device

_ATTN = re.compile(r"blocks_(\d+)_(self_attn|cross_attn)_(q_proj|k_proj|v_proj|output_proj)")
_MLP = re.compile(r"blocks_(\d+)_mlp_(layer1|layer2)")
_ADALN = re.compile(r"blocks_(\d+)_adaln_modulation_(self_attn|cross_attn|mlp)_(1|2)")


def kohya_module_path(name: str) -> str:
    """``blocks_0_self_attn_q_proj`` → ``blocks.0.self_attn.q_proj`` (ComfyUI's Cosmos-2.0 names)."""
    if m := _ATTN.fullmatch(name):
        return f"blocks.{m[1]}.{m[2]}.{m[3]}"
    if m := _MLP.fullmatch(name):
        return f"blocks.{m[1]}.mlp.{m[2]}"
    if m := _ADALN.fullmatch(name):
        return f"blocks.{m[1]}.adaln_modulation_{m[2]}.{m[3]}"
    raise ValueError(f"unknown layer {name}")


def kohya_to_diffusion_model(state: dict) -> dict:
    """A kohya Anima LoRA → ``diffusion_model.<path>.lora_A/lora_B.weight`` (what diffusers converts). ``lora_A`` is
    ``lora_down`` times ``alpha / rank`` (1 without an alpha); other layouts pass through unchanged."""
    if not any(k.startswith("lora_unet_") for k in state):
        return state
    out: dict = {}
    for key in state:
        if key.startswith("lora_te"):
            raise ValueError("text-encoder LoRAs aren't supported")
        if not key.startswith("lora_unet_"):
            raise ValueError(f"unknown layer {key}")
    for key, value in state.items():
        module, _, leaf = key.removeprefix("lora_unet_").partition(".")
        if leaf not in ("lora_down.weight", "lora_up.weight", "alpha"):
            raise ValueError(f"unknown layer {key}")
        path = kohya_module_path(module)
        if leaf == "lora_down.weight":
            alpha = state.get(f"lora_unet_{module}.alpha")
            scale = float(alpha) / value.shape[0] if alpha is not None else 1.0
            out[f"diffusion_model.{path}.lora_A.weight"] = value * scale
        elif leaf == "lora_up.weight":
            out[f"diffusion_model.{path}.lora_B.weight"] = value
    return out


class AnimaWorker(AdapterSlots):
    family = "anima"
    title = "Anima"
    components = ("transformer",)
    variant_components = ("transformer", "text_conditioner")

    def __init__(self):
        self.pipe = None
        self.dtype = None
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        import torch
        from diffusers import ModularPipeline

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        root = models_root / "anima"
        status("Loading Anima")
        # The index names the Hub repo for every component; they load from the volume instead.
        pipe = ModularPipeline.from_pretrained(str(root))
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(root))
        pipe.to(device)
        self.pipe = pipe
        self.models_root = models_root

    def _prepare_lora(self, state: dict) -> dict:
        return kohya_to_diffusion_model(state)

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """Loads a community LoRA, then checks it took: diffusers only warns (and loads nothing) when no key names an
        Anima module. Its factors go to bf16 (kohya files are fp16 or fp32)."""
        import torch

        super()._load_lora(name, parts, slot)
        if not any(name in names for names in self.pipe.get_list_adapters().values()):
            raise ValueError(f"LoRA {slot} doesn't fit {self.title}: none of its layers are Anima's")
        for n, p in self.pipe.transformer.named_parameters():
            if ".lora_" in n and p.dtype in (torch.float32, torch.float16):
                p.data = p.data.to(torch.bfloat16)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from diffusers.modular_pipelines.anima import denoise

        steps, guidance = params["steps"], float(params["guidance"])
        self._use_variant(params.get("model") or "base", sink)
        loras = self._use_loras(params, inputs, sink)
        self.pipe.guider.guidance_scale = guidance  # read at every step; 1.0 turns the negative pass off

        clock = {"done": 0, "total": steps}

        class Progress:  # the denoise loop's progress bar: ours reports and cancels
            def __init__(self, total=None):
                clock["total"] = total or steps

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, n=1):
                clock["done"] += n
                sink.progress(min(clock["done"], clock["total"]), clock["total"])
                sink.check()

        loop = denoise.AnimaDenoiseLoopWrapper
        original = loop.progress_bar
        sink.status("Sampling")
        start = time.monotonic()
        loop.progress_bar = lambda self, iterable=None, total=None: Progress(total=total)
        try:
            with torch.inference_mode():
                images = self.pipe(prompt=params["prompt"], negative_prompt=params["negative"].strip() or None,
                                   height=params["height"], width=params["width"], num_inference_steps=steps,
                                   generator=torch.Generator(device="cpu").manual_seed(params["seed"]),
                                   output_type="pil", output="images")
        finally:
            loop.progress_bar = original
        path = out_dir / "result.png"
        write_png(np.asarray(images[0]), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "guidance": guidance, "loras": loras,
                             "sampleSeconds": round(time.monotonic() - start, 2)})
