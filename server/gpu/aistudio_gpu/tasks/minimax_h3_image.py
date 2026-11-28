"""MiniMax-H3-Image (MiniMax-H3-Image Community Licence: not for the US, EU, UK or Korea): pictures from text
(``text2image``), or a new picture from an instruction and up to four reference pictures the prompt names as
``<Picture 1>`` … ``<Picture 4>`` (``image_conditioned``).

It is the MiniMax H3 transformer without the audio branch (33.1B, bf16, 66 GB), with H3's own Qwen3-VL text encoder
(66.7 GB, read from the ``minimax-h3`` download, not fetched again) and a 2D image VAE. diffusers 0.43.0 ships it as
Modular Diffusers blocks (``MiniMaxH3ImageModularPipeline``); the other workers stay on 0.42.1, so its Modal class has
its own image. The checkpoint is distilled to a fixed grid of six steps with guidance baked in: no steps, no CFG and no
negative prompt. It generates on a canvas of at least ``canvas_mp`` megapixels (the model card: best at 4–16 MP) and
resizes to the asked size, so a 1024×1024 picture costs about what a 2048×2048 one does.

LoRAs (``lora1`` … ``lora4``) load into the transformer through diffusers' MiniMax H3 LoRA converter (the video
model's: musubi-tuner, ai-toolkit, DiffSynth and diffusers layouts). The image transformer is the video one without
its audio branch, so a video LoRA's audio-branch tensors (and any other module the image model lacks) are dropped and
counted; the rest loads.

All of it stays on the GPU (≈ 125 GiB: a B200). Only on a smaller card (an opt-in fallback) do the text encoder and
transformer take turns under a ``ComponentsManager``, which costs ≈ 80 s a picture on an H200 instead of seconds.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..media import write_png
from . import Output, TaskSpec, fp8
from .adapters import AdapterSlots
from .residency import fits, weight_bytes

STEPS = 6  # the checkpoint's sampling grid (one output head per step)
CANVAS_MULTIPLE = 32  # VAE ×16, then 2×2 patches
MAX_PICTURES = 4
PICTURES = tuple(f"image{i}" for i in range(1, MAX_PICTURES + 1))
OFFLOAD_MARGIN_GIB = 16  # a picture's activations are small (≤ 20k tokens), unlike a video's
RESIDENT_MARGIN_GIB = 24  # what must stay free beside all the weights (activations, the VAE decode at 16 MP)


def fits_resident(weights: int, gpu_total: int) -> bool:
    """Whether all the weights fit on the GPU at once with room to run (else the components take turns)."""
    return fits(weights, gpu_total, RESIDENT_MARGIN_GIB)


def output_size(params: dict) -> tuple[int, int]:
    """(width, height) as asked, rounded to multiples of 32."""
    m = CANVAS_MULTIPLE
    return max(m, round(params["width"] / m) * m), max(m, round(params["height"] / m) * m)


def check(task: TaskSpec, params: dict, inputs: dict[str, Path]) -> None:
    """Refuses what the model can't do (the fake worker calls it too). A negative prompt can't reach it: the task has
    no such parameter, so the server refuses one as unknown."""
    if not params["prompt"].strip():
        raise ValueError("describe the picture: MiniMax H3 Image needs a prompt")
    named = [k for k in PICTURES if k in inputs]
    if named != list(PICTURES[: len(named)]):
        raise ValueError("pass the reference pictures in order: image1, image2, …")


def lora_module(key: str) -> str | None:
    """The transformer module a converted LoRA tensor belongs to (``transformer.blocks.0.attn.to_q.lora_A.weight`` →
    ``blocks.0.attn.to_q``); None for a key that isn't a LoRA factor or alpha."""
    name = key.removeprefix("transformer.")
    for part in (".lora_A.", ".lora_B.", ".lora_up.", ".lora_down."):
        if part in name:
            return name.split(part)[0]
    if name.endswith(".alpha"):
        return name.removesuffix(".alpha")
    return None


def keep_existing(state: dict, modules: set[str]) -> tuple[dict, int]:
    """The tensors of ``state`` whose module the image transformer has, and how many were dropped (a video LoRA's audio
    branch)."""
    kept = {k: v for k, v in state.items() if (m := lora_module(k)) is None or m in modules}
    return kept, len(state) - len(kept)


class LoRAPipe:
    """The modular pipeline (``__call__`` and every attribute) plus the pipeline-level LoRA calls ``AdapterSlots``
    makes, done on the transformer by diffusers' MiniMax H3 LoRA mixin (the image pipeline has none of its own)."""

    def __init__(self, pipe):
        from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin
        from diffusers.loaders.peft import PeftAdapterMixin

        t = pipe.transformer
        if not isinstance(t, PeftAdapterMixin):  # diffusers 0.43's image transformer lacks the adapter methods
            t.__class__ = type(type(t).__name__, (type(t), PeftAdapterMixin), {})
        modules = {n for n, _ in t.named_modules()}
        dropped = {"n": 0}

        class Host(MiniMaxH3LoraLoaderMixin):
            def __init__(self, transformer):
                self.transformer = transformer
                self.transformer_ref = None
                # What diffusers' offload check reads from a pipeline: no device map, and its modules.
                self.hf_device_map = None
                self.components = {"transformer": transformer}

            def lora_state_dict(self, *args, **kwargs):  # the video converter, then only what the image model has
                out = MiniMaxH3LoraLoaderMixin.lora_state_dict.__func__(type(self), *args, **kwargs)
                state, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
                state, dropped["n"] = keep_existing(state, modules)
                if not any(lora_module(k) for k in state):
                    raise ValueError("none of its layers exist in MiniMax H3 Image")
                return (state, *rest) if rest else state

        self._pipe = pipe
        self._host = Host(pipe.transformer)
        self.dropped = dropped

    def __call__(self, *args, **kwargs):
        return self._pipe(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def load_lora_weights(self, state, adapter_name, **_):
        self.dropped["n"] = 0
        self._host.load_lora_weights(state, adapter_name=adapter_name)
        if self.dropped["n"]:
            print(f"h3-image: LoRA {adapter_name}: dropped {self.dropped['n']} tensors for layers the image model "
                  "lacks (a video LoRA's audio branch)", flush=True)

    def enable_lora(self):
        self._host.enable_lora()

    def disable_lora(self):
        self._host.disable_lora()

    def set_adapters(self, names, weights):
        self._host.set_adapters(names, weights)

    def delete_adapters(self, name):
        self._host.delete_adapters(name)

    def get_list_adapters(self):
        return self._host.get_list_adapters()


class H3ImageWorker(AdapterSlots):
    family = "minimax-h3-image"
    title = "MiniMax H3 Image"
    components = ("transformer",)

    def __init__(self):
        self.pipe = None
        self.resident = False
        self.dtype = None
        self._init_slots()

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """As every worker, then fp32 factors (DiffSynth's files) back to bf16, so the LoRA path keeps the budget."""
        import torch

        super()._load_lora(name, parts, slot)
        for n, p in self.pipe.transformer.named_parameters():
            if ".lora_" in n and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

    def load(self, models_root: Path, status) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3_image import denoise

        root = models_root / "minimax-h3-image"
        text_root = models_root / "minimax-h3"  # its text_encoder/ is the same model
        if not (text_root / "text_encoder").is_dir():
            raise RuntimeError("MiniMax H3 Image reads MiniMax H3's text encoder: download minimax-h3 too")
        status("Loading MiniMax H3 Image")
        manager = ComponentsManager()
        self.pipe = ModularPipeline.from_pretrained(str(root), components_manager=manager)
        # The transformer and the text encoder (H3's, shared with H3's own copy) from their fp8 copies when they're
        # there: half the read. load_components then loads only the parts still missing.
        sub = getattr(self.pipe._component_specs.get("transformer"), "subfolder", None) or "transformer"
        self.fp8_sources = {"transformer": root / sub, "text_encoder": text_root / "text_encoder"}
        if fp8.copies_usable(self.fp8_sources, "AISTUDIO_PRECISION_H3_IMAGE"):
            status("Loading MiniMax H3 Image (fp8)")
            self.pipe.update_components(**fp8.load_copies(self.fp8_sources))
        # The index names Hub repos; load every part from the volume instead (the text encoder from H3's folder).
        self.pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={
            "text_encoder": str(text_root), "default": str(root)})
        missing = [n for n in ("transformer", "vae", "text_encoder", "tokenizer", "processor")
                   if getattr(self.pipe, n, None) is None]
        if missing:  # load_components only warns
            raise RuntimeError(f"MiniMax H3 Image couldn't load: {', '.join(missing)}")
        parts = {n: getattr(self.pipe, n) for n in ("transformer", "vae", "text_encoder")}
        weights = sum(weight_bytes(m) for m in parts.values())
        total = torch.cuda.mem_get_info()[1]
        self.resident = fits_resident(weights, total)
        if self.resident:
            status("Moving MiniMax H3 Image onto the GPU")
            for module in parts.values():
                module.to("cuda")
            if fp8.wanted("AISTUDIO_PRECISION_H3_IMAGE"):
                # The transformer and text encoder: half their memory (fp4: the transformer a quarter).
                fp8.convert_parts(parts, label="h3-image fp8",
                                  mode=fp8.precision("AISTUDIO_PRECISION_H3_IMAGE"))
                self.fp8_written = fp8.save_copies(parts, self.fp8_sources)  # for the next cold start
        else:  # an opt-in smaller fallback GPU: they take turns (slow; see the module docstring)
            print(f"h3-image: {weights / 2**30:.0f} GiB of weights don't fit {total / 2**30:.0f} GiB with room to run; "
                  "offloading", flush=True)
            manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=OFFLOAD_MARGIN_GIB * 2**30)
        self._denoise = denoise
        self.models_root = models_root
        self.dtype = torch.bfloat16
        self.pipe = LoRAPipe(self.pipe)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from PIL import Image

        from .minimax_h3 import gpu_name

        check(task, params, inputs)
        loras = self._use_loras(params, inputs, sink)
        width, height = output_size(params)
        pictures = [Image.open(inputs[k]).convert("RGB") for k in PICTURES if k in inputs]
        kwargs: dict = {"prompt": params["prompt"].strip(), "width": width, "height": height,
                        "min_generation_area": int(params["canvas_mp"] * 1024 * 1024),
                        "generator": torch.Generator().manual_seed(params["seed"]), "output_type": "pil"}
        if pictures:
            kwargs["image"] = pictures

        done = {"n": 0}

        class Progress:  # the denoise loop's progress bar: ours reports and cancels
            def __init__(self, total=None):
                self.total = total or STEPS

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, n=1):
                done["n"] += n
                sink.progress(min(done["n"], self.total), self.total)
                sink.check()

        loop = self._denoise.MiniMaxH3ImageDenoiseLoopWrapper
        original = loop.progress_bar
        sink.status("Sampling")
        t0 = time.monotonic()
        loop.progress_bar = lambda self, iterable=None, total=None: Progress(total=total)
        try:
            images = self.pipe(**kwargs, output="images")
        finally:
            loop.progress_bar = original
        image = images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image.convert("RGB")), path)
        return Output(path, "image", "image/png",
                      extra={"gpu": gpu_name(), "resident": self.resident,
                             "workflow": "image_conditioned" if pictures else "text2image",
                             "steps": STEPS, "pictures": len(pictures), "loras": loras, "width": image.width, "height": image.height,
                             "canvasMegapixels": params["canvas_mp"],
                             "sampleSeconds": round(time.monotonic() - t0, 2)})
