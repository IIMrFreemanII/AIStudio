"""MiniMax-H3-Image (MiniMax-H3-Image Community Licence: not for the US, EU, UK or Korea): pictures from text
(``text2image``), or a new picture from an instruction and up to four reference pictures the prompt names as
``<Picture 1>`` … ``<Picture 4>`` (``image_conditioned``).

It is the MiniMax H3 transformer without the audio branch (33.1B, bf16, 66 GB), with H3's own Qwen3-VL text encoder
(66.7 GB, read from the ``minimax-h3`` download, not fetched again) and a 2D image VAE. diffusers 0.43.0 ships it as
Modular Diffusers blocks (``MiniMaxH3ImageModularPipeline``); the other workers stay on 0.42.1, so its Modal class has
its own image. The checkpoint is distilled to a fixed grid of six steps with guidance baked in: no steps, no CFG and no
negative prompt. It generates on a canvas of at least ``canvas_mp`` megapixels (the model card: best at 4–16 MP) and
resizes to the asked size, so a 1024×1024 picture costs about what a 2048×2048 one does.

Like H3 video, the text encoder and transformer take turns on an H200 under a ``ComponentsManager``; a B200 holds both.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..media import write_png
from . import Output, TaskSpec

STEPS = 6  # the checkpoint's sampling grid (one output head per step)
CANVAS_MULTIPLE = 32  # VAE ×16, then 2×2 patches
MAX_PICTURES = 4
PICTURES = tuple(f"image{i}" for i in range(1, MAX_PICTURES + 1))
OFFLOAD_MARGIN_GIB = 16  # a picture's activations are small (≤ 20k tokens), unlike a video's


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


class H3ImageWorker:
    def __init__(self):
        self.pipe = None

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
        # The index names Hub repos; load every part from the volume instead (the text encoder from H3's folder).
        self.pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={
            "text_encoder": str(text_root), "default": str(root)})
        missing = [n for n in ("transformer", "vae", "text_encoder", "tokenizer", "processor")
                   if getattr(self.pipe, n, None) is None]
        if missing:  # load_components only warns
            raise RuntimeError(f"MiniMax H3 Image couldn't load: {', '.join(missing)}")
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=OFFLOAD_MARGIN_GIB * 2**30)
        self._denoise = denoise

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from PIL import Image

        from .minimax_h3 import gpu_name

        check(task, params, inputs)
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
                      extra={"gpu": gpu_name(), "workflow": "image_conditioned" if pictures else "text2image",
                             "steps": STEPS, "pictures": len(pictures), "width": image.width, "height": image.height,
                             "canvasMegapixels": params["canvas_mp"],
                             "sampleSeconds": round(time.monotonic() - t0, 2)})
