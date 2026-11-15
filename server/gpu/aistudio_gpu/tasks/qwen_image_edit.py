"""Qwen-Image-Edit 2511 (diffusers ``QwenImageEditPlusPipeline``): a new picture from an instruction and one to three
input pictures ("picture 1 is the woman, picture 2 the room: she sits by the window"). The movie studio makes its
keyframes with it from the cast's reference sheets, so faces and clothes stay the same from shot to shot.

It is Qwen-Image's 20B transformer trained for editing, with the same Qwen2.5-VL text encoder (which also reads the
input pictures) and VAE, so memory is as for Qwen-Image: bf16 on an 80 GB card, fp8 storage on a smaller one. ``fast``
uses its Lightning 8-step LoRA without guidance; otherwise the model card's 40 steps with true CFG at 4. LoRAs work as
on Qwen-Image (edit LoRAs: Qwen-Image's own seldom fit).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..media import write_png
from . import Output, TaskSpec
from .device import generator
from .qwen_image import PREVIEW_EVERY, QwenImageWorker

FULL_STEPS, FULL_CFG = 40, 4.0
FAST_STEPS = 8
PICTURES = ("image1", "image2", "image3")
AUTO_AREA = 1024 * 1024  # width/height 0: picture 1's shape at about a megapixel


def auto_size(width: int, height: int, ratio: float) -> tuple[int, int]:
    """The output size: as asked (rounded to multiples of 16), or picture 1's shape at about a megapixel."""
    if not width or not height:
        width = math.sqrt(AUTO_AREA * ratio)
        height = width / ratio
    return max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)


class QwenImageEditWorker(QwenImageWorker):
    family = "qwen-image-edit"
    title = "Qwen-Image-Edit"
    pipeline_class = "QwenImageEditPlusPipeline"
    folder = "qwen-image-edit"
    lightning = ("qwen-image-edit-lightning", "8steps.safetensors")

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from PIL import Image

        pipe = self.pipe
        pictures = [Image.open(inputs[k]).convert("RGB") for k in PICTURES if k in inputs]
        if not pictures:
            raise ValueError("Qwen-Image-Edit needs at least one picture (image1)")
        fast = params["fast"] and self.has_fast
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        cfg = params["cfg"] or (1.0 if fast else FULL_CFG)
        width, height = auto_size(params["width"], params["height"], pictures[0].width / pictures[0].height)

        loras = self._use_loras(params, inputs, sink, pinned={"fast": 1.0} if fast else None)
        pipe.scheduler = self.fast_scheduler if fast else self.full_scheduler

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, steps)
            if (i + 1) % PREVIEW_EVERY == 0 and i + 1 < steps:
                try:
                    sink.preview(self._preview(kwargs["latents"], width, height), i + 1, steps)
                except Exception:  # noqa: BLE001 — a preview never fails a job
                    pass
            return kwargs

        negative = params["negative"].strip()
        args = dict(image=pictures, prompt=params["prompt"], width=width, height=height, num_inference_steps=steps,
                    true_cfg_scale=cfg, generator=generator(params["seed"]), callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=["latents"])
        if cfg > 1:  # true CFG runs only with a negative prompt; " " is the model card's empty one
            args["negative_prompt"] = negative or " "
        how = "cfg" if cfg > 1 and negative else ("ignored" if negative else "none")
        sink.status("Sampling" + (" (fast)" if fast else ""))
        start = time.monotonic()
        with torch.inference_mode():
            image = pipe(**args).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "cfg": cfg, "fast": fast, "negative": how, "loras": loras,
                             "pictures": len(pictures), "width": width, "height": height, "fp8": self.fp8,
                             "sampleSeconds": round(time.monotonic() - start, 2)})
