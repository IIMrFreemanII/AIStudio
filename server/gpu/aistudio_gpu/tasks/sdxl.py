"""Stable Diffusion XL 1.0 text-to-image (diffusers ``StableDiffusionXLPipeline``), with up to four LoRAs per job and
fine-tunes from links (``adapters.py``).

The download keeps only the fp16 files (≈ 7 GB); they run in bf16 on the GPU, which avoids the fp16 VAE's overflow
(black images) at no cost in speed. A fine-tune (Juggernaut, Pony, Illustrious, NoobAI…) is one checkpoint file from
Civitai: its UNet and both CLIP text encoders are swapped into the loaded pipeline in place, the VAE stays SDXL's.
LoRAs load into the UNet and, when the file has them, both text encoders.

The sampler is chosen per job from three schedulers built once: DPM++ 2M Karras (the default), Euler a, Euler.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_png
from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import generator, pick_device

PREVIEW_EVERY = 5  # a VAE decode per preview: every few steps is plenty


class SDXLWorker(AdapterSlots):
    family = "sdxl"
    title = "SDXL"
    components = ("unet",)  # LoRAs load into the UNet (and the text encoders, by the pipeline itself)
    variant_components = ("unet", "text_encoder", "text_encoder_2")

    def __init__(self):
        self.pipe = None
        self.schedulers: dict = {}
        self.x0: dict = {}  # the last step's estimate of the finished latents, for previews (``_remember_x0``)
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import (DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                               StableDiffusionXLPipeline)

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        status("Loading SDXL")
        pipe = StableDiffusionXLPipeline.from_pretrained(models_root / "sdxl", variant="fp16", dtype=dtype)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        cfg = pipe.scheduler.config
        self.schedulers = {
            "dpmpp-2m-karras": DPMSolverMultistepScheduler.from_config(
                cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++", solver_order=2),
            "euler-a": EulerAncestralDiscreteScheduler.from_config(cfg),
            "euler": EulerDiscreteScheduler.from_config(cfg),
        }
        for scheduler in self.schedulers.values():
            _remember_x0(scheduler, self.x0)
        self.pipe = pipe
        self.models_root = models_root

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        steps, width, height = params["steps"], params["width"], params["height"]
        guidance, sampler = params["guidance"], params["sampler"]

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, steps)
            if (i + 1) % PREVIEW_EVERY == 0 and i + 1 < steps:
                try:
                    x0 = self.x0.get("x0")
                    sink.preview(self._preview(kwargs["latents"] if x0 is None else x0), i + 1, steps)
                except Exception:  # noqa: BLE001 — a preview never fails a job
                    pass
            return kwargs

        self._use_variant(params.get("model") or "base", sink)
        loras = self._use_loras(params, inputs, sink)
        self.pipe.scheduler = self.schedulers[sampler]
        self.x0.clear()
        sink.status("Sampling")
        start = time.monotonic()
        with torch.inference_mode():
            image = self.pipe(prompt=params["prompt"], negative_prompt=params["negative"].strip() or None,
                              width=width, height=height, num_inference_steps=steps, guidance_scale=guidance,
                              generator=generator(params["seed"]), callback_on_step_end=on_step,
                              callback_on_step_end_tensor_inputs=["latents"]).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "guidance": guidance, "sampler": sampler, "loras": loras,
                             "sampleSeconds": round(time.monotonic() - start, 2)})

    def _preview(self, latents) -> np.ndarray:
        import torch

        vae = self.pipe.vae
        with torch.inference_mode():
            image = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)  # -1…1 → 0…1


def _remember_x0(scheduler, slot: dict) -> None:
    """Wraps ``scheduler.step`` (once, at load) to keep the step's estimate of the finished latents in ``slot["x0"]``.
    A preview decodes that: the latents mid-way through a Karras schedule are still mostly noise. Euler samplers
    return the estimate; DPM++ (``dpmsolver++``) keeps it as its last model output."""
    step = scheduler.step

    def wrapped(*args, **kwargs):
        out = step(*args, **kwargs)
        x0 = out[1] if isinstance(out, tuple) and len(out) > 1 else getattr(out, "pred_original_sample", None)
        if x0 is None and getattr(scheduler, "model_outputs", None):
            x0 = scheduler.model_outputs[-1]
        slot["x0"] = x0
        return out

    scheduler.step = wrapped
