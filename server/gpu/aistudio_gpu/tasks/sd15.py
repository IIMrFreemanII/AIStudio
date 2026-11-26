"""Stable Diffusion 1.5 text-to-image (diffusers ``StableDiffusionPipeline``), with up to four LoRAs per job and
fine-tunes from links (``adapters.py``).

The download keeps only the fp16 files (≈ 2 GB, no safety checker); they run in bf16 on the GPU. A fine-tune
(DreamShaper, Realistic Vision, Deliberate…) is one checkpoint file from Civitai: its UNet and CLIP text encoder are
swapped into the loaded pipeline in place, the VAE stays SD 1.5's. LoRAs load into the UNet and, when the file has
them, the text encoder.

Prompts may be longer than CLIP's 77 tokens and carry A1111 weights, ``(red jacket:1.3)``: the worker encodes
them itself (``prompt_weights.py``) and passes the embeddings.

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
from .previews import remember_x0
from .prompt_weights import encode_sd15

PREVIEW_EVERY = 5  # a VAE decode per preview: every few steps is plenty


class SD15Worker(AdapterSlots):
    family = "sd1.5"
    title = "Stable Diffusion 1.5"
    components = ("unet",)  # LoRAs load into the UNet (and the text encoder, by the pipeline itself)
    variant_components = ("unet", "text_encoder")

    def __init__(self):
        self.pipe = None
        self.schedulers: dict = {}
        self.x0: dict = {}  # the last step's estimate of the finished latents, for previews (``previews.remember_x0``)
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import (DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                               StableDiffusionPipeline)

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        status("Loading Stable Diffusion 1.5")
        pipe = StableDiffusionPipeline.from_pretrained(models_root / "sd15", variant="fp16", dtype=dtype,
                                                       safety_checker=None, requires_safety_checker=False)
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
            remember_x0(scheduler, self.x0)
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

        wanted = params.get("model") or "base"
        self._use_variant(wanted, sink)
        loras = self._use_loras(params, inputs, sink)
        self.pipe.scheduler = self.schedulers[sampler]
        self.x0.clear()
        sink.status("Sampling")
        start = time.monotonic()
        with torch.inference_mode():
            # After the LoRAs: the ones with text-encoder parts change the embeddings.
            embeds = encode_sd15(self.pipe, params["prompt"], params["negative"])
            image = self.pipe(**embeds, width=width, height=height, num_inference_steps=steps, guidance_scale=guidance,
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
