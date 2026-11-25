"""Stable Diffusion XL 1.0 text-to-image (diffusers ``StableDiffusionXLPipeline``), with up to four LoRAs per job and
fine-tunes from links (``adapters.py``).

The download keeps only the fp16 files (≈ 7 GB); they run in bf16 on the GPU, which avoids the fp16 VAE's overflow
(black images) at no cost in speed. A fine-tune (Juggernaut, Pony, Illustrious, NoobAI…) is one checkpoint file from
Civitai: its UNet and both CLIP text encoders are swapped into the loaded pipeline in place, the VAE stays SDXL's.
LoRAs load into the UNet and, when the file has them, both text encoders.

An optional ``face`` picture steers the result towards that face (diffusers IP-Adapter "plus face" with its ViT-H image
encoder, ``sdxl-face``, loaded on the first job that has a face and unloaded again for jobs without one or when the
fine-tune changes). Jobs without a face run exactly as before.

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

FACE_DIR = "sdxl-face"
FACE_WEIGHTS = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
PREVIEW_EVERY = 5  # a VAE decode per preview: every few steps is plenty


class SDXLWorker(AdapterSlots):
    family = "sdxl"
    title = "SDXL"
    components = ("unet",)  # LoRAs load into the UNet (and the text encoders, by the pipeline itself)
    variant_components = ("unet", "text_encoder", "text_encoder_2")

    def __init__(self):
        self.pipe = None
        self.schedulers: dict = {}
        self.face_loaded = False
        self.image_encoder = None  # the face model's CLIP ViT-H, kept once loaded
        self.x0: dict = {}  # the last step's estimate of the finished latents, for previews (``previews.remember_x0``)
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
        if self.face_loaded and wanted != self.variant:
            self._unload_face()  # a fine-tune swap changes the UNet the adapter's attention layers sit in
        self._use_variant(wanted, sink)
        loras = self._use_loras(params, inputs, sink)
        face = inputs.get("face")
        extra_args = {}
        if face:
            from PIL import Image

            self._load_face(sink)
            self.pipe.set_ip_adapter_scale(params["face_strength"])
            with Image.open(face) as picture:
                extra_args["ip_adapter_image"] = picture.convert("RGB")
        elif self.face_loaded:
            self._unload_face()
        self.pipe.scheduler = self.schedulers[sampler]
        self.x0.clear()
        sink.status("Sampling")
        start = time.monotonic()
        with torch.inference_mode():
            image = self.pipe(prompt=params["prompt"], negative_prompt=params["negative"].strip() or None,
                              width=width, height=height, num_inference_steps=steps, guidance_scale=guidance,
                              generator=generator(params["seed"]), callback_on_step_end=on_step,
                              callback_on_step_end_tensor_inputs=["latents"], **extra_args).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "guidance": guidance, "sampler": sampler, "loras": loras,
                             "face": bool(face), "sampleSeconds": round(time.monotonic() - start, 2)})

    def _load_face(self, sink) -> None:
        if self.face_loaded:
            return
        import transformers

        root = self.models_root / FACE_DIR
        if not (root / ".complete").exists():
            raise ValueError("The face reference model isn't on this GPU server yet: download sdxl-face "
                             "(python -m aistudio_gpu download --models sdxl-face).")
        sink.status("Loading the face reference")
        if self.image_encoder is None:
            self.image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(
                root / "models" / "image_encoder", dtype=self.dtype).to(self.device)
        self.pipe.load_ip_adapter(str(root), subfolder="sdxl_models", weight_name=FACE_WEIGHTS,
                                  image_encoder_folder=None)
        self.pipe.register_modules(image_encoder=self.image_encoder)
        if getattr(self.pipe, "feature_extractor", None) is None:
            self.pipe.register_modules(feature_extractor=transformers.CLIPImageProcessor())
        self.face_loaded = True

    def _unload_face(self) -> None:
        self.pipe.unload_ip_adapter()
        self.face_loaded = False

    def _preview(self, latents) -> np.ndarray:
        import torch

        vae = self.pipe.vae
        with torch.inference_mode():
            image = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)  # -1…1 → 0…1
