"""Chroma1-HD text-to-image (diffusers ``ChromaPipeline``), bf16, with up to four LoRAs per job and fine-tunes from
links (``adapters.py``, family ``chroma``).

``fast`` runs Chroma1-Flash's transformer (CFG baked in: 16 steps, no guidance pass); otherwise Chroma1-HD (40 steps
at guidance 3, with the negative prompt). Flash is the default when it is installed, so it loads first; the two share
the T5 text encoder, the VAE and the scheduler and differ only in the transformer, which is swapped in place from the
models folder when a job asks for the other (as a variant with the reserved id ``@flash``). A fine-tune always runs as
full Chroma.

An optional ``face`` picture steers the result towards that face: XLabs' FLUX.1 IP-Adapter v2 with its CLIP ViT-L
image encoder (``chroma-face``), loaded on the first job that has a face and unloaded again for jobs without one or
before the transformer's weights change (its extra attention weights would not be in a fine-tune's files).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_png
from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import generator, pick_device

FLASH = "@flash"  # the variant id of Chroma1-Flash's transformer (not a valid library id)
FLASH_DIR = "chroma-flash"
FACE_DIR = "chroma-face"
FAST_STEPS, FULL_STEPS, FULL_GUIDANCE = 16, 40, 3.0
PREVIEW_EVERY = 4  # a VAE decode per preview: every few steps is plenty


class ChromaWorker(AdapterSlots):
    family = "chroma"
    title = "Chroma1-HD"

    def __init__(self):
        self.pipe = None
        self.has_fast = False
        self.face_loaded = False
        self.image_encoder = None  # the face model's CLIP ViT-L, kept once loaded
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import ChromaPipeline, ChromaTransformer2DModel

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        self.models_root = models_root
        extra = {}
        flash = models_root / FLASH_DIR
        if (flash / ".complete").exists():  # the default mode: start with Flash's transformer
            status("Loading Chroma1-Flash")
            extra["transformer"] = ChromaTransformer2DModel.from_pretrained(flash / "transformer", dtype=dtype)
            self.has_fast = True
        else:
            status("Loading Chroma1-HD")
        pipe = ChromaPipeline.from_pretrained(models_root / "chroma", dtype=dtype, **extra)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        if self.has_fast:
            self.variant = FLASH
            self.loaded["transformer"] = flash / "transformer"

    def _variant_installed(self, vid: str) -> bool:
        if vid == FLASH:
            return self.has_fast
        return super()._variant_installed(vid)

    def _variant_dir(self, vid: str, component: str) -> Path:
        if vid == FLASH:
            return self.models_root / FLASH_DIR / component
        return super()._variant_dir(vid, component)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        model = params.get("model") or "base"
        fast = params["fast"] and self.has_fast and model == "base"
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        guidance = params["guidance"] or (1.0 if fast else FULL_GUIDANCE)  # ≤ 1: no guidance pass
        width, height = params["width"], params["height"]

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, steps)
            if (i + 1) % PREVIEW_EVERY == 0 and i + 1 < steps:
                try:
                    sink.preview(self._preview(kwargs["latents"], width, height), i + 1, steps)
                except Exception:  # noqa: BLE001 — a preview never fails a job
                    pass
            return kwargs

        wanted = FLASH if fast else model
        if self.face_loaded and wanted != self.variant:
            self._unload_face()  # the adapter's attention weights aren't in the other transformer's files
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
        negative = params["negative"].strip()
        if guidance > 1 and negative:
            extra_args["negative_prompt"] = negative
        sink.status("Sampling" + (" (fast)" if fast else ""))
        start = time.monotonic()
        with torch.inference_mode():
            image = self.pipe(prompt=params["prompt"], width=width, height=height, num_inference_steps=steps,
                              guidance_scale=guidance, generator=generator(params["seed"]),
                              callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"],
                              **extra_args).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "guidance": guidance, "fast": fast, "loras": loras, "face": bool(face),
                             "sampleSeconds": round(time.monotonic() - start, 2)})

    def _load_face(self, sink) -> None:
        if self.face_loaded:
            return
        import transformers

        root = self.models_root / FACE_DIR
        if not (root / ".complete").exists():
            raise ValueError("The face reference model for Chroma isn't on this GPU server yet: download chroma-face "
                             "(python -m aistudio_gpu download --models chroma-face).")
        sink.status("Loading the face reference")
        if self.image_encoder is None:
            self.image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(
                root / "image_encoder", dtype=self.dtype).to(self.device)
        self.pipe.load_ip_adapter(str(root / "adapter"), weight_name="ip_adapter.safetensors",
                                  image_encoder_pretrained_model_name_or_path=None)
        self.pipe.register_modules(image_encoder=self.image_encoder)
        if getattr(self.pipe, "feature_extractor", None) is None:
            self.pipe.register_modules(
                feature_extractor=transformers.CLIPImageProcessor.from_pretrained(root / "image_encoder"))
        self.face_loaded = True

    def _unload_face(self) -> None:
        self.pipe.unload_ip_adapter()
        self.face_loaded = False

    def _preview(self, latents, width: int, height: int) -> np.ndarray:
        import torch

        pipe = self.pipe
        vae = pipe.vae
        unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        unpacked = unpacked / vae.config.scaling_factor + vae.config.shift_factor
        with torch.inference_mode():
            image = vae.decode(unpacked.to(vae.dtype), return_dict=False)[0]  # [1, 3, H, W] in -1…1
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)
