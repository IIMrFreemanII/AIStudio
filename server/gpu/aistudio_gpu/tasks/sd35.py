"""Stable Diffusion 3.5 Large text-to-image (diffusers ``StableDiffusion3Pipeline``), bf16, with up to four LoRAs per
job. There are no fine-tunes of it to install.

``fast`` runs Large Turbo (4 steps, no guidance); otherwise Large (28 steps, guidance 3.5, a negative prompt used).
Turbo is the default, so it loads first when installed; the two share the text encoders, the VAE and the scheduler
and differ only in the transformer, which is swapped in place from the models Volume when a job asks for the other
one (loaded LoRAs stay, they wrap the layers whatever their weights are).

Memory: the transformer ≈ 16 GB and the three text encoders ≈ 11 GB, all bf16: it fits a 48 GB L40S.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_png
from . import Output, TaskSpec, adapters
from .adapters import AdapterSlots
from .device import generator, pick_device
from .previews import remember_x0

PREVIEW_EVERY = 4
FAST_STEPS, FULL_STEPS, FULL_GUIDANCE = 4, 28, 3.5
FOLDERS = {"turbo": "sd35-large-turbo", "large": "sd35-large"}


class SD35Worker(AdapterSlots):
    family = "sd3.5-large"  # the library family its LoRAs belong to
    title = "SD 3.5 Large"

    def __init__(self):
        self.pipe = None
        self.has_fast = False
        self.current: str | None = None  # "turbo" or "large": the transformer weights loaded (None: half-loaded)
        self.x0: dict = {}  # the last step's estimate of the finished latents, for previews (``previews.remember_x0``)
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        self.models_root = models_root
        extra = {}
        turbo = models_root / "sd35-large-turbo"
        if (turbo / ".complete").exists():  # the default mode: start with Turbo's transformer
            status("Loading SD 3.5 Large Turbo")
            extra["transformer"] = SD3Transformer2DModel.from_pretrained(turbo / "transformer", dtype=dtype)
            self.current, self.has_fast = "turbo", True
        else:
            status("Loading SD 3.5 Large")
            self.current, self.has_fast = "large", False
        pipe = StableDiffusion3Pipeline.from_pretrained(models_root / "sd35-large", dtype=dtype, **extra)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        remember_x0(pipe.scheduler, self.x0)
        self.pipe = pipe

    def _use_transformer(self, which: str, sink) -> None:
        """Makes the loaded transformer the other model's: its weights parked on the GPU swap in (both models stay
        there on an H100, ``adapters.WeightPark``); without room, they're copied from disk in place. A job in the same
        mode costs nothing."""
        if which == self.current:
            return
        src = self.models_root / FOLDERS[which] / "transformer"
        current, self.current = self.current, None  # half-loaded if this fails: the next job reloads
        if self.park.switch(self.pipe.transformer, current, which, src, sink):
            self.current = which
            return
        sink.status("Loading SD 3.5 Large Turbo" if which == "turbo" else "Loading SD 3.5 Large")
        adapters._copy_weights(self.pipe.transformer, src, which)
        self.current = which

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        fast = params["fast"] and self.has_fast
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        guidance = params["guidance"] or (0.0 if fast else FULL_GUIDANCE)
        width, height = params["width"], params["height"]
        negative = params["negative"].strip()

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

        self._use_transformer("turbo" if fast else "large", sink)
        loras = self._use_loras(params, inputs, sink)
        args = dict(prompt=params["prompt"], width=width, height=height, num_inference_steps=steps,
                    guidance_scale=guidance, max_sequence_length=256, generator=generator(params["seed"]),
                    callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"])
        if guidance > 1:  # classifier-free guidance runs only above 1
            args["negative_prompt"] = negative or None
        how = "cfg" if guidance > 1 and negative else ("ignored" if negative else "none")
        self.x0.clear()
        sink.status("Sampling" + (" (fast)" if fast else ""))
        start = time.monotonic()
        with torch.inference_mode():
            image = self.pipe(**args).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png",
                      extra={"steps": steps, "guidance": guidance, "fast": fast, "negative": how, "loras": loras,
                             "sampleSeconds": round(time.monotonic() - start, 2)})

    def _preview(self, latents) -> np.ndarray:
        import torch

        vae = self.pipe.vae
        scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
        with torch.inference_mode():
            image = vae.decode(scaled.to(vae.dtype), return_dict=False)[0]  # [1, 3, H, W] in -1…1
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)
