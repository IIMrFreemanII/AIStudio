"""FLUX.1 dev text-to-image (diffusers ``FluxPipeline``), bf16, with up to four LoRAs per job and fine-tunes from
links (``adapters.py``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_png
from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import generator, pick_device

PREVIEW_EVERY = 4  # a VAE decode per preview: every few steps is plenty


class FluxWorker(AdapterSlots):
    family = "flux.1"
    title = "FLUX.1 dev"

    def __init__(self):
        self.pipe = None
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import FluxPipeline

        device, dtype = pick_device()
        status("Loading FLUX.1 dev")
        pipe = FluxPipeline.from_pretrained(models_root / "flux1-dev", dtype=dtype)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.models_root = models_root

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        steps, width, height = params["steps"], params["width"], params["height"]

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, steps)
            if (i + 1) % PREVIEW_EVERY == 0 and i + 1 < steps:
                try:
                    sink.preview(self._preview(kwargs["latents"], width, height), i + 1, steps)
                except Exception:  # noqa: BLE001
                    pass
            return kwargs

        self._use_variant(params.get("model") or "base", sink)
        loras = self._use_loras(params, inputs, sink)
        sink.status("Sampling")
        start = time.monotonic()
        with torch.inference_mode():
            image = self.pipe(prompt=params["prompt"], width=width, height=height, num_inference_steps=steps,
                              guidance_scale=params["guidance"], generator=generator(params["seed"]),
                              callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"]).images[0]
        path = out_dir / "result.png"
        write_png(np.asarray(image), path)
        return Output(path, "image", "image/png", extra={"sampleSeconds": round(time.monotonic() - start, 2),
                                                          "loras": loras})

    def _preview(self, latents, width: int, height: int) -> np.ndarray:
        import torch

        pipe = self.pipe
        vae = pipe.vae
        unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        unpacked = unpacked / vae.config.scaling_factor + vae.config.shift_factor
        with torch.inference_mode():
            image = vae.decode(unpacked.to(vae.dtype), return_dict=False)[0]  # [1, 3, H, W] in -1…1
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)
