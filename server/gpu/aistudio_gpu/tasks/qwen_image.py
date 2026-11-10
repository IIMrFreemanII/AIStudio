"""Qwen-Image (the original 20B MMDiT, diffusers ``QwenImagePipeline``) text-to-image, with up to four LoRAs per job
and fine-tunes from links (``adapters.py``).

Memory: the transformer is ≈ 41 GB in bf16 and the Qwen2.5-VL text encoder ≈ 17 GB. On an 80 GB card everything
stays bf16; on a smaller one (an L40S, 48 GB) the transformer's weights are stored in fp8 and cast to bf16 per layer.

``fast`` uses the Lightning 8-step LoRA without guidance (≈ 6× fewer GPU seconds); otherwise the model card's 50
steps with true classifier-free guidance at 4. The negative prompt needs guidance (``cfg`` above 1): fast mode
ignores it unless ``cfg`` is raised, which doubles the cost per step. (NAG, which Wan uses without guidance, needs a
separate cross-attention; Qwen-Image's attention is joint.)
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_png
from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import fp8_storage_supported, generator, pick_device

PREVIEW_EVERY = 4
FULL_STEPS, FULL_CFG = 50, 4.0
FAST_STEPS = 8
BF16_MEMORY = 70e9  # GPU memory above which the transformer stays bf16
# The scheduler the Lightning LoRA was distilled with (its README: shift 3, exponential time shift).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


class QwenImageWorker(AdapterSlots):
    family = "qwen-image"
    title = "Qwen-Image"

    def __init__(self):
        self.pipe = None
        self.has_fast = False
        self.fp8 = False
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        self.models_root = models_root
        status("Loading Qwen-Image")
        pipe = QwenImagePipeline.from_pretrained(models_root / "qwen-image", dtype=dtype)
        lightning = models_root / "qwen-image-lightning" / "8steps.safetensors"
        if lightning.exists():
            status("Loading the Lightning LoRA")
            pipe.load_lora_weights(str(lightning.parent), weight_name=lightning.name, adapter_name="fast")
            self.has_fast = True
        small = device != "cuda" or torch.cuda.get_device_properties(0).total_memory < BF16_MEMORY
        if fp8_storage_supported(device) and small:
            from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype,
                skip_modules_pattern=tuple(DEFAULT_SKIP_MODULES_PATTERN) + ("lora_",))
            self.fp8 = True
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.full_scheduler = pipe.scheduler
        self.fast_scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
        self.pipe = pipe

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        pipe = self.pipe
        fast = params["fast"] and self.has_fast
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        cfg = params["cfg"] or (1.0 if fast else FULL_CFG)
        width, height = params["width"], params["height"]

        self._use_variant(params.get("model") or "base", sink)
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
        args = dict(prompt=params["prompt"], width=width, height=height, num_inference_steps=steps,
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
                             "fp8": self.fp8, "sampleSeconds": round(time.monotonic() - start, 2)})

    def _preview(self, latents, width: int, height: int) -> np.ndarray:
        import torch

        pipe, vae = self.pipe, self.pipe.vae
        unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(vae.dtype)  # [1, C, 1, H, W]
        mean = torch.tensor(vae.config.latents_mean, device=unpacked.device, dtype=unpacked.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=unpacked.device, dtype=unpacked.dtype).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            image = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]  # [1, 3, H, W] in -1…1
        return to_uint8((image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)
