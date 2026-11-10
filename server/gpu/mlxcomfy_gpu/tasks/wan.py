"""Wan 2.2 A14B text-to-video and image-to-video (diffusers ``WanPipeline`` / ``WanImageToVideoPipeline``).

Two 14B transformers: ``transformer`` denoises the high-noise steps, ``transformer_2`` the low-noise ones. On CUDA
both stay resident with fp8 weights (cast to bf16 per layer), ≈ 29 GB instead of 57 GB. The Lightning LoRAs
(``fast``) give a usable clip in 4 steps without classifier-free guidance — about 10× fewer GPU seconds.

The negative prompt: with guidance above 1 (the full model's default) classifier-free guidance applies it, at twice
the cost per step. Without guidance (``fast``'s default) Normalized Attention Guidance (``nag.py``) applies it inside
the cross-attention instead, for a few percent more.

Fine-tunes from links and up to four LoRAs per job (``adapters.py``): a LoRA goes to both experts, to one, or comes
as a high/low pair; a fine-tune replaces one expert's weights or both.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import to_uint8, write_mp4
from . import Output, TaskSpec, nag
from .adapters import AdapterSlots
from .device import fp8_storage_supported, generator, pick_device

FPS = 16
# Wan 2.2's defaults without the LoRAs (the model cards): steps, guidance (high noise), guidance (low noise).
FULL = {"wan22-t2v": (40, 4.0, 3.0), "wan22-i2v": (40, 3.5, 3.5)}
LIGHTNING_SHIFT = 5.0


class WanWorker(AdapterSlots):
    components = ("transformer", "transformer_2")
    title = "Wan 2.2"

    def __init__(self, family: str):
        self.task_family = family
        self.family = "wan-2.2-i2v" if family == "wan22-i2v" else "wan-2.2-t2v"  # the library's (variants.py)
        self.pipe = None
        self.has_fast = False
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline, WanPipeline

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        self.models_root = models_root
        t0 = time.monotonic()

        def lap(what: str) -> None:  # load-time breakdown in the server log
            nonlocal t0
            print(f"  {what}: {time.monotonic() - t0:.1f} s", flush=True)
            t0 = time.monotonic()

        cls = WanImageToVideoPipeline if self.task_family == "wan22-i2v" else WanPipeline
        status("Loading Wan 2.2")
        pipe = cls.from_pretrained(models_root / self.task_family, dtype=dtype)
        # The VAE is small and decodes better in float32 (as the Wan reference code does).
        pipe.vae.to(torch.float32)
        lap("pipeline from disk")

        lora_dir = models_root / "wan22-lightning" / ("i2v" if self.task_family == "wan22-i2v" else "t2v")
        if (lora_dir / "high_noise_model.safetensors").exists():
            status("Loading Lightning LoRAs")
            # The same adapter name in both transformers, so one set_adapters/disable_lora switches both.
            pipe.load_lora_weights(lora_dir, weight_name="high_noise_model.safetensors", adapter_name="fast")
            pipe.load_lora_weights(lora_dir, weight_name="low_noise_model.safetensors", adapter_name="fast",
                                   load_into_transformer_2=True)
            self.has_fast = True
            lap("Lightning LoRAs")

        if fp8_storage_supported(device):
            from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

            skip = tuple(DEFAULT_SKIP_MODULES_PATTERN) + ("lora_",)  # LoRA weights stay in bf16
            for t in (pipe.transformer, pipe.transformer_2):
                t.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype,
                                           skip_modules_pattern=skip)
            lap("fp8 casting")
        pipe.to(device)
        lap(f"to {device}")
        pipe.set_progress_bar_config(disable=True)
        self.nag = nag.install((pipe.transformer, pipe.transformer_2))
        self.full_scheduler = pipe.scheduler
        self.fast_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            {"num_train_timesteps": pipe.scheduler.config.get("num_train_timesteps", 1000)}, shift=LIGHTNING_SHIFT)
        self.pipe = pipe

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        pipe = self.pipe
        fast = params["fast"] and self.has_fast
        full_steps, full_g1, full_g2 = FULL[self.task_family]
        self._use_variant(params.get("model") or "base", sink)
        loras = self._use_loras(params, inputs, sink, pinned={"fast": 1.0} if fast else None)
        if fast:
            pipe.scheduler = self.fast_scheduler
            steps = params["steps"] or 4
            g1 = g2 = params["guidance"] or 1.0
        else:
            pipe.scheduler = self.full_scheduler
            steps = params["steps"] or full_steps
            g1 = params["guidance"] or full_g1
            g2 = params["guidance"] or full_g2
        # Wan wants 4k+1 frames.
        length = max(5, (params["length"] - 1) // 4 * 4 + 1)
        width, height = params["width"], params["height"]

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, steps)
            if i + 1 < steps:  # the last step's frames are the result itself
                try:
                    sink.preview(self._preview(kwargs["latents"]), i + 1, steps)
                except Exception:  # noqa: BLE001 — a preview never fails a job
                    pass
            return kwargs

        args = dict(width=width, height=height, num_frames=length, num_inference_steps=steps, guidance_scale=g1,
                    guidance_scale_2=g2, generator=generator(params["seed"]), output_type="np",
                    callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"])
        negative = params["negative"].strip()
        if not negative:
            args["prompt"], how = params["prompt"], "none"
        elif g1 > 1:  # the pipeline runs classifier-free guidance whenever the first guidance scale is above 1
            args["prompt"], args["negative_prompt"], how = params["prompt"], negative, "cfg"
        else:
            with torch.inference_mode():
                pos, neg = pipe.encode_prompt(prompt=params["prompt"], negative_prompt=negative,
                                              do_classifier_free_guidance=True, max_sequence_length=nag.TEXT_LEN,
                                              device=self.device)
            args["prompt_embeds"], how = torch.cat([pos, neg], dim=1), "nag"  # split again in each cross-attention
        if self.task_family == "wan22-i2v":
            args["image"] = fit_image(inputs["image"], width, height)
        sink.status("Sampling" + (" (fast)" if fast else ""))
        start = time.monotonic()
        with torch.inference_mode():
            frames = pipe(**args).frames[0]  # [T, H, W, 3] float in 0…1
        sink.status("Encoding video")
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS)
        return Output(path, "video", "video/mp4",
                      extra={"steps": steps, "fast": fast, "frames": int(len(frames)), "negative": how, "loras": loras,
                             "sampleSeconds": round(time.monotonic() - start, 2)})

    def _preview(self, latents) -> np.ndarray:
        """The first frame of the current latents through the VAE (Wan's VAE decodes one latent frame to one
        video frame)."""
        import torch

        vae = self.pipe.vae
        first = latents[:, :, :1].to(torch.float32)
        mean = torch.tensor(vae.config.latents_mean, device=first.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=first.device).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            frame = vae.decode(first * std + mean, return_dict=False)[0]  # [1, 3, 1, H, W] in -1…1
        return to_uint8((frame[0, :, 0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0)


def fit_image(path: Path, width: int, height: int):
    """The start image scaled to cover width×height and centre-cropped (the clip's size wins)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    scale = max(width / img.width, height / img.height)
    img = img.resize((max(width, round(img.width * scale)), max(height, round(img.height * scale))), Image.LANCZOS)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))
