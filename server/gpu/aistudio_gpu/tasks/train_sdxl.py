"""A character LoRA for SDXL, trained on the GPU from a zip of pictures and captions (``training.py``).

The base is SDXL or one of its fine-tunes from the library (``model``, swapped in as the SDXL worker does), so the
LoRA fits the model the character is drawn with. Every part stays on the GPU. Each picture's VAE latents and text
embeddings are computed once (at an aspect bucket near 1 MP, plus an empty caption for dropout); then a PEFT LoRA on
the UNet's attention layers trains with AdamW, bf16 autocast, gradient checkpointing and min-SNR loss weighting.

The result is a diffusers-format ``.safetensors`` (``StableDiffusionXLPipeline.save_lora_weights``), which the SDXL
worker's LoRA slots load like any other. The adapter is removed after the job, so the container's next job starts
from the plain model.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from . import Output, TaskSpec
from .adapters import AdapterSlots
from .device import pick_device
from .training import bucket, fit, read_dataset, total_steps

ADAPTER = "character"
TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]
CAPTION_DROPOUT = 0.05
MIN_SNR_GAMMA = 5.0
PROGRESS_EVERY = 5


class SDXLTrainWorker(AdapterSlots):
    family = "sdxl"
    title = "SDXL"
    components = ("unet",)
    variant_components = ("unet", "text_encoder", "text_encoder_2")

    def __init__(self):
        self.pipe = None
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        from diffusers import DDPMScheduler, StableDiffusionXLPipeline

        device, dtype = pick_device()
        self.device, self.dtype = device, dtype
        status("Loading SDXL")
        pipe = StableDiffusionXLPipeline.from_pretrained(models_root / "sdxl", variant="fp16", dtype=dtype)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
            m.requires_grad_(False)
        self.noise = DDPMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe
        self.models_root = models_root

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        samples = read_dataset(inputs["dataset"])
        self._use_variant(params.get("model") or "base", sink)
        pipe, device = self.pipe, self.device
        steps = total_steps(len(samples), params["steps"], 80, 1200, 2400)
        area = params["resolution"] ** 2
        rng = random.Random(params["seed"])
        torch.manual_seed(params["seed"])

        sink.status(f"Preparing {len(samples)} pictures")
        cached = []
        with torch.no_grad():
            empty = self._encode("")
            for i, s in enumerate(samples):
                sink.check()
                w, h = bucket(s.image.width, s.image.height, area)
                pixels = torch.from_numpy(_to_array(fit(s.image, (w, h)))).to(device, pipe.vae.dtype)
                latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
                time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=device, dtype=self.dtype)
                cached.append((latents.to(self.dtype), self._encode(s.caption), time_ids))
                sink.progress(i + 1, len(samples))

        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        unet = pipe.unet
        unet.add_adapter(LoraConfig(r=params["rank"], lora_alpha=params["rank"] // 2, init_lora_weights="gaussian",
                                    target_modules=TARGETS), adapter_name=ADAPTER)
        trainable = [p for p in unet.parameters() if p.requires_grad]
        for p in trainable:
            p.data = p.data.float()
        optimizer = torch.optim.AdamW(trainable, lr=params["learning_rate"], weight_decay=1e-2)
        unet.enable_gradient_checkpointing()
        unet.train()
        alphas = self.noise.alphas_cumprod.to(device)
        v_prediction = self.noise.config.prediction_type == "v_prediction"
        autocast = torch.autocast(device_type=device if device in ("cuda", "cpu") else "cpu", dtype=torch.bfloat16,
                                  enabled=device == "cuda")
        sink.status("Training")
        t0, losses = time.monotonic(), []
        try:
            for step in range(steps):
                sink.check()
                latents, (embeds, pooled), time_ids = rng.choice(cached)
                if rng.random() < CAPTION_DROPOUT:
                    embeds, pooled = empty
                noise = torch.randn_like(latents)
                t = torch.randint(0, self.noise.config.num_train_timesteps, (1,), device=device)
                noisy = self.noise.add_noise(latents, noise, t)
                with autocast:
                    pred = unet(noisy, t, encoder_hidden_states=embeds,
                                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                target = self.noise.get_velocity(latents, noise, t) if v_prediction else noise
                snr = alphas[t] / (1 - alphas[t])
                weight = torch.clamp(snr, max=MIN_SNR_GAMMA) / (snr + 1 if v_prediction else snr)
                loss = (weight * ((pred.float() - target.float()) ** 2).mean()).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
                if (step + 1) % PROGRESS_EVERY == 0 or step + 1 == steps:
                    sink.progress(step + 1, steps)
            sink.status("Saving the LoRA")
            from diffusers import StableDiffusionXLPipeline
            from diffusers.utils import convert_state_dict_to_diffusers

            layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=ADAPTER))
            layers = {k: v.to(torch.float16).contiguous() for k, v in layers.items()}
            StableDiffusionXLPipeline.save_lora_weights(out_dir, unet_lora_layers=layers, safe_serialization=True)
        finally:
            unet.eval()
            unet.disable_gradient_checkpointing()
            unet.delete_adapters(ADAPTER)
        path = out_dir / "pytorch_lora_weights.safetensors"
        seconds = time.monotonic() - t0
        tail = losses[-50:] or [0.0]
        return Output(path, "file", "application/octet-stream",
                      extra={"pictures": len(samples), "steps": steps, "rank": params["rank"],
                             "trainSeconds": round(seconds, 1), "loss": round(sum(tail) / len(tail), 5),
                             "model": params.get("model") or "base"})

    def _encode(self, caption: str):
        """(prompt embeddings, pooled embedding) for one caption, as SDXL's pipeline makes them."""
        embeds, _, pooled, _ = self.pipe.encode_prompt(caption, device=self.device, num_images_per_prompt=1,
                                                       do_classifier_free_guidance=False)
        return embeds.to(self.dtype), pooled.to(self.dtype)


def _to_array(image):
    """A PIL picture as a [1, 3, H, W] float array in −1…1."""
    import numpy as np

    a = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return a.transpose(2, 0, 1)[None].copy()
