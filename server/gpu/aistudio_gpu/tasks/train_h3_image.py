"""A character LoRA for MiniMax H3 Image, trained on the GPU from a zip of pictures and captions (``training.py``).

The checkpoint is distilled to six fixed steps, each with its own output head (``pdd_step_index``), and its sampler
re-noises from the estimate at every step (``stochastic_sampling``): step i sees ``x = (1 − σᵢ)·x₀ + σᵢ·ε`` and the
transformer predicts the velocity ``v = x₀ − ε`` (``denoised = x + σ·v``). So each step trains on its own, with the
pipeline's own inputs for it:

* **Inputs.** The pipeline runs up to its denoise loop and stops there (``capture``): a text-to-image run with the
  caption gives the packed noise rows, the prompt embeddings, each step's timestep plan and the sequence description
  (token tags, positions, row indices); an edit run with the picture as ``<Picture 1>`` gives the picture's clean
  latent rows (its reference rows), at the same canvas: a short edge of 1024 (``reference_size``), the size at
  which the edit path keeps a reference as it is.
* **Training.** A PEFT LoRA on the transformer blocks' attention and feed-forward layers (``alpha = rank``: the file
  loads at scale 1, as alpha-less H3 LoRAs do), AdamW, bf16 autocast, gradient checkpointing; each step a random
  picture and a random one of the six steps.

Every part stays on the GPU, as in ``minimax_h3_image.py`` (a B200). The result is the LoRA in diffusers layout
(``transformer.``-prefixed), which the H3 Image task's LoRA slots load; the adapter is removed after the job.
"""

from __future__ import annotations

import inspect
import random
import time
from dataclasses import dataclass
from pathlib import Path

from . import Output, TaskSpec
from .minimax_h3_image import H3ImageWorker
from .training import fit, read_dataset, total_steps

ADAPTER = "character"
TARGETS = r".*transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
PROGRESS_EVERY = 5


@dataclass
class Prepared:
    clean: object  # [rows, C] latent rows of the picture
    embeds: object
    plan: list  # per step: (unique timesteps, timestep indices)
    fields: dict  # the transformer's sequence description
    sigmas: list[float]  # per step


class Captured(Exception):
    def __init__(self, block_state, scheduler):
        super().__init__("captured")
        self.state = block_state
        self.sigmas = [float(s) for s in scheduler.sigmas]


def capture(pipe, denoise, **kwargs) -> Captured:
    """Runs the modular pipeline up to its denoise loop and returns what the loop would have read."""
    wrapper = denoise.MiniMaxH3ImageDenoiseLoopWrapper
    original = wrapper.__call__

    def grab(self, components, state):
        raise Captured(self.get_block_state(state), components.scheduler)

    wrapper.__call__ = grab
    try:
        pipe(**kwargs, output="images")
    except Captured as c:
        return c
    finally:
        wrapper.__call__ = original
    raise RuntimeError("MiniMax H3 Image didn't reach its denoise loop")


def reference_size(width: int, height: int, short: int, multiple: int = 32) -> tuple[int, int]:
    """The training size of a picture: its aspect ratio with a short edge of ``short`` (1024–2048), both sides in
    multiples of 32. The edit path keeps a reference whose short edge is within 1024–2048 at its own size and the
    canvas rounds to 32, so at this size the picture's reference rows and a generation's rows line up (at 832×1248 the
    reference was enlarged to 1024×1536: 1536 rows against 1014)."""
    short = min(max(short, 1024), 2048)
    scale = short / min(width, height)
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def noisy_and_target(clean, noise, sigma: float):
    """The step's input and its velocity target: x = (1 − σ)·x₀ + σ·ε, v = x₀ − ε."""
    return (1.0 - sigma) * clean + sigma * noise, clean - noise


def lora_file_keys(state: dict) -> dict:
    """PEFT's adapter tensors (``transformer_blocks.0.attn.to_q.lora_A.weight``) under the ``transformer.`` prefix the
    H3 LoRA loader reads as diffusers layout."""
    return {k if k.startswith("transformer.") else f"transformer.{k}": v for k, v in state.items()}


class H3ImageTrainWorker(H3ImageWorker):
    """Loads as the H3 Image worker (all on the GPU), then trains instead of drawing."""

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        samples = read_dataset(inputs["dataset"])
        pipe, transformer = self.pipe._pipe, self.pipe.transformer
        steps = total_steps(len(samples), params["steps"], 50, 800, 1600)
        rng = random.Random(params["seed"])
        torch.manual_seed(params["seed"])

        sink.status(f"Preparing {len(samples)} pictures")
        prepared: list[Prepared] = []
        for i, s in enumerate(samples):
            sink.check()
            w, h = reference_size(s.image.width, s.image.height, params["resolution"])
            picture = fit(s.image, (w, h))
            common = {"width": w, "height": h, "min_generation_area": w * h,
                      "generator": torch.Generator().manual_seed(params["seed"] + i), "output_type": "pil"}
            caption = s.caption or "a photo"
            with torch.no_grad():
                text = capture(pipe, self._denoise, prompt=caption, **common)
                edit = capture(pipe, self._denoise, prompt="<Picture 1> " + caption, image=[picture], **common)
            refs = int(getattr(edit.state, "num_reference_rows", 0) or 0)
            clean = edit.state.latents[:refs]
            if refs == 0 or clean.shape != text.state.latents.shape:
                raise RuntimeError(f"{s.name}: its latent rows ({tuple(clean.shape)}) don't match a generation's "
                                   f"({tuple(text.state.latents.shape)}) at {w}×{h}")
            prepared.append(Prepared(clean.detach().to(torch.bfloat16), text.state.prompt_embeds.detach(),
                                     list(text.state.row_timestep_plan), dict(text.state.denoiser_input_fields),
                                     [text.sigmas[k] for k in range(len(text.state.row_timestep_plan))]))
            sink.progress(i + 1, len(samples))

        transformer.requires_grad_(False)
        transformer.add_adapter(LoraConfig(r=params["rank"], lora_alpha=params["rank"], init_lora_weights="gaussian",
                                           target_modules=TARGETS), adapter_name=ADAPTER)
        trainable = [p for p in transformer.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("the LoRA found no layers to train in MiniMax H3 Image")
        for p in trainable:
            p.data = p.data.float()
        optimizer = torch.optim.AdamW(trainable, lr=params["learning_rate"], weight_decay=1e-2)
        if getattr(transformer, "_supports_gradient_checkpointing", False):
            transformer.enable_gradient_checkpointing()
        transformer.train()
        args = set(inspect.signature(transformer.forward).parameters)
        sink.status("Training")
        t0, losses = time.monotonic(), []
        try:
            for step in range(steps):
                sink.check()
                p = rng.choice(prepared)
                k = rng.randrange(len(p.plan))
                noise = torch.randn_like(p.clean, dtype=torch.float32)
                x, target = noisy_and_target(p.clean.float(), noise, p.sigmas[k])
                unique, indices = p.plan[k]
                fields = {n: v for n, v in p.fields.items() if n in args}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = transformer(hidden_states=x.to(torch.bfloat16)[None], encoder_hidden_states=p.embeds,
                                       timestep=unique, timestep_indices=indices, pdd_step_index=k,
                                       return_dict=False, **fields)[0]
                loss = torch.nn.functional.mse_loss(pred[0].float(), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
                if (step + 1) % PROGRESS_EVERY == 0 or step + 1 == steps:
                    sink.progress(step + 1, steps)
            sink.status("Saving the LoRA")
            state = get_peft_model_state_dict(transformer, adapter_name=ADAPTER)
            path = out_dir / "character_lora.safetensors"
            save_file({k: v.to(torch.bfloat16).contiguous() for k, v in lora_file_keys(state).items()}, str(path))
        finally:
            transformer.eval()
            if getattr(transformer, "_supports_gradient_checkpointing", False):
                transformer.disable_gradient_checkpointing()
            transformer.delete_adapters(ADAPTER)
            for p in transformer.parameters():
                p.requires_grad_(False)
        tail = losses[-50:] or [0.0]
        return Output(path, "file", "application/octet-stream",
                      extra={"pictures": len(samples), "steps": steps, "rank": params["rank"],
                             "trainSeconds": round(time.monotonic() - t0, 1), "loss": round(sum(tail) / len(tail), 5)})
