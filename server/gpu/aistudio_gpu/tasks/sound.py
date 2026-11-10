"""Stable Audio 3 medium: sound effects, ambience and instrumental music from text (diffusers
``StableAudio3Pipeline``), 44.1 kHz stereo, up to about six minutes.

The distilled checkpoint samples in 8 steps with its guidance baked in, so there is no guidance or negative prompt.
The weights are the diffusers conversion that ``download.py`` makes from Stability's checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..media import write_wav
from . import Output, TaskSpec
from .device import generator, pick_device


class SoundWorker:
    def __init__(self):
        self.pipe = None

    def load(self, models_root: Path, status) -> None:
        from diffusers import StableAudio3Pipeline

        device, dtype = pick_device()
        status("Loading Stable Audio 3")
        pipe = StableAudio3Pipeline.from_pretrained(models_root / "stable-audio-3", dtype=dtype)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.sample_rate = int(pipe.vae.config.sampling_rate)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        steps = params["steps"] or None  # None: the scheduler's own default (8 for the distilled model)
        total = steps or 8

        def on_step(p, i, t, kwargs):
            sink.check()
            sink.progress(i + 1, total)
            return kwargs

        sink.status("Sampling")
        start = time.monotonic()
        with torch.inference_mode():
            audio = self.pipe(prompt=params["prompt"], duration=params["seconds"], num_inference_steps=steps,
                              generator=generator(params["seed"]), callback_on_step_end=on_step).audios[0]
        path = out_dir / "result.wav"
        write_wav(audio.float().cpu().numpy(), self.sample_rate, path)
        return Output(path, "audio", "audio/wav",
                      extra={"sampleRate": self.sample_rate, "sampleSeconds": round(time.monotonic() - start, 2)})
