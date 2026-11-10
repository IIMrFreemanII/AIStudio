"""CPU test tasks: a moving gradient, with progress, previews, cancellation and an optional failure. They exercise
the whole job path (upload, events, result download) without a GPU or models."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from ..media import read_wav, tone, write_mp4, write_png, write_wav
from . import Output, TaskSpec


def pattern(width: int, height: int, t: float, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    phase = (seed % 97) / 97.0
    r = (x / max(width - 1, 1) + t + phase) % 1.0
    g = (y / max(height - 1, 1) + 0.5 * t) % 1.0
    b = np.full_like(r, 0.5 + 0.5 * np.sin(2 * np.pi * (t + phase)))
    return np.stack([r, g, b], axis=-1)


class TestPatternWorker:
    def load(self, models_root: Path, status) -> None:
        pass

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        steps = params["steps"]
        audio = task.kind == "audio"
        w, h = (0, 0) if audio else (params["width"], params["height"])
        sink.status("Sampling")
        for step in range(1, steps + 1):
            sink.check()
            time.sleep(params["step_seconds"])
            if params.get("fail_at") and step == params["fail_at"]:
                raise RuntimeError(f"test failure at step {step}")
            sink.progress(step, steps)
            if not audio:
                sink.preview(pattern(w, h, step / steps, params["seed"]), step, steps)
        sink.status("Encoding")
        if audio:
            return _audio(params["seconds"], params["sample_rate"], params["channels"], params["seed"], inputs,
                          out_dir)
        if task.kind == "video":
            n = params["length"]
            frames = np.stack([pattern(w, h, i / max(n, 1), params["seed"]) for i in range(n)])
            path = out_dir / "result.mp4"
            write_mp4(frames, path, fps=16)
            return Output(path, "video", "video/mp4")
        path = out_dir / "result.png"
        write_png(pattern(w, h, 0.0, params["seed"]), path)
        return Output(path, "image", "image/png")


class FakeModelWorker:
    """Stands in for a GPU model (``AISTUDIO_FAKE_MODELS=1``): the task's own parameters, steps, previews and output
    kind, from a test pattern on the CPU. Image-to-video starts from the uploaded image, so a client can check the
    upload arrived. For testing clients end to end without models."""

    def load(self, models_root: Path, status) -> None:
        pass

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        from PIL import Image

        steps = params.get("steps") or 4
        sink.status("Sampling")
        for step in range(1, steps + 1):
            sink.check()
            time.sleep(0.02)
            sink.progress(step, steps)
            if task.kind != "audio":
                sink.preview(pattern(params["width"], params["height"], step / steps, params["seed"]), step, steps)
        if task.kind == "audio":
            # Each audio model's own rate and channels; length from its seconds (speech: from the text).
            rate, channels = {"music": (48000, 2), "sound": (44100, 2)}.get(task.family, (24000, 1))
            seconds = params.get("seconds") or min(10.0, max(1.0, len(params.get("text", "")) / 15))
            return _audio(seconds, rate, channels, params["seed"], inputs, out_dir)
        w, h = params["width"], params["height"]
        # LoRAs with a strength: the top row's brightness is their count / 4 (Wan pairs' low-noise files: the second
        # row's); a fine-tune (`model` other than base): the right-hand column turns white. So a client can check they
        # arrived. In video, 8-pixel bands.
        loras = sum(1 for k in inputs if re.fullmatch(r"lora\d", k) and params.get(f"{k}_strength", 1) != 0)
        pairs = sum(1 for k in inputs if re.fullmatch(r"lora\d_low", k))

        band = 1 if task.kind == "image" else 8  # video: 8-pixel bands, so they survive H.264

        def mark(image):
            if loras:
                image[:band] = loras / 4
            if pairs:
                image[band:2 * band] = pairs / 4
            if params.get("model") not in (None, "", "base"):
                image[:, -band:] = 1.0
            return image

        if task.kind == "image":
            image = mark(pattern(w, h, 0.0, params["seed"]))
            path = out_dir / "result.png"
            write_png(image, path)
            return Output(path, "image", "image/png", extra={"loras": loras})
        n = params.get("length", 9)
        frames = np.stack([pattern(w, h, i / max(n, 1), params["seed"]) for i in range(n)])
        if "image" in inputs:
            start = Image.open(inputs["image"]).convert("RGB").resize((w, h))
            frames[0] = np.asarray(start, dtype=np.float32) / 255.0
        frames = np.stack([mark(f) for f in frames])
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=16)
        return Output(path, "video", "video/mp4")


def _audio(seconds: float, rate: int, channels: int, seed: int, inputs: dict[str, Path], out_dir: Path) -> Output:
    """The test tone; an uploaded ``reference`` clip is played first, so a client can check the upload arrived."""
    samples = tone(seconds, rate, channels, seed)
    if "reference" in inputs:
        ref, ref_rate = read_wav(inputs["reference"])
        if ref_rate == rate:
            ref = ref[:1].repeat(channels, axis=0) if ref.shape[0] != channels else ref
            samples = np.concatenate([ref, samples], axis=1)
    path = out_dir / "result.wav"
    write_wav(samples, rate, path)
    return Output(path, "audio", "audio/wav", extra={"sampleRate": rate, "channels": channels})
