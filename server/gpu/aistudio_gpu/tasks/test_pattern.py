"""CPU test tasks: a moving gradient, with progress, previews, cancellation and an optional failure. They exercise
the whole job path (upload, events, result download) without a GPU or models."""

from __future__ import annotations

import math
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


# The talking tasks: their clip is as long as the speech, at the model's frame rate.
TALKING_FPS = {"wan22-s2v": 16, "longcat-avatar": 25, "infinitetalk": 25}


def _dub(inputs: dict[str, Path], seconds: float, out_dir: Path) -> Output:
    """The fake InfiniteTalk: the uploaded clip re-timed to 25 fps (its last frame held past its end), as long as the
    speech, with the speech muxed in."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(inputs["video"]))
    meta = reader.__next__()
    w, h = meta["size"]
    source = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    if not source:
        raise ValueError("the clip has no frames")
    rate = float(meta.get("fps") or 16)
    n = max(1, math.ceil(seconds * 25))
    frames = np.stack([source[min(len(source) - 1, int(i / 25 * rate))] for i in range(n)])
    path = out_dir / "result.mp4"
    write_mp4(frames, path, fps=25, audio=inputs["audio"])
    return Output(path, "video", "video/mp4", extra={"frames": n, "width": w, "height": h,
                                                    "audioSeconds": round(seconds, 2), "sourceFrames": len(source)})


class FakeModelWorker:
    """Stands in for a GPU model (``AISTUDIO_FAKE_MODELS=1``): the task's own parameters, steps, previews and output
    kind, from a test pattern on the CPU. Image-to-video starts from the uploaded image (and ends on ``last_image``),
    and an edit returns picture 1 tinted, so a client can check the upload arrived. Speech-to-video makes 16 frames per
    second of the uploaded audio (refusing more than ``max_seconds``), the first frame the uploaded picture, with the
    audio muxed in. For testing clients end to end without models."""

    def load(self, models_root: Path, status) -> None:
        pass

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        from PIL import Image

        speech_seconds = None
        fps = TALKING_FPS.get(task.id, 16)
        if task.id in TALKING_FPS:  # as the real workers: the clip is as long as the audio, within its limit
            seconds = []
            for name in ("audio", "audio2"):
                if name in inputs:
                    samples, rate = read_wav(inputs[name])
                    seconds.append(samples.shape[1] / rate)
            speech_seconds = max(seconds)
            if speech_seconds > params["max_seconds"]:
                raise ValueError(f"the audio is {speech_seconds:.1f} s; at most {params['max_seconds']:g} s per clip")
        steps = params.get("steps") or 4
        sink.status("Sampling")
        for step in range(1, steps + 1):
            sink.check()
            time.sleep(0.02)
            sink.progress(step, steps)
            if task.kind != "audio":
                sink.preview(pattern(params.get("width") or 64, params.get("height") or 64, step / steps, params["seed"]),
                             step, steps)
        if task.kind == "audio":
            if task.id == "breeze-tts":  # the real worker's checks (text, a transcript with a reference)
                from .breeze import request
                request(params, inputs)
            # Each audio model's own rate and channels; length from its seconds (speech: from the text).
            rate, channels = {"music": (48000, 2), "sound": (44100, 2)}.get(task.family, (24000, 1))
            seconds = params.get("seconds") or min(10.0, max(1.0, len(params.get("text", "")) / 15))
            return _audio(seconds, rate, channels, params["seed"], inputs, out_dir)
        if task.id == "infinitetalk":  # a dub: the uploaded clip's frames at 25 fps, as long as the speech
            return _dub(inputs, speech_seconds, out_dir)
        w, h = params.get("width", 832), params.get("height", 480)  # tasks without a size: Wan's
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

        if task.kind == "image" and "image1" in inputs:
            # An edit: picture 1 at the asked size (0: its own), tinted red, with the bottom row's brightness the
            # number of pictures / 4, so a client can check the uploads arrived.
            first = Image.open(inputs["image1"]).convert("RGB")
            w, h = w or first.width, h or first.height
            image = np.asarray(first.resize((w, h)), dtype=np.float32) / 255.0
            image = image * np.array([0.5, 0.25, 0.25], dtype=np.float32) + np.array([0.5, 0, 0], dtype=np.float32)
            image[-1:] = sum(1 for k in ("image1", "image2", "image3") if k in inputs) / 4
            image = mark(image)
            path = out_dir / "result.png"
            write_png(image, path)
            return Output(path, "image", "image/png", extra={"loras": loras})
        if task.kind == "image":
            image = mark(pattern(w, h, 0.0, params["seed"]))
            path = out_dir / "result.png"
            write_png(image, path)
            return Output(path, "image", "image/png", extra={"loras": loras})
        n = params.get("length", 9) if speech_seconds is None else max(1, math.ceil(speech_seconds * fps))
        frames = np.stack([pattern(w, h, i / max(n, 1), params["seed"]) for i in range(n)])
        if "image" in inputs:
            start = Image.open(inputs["image"]).convert("RGB").resize((w, h))
            frames[0] = np.asarray(start, dtype=np.float32) / 255.0
        if "last_image" in inputs:  # the end frame, so a client can check it arrived
            end = Image.open(inputs["last_image"]).convert("RGB").resize((w, h))
            frames[-1] = np.asarray(end, dtype=np.float32) / 255.0
        frames = np.stack([mark(f) for f in frames])
        path = out_dir / "result.mp4"
        if speech_seconds is not None:
            write_mp4(frames, path, fps=fps, audio=inputs["audio"])
            return Output(path, "video", "video/mp4", extra={"frames": n, "width": w, "height": h,
                                                            "audioSeconds": round(speech_seconds, 2),
                                                            "speakers": 2 if "audio2" in inputs else 1})
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
