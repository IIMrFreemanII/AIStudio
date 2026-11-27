"""Motion transfer: a character picture plus a driving video → a video of the character moving as the person in it.

**Wan-Animate-2** (``wan-animate2``, Apache 2.0, diffusers' modular pipeline ``WanAnimate2DistilledModularPipeline``)
reads the driving video directly (no pose or face trackers): each 81-frame segment caches the driving clip's K/V in every
layer and denoises against it, 10 steps without guidance (the distilled checkpoint), at 24 fps. The scene comes from the
picture, not the video. Every part stays on the GPU (≈ 46 GB of weights plus the per-segment cache, which doesn't fit an
80 GB card at the default size: an H200), and the transformer's repeated blocks are compiled (its in-context attention
runs on the flex backend).

The clip is cut to ``start … start + seconds`` first (ffmpeg); the result can be colour-matched to the picture (the
Monge–Kantorovich linear transform, "mkl", which keeps segments from drifting in colour) and gets the clip's sound back
(``keep_audio``).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from . import Output, TaskSpec
from ..media import mux_audio, write_mp4

FPS = 24
# The model's captions are two parts, the person's look and the background (its model card's format).
LOOK, SCENE = "人物外观描述：", "背景描述："
AREAS = {"640x800": (640, 800), "720x1280": (720, 1280)}  # width × height of the target area (aspect: the picture's)


def caption(prompt: str, scene: str = "") -> str:
    """The model's two-part caption from a look (and a scene); a prompt already in that form is kept as it is."""
    prompt, scene = prompt.strip(), scene.strip()
    if LOOK in prompt:
        return prompt
    return f"{LOOK}{prompt or 'the person in the reference picture'} {SCENE}{scene or 'as in the reference picture'}"


def cut_clip(source: Path, out: Path, start: float, seconds: float) -> bool:
    """``source`` from ``start`` for ``seconds`` as an H.264 MP4 (sound kept as AAC when it has any). True when the
    cut has sound."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", str(source), "-t", f"{seconds:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not out.exists():
        raise ValueError(f"the video couldn't be read: {done.stderr.strip()[-400:]}")
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(out)], capture_output=True, text=True)
    if "Video:" not in probe.stderr:  # a start past the clip's end
        raise ValueError("the video has no frames in that range")
    return "Audio:" in probe.stderr


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """An MP4's frames (uint8 [H, W, 3]) and frame rate."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    w, h = meta["size"]
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    if not frames:
        raise ValueError("the video has no frames in that range")
    return frames, float(meta.get("fps") or FPS)


def color_match(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each frame's colours mapped onto the reference's (mean and covariance, the Monge–Kantorovich linear map).
    ``frames`` float [T, H, W, 3] in 0…1, ``reference`` [H, W, 3] in 0…1."""

    def stats(a):
        x = a.reshape(-1, 3).astype(np.float64)
        return x.mean(0), np.cov(x, rowvar=False) + np.eye(3) * 1e-6

    def sqrtm(m):
        w, v = np.linalg.eigh(m)
        return (v * np.sqrt(np.clip(w, 0, None))) @ v.T

    mu_r, cov_r = stats(reference)
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        mu_s, cov_s = stats(f)
        s = sqrtm(cov_s)
        s_inv = np.linalg.inv(s)
        t = s_inv @ sqrtm(s @ cov_r @ s) @ s_inv
        out[i] = np.clip((f.reshape(-1, 3) - mu_s) @ t.T + mu_r, 0.0, 1.0).reshape(f.shape)
    return out


class _Progress:
    """Stands in for the denoise loop's tqdm: every step checks for a cancel and reports progress across segments
    ("Segment k/N" in its description)."""

    sink = None

    def __init__(self, total=None, desc="", **_):
        self.total = int(total or 1)
        k, _, n = desc.removeprefix("Segment ").partition("/")
        self.k, self.n = (int(k) - 1, int(n)) if k.isdigit() and n.isdigit() else (0, 1)
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, n=1):
        self.i += n
        if _Progress.sink is not None:
            _Progress.sink.check()
            _Progress.sink.progress(self.k * self.total + self.i, self.n * self.total)


class WanAnimate2Worker:
    family = "wan-animate2"

    def __init__(self):
        self.pipe = None

    def load(self, models_root: Path, status) -> None:
        import torch
        from diffusers import ModularPipeline
        from diffusers.modular_pipelines.wan_animate_2 import denoise

        status("Loading Wan-Animate-2")
        pipe = ModularPipeline.from_pretrained(str(models_root / "wan-animate2"))
        pipe.load_components(dtype=torch.bfloat16)
        for name in ("transformer", "text_encoder", "image_encoder", "vae"):  # all resident, no offload
            getattr(pipe, name).to("cuda")
        status("Compiling Wan-Animate-2")
        pipe.transformer.compile_repeated_blocks(fullgraph=False)
        denoise.tqdm = _Progress
        self.pipe = pipe

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from PIL import Image

        sink.status("Reading the video")
        clip = out_dir / "driving.mp4"
        has_audio = cut_clip(inputs["video"], clip, params["start"], params["seconds"])
        frames, fps = read_frames(clip)
        picture = Image.open(inputs["image"]).convert("RGB")
        width, height = AREAS[params["area"]]
        prompt = caption(params["prompt"], params.get("scene", ""))
        sink.status("Animating")
        _Progress.sink = sink
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            videos = self.pipe(image=picture, driving_video=[Image.fromarray(f) for f in frames],
                               driving_video_fps=fps, prompt=prompt, height=height, width=width,
                               num_inference_steps=params["steps"],
                               generator=torch.Generator("cuda").manual_seed(params["seed"]),
                               output_type="np", output="videos")
        finally:
            _Progress.sink = None
        seconds = time.monotonic() - t0
        video = np.asarray(videos[0], dtype=np.float32)
        if params["color_match"]:
            ref = np.asarray(picture.resize((video.shape[2], video.shape[1])), dtype=np.float32) / 255.0
            video = color_match(video, ref)
        path = out_dir / "result.mp4"
        write_mp4(video, path, fps=FPS)
        if params["keep_audio"] and has_audio:
            silent = path.with_name("result.silent.mp4")
            path.rename(silent)
            mux_audio(silent, clip, path, seconds=len(video) / FPS)
            silent.unlink(missing_ok=True)
        return Output(path, "video", "video/mp4", extra={
            "frames": len(video), "width": int(video.shape[2]), "height": int(video.shape[1]), "fps": FPS,
            "drivingFrames": len(frames), "drivingFps": round(fps, 3), "audio": bool(params["keep_audio"] and has_audio),
            "animateSeconds": round(seconds, 1), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 1)})
