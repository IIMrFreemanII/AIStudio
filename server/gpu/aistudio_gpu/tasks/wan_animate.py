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


# MARK: Wan-Animate v1, replacement


REPLACE_FPS = 30  # the official preprocessing's and model's frame rate
REPLACE_CLIP, REPLACE_OVERLAP = 77, 1  # frames per segment, frames each carries over
REPLACE_AREAS = {"832x480": (832, 480), "1280x720": (1280, 720)}  # width × height of the target area
REPLACE_PROMPT = "视频中的人在做动作"  # the model's default caption ("the person in the video moves")


def replace_segments(frames: int, clip: int = REPLACE_CLIP, overlap: int = REPLACE_OVERLAP) -> int:
    """How many segments the official loop renders for ``frames`` driving frames."""
    return max(1, -(-(max(frames, clip) - overlap) // (clip - overlap)))


class WanAnimateReplaceWorker:
    """Wan-Animate v1 (``Wan-AI/Wan2.2-Animate-14B``, Apache 2.0) in replacement mode, with the official repo's code
    (as ``wan_s2v.py`` uses it): the repo's preprocessing tracks the person in the clip (YOLOv10 and ViTPose ONNX
    models: pose and face crops) and masks them out (SAM2), then the model draws the picture's character in their place,
    with the clip's background, lighting (the relighting LoRA) and motion kept. Everything stays on the GPU."""

    family = "wan-animate-replace"

    def __init__(self):
        self.model = None
        self.process = None
        self.attention = "flash_attn"

    def load(self, models_root: Path, status) -> None:
        import sys

        import wan
        from wan.configs import WAN_CONFIGS

        from .wan_s2v import _use_sdpa_attention

        root = models_root / "wan-animate"
        if _use_sdpa_attention():
            self.attention = "sdpa"
        status("Loading the trackers")
        preprocess = Path(wan.__file__).parent / "modules" / "animate" / "preprocess"
        sys.path.insert(0, str(preprocess))  # its modules import each other by their bare names
        from process_pipepline import ProcessPipeline

        ckpt = root / "process_checkpoint"
        self.process = ProcessPipeline(det_checkpoint_path=str(ckpt / "det" / "yolov10m.onnx"),
                                       pose2d_checkpoint_path=str(ckpt / "pose2d" / "vitpose_h_wholebody.onnx"),
                                       sam_checkpoint_path=str(ckpt / "sam2" / "sam2_hiera_large.pt"),
                                       flux_kontext_path=None)
        status("Loading Wan-Animate")
        self.model = wan.WanAnimate(config=WAN_CONFIGS["animate-14B"], checkpoint_dir=str(root), device_id=0, rank=0,
                                    t5_cpu=False, init_on_cpu=False, convert_model_dtype=True,
                                    use_relighting_lora=True)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        import wan.animate as animate

        sink.status("Reading the video")
        clip = out_dir / "driving.mp4"
        has_audio = cut_clip(inputs["video"], clip, params["start"], params["seconds"])
        source = out_dir / "source"
        source.mkdir(exist_ok=True)
        sink.status("Tracking the person")
        t0 = time.monotonic()
        self.process(video_path=str(clip), refer_image_path=str(inputs["image"]), output_path=str(source),
                     resolution_area=list(REPLACE_AREAS[params["area"]]), fps=REPLACE_FPS, replace_flag=True)
        track_seconds = time.monotonic() - t0
        frames_in = len(read_frames(source / "src_pose.mp4")[0])
        steps = params["steps"]
        total = steps * replace_segments(frames_in)
        done = 0

        def tqdm(items, *args, **kwargs):  # the official loop's progress bar: ours reports and honours cancel
            nonlocal done
            for item in items:
                sink.check()
                yield item
                done += 1
                sink.progress(min(done, total), total)

        sink.status("Replacing")
        t1 = time.monotonic()
        original = animate.tqdm
        animate.tqdm = tqdm
        try:
            with torch.inference_mode():
                video = self.model.generate(src_root_path=str(source), replace_flag=True, clip_len=REPLACE_CLIP,
                                            refert_num=REPLACE_OVERLAP, shift=5.0, sample_solver="dpm++",
                                            sampling_steps=steps, guide_scale=1.0,
                                            input_prompt=params["prompt"] or REPLACE_PROMPT, n_prompt="",
                                            seed=params["seed"], offload_model=False)
        finally:
            animate.tqdm = original
        replace_seconds = time.monotonic() - t1
        # [C, N, H, W] in -1…1 → [N, H, W, 3] in 0…1.
        frames = ((video.float().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 3, 0).cpu().numpy()
        sink.status("Encoding video")
        path = out_dir / "result.mp4"
        keep = bool(params["keep_audio"] and has_audio)
        write_mp4(frames, path, fps=REPLACE_FPS, audio=clip if keep else None)
        return Output(path, "video", "video/mp4", extra={
            "frames": int(len(frames)), "width": int(frames.shape[2]), "height": int(frames.shape[1]),
            "fps": REPLACE_FPS, "audio": keep, "trackSeconds": round(track_seconds, 1),
            "replaceSeconds": round(replace_seconds, 1), "attention": self.attention,
            "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 1)})
