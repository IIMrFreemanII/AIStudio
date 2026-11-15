"""Wan 2.2 S2V-14B: a still picture and a speech recording make a lip-synced clip (a talking shot).

diffusers has no pipeline for it, so this calls the official Wan2.2 repo's ``wan.WanS2V`` (installed in its own Modal
image at a pinned commit, see ``modal_app.py``). One 14B transformer, bf16, ≈ 28 GB, plus the UMT5 text encoder, the
Wan 2.1 VAE and a wav2vec2 audio encoder. It generates 80 frames (5 s at 16 fps) per pass and chains passes for longer
audio, each continuing from the last frames of the one before; ``max_seconds`` bounds the passes.

The repo calls flash-attention directly. The Modal image installs flash_attn; where it is missing, ``load`` routes
every use of it to PyTorch's scaled-dot-product attention (the repo's own fallback in
``wan.modules.attention.attention``, which ignores the padding masks).

Each pass's steps are timed and logged (stdout, so the container log shows them) and the result reports the setup
time before the first step and the average step, to price and tune the task.

The result is an H.264 MP4 at 16 fps with the uploaded audio muxed in, as long as the audio. Its size is width and
height rounded down to multiples of 64 (832×480 → 832×448): the start picture is cropped to that, because the model
pads anything else with black bars. The app scales clips to fill the frame.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from ..media import read_wav, write_mp4
from . import Output, TaskSpec
from .wan import fit_image

FPS = 16
PASS_FRAMES = 80  # frames per pass of the model (a multiple of 4)
MIN_SECONDS = 0.2
STEPS, GUIDANCE, SHIFT = 40, 4.5, 3.0  # the model's defaults (shift 3 is its advice for 480p)


class WanS2VWorker:
    def __init__(self):
        self.model = None
        self.attention = "flash_attn"

    def load(self, models_root: Path, status) -> None:
        import wan
        from wan.configs import WAN_CONFIGS

        if _use_sdpa_attention():
            self.attention = "sdpa"
        print(f"wan22-s2v attention: {self.attention}", flush=True)
        status("Loading Wan 2.2 S2V")
        self.model = wan.WanS2V(config=WAN_CONFIGS["s2v-14B"], checkpoint_dir=str(models_root / "wan22-s2v"),
                                device_id=0, rank=0, t5_cpu=False, init_on_cpu=False, convert_model_dtype=True)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        import wan.speech2video as s2v

        audio = inputs["audio"]
        samples, rate = read_wav(audio)
        seconds = samples.shape[1] / rate
        if seconds > params["max_seconds"]:
            raise ValueError(f"the audio is {seconds:.1f} s; at most {params['max_seconds']:g} s per clip")
        if seconds < MIN_SECONDS:
            raise ValueError(f"the audio is {seconds:.2f} s; it needs at least {MIN_SECONDS:g} s")
        width, height = params["width"], params["height"]
        steps, guidance = params["steps"] or STEPS, params["guidance"] or GUIDANCE
        want = math.ceil(seconds * FPS)
        # The sampling loop runs `steps` per pass; the passes are not known up front, so this is the estimate.
        total = steps * math.ceil(want / PASS_FRAMES)

        # The model works at multiples of 64 and pads a picture that doesn't fit with black bars: crop the start
        # picture to the largest such size within width × height instead (832×480 → 832×448), so it fills the clip.
        width, height = model_size(width, height)
        start = out_dir / "start.png"
        fit_image(inputs["image"], width, height).save(start)

        done = 0
        step_times: list[float] = []
        first_step: float | None = None

        def tqdm(items, *args, **kwargs):  # the official loop's progress bar: ours reports, times and honours cancel
            nonlocal done, first_step
            for item in items:
                sink.check()
                began = time.monotonic()
                if first_step is None:
                    first_step = began
                yield item
                torch.cuda.synchronize()  # the step's kernels are queued; wait so its time is the GPU's
                step_times.append(time.monotonic() - began)
                done += 1
                print(f"wan22-s2v step {done}/{total}: {step_times[-1]:.2f} s", flush=True)
                sink.progress(min(done, total), total)

        sink.status("Sampling")
        t0 = time.monotonic()
        original = s2v.tqdm
        s2v.tqdm = tqdm
        try:
            with torch.inference_mode():
                video = self.model.generate(
                    input_prompt=params["prompt"], ref_image_path=str(start), audio_path=str(audio),
                    enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, num_repeat=None,
                    pose_video=None, max_area=width * height, infer_frames=PASS_FRAMES, shift=SHIFT,
                    sample_solver="unipc", sampling_steps=steps, guide_scale=guidance, n_prompt=params["negative"],
                    seed=params["seed"], offload_model=False, init_first_frame=True)
        finally:
            s2v.tqdm = original
        sample_seconds = round(time.monotonic() - t0, 2)
        setup_seconds = round((first_step or t0) - t0, 2)
        step_seconds = round(sum(step_times) / len(step_times), 2) if step_times else 0.0
        print(f"wan22-s2v {done} steps in {sample_seconds} s: setup {setup_seconds} s, {step_seconds} s per step "
              f"({self.attention})", flush=True)

        # [C, N, H, W] in -1…1 → [N, H, W, 3] in 0…1; the last pass can run past the audio.
        frames = ((video.float().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 3, 0).cpu().numpy()[:want]
        sink.status("Encoding video")
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS, audio=audio)
        return Output(path, "video", "video/mp4",
                      extra={"steps": steps, "guidance": guidance, "frames": int(len(frames)),
                             "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                             "audioSeconds": round(seconds, 2), "sampleSeconds": sample_seconds,
                             "setupSeconds": setup_seconds, "stepSeconds": step_seconds,
                             "attention": self.attention})


def model_size(width: int, height: int) -> tuple[int, int]:
    """The largest size of multiples of 64 within width × height (at least 64×64): what S2V renders without
    padding."""
    return max(64, width // 64 * 64), max(64, height // 64 * 64)


def _use_sdpa_attention() -> bool:
    """Replaces the repo's ``flash_attention`` (which needs the flash_attn package) wherever it is imported by name
    with a call to ``wan.modules.attention.attention``, which falls back to scaled-dot-product attention. Does nothing
    and returns False when flash-attention is installed."""
    import wan.modules.attention as attn

    if attn.FLASH_ATTN_2_AVAILABLE or attn.FLASH_ATTN_3_AVAILABLE:
        return False  # keep the real one (the wrapper would recurse into it)
    attention = attn.attention

    def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                        causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
        kwargs = {} if dtype is None else {"dtype": dtype}
        return attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
                         q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic,
                         fa_version=version, **kwargs)

    for name, module in list(sys.modules.items()):
        if name.startswith("wan.") and module is not None and hasattr(module, "flash_attention"):
            module.flash_attention = flash_attention
    return True
