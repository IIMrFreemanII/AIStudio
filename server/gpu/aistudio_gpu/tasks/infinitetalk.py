"""InfiniteTalk (MeiGen-AI, Apache 2.0): dubs a clip, re-syncing its lips, head and body to new speech while keeping
its camera move.

The official repo (``MeiGen-AI/InfiniteTalk`` at a pinned commit, in its own Modal image, see ``modal_app.py``) runs
on Wan 2.1 I2V 14B (480p) with InfiniteTalk's audio-conditioning weights and a Chinese wav2vec2 speech encoder; this
calls its ``wan.InfiniteTalkPipeline.generate_infinitetalk`` in-process the way ``generate_infinitetalk.py`` does,
with the lightx2v step-distillation LoRA (4 steps, text guidance 1, audio guidance 2, shift 2: the repo's advice).

The model reads the source clip frame by frame at 25 fps (the speech features' rate), so the uploaded clip is first
re-timed to 25 fps; past its end the last frame holds. One pass makes 81 frames (3.24 s), longer speech continues in
passes of 72 new frames. The repo needs more than 81 frames of speech features, so short speech is padded with
silence; the result is cut to the speech's length and the uploaded audio muxed in.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..media import write_mp4
from . import Output, TaskSpec

FPS = 25
PASS_FRAMES, MOTION_FRAMES = 81, 9
SAMPLE_RATE = 16000
MIN_SECONDS = 0.2
STEPS, TEXT_GUIDANCE, AUDIO_GUIDANCE, SHIFT = 4, 1.0, 2.0, 2.0  # with the lightx2v LoRA


class InfiniteTalkWorker:
    def __init__(self):
        self.pipe = None
        self.extractor = None
        self.encoder = None

    def load(self, models_root: Path, status) -> None:
        from transformers import Wav2Vec2FeatureExtractor

        import wan
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        from wan.configs import WAN_CONFIGS

        root = models_root / "infinitetalk"
        status("Loading InfiniteTalk")
        self.pipe = wan.InfiniteTalkPipeline(
            config=WAN_CONFIGS["infinitetalk-14B"], checkpoint_dir=str(root / "wan21-i2v"), device_id=0, rank=0,
            t5_cpu=False, lora_dir=[str(root / "lightx2v" / "lightx2v.safetensors")], lora_scales=[1.0],
            infinitetalk_dir=str(root / "infinitetalk" / "single" / "infinitetalk.safetensors"))
        wav2vec = str(root / "wav2vec2")
        self.encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to("cpu")
        self.encoder.feature_extractor._freeze_parameters()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        import wan.multitalk as module

        speech = load_speech(inputs["audio"])
        seconds = len(speech) / SAMPLE_RATE
        if seconds > params["max_seconds"]:
            raise ValueError(f"the audio is {seconds:.1f} s; at most {params['max_seconds']:g} s per clip")
        if seconds < MIN_SECONDS:
            raise ValueError(f"the audio is {seconds:.2f} s; it needs at least {MIN_SECONDS:g} s")
        want = math.ceil(seconds * FPS)
        # More than one pass's worth of features, or the repo skips the speech.
        speech = np.pad(speech, (0, max(0, math.ceil((PASS_FRAMES + 4) / FPS * SAMPLE_RATE) - len(speech))))
        sink.status("Reading the clip")
        source = out_dir / "source-25fps.mp4"
        retime(inputs["video"], source)
        emb = embed(speech, self.extractor, self.encoder)
        emb_path = out_dir / "speech.pt"
        torch.save(emb, emb_path)

        steps = params["steps"] or STEPS
        passes = 1 + max(0, math.ceil((want - PASS_FRAMES) / (PASS_FRAMES - MOTION_FRAMES)))
        total = steps * passes
        done = 0
        step_times: list[float] = []
        first_step: float | None = None

        def tqdm(items, *args, **kwargs):  # the sampling loop's progress bar: ours reports, times and cancels
            nonlocal done, first_step
            for item in items:
                sink.check()
                began = time.monotonic()
                if first_step is None:
                    first_step = began
                yield item
                torch.cuda.synchronize()
                step_times.append(time.monotonic() - began)
                done += 1
                print(f"infinitetalk step {done}/{total}: {step_times[-1]:.2f} s", flush=True)
                sink.progress(min(done, total), total)

        size = "infinitetalk-720" if params["resolution"] == "720p" else "infinitetalk-480"
        extra = SimpleNamespace(use_teacache=False, teacache_thresh=0.2, size=size, use_apg=False, apg_momentum=-0.75,
                                apg_norm_threshold=55)
        sink.status("Sampling")
        t0 = time.monotonic()
        original = module.tqdm
        module.tqdm = tqdm
        try:
            with torch.inference_mode():
                video = self.pipe.generate_infinitetalk(
                    {"prompt": params["prompt"], "cond_video": str(source), "cond_audio": {"person1": str(emb_path)},
                     "video_audio": str(inputs["audio"])},
                    size_buckget=size, motion_frame=MOTION_FRAMES, frame_num=PASS_FRAMES, shift=SHIFT,
                    sampling_steps=steps, text_guide_scale=TEXT_GUIDANCE,
                    audio_guide_scale=params["audio_guidance"] or AUDIO_GUIDANCE, seed=params["seed"],
                    offload_model=False, max_frames_num=max(want, PASS_FRAMES), extra_args=extra)
        finally:
            module.tqdm = original
        sample_seconds = round(time.monotonic() - t0, 2)
        setup_seconds = round((first_step or t0) - t0, 2)
        step_seconds = round(sum(step_times) / len(step_times), 2) if step_times else 0.0
        print(f"infinitetalk {done} steps in {sample_seconds} s: setup {setup_seconds} s, {step_seconds} s per step",
              flush=True)

        # [C, T, H, W] in -1…1 → [T, H, W, 3] in 0…1, cut to the speech.
        frames = ((video.float().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 3, 0).cpu().numpy()[:want]
        sink.status("Encoding video")
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS, audio=inputs["audio"])
        return Output(path, "video", "video/mp4",
                      extra={"steps": steps, "passes": passes, "frames": int(len(frames)),
                             "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                             "audioSeconds": round(seconds, 2), "sampleSeconds": sample_seconds,
                             "setupSeconds": setup_seconds, "stepSeconds": step_seconds})


def load_speech(path: Path) -> np.ndarray:
    """Mono 16 kHz speech at −23 LUFS (the repo's ``audio_prepare_single``)."""
    import librosa
    import pyloudnorm as pyln

    speech, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    meter = pyln.Meter(SAMPLE_RATE)
    loudness = meter.integrated_loudness(speech) if len(speech) > SAMPLE_RATE // 2 else -200.0
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23)
    return speech.astype(np.float32)


def embed(speech: np.ndarray, extractor, encoder):
    """wav2vec2 features per video frame (25 fps), [frames, layers, dim] (the repo's ``get_embedding``)."""
    import torch
    from einops import rearrange

    frames = int(len(speech) / SAMPLE_RATE * FPS)
    values = np.squeeze(extractor(speech, sampling_rate=SAMPLE_RATE).input_values)
    values = torch.from_numpy(values).float().unsqueeze(0)
    with torch.no_grad():
        out = encoder(values, seq_len=frames, output_hidden_states=True)
    emb = torch.stack(out.hidden_states[1:], dim=1).squeeze(0)
    return rearrange(emb, "b s d -> s b d").cpu().detach()


def retime(video: Path, out: Path) -> None:
    """The clip at 25 fps, H.264 (frames dropped or repeated; no audio)."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-an", "-vf", f"fps={FPS}",
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read the clip: {done.stderr.strip()[-600:]}")
