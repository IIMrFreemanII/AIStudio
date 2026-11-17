"""LongCat-Video-Avatar 1.6 (Meituan, MIT): a picture and speech make a lip-synced clip, one speaker or two.

The official repo (``meituan-longcat/LongCat-Video`` at a pinned commit, installed in its own Modal image, see
``modal_app.py``) has no diffusers pipeline; this runs its ``LongCatVideoAvatarPipeline`` in-process the way its
``run_demo_avatar_{single,multi}_audio_to_video.py`` do, on one GPU (context parallel of 1):

- the 15.8B avatar DiT in bf16 (``avatar/base_model``) with its two distillation LoRAs mixed per step (the first
  ``mix_steps`` steps ``mix_alpha``·DMD + the rest OPSD, the model card's defaults 4 and 1.0), 8 steps, no guidance;
- LongCat-Video's UMT5 text encoder, tokenizer and Wan VAE (``base/``), Whisper-large-v3 for the speech.

v1.6 is the v1.5 architecture (the demo maps its model type to "avatar-v1.5") without loudness normalisation. It
renders 93 frames (3.72 s at 25 fps) per segment and continues in segments of 80 new frames for longer speech; the
clip is cut to the audio's length and the uploaded audio muxed in.

Two speakers: ``audio`` is the left-hand person, ``audio2`` the right-hand one (the demo's default masks: each half of
the picture), both time-aligned on one timeline (the demo's "para" mode), so silence in one is the other's turn.

The demo first runs a vocal separator over the speech (for recordings with music under them). The app sends clean
TTS lines, so it is skipped: the speech is resampled to 16 kHz and peak-normalised as the separator's output is.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from ..media import write_mp4, write_wav
from . import Output, TaskSpec

FPS = 25
SEGMENT_FRAMES, COND_FRAMES = 93, 13  # frames per pass, and how many of them continue the pass before
SAMPLE_RATE = 16000
MIN_SECONDS = 0.2
STEPS = 8
MODEL_TYPE = "avatar-v1.5"  # what v1.6 runs as (its demo maps it so)
NEGATIVE = ("Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra "
            "fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, walking backwards")
REF_IMG_INDEX, MASK_FRAME_RANGE = 10, 0  # the demo's continuation settings


class LongCatAvatarWorker:
    def __init__(self):
        self.pipe = None
        self.mix = (4, 1.0)

    def load(self, models_root: Path, status) -> None:
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer, UMT5EncoderModel

        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        from longcat_video.context_parallel import context_parallel_util
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

        root = models_root / "longcat-avatar"
        base, avatar = str(root / "base"), root / "avatar"
        # The repo's attention and audio code ask the context-parallel groups even on one GPU: a one-process group.
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29517")
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
        context_parallel_util.init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
        split = context_parallel_util.get_optimal_split(1)

        status("Loading LongCat Avatar")
        bf16 = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", torch_dtype=bf16)
        text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=bf16)
        vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=bf16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(avatar), subfolder="scheduler",
                                                                    torch_dtype=bf16)
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(str(avatar), subfolder="base_model",
                                                                   cp_split_hw=split, torch_dtype=bf16).to(0)
        dit.load_lora(str(avatar / "loraA" / "lora.safetensors"), "dmd", multiplier=1.0, lora_network_dim=128,
                      lora_network_alpha=64)
        dit.load_lora(str(avatar / "loraB" / "lora.safetensors"), "opsd", multiplier=1.0, lora_network_dim=128,
                      lora_network_alpha=64)
        dit.enable_lora_mix("dmd", "opsd", self.mix[1], self.mix[0])
        whisper = str(avatar / "whisper-large-v3")
        encoder = get_audio_encoder(whisper, MODEL_TYPE).to(0)
        extractor = get_audio_feature_extractor(whisper, MODEL_TYPE)
        self.pipe = LongCatVideoAvatarPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                               scheduler=scheduler, dit=dit, audio_encoder=encoder,
                                               audio_feature_extractor=extractor, model_type=MODEL_TYPE)
        self.pipe.to(0)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import PIL.Image
        import torch

        import longcat_video.pipeline_longcat_video_avatar as module

        speakers = [load_speech(inputs["audio"])] + ([load_speech(inputs["audio2"])] if "audio2" in inputs else [])
        n = max(len(s) for s in speakers)
        speakers = [np.pad(s, (0, n - len(s))) for s in speakers]  # time-aligned: the shorter one ends in silence
        seconds = n / SAMPLE_RATE
        if seconds > params["max_seconds"]:
            raise ValueError(f"the audio is {seconds:.1f} s; at most {params['max_seconds']:g} s per clip")
        if seconds < MIN_SECONDS:
            raise ValueError(f"the audio is {seconds:.2f} s; it needs at least {MIN_SECONDS:g} s")
        want = math.ceil(seconds * FPS)
        passes = segments(want)
        steps = params["steps"] or STEPS
        mix = (params["mix_steps"], params["mix_alpha"])
        if mix != self.mix:
            self.pipe.dit.enable_lora_mix("dmd", "opsd", mix[1], mix[0])
            self.mix = mix

        # Whisper features per video frame; the demo pads the speech to what the passes cover plus a second.
        covered = (SEGMENT_FRAMES + (passes - 1) * (SEGMENT_FRAMES - COND_FRAMES)) / FPS + 1.0
        pad = max(0, math.ceil(covered * SAMPLE_RATE) - n)
        embeds = []
        for speech in speakers:
            emb = self.pipe.get_audio_embedding(np.pad(speech, (0, pad)), fps=FPS, device=0, sample_rate=SAMPLE_RATE,
                                                model_type=MODEL_TYPE, use_loudness_norm=False)
            if torch.isnan(emb).any():
                raise ValueError("the speech gave a broken audio embedding")
            embeds.append(emb)

        image = PIL.Image.open(inputs["image"]).convert("RGB")
        masks = two_speaker_masks(image.height, image.width) if len(speakers) == 2 else None
        generator = torch.Generator(device=0).manual_seed(params["seed"])
        total = steps * passes
        done = 0
        step_times: list[float] = []
        first_step: float | None = None

        def tqdm(*args, **kwargs):  # the pipeline's progress bar: ours reports, times and honours cancel
            return _Progress(sink, on_step)

        def on_step(seconds_taken: float):
            nonlocal done, first_step
            done += 1
            if first_step is None:  # the setup before it: text and image encoding
                first_step = time.monotonic() - seconds_taken
            step_times.append(seconds_taken)
            print(f"longcat-avatar step {done}/{total}: {seconds_taken:.2f} s", flush=True)
            sink.progress(min(done, total), total)

        def window(start: int):  # the audio features for the pass starting at frame `start`
            offsets = torch.arange(5) - 2
            centres = torch.arange(start, start + SEGMENT_FRAMES).unsqueeze(1) + offsets.unsqueeze(0)
            centres = torch.clamp(centres, min=0, max=embeds[0].shape[0] - 1)
            return torch.cat([e[centres][None, ...].to(0) for e in embeds])

        common = dict(prompt=params["prompt"], negative_prompt=NEGATIVE, num_inference_steps=steps,
                      text_guidance_scale=1.0, audio_guidance_scale=1.0, generator=generator, output_type="both",
                      use_distill=True)
        if masks is not None:
            common["ref_target_masks"] = masks.to(0)
        sink.status("Sampling")
        t0 = time.monotonic()
        original = module.tqdm
        module.tqdm = tqdm
        try:
            with torch.inference_mode():
                output, latent = self.pipe.generate_ai2v(image=image, resolution=params["resolution"],
                                                         num_frames=SEGMENT_FRAMES, audio_emb=window(0), **common)
                frames = (output[0] * 255).astype(np.uint8)
                video = [PIL.Image.fromarray(f) for f in frames]
                ref_latent = latent[:, :, :1].clone()
                start = 0
                height, width = frames.shape[1:3]
                for _ in range(1, passes):
                    start += SEGMENT_FRAMES - COND_FRAMES
                    output, latent = self.pipe.generate_avc(
                        video=video, video_latent=latent, height=height, width=width, num_frames=SEGMENT_FRAMES,
                        num_cond_frames=COND_FRAMES, use_kv_cache=False, offload_kv_cache=False, enhance_hf=False,
                        audio_emb=window(start), ref_latent=ref_latent, ref_img_index=REF_IMG_INDEX,
                        mask_frame_range=MASK_FRAME_RANGE, **common)
                    new = (output[0] * 255).astype(np.uint8)
                    frames = np.concatenate([frames, new[COND_FRAMES:]])
                    video = [PIL.Image.fromarray(f) for f in new]
        finally:
            module.tqdm = original
        sample_seconds = round(time.monotonic() - t0, 2)
        setup_seconds = round((first_step or t0) - t0, 2)
        step_seconds = round(sum(step_times) / len(step_times), 2) if step_times else 0.0
        print(f"longcat-avatar {done} steps, {passes} passes in {sample_seconds} s: {step_seconds} s per step",
              flush=True)

        frames = frames[:want]
        sink.status("Encoding video")
        audio = inputs["audio"]
        if len(speakers) == 2:  # both voices in the clip's sound
            audio = out_dir / "speech.wav"
            write_wav(np.clip(speakers[0] + speakers[1], -1, 1), SAMPLE_RATE, audio)
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS, audio=audio)
        return Output(path, "video", "video/mp4",
                      extra={"steps": steps, "passes": passes, "frames": int(len(frames)),
                             "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                             "speakers": len(speakers), "audioSeconds": round(seconds, 2),
                             "sampleSeconds": sample_seconds, "setupSeconds": setup_seconds,
                             "stepSeconds": step_seconds})


def segments(frames: int) -> int:
    """How many passes cover ``frames``: the first makes 93, each further one 80 new frames (13 continue)."""
    return 1 + max(0, math.ceil((frames - SEGMENT_FRAMES) / (SEGMENT_FRAMES - COND_FRAMES)))


def load_speech(path: Path) -> np.ndarray:
    """Mono 16 kHz float speech, peak-normalised (the level the demo's vocal separator hands the model)."""
    import librosa

    speech, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    peak = float(np.max(np.abs(speech))) if speech.size else 0.0
    return (speech / peak * 0.9 if peak > 1e-4 else speech).astype(np.float32)


def two_speaker_masks(height: int, width: int):
    """The demo's default masks without boxes: each speaker the middle of their half of the picture, the rest
    background."""
    import torch

    scale = 0.1
    half = width // 2
    y0, y1 = int(height * scale), int(height * (1 - scale))
    left = torch.zeros([height, width])
    right = torch.zeros([height, width])
    left[y0:y1, int(half * scale):int(half * (1 - scale))] = 1
    right[y0:y1, int(half * scale + half):int(half * (1 - scale) + half)] = 1
    background = torch.where(left + right > 0, torch.tensor(0.0), torch.tensor(1.0))
    return torch.stack([left, right, background], dim=0)


class _Progress:
    """Stands in for the pipeline's ``with tqdm(total=…) as progress_bar`` / ``progress_bar.update()``: each update
    is one step. Checks for a cancel and times the step (synchronising, so it is the GPU's time)."""

    def __init__(self, sink, on_step):
        self.sink, self.on_step = sink, on_step
        self.began = time.monotonic()

    def __enter__(self):
        self.sink.check()
        self.began = time.monotonic()
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n: int = 1):
        import torch

        torch.cuda.synchronize()
        now = time.monotonic()
        self.on_step(now - self.began)
        self.sink.check()
        self.began = time.monotonic()
