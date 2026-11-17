"""MiniMax H3 (MiniMax H3 Community Licence): video with its own soundtrack (dialogue, effects, ambience) from one
denoising loop, 24 fps, 5–15 s, a 768-pixel short edge, 32 kHz stereo.

diffusers 0.42.1 ships it as Modular Diffusers blocks (``MiniMaxH3ModularPipeline``). The checkpoint has two
transformer partitions sharing everything else, so there are two workers, each holding one (61.7 GB) with the shared
Qwen3-VL conditioner (62 GB) and VAEs, registered in a ``ComponentsManager`` that moves each onto the GPU as its block
runs (they don't fit an H200 together):

- ``h3-video`` (``transformer/``): text only (``t2va``), or from a first and/or last keyframe (``fl2va``);
- ``h3-references`` (``transformer_ref/``): ``ref2va``, up to four pictures and three voice or music clips the prompt
  names as ``<Picture n>`` / ``<Audio n>``.

``fast`` (the default) runs LightX2V's Turbo LoRA for that partition at 8 model evaluations; off, the guidance-distilled
base model at 49 (diffusers' default 50 sigma grid points; the scheduler counts the terminal 0 too). Both LoRAs use the
checkpoint's own shifts (12 video, 3 audio), as LightX2V's configs for these files do. Their alpha comes per layer from
the file (FL2VA v1.2) or from its metadata (Ref2VA v1.1); diffusers' loader doesn't take the per-layer form, so
``prepare_lora`` folds the scale into ``lora_B`` and hands it a plain state dict.

Attention is PyTorch's default: the faster ``_flash_3_hub`` backend fetches its kernels from the Hub at run time, which
the containers (``HF_HUB_OFFLINE``) don't do.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..media import read_wav, resample, write_mp4, write_wav
from . import Output, TaskSpec

FPS = 24
CHUNK, KEEP = 17, 5  # the video VAE decodes 17·n + 5 frames
MIN_SECONDS, MAX_SECONDS = 5.0, 15.0
SAMPLE_RATE = 32000
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344  # the checkpoint's `canvas_max_pixels`, for a 768 short edge
CANVAS_MULTIPLE = 32
SIZES = {"768p": 768, "544p": 544}  # 544p: about 2.3× faster per step (diffusers' docs)
ASPECTS = ("auto", "16:9", "9:16", "1:1")
FAST_STEPS, FULL_STEPS = 8, 49  # model evaluations
LORAS = {"t2va": "turbo/fl2v.safetensors", "ref2va": "turbo/ref2v.safetensors"}
MAX_REFERENCE_IMAGES, MAX_REFERENCE_AUDIO = 4, 3
MIN_AUDIO_SECONDS = 2.0  # each audio reference: 2–15 s (the model card)


def num_frames(seconds: float) -> int:
    """`seconds` at 24 fps snapped up to the next 17·n + 5, kept within 5–15 s (124…345 frames)."""
    n = max(1, round(seconds * FPS))
    while n % CHUNK != KEEP:
        n += 1
    while n / FPS > MAX_SECONDS:
        n -= CHUNK
    while n / FPS < MIN_SECONDS:
        n += CHUNK
    return n


def canvas(ratio_w: float, ratio_h: float, short_edge: int = SHORT_EDGE) -> tuple[int, int]:
    """(height, width) for an aspect ratio: diffusers' ``resolve_canvas_size`` with the area budget scaled to the
    short edge (1344×768 for 16:9 at 768, 960×544 at 544)."""
    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError(f"the aspect ratio must be positive, got {ratio_w}:{ratio_h}")
    ratio = ratio_w / ratio_h
    if not 0.25 <= ratio <= 4:
        raise ValueError(f"MiniMax H3 takes aspect ratios from 1:4 to 4:1, got {ratio_w:g}:{ratio_h:g}")
    w, h = (short_edge * ratio, float(short_edge)) if ratio >= 1 else (float(short_edge), short_edge / ratio)
    budget = MAX_PIXELS * (short_edge / SHORT_EDGE) ** 2
    if w * h > budget:
        scale = (budget / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    m = CANVAS_MULTIPLE
    return max(m, round(h / m) * m), max(m, round(w / m) * m)


def target_size(params: dict, picture_size: tuple[int, int] | None) -> tuple[int, int]:
    """(height, width): the asked aspect at the asked size; "auto" follows the first picture (16:9 without one)."""
    edge = SIZES[params["size"]]
    if params["aspect"] == "auto":
        return canvas(*(picture_size or (16, 9)), edge)
    a, b = params["aspect"].split(":")
    return canvas(float(a), float(b), edge)


def prepare_lora(state_dict: dict, metadata: dict | None) -> dict:
    """A LightX2V Turbo LoRA as diffusers' loader takes it: each layer's ``alpha / rank`` folded into its ``lora_B``
    (alpha from the layer's own ``.alpha`` tensor, else the file's metadata, else the rank: scale 1), the alpha
    tensors dropped, peft's ``.default`` infix removed and the keys under ``transformer.`` (a pipeline holding only
    ``transformer_ref`` routes them there)."""
    alphas = {k[: -len(".alpha")]: float(v) for k, v in state_dict.items() if k.endswith(".alpha")}
    file_alpha = float(metadata["alpha"]) if metadata and metadata.get("alpha") not in (None, "") else None
    out = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            continue
        name = key.replace(".default.", ".")
        if ".lora_B." in name:
            module = key.split(".lora_B.")[0]
            rank = state_dict[key.replace(".lora_B.", ".lora_A.")].shape[0]
            alpha = alphas.get(module, file_alpha if file_alpha is not None else rank)
            if alpha != rank:
                value = value * (alpha / rank)
        out[name if name.startswith(("transformer.", "transformer_ref.")) else f"transformer.{name}"] = value
    return out


def check(task: TaskSpec, params: dict, inputs: dict[str, Path]) -> None:
    """Refuses what the model can't do (the fake worker calls it too)."""
    if not params["prompt"].strip():
        raise ValueError("describe the shot: MiniMax H3 needs a prompt")
    if task.id == "h3-references":
        for name in (f"audio{i}" for i in range(1, MAX_REFERENCE_AUDIO + 1)):
            if name in inputs:
                samples, rate = read_wav(inputs[name])
                seconds = samples.shape[1] / rate
                if not MIN_AUDIO_SECONDS <= seconds <= MAX_SECONDS:
                    raise ValueError(f"{name} is {seconds:.1f} s; an audio reference has to be 2–15 s")


def gpu_name() -> str:
    """The GPU this container got (with a fallback list it may not be the first one): "H200", "B200", … or the
    device's own name."""
    import torch

    name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return next((k for k in ("B200", "H200", "H100", "A100", "L40S") if k in name), name)


RESIDENT_GIB = 80  # the most model weight kept on the GPU at once: one 61.7 GiB transformer or conditioner, plus the VAE


def offload_margin(total_bytes: int) -> int:
    """The components manager's reserve, in bytes: it moves a component onto the GPU only if this much stays free, else
    offloads others first. With all but ``RESIDENT_GIB`` reserved, the 62 GiB conditioner leaves before the transformer
    comes on, so denoising has the rest for activations (≈ 60 GiB on an H200, ≈ 98 on a B200). A 12 GB margin kept both
    on an H200 and ran out of memory at step 5 of a 6 s clip."""
    return max(12 * 2**30, int(total_bytes - RESIDENT_GIB * 2**30))


class H3Worker:
    def __init__(self, workflow: str):
        self.workflow = workflow  # "t2va" (serves fl2va too) or "ref2va": which transformer partition it loads
        self.pipe = None

    def load(self, models_root: Path, status) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # read at the first CUDA allocation
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3 import denoise
        from safetensors import safe_open
        from safetensors.torch import load_file

        root = models_root / "minimax-h3"
        status("Loading MiniMax H3")
        manager = ComponentsManager()
        # All the blocks (each call picks t2va / fl2va / ref2va from its inputs); only this partition's weights. The
        # index names the Hub repo for every component, so they load from the volume instead.
        self.pipe = ModularPipeline.from_pretrained(str(root), components_manager=manager)
        self.pipe.load_components(workflow=self.workflow, dtype=torch.bfloat16, pretrained_model_name_or_path=str(root))
        status("Loading the Turbo LoRA")
        lora = root / LORAS[self.workflow]
        with safe_open(str(lora), framework="pt") as f:
            metadata = f.metadata()
        self.pipe.load_lora_weights(prepare_lora(load_file(str(lora)), metadata), adapter_name="turbo")
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=offload_margin(torch.cuda.mem_get_info()[1]))
        self._denoise = denoise

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference
        from PIL import Image

        check(task, params, inputs)
        fast = params["fast"]
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        (self.pipe.enable_lora if fast else self.pipe.disable_lora)()
        frames_wanted = num_frames(params["seconds"])
        kwargs: dict = {"prompt": params["prompt"].strip(), "num_frames": frames_wanted,
                        "num_inference_steps": steps + 1, "output_type": "np",
                        "generator": torch.Generator().manual_seed(params["seed"]),
                        "output": ["videos", "audio", "sampling_rate"]}
        if task.id == "h3-video":
            pictures = {k: Image.open(inputs[k]).convert("RGB") for k in ("image", "last_image") if k in inputs}
            kwargs.update(pictures)
            first = pictures.get("image") or pictures.get("last_image")
            workflow = "fl2va" if pictures else "t2va"
        else:
            refs: list = [MiniMaxH3ImageReference(image=Image.open(inputs[k]).convert("RGB"))
                          for k in (f"image{i}" for i in range(1, MAX_REFERENCE_IMAGES + 1)) if k in inputs]
            for k in (f"audio{i}" for i in range(1, MAX_REFERENCE_AUDIO + 1)):
                if k in inputs:
                    samples, rate = read_wav(inputs[k])  # diffusers needs torchaudio to resample; we do it here
                    samples = resample(samples, rate, SAMPLE_RATE)
                    refs.append(MiniMaxH3AudioReference(audio=torch.from_numpy(samples), sample_rate=SAMPLE_RATE))
            kwargs["references"] = refs
            first = refs[0].image
            workflow = "ref2va"
        h, w = target_size(params, first.size if first is not None else None)
        kwargs.update(height=h, width=w)

        step_times: list[float] = []
        clock = {"last": 0.0, "done": 0, "total": steps}

        class Progress:  # the denoise loop's progress bar: ours reports, times and cancels
            def __init__(self, total=None):
                clock["total"] = total or steps

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, n=1):
                torch.cuda.synchronize()
                now = time.monotonic()
                step_times.append(now - clock["last"])
                clock["last"] = now
                clock["done"] += n
                print(f"h3 step {clock['done']}/{clock['total']}: {step_times[-1]:.2f} s", flush=True)
                sink.progress(min(clock["done"], clock["total"]), clock["total"])
                sink.check()

        loop = self._denoise.MiniMaxH3DenoiseLoopWrapper
        original = loop.progress_bar
        sink.status("Sampling")
        t0 = clock["last"] = time.monotonic()
        loop.progress_bar = lambda self, iterable=None, total=None: Progress(total=total)
        try:
            result = self.pipe(**kwargs)
        finally:
            loop.progress_bar = original
        sample_seconds = round(time.monotonic() - t0, 2)
        # The first "step" includes the conditioner and VAE encode (and moving the transformer onto the GPU).
        setup_seconds = round(step_times[0], 2) if step_times else 0.0
        step_seconds = round(sum(step_times[1:]) / max(1, len(step_times) - 1), 2) if len(step_times) > 1 else 0.0

        sink.status("Encoding video")
        frames = np.asarray(result["videos"][0], dtype=np.float32)  # [T, H, W, 3] in 0…1
        audio = result["audio"][0].float().cpu().numpy()  # [2, n]
        rate = int(result["sampling_rate"])
        sound = out_dir / "sound.wav"
        write_wav(audio, rate, sound)
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS, audio=sound)
        return Output(path, "video", "video/mp4",
                      extra={"gpu": gpu_name(), "workflow": workflow, "fast": fast, "steps": steps, "frames": int(len(frames)),
                             "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                             "seconds": round(len(frames) / FPS, 3), "sampleRate": rate,
                             "audioChannels": int(audio.shape[0]), "sampleSeconds": sample_seconds,
                             "setupSeconds": setup_seconds, "stepSeconds": step_seconds})
