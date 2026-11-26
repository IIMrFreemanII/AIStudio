"""The tasks the server runs: their parameters (with limits), GPU, price, and the models they need.

The front desk only reads these specs (no torch); the GPU workers import the task modules that do the work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# Modal's on-demand prices in USD per second (modal.com/pricing, checked 2026-11-10). Used for the cost estimate
# shown after each job; billing itself is Modal's.
GPU_PRICES = {
    "B200": 0.001736, "H200": 0.001261, "H100": 0.001097, "A100-80GB": 0.000694, "A100-40GB": 0.000583,
    "L40S": 0.000542, "A10": 0.000306, "L4": 0.000222, "T4": 0.000164,
}
# A container's memory and CPU are billed too, for the larger of what it requests and what it uses (modal.com/pricing,
# checked 2026-11-17): USD per GiB-second and per physical core-second. Modal's minimum request is 0.125 cores.
MEMORY_PRICE = 0.00000222
CPU_PRICE = 0.0000131
MIN_CPU = 0.125

# Each worker family's Modal container: memory request and limit (MiB) and CPU cores requested. modal_app.py sizes its
# classes from this, and a task's price per second counts the request (the least its container is billed for).
CONTAINERS: dict[str, tuple[int, int, float]] = {
    "wan22-t2v": (16384, 98304, MIN_CPU), "wan22-i2v": (16384, 98304, MIN_CPU), "wan22-s2v": (16384, 131072, 4),
    "longcat-avatar": (16384, 131072, 4), "infinitetalk": (32768, 196608, 4), "flux1-dev": (8192, 65536, MIN_CPU),
    "qwen-image": (16384, 98304, MIN_CPU), "qwen-image-edit": (16384, 98304, MIN_CPU),
    # MiniMax H3 keeps ≈ 140 GB of weights in host RAM (its two 61.7 GB models take turns on the GPU).
    "h3-video": (163840, 262144, 8), "h3-references": (163840, 262144, 8),
    "sd35-large": (16384, 65536, MIN_CPU), "sdxl": (8192, 32768, MIN_CPU), "sd15": (8192, 32768, MIN_CPU), "anima": (8192, 32768, MIN_CPU), "chroma": (16384, 65536, MIN_CPU), "music": (16384, 65536, MIN_CPU),
    "sound": (8192, 32768, MIN_CPU), "speech": (8192, 32768, MIN_CPU), "breeze": (8192, 32768, MIN_CPU),
}


def container_price(family: str) -> float:
    """USD per second for a family's container beyond its GPU: its memory request and CPU cores."""
    mib, _, cpu = CONTAINERS.get(family, (0, 0, MIN_CPU))
    return mib / 1024 * MEMORY_PRICE + max(cpu, MIN_CPU) * CPU_PRICE


@dataclass(frozen=True)
class Param:
    type: str  # "int", "float", "string", "bool"
    default: Any
    min: float | None = None
    max: float | None = None
    multiple: int | None = None  # ints are rounded to a multiple of this (frame sizes)
    choices: tuple[str, ...] | None = None  # strings: the allowed values (matched case-insensitively)

    def coerce(self, name: str, value: Any) -> Any:
        if value is None:
            return self.default
        try:
            if self.type == "int":
                v = int(round(float(value)))
                if self.multiple:
                    v = max(self.multiple, round(v / self.multiple) * self.multiple)
            elif self.type == "float":
                v = float(value)
            elif self.type == "bool":
                v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            else:
                v = str(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected {self.type}, got {value!r}") from None
        if self.choices is not None:
            match = next((c for c in self.choices if c.lower() == v.lower()), None)
            if match is None:
                raise ValueError(f"{name} must be one of {', '.join(self.choices)}")
            v = match
        if self.type in ("int", "float"):
            if self.min is not None and v < self.min:
                raise ValueError(f"{name} must be at least {self.min:g}")
            if self.max is not None and v > self.max:
                raise ValueError(f"{name} must be at most {self.max:g}")
        return v

    def describe(self) -> dict:
        d: dict[str, Any] = {"type": self.type, "default": self.default}
        for k in ("min", "max", "multiple"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.choices is not None:
            d["choices"] = list(self.choices)
        return d


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    kind: str  # "video", "image" or "audio": what the job returns
    family: str  # the worker that runs it (one GPU container type per family)
    gpu: str  # Modal GPU type
    params: dict[str, Param]
    inputs: tuple[str, ...] = ()  # uploaded files the task needs ("image")
    optional_inputs: tuple[str, ...] = ()  # uploaded files the task can use ("reference")
    models: tuple[str, ...] = ()  # model folders under the models root (see download.MODELS)
    cpu_only: bool = False  # the test tasks
    fallback_gpus: tuple[str, ...] = ()  # GPUs Modal may give it when none of `gpu` is free

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameter(s) for {self.id}: {', '.join(sorted(unknown))}")
        return {name: p.coerce(name, params.get(name)) for name, p in self.params.items()}

    @property
    def price_per_second(self) -> float | None:
        return self.price_on(self.gpu)

    def price_on(self, gpu: str) -> float | None:
        """USD per second on `gpu` (the task's own, or a fallback it got): the GPU plus its container's memory and
        CPU; None for CPU-only tasks and unknown GPUs."""
        if self.cpu_only or gpu not in GPU_PRICES:
            return None
        return GPU_PRICES[gpu] + container_price(self.family)

    def describe(self, installed: bool) -> dict:
        d = {"id": self.id, "title": self.title, "kind": self.kind, "gpu": "CPU" if self.cpu_only else self.gpu,
             "pricePerSecond": self.price_per_second, "installed": installed, "inputs": list(self.inputs),
             "optionalInputs": list(self.optional_inputs),
             "params": {k: v.describe() for k, v in self.params.items()}}
        priced = [(g, self.price_on(g)) for g in self.fallback_gpus]
        priced = [(g, p) for g, p in priced if p is not None]
        if priced:  # the dearest GPU it may take instead, for the estimate's worst case
            gpu, price = max(priced, key=lambda gp: gp[1])
            d.update(fallbackGpu=gpu, fallbackPricePerSecond=price)
        return d


FLUX_LORAS = 4  # LoRA slots on the FLUX, Qwen-Image (and Edit), SD 3.5, SDXL, Chroma, Wan and MiniMax H3 tasks


def _lora_params() -> dict[str, Param]:
    return {f"lora{i}_strength": Param("float", 1.0, min=-2.0, max=2.0) for i in range(1, FLUX_LORAS + 1)}


LORA_INPUTS = tuple(f"lora{i}" for i in range(1, FLUX_LORAS + 1))
# Wan 2.2 pairs: the low-noise expert's file of slot N.
WAN_LORA_INPUTS = LORA_INPUTS + tuple(f"lora{i}_low" for i in range(1, FLUX_LORAS + 1))


def _wan_params() -> dict[str, Param]:
    return {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 832, min=256, max=1280, multiple=16),
        "height": Param("int", 480, min=256, max=1280, multiple=16),
        # Wan wants 4k+1 frames; 81 = 5 s at 16 fps.
        "length": Param("int", 81, min=5, max=121),
        # The Lightning 4-step LoRAs: ~10× fewer GPU seconds, a little less detail.
        "fast": Param("bool", True),
        # 0 = automatic: 4 steps without guidance when fast, else Wan's defaults (40 steps, guidance 4/3).
        "steps": Param("int", 0, min=0, max=60),
        "guidance": Param("float", 0.0, min=0.0, max=10.0),
        # A fine-tune installed from a link (variants.py); "base" is Wan 2.2 itself.
        "model": Param("string", "base"),
        # LoRAs: lora1 … lora4, each for both experts or one (`high` / `low`), or a pair (loraN + loraN_low).
        **_lora_params(),
        **{f"lora{i}_expert": Param("string", "both", choices=("both", "high", "low")) for i in range(1, FLUX_LORAS + 1)},
    }


GPU_WAN = os.environ.get("AISTUDIO_GPU_WAN", "H100")
GPU_FLUX = os.environ.get("AISTUDIO_GPU_FLUX", "L40S")
GPU_AUDIO = os.environ.get("AISTUDIO_GPU_AUDIO", "L40S")
GPU_H3 = os.environ.get("AISTUDIO_GPU_H3", "H200")  # its 61.7 GB transformer + 62 GB conditioner take turns on it
# Modal takes the first of these GPUs it has free: H200s can queue for long stretches, and a B200 holds H3 as well.
GPU_H3_FALLBACK = os.environ.get("AISTUDIO_GPU_H3_FALLBACK", "B200")
H3_FALLBACKS = tuple(g for g in (x.strip() for x in GPU_H3_FALLBACK.split(",")) if g and g != GPU_H3)
# Qwen-Image's 20B transformer: bf16 on an H100 (80 GB); on a 48 GB card (L40S) its weights are stored in fp8.
GPU_QWEN = os.environ.get("AISTUDIO_GPU_QWEN", "H100")
GPU_SD = os.environ.get("AISTUDIO_GPU_SD", "L40S")  # Stable Diffusion 3.5 Large and SDXL
GPU_SD15 = os.environ.get("AISTUDIO_GPU_SD15", "L40S")  # Stable Diffusion 1.5
GPU_ANIMA = os.environ.get("AISTUDIO_GPU_ANIMA", "L40S")  # Anima (2B)
GPU_CHROMA = os.environ.get("AISTUDIO_GPU_CHROMA", "L40S")  # Chroma1-HD: ≈ 28 GB of bf16 weights
# The lip-sync models (LongCat-Video-Avatar's 15.8B DiT, InfiniteTalk on Wan 2.1's 14B), bf16 on an 80 GB card.
GPU_AVATAR = os.environ.get("AISTUDIO_GPU_AVATAR", "H100")

# ACE-Step's planner and DiT: keys as "C major" … "B minor", time signatures as beats per bar, ISO 639-1 languages
# (the same lists as the app's local Text to Music node; "auto" lets the planner choose).
MUSIC_KEYS = ("auto",) + tuple(f"{n} {m}" for n in ("C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A",
                                                     "Bb", "B") for m in ("major", "minor"))
MUSIC_TIME_SIGNATURES = ("auto", "2", "3", "4", "6")
MUSIC_LANGUAGES = ("auto", "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi", "nl", "pl", "sv",
                   "tr", "uk", "vi", "id", "th", "yue")
# Qwen3-TTS CustomVoice's built-in speakers, plus "designed": a voice made from the description (VoiceDesign).
SPEECH_VOICES = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee", "designed")
SPEECH_LANGUAGES = ("Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese",
                    "Spanish", "Italian")
SEED = Param("int", 0, min=0, max=2**32 - 1)


def _h3_params() -> dict[str, Param]:
    return {
        # A structured prompt reads best (see MiniMax's prompting guide): the shot, then its dialogue as
        # <d>[English] …</d>, the soundscape and any music. With references, name them <Picture n> / <Audio n>.
        "prompt": Param("string", ""),
        "seed": SEED,
        "seconds": Param("float", 5.0, min=5.0, max=15.0),  # snapped to the VAE's 17·n + 5 frames at 24 fps
        "size": Param("string", "768p", choices=("768p", "544p")),  # 544p: about 2.3× faster per step
        "aspect": Param("string", "auto", choices=("auto", "16:9", "9:16", "1:1")),  # auto: the first picture's
        # LightX2V's Turbo LoRA, 8 model evaluations; off: the base model at 49 (≈ 6× the GPU time).
        "fast": Param("bool", True),
        "steps": Param("int", 0, min=0, max=100),  # model evaluations; 0 = 8 fast, 49 full
        # Community LoRAs (lora1 … lora4: musubi-tuner, ai-toolkit, kohya, DiffSynth files; diffusers converts them),
        # for this task's partition; with `fast` they stack on the Turbo LoRA.
        **_lora_params(),
    }

TASKS: dict[str, TaskSpec] = {t.id: t for t in [
    TaskSpec("wan22-t2v", "Wan 2.2 Text to Video", "video", "wan22-t2v", GPU_WAN, _wan_params(),
             optional_inputs=WAN_LORA_INPUTS, models=("wan22-t2v", "wan22-lightning")),
    TaskSpec("wan22-i2v", "Wan 2.2 Image to Video", "video", "wan22-i2v", GPU_WAN, _wan_params(),
             # last_image: the frame the clip should end on (first-last-frame), optional.
             inputs=("image",), optional_inputs=("last_image",) + WAN_LORA_INPUTS,
             models=("wan22-i2v", "wan22-lightning")),
    # A talking shot: a still picture and a speech recording make a lip-synced clip (the audio is muxed in).
    TaskSpec("wan22-s2v", "Wan 2.2 Speech to Video", "video", "wan22-s2v", GPU_WAN, {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 832, min=256, max=1280, multiple=16),
        "height": Param("int", 480, min=256, max=1280, multiple=16),
        "steps": Param("int", 0, min=0, max=60),  # 0 = the model's 40
        "guidance": Param("float", 0.0, min=0.0, max=10.0),  # 0 = the model's 4.5
        # Longer audio is refused, so a job's cost stays bounded (≈ 80 frames, 5 s, per pass of the model).
        "max_seconds": Param("float", 15.0, min=1.0, max=30.0),
    }, inputs=("image", "audio"), models=("wan22-s2v",)),
    # LongCat-Video-Avatar 1.6 (MIT): a picture and speech make a lip-synced clip, like S2V; audio2 is a second
    # speaker (the picture's right-hand person), time-aligned with the first. Distilled: 8 steps, no guidance.
    TaskSpec("longcat-avatar", "LongCat Avatar talking video", "video", "longcat-avatar", GPU_AVATAR, {
        "prompt": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "resolution": Param("string", "480p", choices=("480p", "720p")),  # the size follows the picture's shape
        "steps": Param("int", 0, min=0, max=50),  # 0 = the distilled 8
        # The first mix_steps steps blend the DMD LoRA (mix_alpha) with OPSD; the rest are OPSD (the model's advice).
        "mix_steps": Param("int", 4, min=0, max=8),
        "mix_alpha": Param("float", 1.0, min=0.0, max=1.0),
        "max_seconds": Param("float", 15.0, min=1.0, max=30.0),
    }, inputs=("image", "audio"), optional_inputs=("audio2",), models=("longcat-avatar",)),
    # InfiniteTalk (Apache 2.0, on Wan 2.1 I2V 14B): re-syncs a clip's lips, head and body to new speech (dubbing),
    # keeping its camera move. 4 steps with the lightx2v LoRA; the clip is as long as the audio (≤ the video).
    TaskSpec("infinitetalk", "InfiniteTalk lip-sync (dub a clip)", "video", "infinitetalk", GPU_AVATAR, {
        "prompt": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "resolution": Param("string", "480p", choices=("480p", "720p")),
        "steps": Param("int", 0, min=0, max=40),  # 0 = 4 (the LoRA's)
        "audio_guidance": Param("float", 0.0, min=0.0, max=10.0),  # 0 = 2 (with the LoRA; raise for tighter lips)
        "max_seconds": Param("float", 15.0, min=1.0, max=30.0),
    }, inputs=("video", "audio"), models=("infinitetalk",)),
    TaskSpec("flux1-dev", "FLUX.1 dev", "image", "flux1-dev", GPU_FLUX, {
        "prompt": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 1024, min=256, max=2048, multiple=16),
        "height": Param("int", 1024, min=256, max=2048, multiple=16),
        "steps": Param("int", 28, min=1, max=60),
        "guidance": Param("float", 3.5, min=0.0, max=10.0),
        # A fine-tune installed from a link (variants.py), by its id; "base" is FLUX.1 dev itself.
        "model": Param("string", "base"),
        # LoRAs: uploaded .safetensors files lora1 … lora4 (kohya, diffusers/PEFT, ai-toolkit…), each with a strength.
        **_lora_params(),
    }, optional_inputs=LORA_INPUTS, models=("flux1-dev",)),
    TaskSpec("qwen-image", "Qwen-Image", "image", "qwen-image", GPU_QWEN, {
        "prompt": Param("string", ""),
        # Applied by true classifier-free guidance (cfg above 1), at twice the cost per step.
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        # Its training sizes: 1328², 1664×928, 1472×1140, 1584×1056 (and portrait).
        "width": Param("int", 1328, min=256, max=2048, multiple=16),
        "height": Param("int", 1328, min=256, max=2048, multiple=16),
        # The Lightning 8-step LoRA, without guidance: ≈ 6× fewer GPU seconds, a little less detail.
        "fast": Param("bool", True),
        # 0 = automatic: 8 steps at cfg 1 when fast, else 50 steps at cfg 4 (the model card's).
        "steps": Param("int", 0, min=0, max=100),
        "cfg": Param("float", 0.0, min=0.0, max=10.0),
        "model": Param("string", "base"),
        **_lora_params(),
    }, optional_inputs=LORA_INPUTS, models=("qwen-image", "qwen-image-lightning")),
    # A new picture from an instruction and one to three pictures ("picture 1 is the woman, picture 2 the room…").
    TaskSpec("qwen-image-edit", "Qwen-Image-Edit 2511", "image", "qwen-image-edit", GPU_QWEN, {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),  # with cfg above 1, as on Qwen-Image
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        # 0 = picture 1's shape at about a megapixel; otherwise rounded to a multiple of 16.
        "width": Param("int", 0, min=0, max=2048),
        "height": Param("int", 0, min=0, max=2048),
        # Its Lightning 8-step LoRA, without guidance: ≈ 10× fewer GPU seconds than 40 steps with CFG.
        "fast": Param("bool", True),
        # 0 = automatic: 8 steps at cfg 1 when fast, else 40 steps at cfg 4 (the model card's).
        "steps": Param("int", 0, min=0, max=100),
        "cfg": Param("float", 0.0, min=0.0, max=10.0),
        **_lora_params(),
    }, inputs=("image1",), optional_inputs=("image2", "image3") + LORA_INPUTS,
        models=("qwen-image-edit", "qwen-image-edit-lightning")),
    TaskSpec("sd35-large", "Stable Diffusion 3.5 Large", "image", "sd35-large", GPU_SD, {
        "prompt": Param("string", ""),
        # Used with guidance above 1 (Large's classifier-free guidance); Turbo runs without guidance and ignores it.
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 1024, min=256, max=2048, multiple=16),
        "height": Param("int", 1024, min=256, max=2048, multiple=16),
        # Large Turbo: 4 steps without guidance, ≈ 7× fewer GPU seconds than Large's 28 steps.
        "fast": Param("bool", True),
        # 0 = automatic: 4 steps when fast, else 28.
        "steps": Param("int", 0, min=0, max=100),
        # 0 = automatic: 0 when fast, else 3.5.
        "guidance": Param("float", 0.0, min=0.0, max=20.0),
        **_lora_params(),
    }, optional_inputs=LORA_INPUTS, models=("sd35-large", "sd35-large-turbo")),
    TaskSpec("sdxl", "Stable Diffusion XL", "image", "sdxl", GPU_SD, {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 1024, min=512, max=2048, multiple=8),
        "height": Param("int", 1024, min=512, max=2048, multiple=8),
        "steps": Param("int", 30, min=1, max=100),
        "guidance": Param("float", 6.0, min=0.0, max=20.0),
        "sampler": Param("string", "dpmpp-2m-karras", choices=("dpmpp-2m-karras", "euler-a", "euler")),
        # A fine-tune installed from a link (variants.py), by its id; "base" is SDXL 1.0 itself.
        "model": Param("string", "base"),
        # How strongly the face picture (the optional `face` input) steers the result.
        "face_strength": Param("float", 0.6, min=0.0, max=1.0),
        **_lora_params(),
    }, optional_inputs=("face",) + LORA_INPUTS, models=("sdxl",)),
    TaskSpec("sd15", "Stable Diffusion 1.5", "image", "sd15", GPU_SD15, {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 512, min=256, max=1024, multiple=8),
        "height": Param("int", 768, min=256, max=1024, multiple=8),
        "steps": Param("int", 25, min=1, max=100),
        "guidance": Param("float", 7.0, min=0.0, max=20.0),
        "sampler": Param("string", "dpmpp-2m-karras", choices=("dpmpp-2m-karras", "euler-a", "euler")),
        # A fine-tune installed from a link (variants.py), by its id; "base" is SD 1.5 itself.
        "model": Param("string", "base"),
        **_lora_params(),
    }, optional_inputs=LORA_INPUTS, models=("sd15",)),
    TaskSpec("anima", "Anima", "image", "anima", GPU_ANIMA, {
        "prompt": Param("string", ""),
        "negative": Param("string", "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
                                    "jpeg artifacts, chromatic aberration"),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 1024, min=512, max=1536, multiple=16),
        "height": Param("int", 1024, min=512, max=1536, multiple=16),
        "steps": Param("int", 30, min=1, max=80),
        "guidance": Param("float", 4.0, min=1.0, max=10.0),
        # A fine-tune installed from a link (variants.py), by its id; "base" is Anima itself.
        "model": Param("string", "base"),
        **_lora_params(),
    }, optional_inputs=LORA_INPUTS, models=("anima",)),
    # Chroma1-HD (Apache-2.0, Flux-based, uncensored). `fast` swaps in Chroma1-Flash's transformer (CFG baked in) when
    # it is installed; fine-tunes from links always run as full Chroma.
    TaskSpec("chroma", "Chroma1-HD", "image", "chroma", GPU_CHROMA, {
        "prompt": Param("string", ""),
        "negative": Param("string", ""),  # used with guidance above 1 (true classifier-free guidance, 2× per step)
        "seed": Param("int", 0, min=0, max=2**32 - 1),
        "width": Param("int", 1024, min=256, max=2048, multiple=16),
        "height": Param("int", 1024, min=256, max=2048, multiple=16),
        "fast": Param("bool", True),
        # 0 = automatic: 16 steps without guidance when fast, else 40 steps at guidance 3 (the model card's).
        "steps": Param("int", 0, min=0, max=100),
        "guidance": Param("float", 0.0, min=0.0, max=20.0),
        "model": Param("string", "base"),
        # How strongly the face picture (the optional `face` input, chroma-face) steers the result.
        "face_strength": Param("float", 0.6, min=0.0, max=1.0),
        **_lora_params(),
    }, optional_inputs=("face",) + LORA_INPUTS, models=("chroma",)),
    TaskSpec("ace-step-xl", "ACE-Step 1.5 XL music", "audio", "music", GPU_AUDIO, {
        "tags": Param("string", ""),  # the style: genre, instruments, mood, voice
        "lyrics": Param("string", "[Instrumental]"),
        "seconds": Param("float", 30.0, min=10, max=600),
        "seed": SEED,
        "steps": Param("int", 0, min=0, max=100),  # 0 = the turbo model's 8
        "planner": Param("bool", True),  # the 4B language model plans the song (structure, metadata, audio codes)
        "bpm": Param("int", 0, min=0, max=300),  # 0 = auto
        "key": Param("string", "auto", choices=MUSIC_KEYS),
        "time_signature": Param("string", "auto", choices=MUSIC_TIME_SIGNATURES),
        "language": Param("string", "auto", choices=MUSIC_LANGUAGES),
    }, models=("ace-step-xl",)),
    TaskSpec("stable-audio-3", "Stable Audio 3 sound", "audio", "sound", GPU_AUDIO, {
        "prompt": Param("string", ""),
        "seconds": Param("float", 10.0, min=1, max=380),
        "seed": SEED,
        "steps": Param("int", 0, min=0, max=100),  # 0 = the distilled model's 8
    }, models=("stable-audio-3",)),
    TaskSpec("qwen3-tts", "Qwen3-TTS speech", "audio", "speech", GPU_AUDIO, {
        "text": Param("string", ""),
        "language": Param("string", "Auto", choices=SPEECH_LANGUAGES),
        "voice": Param("string", "Ryan", choices=SPEECH_VOICES),
        # With "designed": the voice to make ("a calm older man, British accent"); with a built-in voice: how it
        # speaks ("excited, a little breathless"). Ignored when cloning.
        "description": Param("string", ""),
        "reference_text": Param("string", ""),  # what the reference clip says (better clones; optional)
        "seed": SEED,
    }, optional_inputs=("reference",), models=("qwen3-tts",)),
    # MiniMax H3 (Community Licence: not for the US, EU, UK or Korea without MiniMax's licence; above $20M revenue
    # MiniMax's written permission; products made with it show "MiniMax H3"): video with its own sound, 24 fps.
    TaskSpec("h3-video", "MiniMax H3 video with sound (licence: not US/EU/UK/Korea)", "video", "h3-video", GPU_H3,
             _h3_params(), optional_inputs=("image", "last_image") + LORA_INPUTS, models=("minimax-h3",), fallback_gpus=H3_FALLBACKS),
    TaskSpec("h3-references", "MiniMax H3 from references, with sound (licence: not US/EU/UK/Korea)", "video",
             "h3-references", GPU_H3, _h3_params(), inputs=("image1",),
             optional_inputs=("image2", "image3", "image4", "audio1", "audio2", "audio3") + LORA_INPUTS,
             models=("minimax-h3",),
             fallback_gpus=H3_FALLBACKS),
    # Breeze TTS 2: English and Chinese; research and non-commercial use only (its weights and what it makes).
    TaskSpec("breeze-tts", "Breeze TTS 2 speech (non-commercial)", "audio", "breeze", GPU_AUDIO, {
        "text": Param("string", ""),  # vocal events inline: (laugh), (sigh), (clears throat); in Chinese [笑]
        # With a reference clip: how to say the text ("slowly, restrained"; voice direction); without one: the voice to
        # make ("a warm young woman, calm"; voice design).
        "instruction": Param("string", ""),
        "reference_text": Param("string", ""),  # the reference clip's exact transcript; needed with a reference
        "cfg_scale": Param("float", 0.0, min=0, max=10),  # 0: 4 with an instruction (the repo's advice), else 1
        "seed": SEED,
    }, optional_inputs=("reference",), models=("breeze-tts",)),
    # CPU tasks for tests and for checking a deployment end to end without spending GPU time.
    TaskSpec("test-video", "Test pattern video", "video", "test", "CPU", {
        "width": Param("int", 320, min=16, max=1920, multiple=16),
        "height": Param("int", 192, min=16, max=1080, multiple=16),
        "length": Param("int", 24, min=1, max=240),
        "steps": Param("int", 4, min=1, max=100),
        "step_seconds": Param("float", 0.05, min=0, max=10),
        "fail_at": Param("int", 0, min=0, max=100),  # 0 = never
        "seed": Param("int", 0, min=0, max=2**32 - 1),
    }, cpu_only=True),
    TaskSpec("test-image", "Test pattern image", "image", "test", "CPU", {
        "width": Param("int", 256, min=16, max=4096, multiple=16),
        "height": Param("int", 256, min=16, max=4096, multiple=16),
        "steps": Param("int", 4, min=1, max=100),
        "step_seconds": Param("float", 0.05, min=0, max=10),
        "seed": Param("int", 0, min=0, max=2**32 - 1),
    }, inputs=(), cpu_only=True),
    TaskSpec("test-audio", "Test tone", "audio", "test", "CPU", {
        "seconds": Param("float", 2.0, min=0.1, max=60),
        "sample_rate": Param("int", 24000, min=8000, max=96000),
        "channels": Param("int", 1, min=1, max=2),
        "steps": Param("int", 4, min=1, max=100),
        "step_seconds": Param("float", 0.05, min=0, max=10),
        "fail_at": Param("int", 0, min=0, max=100),
        "seed": SEED,
    }, optional_inputs=("reference",), cpu_only=True),
]}


class JobCancelled(Exception):
    pass


class Sink(Protocol):
    """How a running task reports back. Called from the worker thread."""

    def status(self, text: str | None) -> None: ...
    def progress(self, step: int, total: int) -> None: ...
    # An RGB uint8 array [H, W, 3] (or a PIL image); encoded and rate-limited by the sink.
    def preview(self, image: Any, step: int, total: int) -> None: ...
    # Raises JobCancelled when the job was cancelled; tasks call it between steps.
    def check(self) -> None: ...


@dataclass
class Output:
    path: Path
    kind: str  # "video" | "image" | "audio"
    content_type: str
    load_seconds: float = 0.0  # time this container spent loading models before its first job (cold start)
    extra: dict = field(default_factory=dict)


class Worker(Protocol):
    def load(self, models_root: Path, status: Callable[[str], None]) -> None: ...
    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink: Sink, out_dir: Path) -> Output: ...


def make_worker(family: str) -> Worker:
    """The worker for a task family; imports torch/diffusers only for GPU families."""
    if family == "test":
        from .test_pattern import TestPatternWorker
        return TestPatternWorker()
    if fake_models():
        from .test_pattern import FakeModelWorker
        return FakeModelWorker()
    if family in ("wan22-t2v", "wan22-i2v"):
        from .wan import WanWorker
        return WanWorker(family)
    if family == "wan22-s2v":
        from .wan_s2v import WanS2VWorker
        return WanS2VWorker()
    if family == "longcat-avatar":
        from .longcat_avatar import LongCatAvatarWorker
        return LongCatAvatarWorker()
    if family == "infinitetalk":
        from .infinitetalk import InfiniteTalkWorker
        return InfiniteTalkWorker()
    if family == "flux1-dev":
        from .flux import FluxWorker
        return FluxWorker()
    if family == "qwen-image":
        from .qwen_image import QwenImageWorker
        return QwenImageWorker()
    if family == "qwen-image-edit":
        from .qwen_image_edit import QwenImageEditWorker
        return QwenImageEditWorker()
    if family == "sd35-large":
        from .sd35 import SD35Worker
        return SD35Worker()
    if family == "sdxl":
        from .sdxl import SDXLWorker
        return SDXLWorker()
    if family == "sd15":
        from .sd15 import SD15Worker
        return SD15Worker()
    if family == "anima":
        from .anima import AnimaWorker
        return AnimaWorker()
    if family == "chroma":
        from .chroma import ChromaWorker
        return ChromaWorker()
    if family == "music":
        from .music import MusicWorker
        return MusicWorker()
    if family == "sound":
        from .sound import SoundWorker
        return SoundWorker()
    if family == "speech":
        from .speech import SpeechWorker
        return SpeechWorker()
    if family == "breeze":
        from .breeze import BreezeWorker
        return BreezeWorker()
    if family == "h3-video":
        from .minimax_h3 import H3Worker
        return H3Worker("t2va")
    if family == "h3-references":
        from .minimax_h3 import H3Worker
        return H3Worker("ref2va")
    raise KeyError(family)


def fake_models() -> bool:
    """``AISTUDIO_FAKE_MODELS=1``: every GPU task runs a CPU stand-in (tests of clients; never on a real server)."""
    return os.environ.get("AISTUDIO_FAKE_MODELS") == "1"


def models_installed(spec: TaskSpec, models_root: Path) -> bool:
    """Whether every model folder the task needs has finished downloading (``.complete`` marker)."""
    return fake_models() or all((models_root / m / ".complete").exists() for m in spec.models)
