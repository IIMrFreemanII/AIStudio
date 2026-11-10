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

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameter(s) for {self.id}: {', '.join(sorted(unknown))}")
        return {name: p.coerce(name, params.get(name)) for name, p in self.params.items()}

    @property
    def price_per_second(self) -> float | None:
        return None if self.cpu_only else GPU_PRICES.get(self.gpu)

    def describe(self, installed: bool) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind, "gpu": "CPU" if self.cpu_only else self.gpu,
                "pricePerSecond": self.price_per_second, "installed": installed, "inputs": list(self.inputs),
                "optionalInputs": list(self.optional_inputs),
                "params": {k: v.describe() for k, v in self.params.items()}}


FLUX_LORAS = 4  # LoRA slots on the FLUX, Qwen-Image and Wan tasks


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


GPU_WAN = os.environ.get("MLXCOMFY_GPU_WAN", "H100")
GPU_FLUX = os.environ.get("MLXCOMFY_GPU_FLUX", "L40S")
GPU_AUDIO = os.environ.get("MLXCOMFY_GPU_AUDIO", "L40S")
# Qwen-Image's 20B transformer: bf16 on an H100 (80 GB); on a 48 GB card (L40S) its weights are stored in fp8.
GPU_QWEN = os.environ.get("MLXCOMFY_GPU_QWEN", "H100")

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

TASKS: dict[str, TaskSpec] = {t.id: t for t in [
    TaskSpec("wan22-t2v", "Wan 2.2 Text to Video", "video", "wan22-t2v", GPU_WAN, _wan_params(),
             optional_inputs=WAN_LORA_INPUTS, models=("wan22-t2v", "wan22-lightning")),
    TaskSpec("wan22-i2v", "Wan 2.2 Image to Video", "video", "wan22-i2v", GPU_WAN, _wan_params(),
             inputs=("image",), optional_inputs=WAN_LORA_INPUTS, models=("wan22-i2v", "wan22-lightning")),
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
    if family == "flux1-dev":
        from .flux import FluxWorker
        return FluxWorker()
    if family == "qwen-image":
        from .qwen_image import QwenImageWorker
        return QwenImageWorker()
    if family == "music":
        from .music import MusicWorker
        return MusicWorker()
    if family == "sound":
        from .sound import SoundWorker
        return SoundWorker()
    if family == "speech":
        from .speech import SpeechWorker
        return SpeechWorker()
    raise KeyError(family)


def fake_models() -> bool:
    """``MLXCOMFY_FAKE_MODELS=1``: every GPU task runs a CPU stand-in (tests of clients; never on a real server)."""
    return os.environ.get("MLXCOMFY_FAKE_MODELS") == "1"


def models_installed(spec: TaskSpec, models_root: Path) -> bool:
    """Whether every model folder the task needs has finished downloading (``.complete`` marker)."""
    return fake_models() or all((models_root / m / ".complete").exists() for m in spec.models)
