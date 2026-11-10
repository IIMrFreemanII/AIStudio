"""Qwen3-TTS 12Hz 1.7B: speech in 10 languages, 24 kHz mono, with three checkpoints kept in memory (≈ 13 GB):

- **Base** clones the voice of a reference clip (a few seconds is enough; its transcript helps but is optional);
- **CustomVoice** speaks with one of nine built-in voices, optionally in a described style ("excited, whispering");
- **VoiceDesign** makes a new voice from a description ("a calm older man with a British accent").

Which one runs: a ``reference`` clip → Base; ``voice == "designed"`` → VoiceDesign; otherwise CustomVoice. The models
are autoregressive (no denoising steps), so progress is reported as status lines only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import write_wav
from . import Output, TaskSpec
from .device import pick_device

CHECKPOINTS = {"base": "Base", "custom": "CustomVoice", "design": "VoiceDesign"}  # folder → model


class SpeechWorker:
    def __init__(self):
        self.models = {}

    def load(self, models_root: Path, status) -> None:
        from qwen_tts import Qwen3TTSModel

        device, dtype = pick_device()
        self.device = device
        for folder, name in CHECKPOINTS.items():
            status(f"Loading Qwen3-TTS {name}")
            self.models[folder] = Qwen3TTSModel.from_pretrained(
                str(models_root / "qwen3-tts" / folder), device_map=device, dtype=dtype, attn_implementation="sdpa")

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch

        text = params["text"].strip()
        if not text:
            raise ValueError("there is no text to speak")
        language, description = params["language"], params["description"].strip()
        torch.manual_seed(params["seed"])
        start = time.monotonic()
        if "reference" in inputs:
            sink.status("Cloning the voice")
            ref_text = params["reference_text"].strip()
            wavs, rate = self.models["base"].generate_voice_clone(
                text=text, language=language, ref_audio=str(inputs["reference"]), ref_text=ref_text or None,
                x_vector_only_mode=not ref_text)  # without a transcript: the speaker embedding alone
            mode = "clone"
        elif params["voice"] == "designed":
            if not description:
                raise ValueError("describe the voice to design (for example: a calm older man, British accent)")
            sink.status("Designing the voice")
            wavs, rate = self.models["design"].generate_voice_design(text=text, instruct=description,
                                                                     language=language)
            mode = "design"
        else:
            sink.status(f"Speaking as {params['voice'].replace('_', ' ')}")
            wavs, rate = self.models["custom"].generate_custom_voice(text=text, speaker=params["voice"],
                                                                     language=language, instruct=description or None)
            mode = "voice"
        sink.check()
        path = out_dir / "result.wav"
        write_wav(np.asarray(wavs[0], dtype=np.float32), rate, path)
        return Output(path, "audio", "audio/wav", extra={"sampleRate": int(rate), "mode": mode,
                                                         "sampleSeconds": round(time.monotonic() - start, 2)})
