"""Breeze TTS 2 (BreezeBlue; research and non-commercial licence): English and Chinese speech, 24 kHz mono.

The official repo (``breezeblue-ai/breeze-tts`` at a pinned commit, on the Qwen3-TTS image, see ``modal_app.py``) runs
in-process the way its ``infer.py`` does, in eager mode (no flash-attn or CUDA graphs). What it is given picks the mode:

- a ``reference`` clip and its exact transcript → **clone** (the clip's voice, rhythm and style);
- the same plus an ``instruction`` → **direction** (that voice, said as instructed: "slowly, restrained");
- an ``instruction`` alone → **design** (a new voice from a description);
- neither → plain speech in the model's default voice.

Vocal events go inline in the text: ``(laugh)``, ``(sigh)``, ``(clears throat)``; in Chinese ``[笑]``. The model is
autoregressive and streams its audio in chunks, so progress is reported as status lines and cancel is checked between
chunks.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import write_wav
from . import Output, TaskSpec

MAX_NEW_TOKENS, MAX_SEQ_LEN, REPETITION_PENALTY = 1500, 2048, 1.1  # infer.py's
INSTRUCTION_CFG = 4.0  # the README's advice for design and direction


def request(params: dict, inputs: dict[str, Path]) -> tuple[dict, str, float]:
    """The repo's request dict, the mode and the guidance scale; refuses what it can't do."""
    text = params["text"].strip()
    if not text:
        raise ValueError("there is no text to speak")
    instruction = params["instruction"].strip()
    ref_text = params["reference_text"].strip()
    req: dict = {"id": "aistudio", "text": text, "speaker": "S0"}
    if instruction:
        req["instruction"] = instruction
    if "reference" in inputs:
        if not ref_text:
            raise ValueError("Breeze needs the reference clip's exact transcript (reference_text) to clone its voice")
        req["ref_audio_path"] = str(inputs["reference"])
        req["ref_text"] = ref_text
        mode = "direction" if instruction else "clone"
    else:
        mode = "design" if instruction else "plain"
    cfg = params["cfg_scale"] or (INSTRUCTION_CFG if instruction else 1.0)
    return req, mode, cfg


class BreezeWorker:
    def __init__(self):
        self.tokenizer = self.model = self.audio_tokenizer = self.runtime = None

    def load(self, models_root: Path, status) -> None:
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

        status("Loading Breeze TTS 2")
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            models_root / "breeze-tts", device=resolve_device(), attn_implementation="eager")
        update_generation_config_for_breeze(self.model)
        config = FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=False,
                                     repetition_penalty=REPETITION_PENALTY)
        self.runtime = FastBreezeStreamingRuntime(self.model, self.audio_tokenizer, config, tokenizer=self.tokenizer)

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        req, mode, cfg = request(params, inputs)
        sink.status({"clone": "Cloning the voice", "direction": "Directing the voice", "design": "Designing the voice",
                     "plain": "Speaking"}[mode])
        start = time.monotonic()
        set_all_seeds(params["seed"])
        prepared = prepare_inputs(self.tokenizer, self.audio_tokenizer, self.model, [req],
                                  get_template(select_template_name(req)), guidance_scale=cfg,
                                  guidance_scale_ref=None, guidance_scale_ins=None)
        chunks = []
        for chunk in self.runtime.iter_audio_chunks(prepared, request_id=req["id"], seed=params["seed"]):
            sink.check()
            chunks.append(np.asarray(chunk.audio, dtype=np.float32).reshape(-1))
        if not chunks:
            raise RuntimeError("Breeze made no audio")
        audio = np.concatenate(chunks)
        rate = int(self.runtime.sample_rate)
        path = out_dir / "result.wav"
        write_wav(audio, rate, path)
        return Output(path, "audio", "audio/wav",
                      extra={"sampleRate": rate, "mode": mode, "cfgScale": cfg, "seconds": round(len(audio) / rate, 2),
                             "sampleSeconds": round(time.monotonic() - start, 2)})
