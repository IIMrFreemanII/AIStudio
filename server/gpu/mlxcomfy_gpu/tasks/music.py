"""ACE-Step 1.5 XL: songs with vocals and lyrics, or instrumentals, 48 kHz stereo, 10 s – 10 min.

ACE-Step's own inference code (the ``acestep`` package, pinned in the Modal image), not a diffusers port: its 4B
language model (the "planner") first writes the song's metadata and semantic audio codes, then the 4B XL turbo DiT
renders it in 8 steps. ``planner=false`` skips the language model (faster, looser structure).

Models folder layout (``download.py``): ``ace-step-xl/checkpoints/{vae, Qwen3-Embedding-0.6B, acestep-v15-turbo,
acestep-5Hz-lm-1.7B}`` (the main repo, which ACE-Step checks for) plus ``acestep-v15-xl-turbo`` and
``acestep-5Hz-lm-4B``.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

from ..media import write_wav
from . import Output, TaskSpec
from .device import pick_device

DIT = "acestep-v15-xl-turbo"
LM = "acestep-5Hz-lm-4B"


class MusicWorker:
    def __init__(self):
        self.dit = self.llm = None

    def load(self, models_root: Path, status) -> None:
        checkpoints = models_root / "ace-step-xl" / "checkpoints"
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)  # never auto-download into the read-only volume
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        device, _ = pick_device()
        status("Loading ACE-Step 1.5 XL")
        self.dit = AceStepHandler()
        message, ok = self.dit.initialize_service(project_root=str(checkpoints.parent), config_path=DIT,
                                                  device=device)
        if not ok:
            raise RuntimeError(f"ACE-Step DiT: {message}")
        status("Loading the song planner")
        self.llm = LLMHandler()
        # nano-vllm (paged attention, CUDA graphs) on NVIDIA; plain PyTorch elsewhere.
        message, ok = self.llm.initialize(checkpoint_dir=str(checkpoints), lm_model_path=LM,
                                          backend="vllm" if device == "cuda" else "pt", device=device)
        if not ok:
            raise RuntimeError(f"ACE-Step planner: {message}")

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        main = threading.current_thread()
        last = {"desc": None}

        def progress(value=0.0, desc=None, *args, **kwargs):
            # ACE-Step reports a fraction and a phase ("Phase 1: Generating CoT metadata…", "Decoding audio…"),
            # sometimes from a timer thread; cancellation is only raised on the thread running the job.
            if threading.current_thread() is main:
                sink.check()
            if desc and desc != last["desc"]:
                last["desc"] = desc
                sink.status(str(desc).rstrip(". …"))
            try:
                sink.progress(max(0, min(100, int(round(float(value) * 100)))), 100)
            except (TypeError, ValueError):
                pass

        planner = params["planner"]
        gp = GenerationParams(
            task_type="text2music", caption=params["tags"], lyrics=params["lyrics"],
            instrumental=params["lyrics"].strip().lower() in ("", "[instrumental]"),
            vocal_language="unknown" if params["language"] == "auto" else params["language"],
            bpm=params["bpm"] or None, keyscale=_key(params["key"]),
            timesignature="" if params["time_signature"] == "auto" else params["time_signature"],
            duration=params["seconds"], inference_steps=params["steps"] or 8, seed=params["seed"],
            thinking=planner, use_cot_metas=planner, use_cot_caption=planner, use_cot_language=planner)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[params["seed"]], audio_format="wav")
        sink.status("Planning the song" if planner else "Generating")
        start = time.monotonic()
        with tempfile.TemporaryDirectory(dir=out_dir) as save:
            result = generate_music(self.dit, self.llm, gp, config, save_dir=save, progress=progress)
        sink.check()
        if not result.success or not result.audios:
            raise RuntimeError(result.error or result.status_message or "ACE-Step returned no audio")
        audio = result.audios[0]
        rate = int(audio.get("sample_rate") or 48000)
        path = out_dir / "result.wav"
        write_wav(audio["tensor"].float().cpu().numpy(), rate, path)
        meta = (result.extra_outputs or {}).get("lm_metadata") or {}
        return Output(path, "audio", "audio/wav", extra={
            "sampleRate": rate, "sampleSeconds": round(time.monotonic() - start, 2),
            **{k: meta[k] for k in ("bpm", "keyscale", "timesignature", "duration") if isinstance(meta, dict)
               and meta.get(k) not in (None, "")}})


def _key(key: str) -> str:
    """"C# minor" → "C# Minor" (ACE-Step's spelling); "auto" → "" (the planner chooses)."""
    if key == "auto":
        return ""
    note, _, mode = key.partition(" ")
    return f"{note} {mode.capitalize()}".strip()

