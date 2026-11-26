"""LTX-2.3 (Lightricks, LTX-2 open licence; a 22B DiT): video with its own synchronized sound, 24 fps, up to 10 s,
text to video or from a start picture (diffusers 0.42.1: ``LTX2Pipeline`` and ``LTX2ImageToVideoPipeline``), with up to
four LoRAs per job and fine-tunes from links (``adapters.py``, family ``ltx-2.3``).

``fast`` (the default) runs Lightricks' distilled transformer: 8 steps on the distilled sigma schedule, no guidance (one
transformer pass per step). Off, the base model at 30 steps with CFG, spatio-temporal and modality guidance (three
passes per step). The distilled transformer is a pseudo-variant with the reserved id ``@distilled``, swapped into the
loaded transformer in place when a job asks for the other (as Chroma's ``@flash``); it comes first when it is
installed. A fine-tune keeps its own weights in either mode: ``fast`` then only picks the schedule.

The text-to-video and image-to-video pipelines share every module object (the 49 GB Gemma 3 text encoder, the 38 GB
transformer, the VAEs, the vocoder). All of them stay on the GPU together (the user's rule, residency.py): ≈ 84 GiB on
an H200 (140 GiB) leaves ≈ 56 GiB for a clip. Only on a smaller card (an opt-in fallback) or for a clip that runs out of
memory beside the weights is model offload enabled, on the text-to-video pipeline only: accelerate's hooks sit on the
shared modules, so the other pipeline runs through them (a second hook chain would replace the first one's). LoRAs and
fine-tune weights loaded through the text-to-video pipeline land in the shared modules, so both see them.

The base scheduler stretches and shifts whatever sigmas it is given (dynamic shifting, ``shift_terminal`` 0.1); the
distilled schedule must run as it is, so fast jobs use a scheduler configured like the distilled repo's (no shifting).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..media import write_mp4, write_wav
from . import Output, TaskSpec
from .residency import fits, modules_of, weight_bytes

RESIDENT_MARGIN_GIB = 32  # room for a clip's activations beside all the weights
from .adapters import AdapterSlots
from .device import pick_device
from .minimax_h3 import gpu_name

DISTILLED = "@distilled"  # the variant id of the distilled transformer (not a valid library id)
DISTILLED_DIR = "ltx23-distilled"
FPS = 24
FAST_STEPS, FULL_STEPS = 8, 30
TEMPORAL = 8  # the video VAE's temporal compression: a clip is 8k + 1 frames


def num_frames(seconds: float) -> int:
    """``seconds`` at 24 fps snapped to the nearest 8k + 1 (what LTX needs), at least 9."""
    n = max(1, round(seconds * FPS))
    return max(TEMPORAL + 1, int((n - 1) / TEMPORAL + 0.5) * TEMPORAL + 1)


class LTX23Worker(AdapterSlots):
    family = "ltx-2.3"
    title = "LTX-2.3"
    components = ("transformer",)

    def __init__(self):
        self.pipe = None  # text to video; owns the offload hooks (fallback only)
        self.resident = False  # every module on the GPU
        self.i2v = None  # image to video, over the same modules
        self.has_fast = False
        self.dtype = None
        self._init_slots()

    def load(self, models_root: Path, status) -> None:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ImageToVideoPipeline, LTX2Pipeline
        from diffusers import LTX2VideoTransformer3DModel

        device, dtype = pick_device()
        dtype = torch.bfloat16 if device != "cpu" else dtype
        self.device, self.dtype = device, dtype
        self.models_root = models_root
        extra = {}
        distilled = models_root / DISTILLED_DIR
        if (distilled / ".complete").exists():  # the default mode: start with the distilled transformer
            status("Loading LTX-2.3 (distilled)")
            extra["transformer"] = LTX2VideoTransformer3DModel.from_pretrained(distilled / "transformer",
                                                                               torch_dtype=dtype)
            self.has_fast = True
        else:
            status("Loading LTX-2.3")
        pipe = LTX2Pipeline.from_pretrained(models_root / "ltx23", torch_dtype=dtype, **extra)
        self.i2v = LTX2ImageToVideoPipeline(**pipe.components)
        if device == "cuda":
            # diffusers caches its accelerator lookup, and Modal's memory snapshot runs the imports on a CPU-only
            # machine: forget that answer, or this (and the re-enable after each LoRA load) finds "no accelerator".
            from diffusers.utils.torch_utils import TorchDeviceBackend

            TorchDeviceBackend._detect_device_type.cache_clear()
            weights = sum(weight_bytes(m) for m in modules_of(pipe.components).values())
            total = torch.cuda.mem_get_info()[1]
            if fits(weights, total, RESIDENT_MARGIN_GIB):
                status("Moving LTX-2.3 onto the GPU")
                pipe.to("cuda")
                self.resident = True
            else:
                print(f"ltx23: {weights / 2**30:.0f} GiB of weights don't fit {total / 2**30:.0f} GiB with room for a "
                      "clip; offloading", flush=True)
                pipe.enable_model_cpu_offload(device="cuda")  # only here: the hooks live on the shared modules
        else:
            pipe.to(device)
        for p in (pipe, self.i2v):
            p.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.schedulers = {
            "full": pipe.scheduler,
            # The distilled repo's scheduler: its sigmas are used as given.
            "fast": FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False,
                                                                shift_terminal=None),
        }
        if self.has_fast:
            self.variant = DISTILLED
            self.loaded["transformer"] = distilled / "transformer"

    def _variant_installed(self, vid: str) -> bool:
        if vid == DISTILLED:
            return self.has_fast
        return super()._variant_installed(vid)

    def _variant_dir(self, vid: str, component: str) -> Path:
        if vid == DISTILLED:
            return self.models_root / DISTILLED_DIR / component
        return super()._variant_dir(vid, component)

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """Loads a community LoRA, then checks it took (diffusers only warns when no key fits)."""
        super()._load_lora(name, parts, slot)
        if not any(name in names for names in self.pipe.get_list_adapters().values()):
            self._drop_adapter(name)
            raise ValueError(f"LoRA {slot} doesn't fit {self.title}: none of its layers are LTX-2.3's")

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
        from diffusers.utils import is_av_available
        from PIL import Image

        wanted = params.get("model") or "base"
        # The distilled transformer is the base's fast mode; with it missing, a base job runs the full schedule. A
        # fine-tune keeps its own weights either way.
        fast = bool(params["fast"]) and (wanted != "base" or self.has_fast)
        if wanted == "base" and fast:
            wanted = DISTILLED
        self._use_variant(wanted, sink)
        loras = self._use_loras(params, inputs, sink)

        frames_wanted = num_frames(params["seconds"])
        # One transformer pass per step when fast: the pipeline runs a guidance pass for any scale above 1 (the audio
        # guidance scale defaults to 7, and `audio_x or x` falls back to the video's only for 0), a spatio-temporal
        # pass for a positive STG scale (default 1) and a modality-isolation pass for a modality scale above 1
        # (default 3); the rescale is the identity without them.
        if fast:
            steps = len(DISTILLED_SIGMA_VALUES)
            schedule = {"sigmas": list(DISTILLED_SIGMA_VALUES), "guidance_scale": 1.0, "audio_guidance_scale": 1.0,
                        "stg_scale": 0.0, "audio_stg_scale": 0.0, "modality_scale": 1.0, "audio_modality_scale": 1.0,
                        "guidance_rescale": 0.0, "audio_guidance_rescale": 0.0}
        else:
            steps = params["steps"] or FULL_STEPS
            schedule = {"guidance_scale": params["guidance"]}
        for p in (self.pipe, self.i2v):
            p.register_modules(scheduler=self.schedulers["fast" if fast else "full"])

        def on_step(pipeline, i, t, callback_kwargs):
            sink.progress(i + 1, steps)
            sink.check()
            return callback_kwargs

        kwargs = {"prompt": params["prompt"], "negative_prompt": params["negative"].strip() or DEFAULT_NEGATIVE_PROMPT,
                  "height": params["height"], "width": params["width"], "num_frames": frames_wanted,
                  "frame_rate": float(FPS), "num_inference_steps": steps,
                  "generator": torch.Generator().manual_seed(params["seed"]), "output_type": "np",
                  "return_dict": False, "callback_on_step_end": on_step, **schedule}
        pipe = self.pipe
        if "image" in inputs:
            with Image.open(inputs["image"]) as picture:
                kwargs["image"] = picture.convert("RGB")
            if not is_av_available():
                kwargs["image_crf"] = 0  # the default re-compresses the picture through H.264, which needs PyAV
            pipe = self.i2v
        sink.status("Sampling" + (" (fast)" if fast else ""))
        start = time.monotonic()
        offloaded = False
        try:
            try:
                video, audio = pipe(**kwargs)
            except torch.OutOfMemoryError:
                if not self.resident:
                    raise
                # Too big to run beside all the weights: this clip only, with the modules taking turns.
                offloaded = True
                sink.status("Sampling (offloaded: too big for the GPU beside the weights)")
                torch.cuda.empty_cache()
                self.pipe.enable_model_cpu_offload(device="cuda")
                video, audio = pipe(**kwargs)
        finally:
            if offloaded or not self.resident:
                self.pipe.maybe_free_model_hooks()  # everything back on the CPU, also after a cancel or an error
            if offloaded:
                self.pipe.remove_all_hooks()
                self.pipe.to("cuda")
        sample_seconds = round(time.monotonic() - start, 2)

        sink.status("Encoding video")
        frames = np.asarray(video[0], dtype=np.float32)  # [T, H, W, 3] in 0…1
        samples = audio[0].float().cpu().numpy()  # [channels, n]
        rate = int(self.pipe.vocoder.config.output_sampling_rate)
        sound = out_dir / "sound.wav"
        write_wav(samples, rate, sound)
        path = out_dir / "result.mp4"
        write_mp4(frames, path, fps=FPS, audio=sound)
        return Output(path, "video", "video/mp4",
                      extra={"gpu": gpu_name(), "resident": self.resident and not offloaded, "fast": fast, "steps": steps, "loras": loras,
                             "frames": int(len(frames)), "width": int(frames.shape[2]),
                             "height": int(frames.shape[1]), "seconds": round(len(frames) / FPS, 3),
                             "sampleRate": rate, "sampleSeconds": sample_seconds})
