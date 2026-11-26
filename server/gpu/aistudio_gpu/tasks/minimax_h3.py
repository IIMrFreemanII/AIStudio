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

Community LoRAs (``lora1`` … ``lora4``) load through diffusers' ``MiniMaxH3LoraLoaderMixin``, which converts the
layouts they're published in (musubi-tuner's ``lora_unet_blocks_0_attn_qkv_proj``, ai-toolkit's
``diffusion_model.blocks.0.attn.qkv_proj``, kohya alphas, DiffSynth's interleaved QKV) onto diffusers' names and,
since a worker holds one partition, into that one. A LoRA trained for the other partition still loads (the module
names match) but fits worse, so the app keeps FL2VA and REF2VA LoRAs apart. They stay loaded as adapters
(``adapters.py``) and stack on the Turbo LoRA when ``fast`` is on.

Attention is PyTorch's default: the faster ``_flash_3_hub`` backend fetches its kernels from the Hub at run time, which
the containers (``HF_HUB_OFFLINE``) don't do.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np

from ..media import read_wav, resample, write_mp4, write_wav
from . import Output, TaskSpec
from .adapters import AdapterSlots

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


# Everything resident (the user's rule, residency.py): this partition's 61.7 GiB transformer, the 62.1 GiB conditioner
# and the 9.7 GiB of VAEs (≈ 134 GiB) stay on a B200 (178 GiB), leaving ≈ 45 GiB for a clip's activations. A smaller
# card (an opt-in fallback) or a clip that runs out of memory beside the weights falls back to the offload below.
RESIDENT_MARGIN_GIB = 40

RESIDENT_GIB = 80  # offload fallback: the most model weight kept on the GPU at once: one 61.7 GiB transformer or conditioner, plus the VAE


def offload_margin(total_bytes: int) -> int:
    """The components manager's reserve, in bytes: it moves a component onto the GPU only if this much stays free, else
    offloads others first. With all but ``RESIDENT_GIB`` reserved, the 62 GiB conditioner leaves before the transformer
    comes on, so denoising has the rest for activations (≈ 60 GiB on an H200, ≈ 98 on a B200). A 12 GB margin kept both
    on an H200 and ran out of memory at step 5 of a 6 s clip."""
    return max(12 * 2**30, int(total_bytes - RESIDENT_GIB * 2**30))


class H3Worker(AdapterSlots):
    family = "minimax-h3"
    title = "MiniMax H3"

    def __init__(self, workflow: str):
        self.workflow = workflow  # "t2va" (serves fl2va too) or "ref2va": which transformer partition it loads
        self.components = ("transformer_ref",) if workflow == "ref2va" else ("transformer",)
        self.family = "minimax-h3-ref" if workflow == "ref2va" else "minimax-h3"  # the library's (variants.py)
        self.layout = "full"  # what the loaded transformer is: "full" (released) or "pruned" (variants.py)
        self.manager = None
        self.dtype = None
        self.pipe = None
        self.resident = False  # all parts on the GPU (else the components manager's offload)
        self._init_slots()

    def _load_lora(self, name: str, parts: list[tuple[Path, str]], slot: int) -> None:
        """Loads a community LoRA, then checks it took: diffusers only warns (and loads nothing) when no key names an
        H3 module. Its factors go to bf16 (many files are fp32, which would run the LoRA path in fp32)."""
        import torch

        super()._load_lora(name, parts, slot)
        if not any(name in names for names in self.pipe.get_list_adapters().values()):
            raise ValueError(f"LoRA {slot} doesn't fit {self.title}: none of its layers are MiniMax H3's")
        for component in self.components:
            for n, p in getattr(self.pipe, component).named_parameters():
                if ".lora_" in n and p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)

    def load(self, models_root: Path, status) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # read at the first CUDA allocation
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3 import denoise

        self.models_root = models_root
        self.dtype = torch.bfloat16
        root = models_root / "minimax-h3"
        status("Loading MiniMax H3")
        manager = ComponentsManager()
        # All the blocks (each call picks t2va / fl2va / ref2va from its inputs); only this partition's weights. The
        # index names the Hub repo for every component, so they load from the volume instead.
        self.pipe = ModularPipeline.from_pretrained(str(root), components_manager=manager)
        self.pipe.load_components(workflow=self.workflow, dtype=torch.bfloat16, pretrained_model_name_or_path=str(root))
        self.manager = manager
        self._load_turbo(status)
        self._place(status)
        self._denoise = denoise

    def _modules(self) -> dict:
        from .residency import modules_of

        return modules_of({n: getattr(self.pipe, n, None) for n in self.pipe.components})

    def _place(self, status=lambda _: None) -> None:
        """All parts onto the GPU when they fit with room for a clip (a B200); else the offload fallback."""
        import torch

        from .residency import fits, weight_bytes

        weights = sum(weight_bytes(m) for m in self._modules().values())
        total = torch.cuda.mem_get_info()[1]
        if fits(weights, total, RESIDENT_MARGIN_GIB):
            status("Moving MiniMax H3 onto the GPU")
            self._resident_mode()
        else:
            print(f"h3: {weights / 2**30:.0f} GiB of weights don't fit {total / 2**30:.0f} GiB with room for a clip; "
                  "offloading", flush=True)
            self.resident = False
            self._enable_offload()

    def _resident_mode(self) -> None:
        for module in self._modules().values():
            module.to("cuda")
        self.resident = True

    def _load_turbo(self, status) -> None:
        """LightX2V's Turbo LoRA for this partition, as the adapter "turbo" (the loaded transformer takes it, also a
        pruned one: its ``load_lora_adapter`` projects the released-layout AdaLN factors)."""
        from safetensors import safe_open
        from safetensors.torch import load_file

        status("Loading the Turbo LoRA")
        lora = self.models_root / "minimax-h3" / LORAS[self.workflow]
        with safe_open(str(lora), framework="pt") as f:
            metadata = f.metadata()
        self.pipe.load_lora_weights(prepare_lora(load_file(str(lora)), metadata), adapter_name="turbo")

    def _enable_offload(self) -> None:
        import torch

        self.manager.enable_auto_cpu_offload(device="cuda",
                                             memory_reserve_margin=offload_margin(torch.cuda.mem_get_info()[1]))

    # MARK: fine-tunes

    def _use_h3_variant(self, vid: str, sink) -> None:
        """Makes the loaded transformer this job's model: the base's, or a fine-tune's. Same layout as what's loaded
        (full → full, pruned → pruned): its weights are copied in place (``AdapterSlots._use_variant``, the Turbo LoRA
        and loaded LoRAs stay). A different layout: the partition's transformer is rebuilt."""
        from .. import variants

        vid = vid or "base"
        if vid == self.variant and self.layout:
            return
        if vid != "base" and not self._variant_installed(vid):
            raise ValueError(f"the model {vid!r} isn't installed on this server: install it from the app's Models window")
        layout = "full" if vid == "base" else variants.variant_layout(self.models_root, self.family, vid)
        if layout == self.layout:
            return self._use_variant(vid, sink)
        self._rebuild(vid, layout, sink)

    def _rebuild(self, vid: str, layout: str, sink) -> None:
        """Swaps this partition's transformer for another layout's. With the manager's offload off, the old module is
        unloaded (``unload_components``: attribute to None, out of the ``ComponentsManager``, collected), the new one
        loaded on the CPU and registered (``update_components`` adds it to the manager), the Turbo LoRA attached, and
        offload switched on again with this worker's margin. Memory: the old transformer is gone before the new loads."""
        import torch
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

        from .h3_pruned import MiniMaxH3PrunedTransformer3DModel

        component = self.components[0]
        folder = self._variant_dir(vid, component)
        model_class = MiniMaxH3PrunedTransformer3DModel if layout == "pruned" else MiniMaxH3Transformer3DModel
        sink.status(f"Loading {vid} ({layout} layout)" if vid != "base" else "Loading the base model (full layout)")
        # Unknown until this finishes: a failure leaves no transformer, so the next job rebuilds whatever it needs.
        self.variant, self.layout, self.loaded[component] = None, None, None
        if not self.resident:
            self.manager.disable_auto_cpu_offload()
        self.pipe.unload_components(component)
        self.loras.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = model_class.from_pretrained(str(folder), torch_dtype=self.dtype)
        self.pipe.update_components(**{component: model})
        del model
        if self.resident:
            getattr(self.pipe, component).to("cuda")
        self._load_turbo(lambda _: None)
        if not self.resident:
            self._enable_offload()
        self.loaded[component] = folder
        self.variant, self.layout = vid, layout

    def run(self, task: TaskSpec, params: dict, inputs: dict[str, Path], sink, out_dir: Path) -> Output:
        import torch
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference
        from PIL import Image

        check(task, params, inputs)
        fast = params["fast"]
        steps = params["steps"] or (FAST_STEPS if fast else FULL_STEPS)
        self._use_h3_variant(params.get("model") or "base", sink)
        # The Turbo LoRA when fast, plus this job's LoRAs (loaded once, then only switched).
        loras = self._use_loras(params, inputs, sink, pinned={"turbo": 1.0} if fast else None)
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
        offloaded = False
        try:
            try:
                result = self.pipe(**kwargs)
            except torch.OutOfMemoryError:
                if not self.resident:
                    raise
                # Too big to run beside all the weights: this clip only, with the parts taking turns.
                offloaded = True
                sink.status("Sampling (offloaded: too big for the GPU beside the weights)")
                gc.collect()
                torch.cuda.empty_cache()
                self.resident = False
                self._enable_offload()
                step_times.clear()
                clock.update(last=time.monotonic(), done=0)
                result = self.pipe(**kwargs)
        finally:
            loop.progress_bar = original
            if offloaded:
                self.manager.disable_auto_cpu_offload()
                self._resident_mode()
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
                      extra={"gpu": gpu_name(), "resident": not offloaded and self.resident, "workflow": workflow, "fast": fast, "steps": steps, "loras": loras,
                             "frames": int(len(frames)),
                             "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                             "seconds": round(len(frames) / FPS, 3), "sampleRate": rate,
                             "audioChannels": int(audio.shape[0]), "sampleSeconds": sample_seconds,
                             "setupSeconds": setup_seconds, "stepSeconds": step_seconds})
