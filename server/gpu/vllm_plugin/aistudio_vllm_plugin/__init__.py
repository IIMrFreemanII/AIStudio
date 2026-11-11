"""AIStudio's vLLM plugin: weights saved after vLLM's post-processing, for faster cold starts.

A normal load reads the checkpoint tensor by tensor and runs each through the model's Python weight loader; for a
mixture-of-experts model that is ~150k tensors (Qwen3.8-Flash-Next: 512 experts × 48 layers × gate/up/down ×
weight/scale) and ~0.6 ms of Python each, ~110 s however fast the storage is. Then ``process_weights_after_loading``
rearranges the weights into the kernels' layout.

The ``aistudio_processed`` load format keeps a copy of the model's tensors as they are after that processing: a few
thousand large tensors (the experts already fused per layer) plus a manifest of where each lives (GPU, pinned CPU
memory) and what made them (the fingerprint: vLLM and torch versions, model, dtype, KV cache dtype, parallelism,
GPU). A start with a matching copy reads it back into a model built as usual, then runs the processing step in
vLLM's "weights already processed" mode, which only rebuilds kernel state (the mode vLLM's weight-cache IPC loader
uses). Without one (the first start, or after any of those changed) it loads the checkpoint normally and saves a new
copy before the engine finishes starting.

Quant methods that haven't declared they support pre-processed weights make vLLM refuse the fast load, so a model it
can't handle fails at start rather than producing wrong answers.

Usage: ``vllm serve <weights> --load-format aistudio_processed --model-loader-extra-config '{"dir": "<copy>"}'``
(``"save": false`` loads normally without saving when there's no usable copy).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time

MANIFEST = "aistudio-processed.json"
PART = "processed-{:05d}.safetensors"
MAX_PART_BYTES = 4 << 30
PREFETCH_THREADS = 8
LOGGER = "vllm.aistudio"  # under vllm's logger, so its handlers print it


def register() -> None:
    """vLLM general plugin entry point (runs in every vLLM process; must be safe to run more than once)."""
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader("aistudio_processed")(_loader_class())


# Quant methods whose processing step only checks the weights (changes no tensor), so it can run in pre-processed
# mode as is; vLLM just doesn't declare it. By class name, each confirmed from its code before it's marked.
VALIDATION_ONLY = {"Qwen4ExpPLEFp8EmbeddingMethod": "missing its global scale"}
# Anything that could change a tensor: replacing or registering parameters, in-place copies, assigning to the layer.
WRITES = re.compile(r"replace_parameter|register_(parameter|buffer)|setattr|\.data\b|\.copy_\(|\b\w+_\(|"
                    r"\blayer\.\w+(\.\w+)*\s*(=(?!=)|\+=|-=|\*=|/=)")


def validation_only(src: str, marker: str) -> bool:
    """Whether a ``process_weights_after_loading`` source only checks the weights: it has the marker and, after its
    docstring, nothing that could write a tensor."""
    body = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
    return marker in src and not WRITES.search(body)


def allow_validation_only_methods(model) -> list[str]:
    """Marks the model's validation-only quant methods (see ``VALIDATION_ONLY``) as supporting pre-processed weights.
    Done on the built model, when their modules are surely imported. Returns the classes marked."""
    import inspect

    marked = []
    for module in model.modules():
        cls = type(getattr(module, "quant_method", None))
        marker = VALIDATION_ONLY.get(cls.__name__)
        if marker is None or getattr(cls, "supports_pre_processed_weights", False):
            continue
        if validation_only(inspect.getsource(cls.process_weights_after_loading), marker):
            cls.supports_pre_processed_weights = True
            marked.append(cls.__name__)
    return marked


def draft_of(vllm_config, model_config) -> str | None:
    """The architecture of a speculative-decoding draft (an MTP head, an EAGLE model) that vLLM loads with the same
    load format as the target model; None for the target model itself."""
    def arch(mc):
        return ((getattr(getattr(mc, "hf_config", None), "architectures", None) or [None])[0]) if mc else None

    a = arch(model_config)
    return a if model_config is not None and a != arch(vllm_config.model_config) else None


def copy_dir(base: str, vllm_config, model_config) -> str:
    """The target model's copy is in ``base``; a draft's in its own folder inside it."""
    draft = draft_of(vllm_config, model_config)
    return os.path.join(base, "draft-" + re.sub(r"[^A-Za-z0-9_.-]", "_", draft)) if draft else base


def fingerprint(vllm_config, model_config=None) -> dict:
    """What a saved copy depends on: it's used only when all of this matches."""
    import torch
    import vllm

    mc = vllm_config.model_config
    pc = vllm_config.parallel_config
    draft = {"draft": d} if (d := draft_of(vllm_config, model_config)) else {}
    return {**draft, "vllm": vllm.__version__, "torch": torch.__version__, "model": os.path.basename(os.path.normpath(mc.model)),
            "dtype": str(mc.dtype), "quantization": str(mc.quantization),
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            "tp": pc.tensor_parallel_size, "pp": pc.pipeline_parallel_size,
            "expert_parallel": bool(getattr(pc, "enable_expert_parallel", False)),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def mismatch(path: str, vllm_config, model_config=None) -> str | None:
    """Why the copy in ``path`` can't be used (None: it can)."""
    try:
        with open(os.path.join(path, MANIFEST)) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return "there is none yet"
    except (OSError, ValueError) as e:
        return f"its manifest is unreadable ({e})"
    want, have = manifest.get("fingerprint") or {}, fingerprint(vllm_config, model_config)
    diff = {k: (want.get(k), have.get(k)) for k in set(want) | set(have) if want.get(k) != have.get(k)}
    if diff:
        return "it was saved for something else: " + ", ".join(f"{k} {a} → {b}" for k, (a, b) in sorted(diff.items()))
    missing = [i for i in range(manifest.get("parts", 0)) if not os.path.exists(os.path.join(path, PART.format(i)))]
    return f"{len(missing)} of its files are missing" if missing else None


def save_model(model, path: str, fp: dict) -> dict:
    """Writes a processed model's tensors to ``path`` (replacing what's there): first into ``path.tmp``, moved into
    place when complete, so a copy is never half written. Tied tensors are saved once. Returns a summary."""
    from safetensors.torch import save_file

    tmp = path.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    entries: dict[str, dict] = {}
    seen: dict[tuple, str] = {}  # (device, data_ptr, shape, stride) → first name
    named = [(n, t, True) for n, t in model.named_parameters(remove_duplicate=False)]
    named += [(n, t, False) for n, t in model.named_buffers(remove_duplicate=False)]
    part, part_bytes, parts, total = {}, 0, 0, 0
    t0 = time.time()

    def flush():
        nonlocal part, part_bytes, parts
        if part:
            save_file(part, os.path.join(tmp, PART.format(parts)))
            parts += 1
            part, part_bytes = {}, 0

    for name, t, is_param in named:
        if t is None or t.device.type == "meta":
            continue
        key = (str(t.device), t.data_ptr(), tuple(t.shape), tuple(t.stride()))
        info = {"param": is_param, "device": t.device.type, "pinned": bool(t.device.type == "cpu" and t.is_pinned()),
                "dtype": str(t.dtype), "shape": list(t.shape)}
        if t.numel() and key in seen:
            entries[name] = {**info, "alias": seen[key]}
            continue
        seen[key] = name
        size = t.numel() * t.element_size()
        if part_bytes and part_bytes + size > MAX_PART_BYTES:
            flush()
        part[name] = t.detach().contiguous().cpu()
        part_bytes += size
        total += size
        entries[name] = {**info, "part": parts}
    flush()
    with open(os.path.join(tmp, MANIFEST), "w") as f:
        json.dump({"fingerprint": fp, "tensors": entries, "parts": parts, "bytes": total, "saved": time.time()}, f)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return {"tensors": len(entries), "parts": parts, "gb": round(total / 1e9, 1), "seconds": round(time.time() - t0, 1)}


def save_processed(worker, path: str) -> dict:
    """Saves a running engine's model (``LLM.collective_rpc(save_processed, args=(path,))``, one GPU)."""
    return save_model(worker.model_runner.model, path, fingerprint(worker.vllm_config))


def _loader_class():
    import dataclasses

    import torch
    from torch import nn
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
    from vllm.model_executor.utils import weights_already_processed
    from vllm.utils.torch_utils import set_default_torch_dtype

    logger = init_logger(LOGGER)

    class ProcessedWeightsLoader(BaseModelLoader):
        """Loads the saved processed copy, or the checkpoint (then saves a copy). Extra config:
        ``{"dir": "<folder of the copy>", "save": true}``."""

        def __init__(self, load_config):
            super().__init__(load_config)
            extra = dict(load_config.model_loader_extra_config or {})
            self.dir = extra.pop("dir", None)
            self.save = bool(extra.pop("save", True))
            if not self.dir or extra:
                raise ValueError('aistudio_processed needs --model-loader-extra-config \'{"dir": "<folder>"}\' '
                                 f"(and optionally \"save\"), got {load_config.model_loader_extra_config}")

        def download_model(self, model_config) -> None:
            pass

        def load_weights(self, model, model_config) -> None:
            raise NotImplementedError("aistudio_processed loads whole models (load_model)")

        def load_model(self, vllm_config, model_config, prefix: str = "") -> nn.Module:
            path = copy_dir(self.dir, vllm_config, model_config)
            why = mismatch(path, vllm_config, model_config)
            if why is None:
                logger.info("Loading processed weights from %s", path)
                return self._load_processed(path, vllm_config, model_config, prefix)
            logger.info("No usable processed weights in %s (%s): loading the checkpoint", path, why)
            plain = dataclasses.replace(self.load_config, load_format="auto", model_loader_extra_config={})
            model = DefaultModelLoader(plain).load_model(vllm_config, model_config, prefix)
            if self.save:
                logger.info("Saving processed weights to %s (later starts load them instead)", path)
                try:
                    s = save_model(model, path, fingerprint(vllm_config, model_config))
                    logger.info("Saved processed weights: %d tensors, %.1f GB, %d files in %.0f s", s["tensors"],
                                s["gb"], s["parts"], s["seconds"])
                except OSError as e:  # a full volume, say: the engine still starts
                    logger.warning("Couldn't save processed weights to %s: %s", path, e)
                    shutil.rmtree(path.rstrip("/") + ".tmp", ignore_errors=True)
            return model

        def _load_processed(self, path: str, vllm_config, model_config, prefix: str) -> nn.Module:
            from concurrent.futures import ThreadPoolExecutor

            from safetensors import safe_open

            with open(os.path.join(path, MANIFEST)) as f:
                manifest = json.load(f)
            device = vllm_config.device_config.device if self.load_config.device is None else self.load_config.device
            target = torch.device(device)
            t0 = time.time()
            with set_default_torch_dtype(model_config.dtype):
                with target:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                t1 = time.time()
                modules = dict(model.named_modules(remove_duplicate=False))
                entries = manifest["tensors"]
                registered: dict[str, torch.Tensor] = {}

                def put(name: str, tensor: torch.Tensor, info: dict) -> None:
                    module_name, _, leaf = name.rpartition(".")
                    module = modules.get(module_name)
                    if module is None:
                        raise RuntimeError(f"saved tensor {name} has no module in this model")
                    old = (module._parameters if info["param"] else module._buffers).get(leaf)
                    dev = torch.device(info["device"]) if info["device"] != "cuda" else target
                    if (old is not None and old.device.type == dev.type and old.shape == tensor.shape
                            and old.dtype == tensor.dtype):
                        old.data.copy_(tensor, non_blocking=True)  # in place: no second copy of the weights
                        registered[name] = old
                        return
                    module._parameters.pop(leaf, None)
                    module._buffers.pop(leaf, None)
                    moved = tensor.to(dev) if dev.type != "cpu" else (tensor.pin_memory() if info["pinned"] else tensor)
                    if info["param"]:
                        module.register_parameter(leaf, nn.Parameter(moved, requires_grad=False))
                    else:
                        module.register_buffer(leaf, moved)
                    registered[name] = (module._parameters if info["param"] else module._buffers)[leaf]

                # The files are read ahead on several threads (into the page cache: a network volume reads one stream
                # at a time slowly), while this thread moves each finished file's tensors into the model.
                def prefetch(i: int) -> None:
                    with open(os.path.join(path, PART.format(i)), "rb", buffering=0) as fh:
                        while fh.read(16 << 20):
                            pass

                with ThreadPoolExecutor(PREFETCH_THREADS) as pool:
                    ahead = [pool.submit(prefetch, i) for i in range(manifest["parts"])]
                    for i in range(manifest["parts"]):
                        ahead[i].result()
                        with safe_open(os.path.join(path, PART.format(i)), framework="pt", device="cpu") as f:
                            for name in f.keys():  # noqa: SIM118
                                put(name, f.get_tensor(name), entries[name])
                for name, info in entries.items():
                    if "alias" in info:
                        put(name, registered[info["alias"]], info)
                torch.cuda.synchronize()
                t2 = time.time()
                marked = allow_validation_only_methods(model)
                if marked:
                    logger.info("Validation-only processing allowed on pre-processed weights for %s", marked)
                # Only kernel state is rebuilt; methods that can't skip their transforms make this raise.
                with weights_already_processed():
                    process_weights_after_loading(model, model_config, target)
            logger.info("Loaded %d processed tensors (%.1f GB): model built in %.1f s, tensors in %.1f s, kernels in "
                        "%.1f s", len(entries), manifest["bytes"] / 1e9, t1 - t0, t2 - t1, time.time() - t2)
            return model.eval()

    return ProcessedWeightsLoader
