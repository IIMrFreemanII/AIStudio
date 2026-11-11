"""Installed LLMs on the models volume: ``<models>/llm/<id>/llm.json`` (the manifest) and ``weights/`` (the repo's
files as Hugging Face has them). ``weights/.complete`` marks a finished download."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# Fields the app may change after installing (PATCH /v1/llm/models/{id}).
PATCHABLE = {"name", "gpu", "count", "context", "toolParser", "reasoningParser", "engineArgs", "loadFormat", "snapshot",
             "mtp"}
# How vLLM reads the weights: its loader, Run:ai's streamer, or "processed" (AIStudio's vLLM plugin: a copy saved
# after vLLM's processing, made by the first start and loaded by the later ones; see vllm_plugin/).
LOAD_FORMATS = ("auto", "runai", "processed")


def root(models_root: Path) -> Path:
    return models_root / "llm"


def model_dir(models_root: Path, mid: str) -> Path:
    if not isinstance(mid, str) or not ID.match(mid):
        raise ValueError("a model id is 1–64 lowercase letters, digits, '.', '_' or '-'")
    return root(models_root) / mid


def processed_dir(models_root: Path, mid: str) -> Path:
    """The fast-start copies of a model's weights (``loadFormat: "processed"``), one folder per GPU kind, next to its
    checkpoint."""
    return model_dir(models_root, mid) / "processed"


def weights_dir(models_root: Path, mid: str) -> Path:
    return model_dir(models_root, mid) / "weights"


def suggest_id(repo: str) -> str:
    name = repo.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-._") or "model"
    return name[:64]


def read(models_root: Path, mid: str) -> dict | None:
    try:
        return json.loads((model_dir(models_root, mid) / "llm.json").read_text())
    except (OSError, ValueError):
        return None


def write(models_root: Path, manifest: dict) -> None:
    d = model_dir(models_root, manifest["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "llm.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, d / "llm.json")


KV_FILE = "kv-cache-memory.json"  # next to llm.json: {setup key: bytes}, see kv_cache_memory
KV_KEEP = 8  # setups remembered per model


def kv_setup(manifest: dict, *extra) -> str:
    """What the KV cache size vLLM measured depends on: the GPU kind and count, the context, the engine arguments
    (and ``extra``: the server's own defaults that change memory use)."""
    return json.dumps([manifest.get("gpu"), int(manifest.get("count") or 1), manifest.get("context"),
                       list(manifest.get("engineArgs") or []), bool(manifest.get("vision")), *extra])


def kv_cache_memory(models_root: Path, mid: str, setup: str) -> int | None:
    """The KV cache size (bytes) vLLM measured on an earlier start of this setup, or None."""
    try:
        v = json.loads((model_dir(models_root, mid) / KV_FILE).read_text()).get(setup)
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, int) and v > 0 else None


def save_kv_cache_memory(models_root: Path, mid: str, setup: str, nbytes: int | None) -> None:
    """Remembers (or with None forgets) the measured KV cache size of a setup; keeps the latest ``KV_KEEP``."""
    d = model_dir(models_root, mid)
    try:
        known = json.loads((d / KV_FILE).read_text())
        known = known if isinstance(known, dict) else {}
    except (OSError, ValueError):
        known = {}
    known.pop(setup, None)
    if nbytes:
        known[setup] = int(nbytes)
    known = dict(list(known.items())[-KV_KEEP:])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (KV_FILE + ".tmp")
    tmp.write_text(json.dumps(known, indent=2))
    os.replace(tmp, d / KV_FILE)


def mtp_layers(weights: Path) -> int:
    """How many multi-token-prediction (MTP) layers an installed safetensors checkpoint has: its config declares them
    (``mtp_num_hidden_layers``, ``num_nextn_predict_layers``) and its tensors include them (``mtp.…``, or DeepSeek's
    extra layer ``model.layers.<num_hidden_layers>.…``). Quantized re-uploads often drop those tensors while keeping
    the config, and vLLM would then fail to start, so both must agree. 0 when not, or when unreadable."""
    from .facts import safetensors_header

    try:
        config = json.loads((weights / "config.json").read_text())
        text = config.get("text_config") or config.get("llm_config") or config
        merged = {**config, **text}
        n = int(merged.get("mtp_num_hidden_layers") or merged.get("num_nextn_predict_layers") or 0)
        if n <= 0:
            return 0
        index = weights / "model.safetensors.index.json"
        if index.is_file():
            names = list((json.loads(index.read_text()).get("weight_map") or {}).keys())
        else:
            names = []
            for f in sorted(weights.glob("*.safetensors")):
                with open(f, "rb") as fh:
                    names += [k for k in safetensors_header(fh) if k != "__metadata__"]
    except (OSError, ValueError, TypeError):
        return 0
    extra = f"layers.{int(merged.get('num_hidden_layers') or -1)}."
    return n if any(k.startswith("mtp.") or ".mtp." in k or extra in k for k in names) else 0


def with_mtp_layers(models_root: Path, manifest: dict) -> dict:
    """The manifest with ``mtpLayers`` (worked out once from the checkpoint and saved, for models installed before
    the field existed); GGUF models have none."""
    if "mtpLayers" in manifest or manifest.get("engine") == "llamacpp":
        return manifest
    if not installed(models_root, manifest["id"]):
        return manifest
    out = {**manifest, "mtpLayers": mtp_layers(weights_dir(models_root, manifest["id"]))}
    try:
        write(models_root, out)
    except OSError:
        pass
    return out


def installed(models_root: Path, mid: str) -> bool:
    return (weights_dir(models_root, mid) / ".complete").exists()


def listing(models_root: Path) -> list[dict]:
    out = []
    r = root(models_root)
    if not r.is_dir():
        return out
    for d in sorted(r.iterdir()):
        if not d.is_dir() or not ID.match(d.name):
            continue
        m = read(models_root, d.name)
        if m is not None:
            out.append({**with_mtp_layers(models_root, m), "installed": installed(models_root, d.name)})
    return out


def remove(models_root: Path, mid: str) -> bool:
    d = model_dir(models_root, mid)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def patch(manifest: dict, changes: dict) -> dict:
    bad = set(changes) - PATCHABLE
    if bad:
        raise ValueError(f"can't change {', '.join(sorted(bad))}")
    out = {**manifest, **changes}
    if not isinstance(out.get("count"), int) or out["count"] not in (1, 2, 4, 8):
        raise ValueError("count is 1, 2, 4 or 8 GPUs")
    if not isinstance(out.get("context"), int) or out["context"] < 1024:
        raise ValueError("context is at least 1024 tokens")
    if not isinstance(out.get("engineArgs"), list) or not all(isinstance(a, str) for a in out["engineArgs"]):
        raise ValueError("engineArgs is a list of strings")
    if not isinstance(out.get("mtp", True), bool):
        raise ValueError("mtp is true or false")
    if out.get("loadFormat", "auto") not in LOAD_FORMATS:
        raise ValueError("loadFormat is auto, runai or processed")
    if out.get("loadFormat", "auto") != "auto" and out.get("engine") == "llamacpp":
        raise ValueError("the fast loaders are for vLLM models (GGUF files are read by llama.cpp)")
    if out.get("loadFormat") == "processed" and out["count"] != 1:
        raise ValueError("fast start (saved processed weights) works on one GPU only")
    if not isinstance(out.get("snapshot", False), bool):
        raise ValueError("snapshot is true or false")
    if out.get("snapshot") and (out.get("engine") == "llamacpp" or out["count"] != 1):
        raise ValueError("GPU snapshots work for vLLM models on one GPU only")
    out["updated"] = time.time()
    return out


def new_manifest(mid: str, name: str, url: str, facts, profile: dict, gpu: str, count: int, context: int) -> dict:
    return {
        "id": mid, "name": name or facts.repo, "url": url, "repo": facts.repo, "revision": facts.revision,
        "sha": facts.sha, "engine": facts.engine, "source": facts.source, "ggufFile": facts.gguf_file,
        "mmproj": facts.mmproj, "facts": facts.stored(),
        "architecture": facts.architecture, "modelType": facts.model_type,
        "quant": facts.quant, "weightBytes": facts.weight_bytes, "nativeContext": facts.native_context,
        "vision": facts.vision, "experts": facts.experts, "toolParser": profile["toolParser"],
        "reasoningParser": profile["reasoningParser"], "engineArgs": profile["engineArgs"], "gpu": gpu,
        "count": count, "context": context, "created": time.time(),
    }
