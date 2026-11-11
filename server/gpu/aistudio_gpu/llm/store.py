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
PATCHABLE = {"name", "gpu", "count", "context", "toolParser", "reasoningParser", "engineArgs"}


def root(models_root: Path) -> Path:
    return models_root / "llm"


def model_dir(models_root: Path, mid: str) -> Path:
    if not isinstance(mid, str) or not ID.match(mid):
        raise ValueError("a model id is 1–64 lowercase letters, digits, '.', '_' or '-'")
    return root(models_root) / mid


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
            out.append({**m, "installed": installed(models_root, d.name)})
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
