"""Downloading a model's files onto the models volume, one file at a time (so progress and cancel work), straight
from Hugging Face. Only the top-level safetensors, configs, tokenizer and chat template files are fetched; no code
from the repo is run (``trust_remote_code`` stays off)."""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import store

Progress = Callable[[dict], None]
# (repo, revision, file name, destination folder, token) → downloads that file into the folder.
FetchFile = Callable[[str, str, str, Path, "str | None"], None]


def hf_fetch_file(repo: str, revision: str, name: str, dest: Path, token: str | None) -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo, name, revision=revision, local_dir=dest, token=token)


def _size(folder: Path) -> int:
    total = 0
    for p in folder.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def install(models_root: Path, manifest: dict, files: list[tuple[str, int]], token: str | None, progress: Progress,
            cancelled: Callable[[], bool] = lambda: False, fetch: FetchFile | None = None) -> None:
    """Writes the manifest, downloads ``files`` into ``weights/`` and marks it complete. Raises on failure (the
    folder is removed) or ``InterruptedError`` when cancelled."""
    fetch = fetch or hf_fetch_file
    mid = manifest["id"]
    weights = store.weights_dir(models_root, mid)
    if store.installed(models_root, mid):
        raise ValueError(f"a model named {mid} is already installed")
    weights.mkdir(parents=True, exist_ok=True)
    store.write(models_root, manifest)
    total = sum(b for _, b in files)
    stop = threading.Event()

    def report():  # bytes on disk, also while a big shard is still downloading
        while not stop.wait(2.0):
            progress({"state": "downloading", "bytes": min(_size(weights), total), "total": total})

    watcher = threading.Thread(target=report, daemon=True)
    watcher.start()
    try:
        progress({"state": "downloading", "bytes": 0, "total": total, "files": len(files)})
        rev = manifest.get("sha") or manifest.get("revision") or "main"
        for name, _ in files:
            if cancelled():
                raise InterruptedError("cancelled")
            fetch(manifest["repo"], rev, name, weights, token)
        shutil.rmtree(weights / ".cache", ignore_errors=True)
        if not (weights / "config.json").exists() or not any(weights.glob("*.safetensors")):
            raise RuntimeError("the download is incomplete (no config.json or weights)")
        (weights / ".complete").write_text(str(time.time()))
        stop.set()
        watcher.join(timeout=5)
        progress({"state": "done", "bytes": total, "total": total})
    except BaseException:
        stop.set()
        shutil.rmtree(store.model_dir(models_root, mid), ignore_errors=True)
        raise
    finally:
        stop.set()
