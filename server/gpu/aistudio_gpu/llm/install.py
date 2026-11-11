"""Downloading a model's files onto the models volume, one file at a time (so progress and cancel work):

- Hugging Face files by name (``hf_hub_download``; subfolders kept, as split GGUF files may sit in one);
- direct links (Civitai) with redirects followed (the login only to the site), checked against their SHA-256;
- a zipped model folder, unpacked (only that folder) and the zip deleted.

No code from a repo is run (``trust_remote_code`` stays off).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

from . import store
from .remote import REDIRECTS, auth_headers

Progress = Callable[[dict], None]
# (repo, revision, file name, destination folder, token) → downloads that file into the folder.
FetchFile = Callable[[str, str, str, Path, "str | None"], None]
# (url, destination file, tokens, sha256 or None, cancelled) → downloads it.
FetchURL = Callable[[str, Path, dict, "str | None", Callable[[], bool]], None]


def hf_fetch_file(repo: str, revision: str, name: str, dest: Path, token: str | None) -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo, name, revision=revision, local_dir=dest, token=token)


def http_fetch(url: str, dest: Path, tokens: dict, sha256: str | None, cancelled: Callable[[], bool]) -> None:
    import httpx

    hasher = hashlib.sha256()
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=False) as client:
        headers = {"User-Agent": "aistudio-gpu", **auth_headers(url, tokens)}
        for _ in range(10):
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code in REDIRECTS and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    headers = {"User-Agent": "aistudio-gpu"}
                    continue
                if r.status_code in (401, 403):
                    raise PermissionError("the site wants a login for this file (a Civitai API key: the civitai "
                                          "Modal secret, or the key in the app's Models window)")
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code} downloading {dest.name}")
                with open(dest, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        if cancelled():
                            raise InterruptedError("cancelled")
                        out.write(chunk)
                        hasher.update(chunk)
                break
        else:
            raise RuntimeError("too many redirects")
    if sha256 and hasher.hexdigest() != sha256.lower():
        raise ValueError(f"{dest.name} doesn't match its published SHA-256; try again")


def unzip_folder(zip_path: Path, prefix: str, dest: Path) -> None:
    """Unpacks the zip's ``prefix`` folder into ``dest`` (paths checked, so nothing lands outside it)."""
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            rel = PurePosixPath(info.filename[len(prefix):])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in the zip: {info.filename}")
            target = (root / rel).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe path in the zip: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    zip_path.unlink()


def _size(folder: Path) -> int:
    total = 0
    for p in folder.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def install(models_root: Path, manifest: dict, files: list[dict], tokens: dict, progress: Progress,
            cancelled: Callable[[], bool] = lambda: False, fetch: FetchFile | None = None,
            fetch_url: FetchURL | None = None) -> None:
    """Writes the manifest, downloads ``files`` into ``weights/`` and marks it complete. Raises on failure (the
    folder is removed) or ``InterruptedError`` when cancelled."""
    fetch = fetch or hf_fetch_file
    fetch_url = fetch_url or http_fetch
    mid = manifest["id"]
    weights = store.weights_dir(models_root, mid)
    if store.installed(models_root, mid):
        raise ValueError(f"a model named {mid} is already installed")
    weights.mkdir(parents=True, exist_ok=True)
    store.write(models_root, manifest)
    total = sum(int(f.get("size") or 0) for f in files)
    stop = threading.Event()

    def report():  # bytes on disk, also while a big file is still downloading
        while not stop.wait(2.0):
            progress({"state": "downloading", "bytes": min(_size(weights), total), "total": total})

    watcher = threading.Thread(target=report, daemon=True)
    watcher.start()
    try:
        progress({"state": "downloading", "bytes": 0, "total": total, "files": len(files)})
        rev = manifest.get("sha") or manifest.get("revision") or "main"
        for f in files:
            if cancelled():
                raise InterruptedError("cancelled")
            if f.get("url"):
                dest = weights / f["name"]
                fetch_url(f["url"], dest, tokens, f.get("sha256"), cancelled)
                if f.get("extract") is not None:
                    progress({"state": "unpacking"})
                    unzip_folder(dest, f["extract"], weights)
            else:
                fetch(manifest["repo"], rev, f["name"], weights, tokens.get("huggingface"))
        shutil.rmtree(weights / ".cache", ignore_errors=True)
        if manifest.get("engine") == "llamacpp":
            if not (weights / manifest["ggufFile"]).exists():
                raise RuntimeError("the download is incomplete (no GGUF file)")
        elif not (weights / "config.json").exists() or not any(weights.glob("*.safetensors")):
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
