"""Fine-tuned models ("variants") installed from links: a FLUX.1 dev or Qwen-Image fine-tune from Civitai or
Hugging Face becomes ``<models>/variants/<family>/<id>/transformer`` (diffusers layout, bf16), next to the base model
it reuses for the text encoders and VAE. A job picks one with the task's ``model`` parameter; the worker swaps only the transformer.

Sources: a single checkpoint file (FLUX: BFL or ComfyUI names, fp8/bf16/fp16 — converted by diffusers' single-file
loader; diffusers-named files are loaded directly. Qwen-Image: ComfyUI and diffusers files share the names, so they
are copied, scaled fp8 dequantized), or a Hugging Face diffusers repo (its ``transformer/`` folder).
Downloads follow ``fetch.py``'s rules: only Civitai and Hugging Face links, safetensors only, SHA-256 checked.

On Modal the import runs as a CPU function next to the models Volume (``modal_app.py``); on a GPU box, in a thread.
Progress goes to ``progress(dict)``: ``{"state": "downloading|converting|done|failed", "bytes", "total", "error"}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

from .fetch import DEFAULT_HOSTS, MAX_BYTES, REDIRECTS, _family, looks_like_safetensors

# The library's family → the base model folder whose text encoders and VAE a variant reuses.
FAMILIES = {"flux.1": "flux1-dev", "qwen-image": "qwen-image", "wan-2.2-t2v": "wan22-t2v", "wan-2.2-i2v": "wan22-i2v"}
# The task family (tasks/__init__.py) → the library family its `model` parameter picks from.
TASK_FAMILIES = {"flux1-dev": "flux.1", "qwen-image": "qwen-image", "wan22-t2v": "wan-2.2-t2v",
                 "wan22-i2v": "wan-2.2-i2v"}
# The transformers a family's variant can replace. Wan 2.2 A14B has two experts: `transformer` denoises the
# high-noise steps, `transformer_2` the low-noise ones; a fine-tune may ship either or both (the other stays the base's).
COMPONENTS = {"wan-2.2-t2v": ("transformer", "transformer_2"), "wan-2.2-i2v": ("transformer", "transformer_2")}
ROLES = {"high": "transformer", "low": "transformer_2"}


def components(family: str) -> tuple[str, ...]:
    return COMPONENTS.get(family, ("transformer",))
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_MODEL_BYTES = 48_000_000_000  # a bf16 Qwen-Image checkpoint is ≈ 41 GB

Progress = Callable[[dict], None]


def root(models_root: Path) -> Path:
    return models_root / "variants"


def variant_dir(models_root: Path, family: str, vid: str) -> Path:
    if family not in FAMILIES:
        raise ValueError(f"fine-tunes of {family!r} aren't supported (yet): {', '.join(FAMILIES)}")
    if not ID.match(vid):
        raise ValueError(f"bad model id {vid!r}")
    return root(models_root) / family / vid


def transformer_dir(models_root: Path, family: str, vid: str | None, component: str = "transformer") -> Path:
    """The weights to load for one transformer: the base model's for ``base`` (or nothing), else the variant's — or
    the base's, when the variant doesn't replace that one (a Wan fine-tune of one expert)."""
    base = models_root / FAMILIES[family] / component
    if not vid or vid == "base":
        return base
    d = variant_dir(models_root, family, vid)
    if not (d / ".complete").exists():
        return d / component  # not installed: the caller finds no weights there
    return d / component if (d / component).exists() else base


def installed(models_root: Path, family: str, vid: str) -> bool:
    return (variant_dir(models_root, family, vid) / ".complete").exists()


def list_variants(models_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for family in FAMILIES:
        d = root(models_root) / family
        items = []
        for v in sorted(d.iterdir()) if d.exists() else []:
            if (v / ".complete").exists():
                meta = _read_json(v / "variant.json")
                items.append({"id": v.name, "name": meta.get("name", v.name), "bytes": meta.get("bytes")})
        out[family] = items
    return out


def remove(models_root: Path, family: str, vid: str) -> bool:
    d = variant_dir(models_root, family, vid)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


# MARK: importing

def import_variant(models_root: Path, family: str, vid: str, name: str, files: list[dict] | None = None,
                   repo: str | None = None, progress: Progress = lambda _: None, tokens: dict | None = None,
                   extra_hosts: tuple[str, ...] = (), cancelled: Callable[[], bool] = lambda: False) -> Path:
    """Downloads and converts one variant; returns its folder. Raises on failure (and leaves nothing behind)."""
    from .tasks import fake_models

    dest = variant_dir(models_root, family, vid)
    base = models_root / FAMILIES[family]
    fake = fake_models()  # tests: no base model, no conversion — the file is kept as it is
    if not fake and not (base / ".complete").exists():
        raise RuntimeError(f"install the base model first ({FAMILIES[family]})")
    if (dest / ".complete").exists():
        progress({"state": "done"})
        return dest
    tokens = tokens if tokens is not None else {"civitai": os.environ.get("CIVITAI_TOKEN") or None,
                                                "huggingface": os.environ.get("HF_TOKEN") or None}
    parts = components(family)
    work = dest / ".work"
    shutil.rmtree(dest, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        done: list[str] = []
        if repo:
            _download_repo_transformer(repo, work, tokens.get("huggingface"), progress, parts)
            done = [c for c in parts if (work / c).exists()]
        else:
            if not files:
                raise ValueError("nothing to import: give a file link or a Hugging Face repo")
            jobs = []
            for f in files:
                role = (f.get("role") or "").lower()
                if len(parts) > 1 and role not in ROLES:
                    raise ValueError("a Wan 2.2 fine-tune file must say which expert it is (high or low noise)")
                jobs.append((ROLES.get(role, "transformer") if len(parts) > 1 else "transformer", f))
            if len({c for c, _ in jobs}) != len(jobs):
                raise ValueError("two files for the same transformer")
            for i, (component, f) in enumerate(jobs):
                ckpt = work / f"checkpoint-{component}.safetensors"
                download_file(f["url"], ckpt, f.get("sha256"), tokens,
                              (lambda p, i=i: progress({**p, "file": i + 1, "files": len(jobs)})), extra_hosts,
                              cancelled)
                progress({"state": "converting", "file": i + 1, "files": len(jobs)})
                out = work / component
                if fake:
                    out.mkdir()
                    os.replace(ckpt, out / "diffusion_pytorch_model.safetensors")
                else:
                    convert_checkpoint(family, ckpt, base, out, component)
                    ckpt.unlink(missing_ok=True)
                done.append(component)
        if not done:
            raise ValueError("nothing was imported")
        size = 0
        for component in done:
            os.replace(work / component, dest / component)
            size += sum(p.stat().st_size for p in (dest / component).glob("*") if p.is_file())
        (dest / "variant.json").write_text(json.dumps({"id": vid, "name": name, "family": family, "bytes": size,
                                                       "components": done,
                                                       "source": repo or [f.get("url") for f in files or []],
                                                       "added": time.time()}))
        (dest / ".complete").write_text("{}")
        progress({"state": "done", "bytes": size})
        return dest
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def download_file(url: str, dest: Path, sha256: str | None, tokens: dict, progress: Progress,
                  extra_hosts: tuple[str, ...] = (), cancelled: Callable[[], bool] = lambda: False) -> None:
    """One file from a Civitai / Hugging Face link (redirects to their CDNs followed, the key sent only to the site)."""
    import httpx

    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if host not in set(DEFAULT_HOSTS) | set(extra_hosts):
        raise ValueError(f"the server only downloads from {', '.join(DEFAULT_HOSTS)}, not {host or url!r}")
    if parts.scheme != "https" and not (parts.scheme == "http" and host in extra_hosts):
        raise ValueError("only https links")
    token = tokens.get(_family(host) or "")
    hasher, done, head = hashlib.sha256(), 0, b""
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=False) as client:
        for hop in range(10):
            headers = {"User-Agent": "mlxcomfy-gpu"}
            if hop == 0 and token:
                headers["Authorization"] = f"Bearer {token}"
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code in REDIRECTS and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code in (401, 403):
                    raise PermissionError("the site wants an API key for this file (the civitai / huggingface "
                                          "Modal secret; a GPU box: CIVITAI_TOKEN / HF_TOKEN)")
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code} from {urlsplit(url).hostname}")
                total = int(r.headers.get("content-length") or 0) or None
                if total and total > MAX_MODEL_BYTES:
                    raise ValueError(f"the file is {total / 1e9:.1f} GB; the limit is {MAX_MODEL_BYTES / 1e9:.0f} GB")
                last = 0.0
                with open(dest, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        if cancelled():
                            raise InterruptedError("cancelled")
                        if len(head) < 9:
                            head += chunk[:9 - len(head)]
                            if len(head) == 9 and not looks_like_safetensors(head):
                                raise ValueError("the link isn't a .safetensors file")
                        out.write(chunk)
                        hasher.update(chunk)
                        done += len(chunk)
                        if time.monotonic() - last > 1.0:
                            last = time.monotonic()
                            progress({"state": "downloading", "bytes": done, "total": total})
                break
        else:
            raise RuntimeError("too many redirects")
    if sha256 and hasher.hexdigest() != sha256.lower():
        raise ValueError("the download doesn't match the file's published SHA-256; try again")
    progress({"state": "downloading", "bytes": done, "total": done})


def _download_repo_transformer(repo: str, work: Path, token: str | None, progress: Progress,
                               parts: tuple[str, ...] = ("transformer",)) -> None:
    """A Hugging Face diffusers repo's transformer folders (``owner/name`` or ``owner/name@revision``)."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download

    repo_id, _, revision = repo.partition("@")
    progress({"state": "downloading"})
    snapshot_download(repo_id, revision=revision or None, allow_patterns=[f"{c}/*" for c in parts], local_dir=work,
                      token=token)
    for c in parts:
        if (work / c).exists() and not any((work / c).glob("*.safetensors")):
            shutil.rmtree(work / c)
    if not any((work / c).exists() for c in parts):
        raise ValueError(f"{repo} has no {' or '.join(parts)}/*.safetensors")


def convert_checkpoint(family: str, ckpt: Path, base: Path, out: Path, component: str = "transformer") -> None:
    """A single-file transformer checkpoint → a diffusers transformer folder (bf16), with the base's config."""
    import torch
    from diffusers import FluxTransformer2DModel
    from safetensors import safe_open

    if family == "qwen-image":
        return copy_diffusers_named(ckpt, base / component, out)
    if family.startswith("wan-2.2"):
        return convert_wan(ckpt, base / component, out)
    assert family == "flux.1"
    with safe_open(str(ckpt), "pt") as f:
        keys = list(f.keys())
    prefix = "transformer." if any(k.startswith("transformer.transformer_blocks.") for k in keys) else ""
    if any(k.startswith(prefix + "transformer_blocks.") for k in keys):
        # Already diffusers names: build from the base's config and load the weights as they are.
        from safetensors.torch import load_file

        config = FluxTransformer2DModel.load_config(str(base / "transformer"))
        with torch.device("meta"):
            model = FluxTransformer2DModel.from_config(config)
        state = {k[len(prefix):]: v.to(torch.bfloat16) for k, v in load_file(str(ckpt)).items()
                 if k.startswith(prefix)}
        model.load_state_dict(state, strict=True, assign=True)
    else:
        # BFL / ComfyUI names (`double_blocks.*`, `model.diffusion_model.*`), fp8 included: diffusers converts them.
        model = FluxTransformer2DModel.from_single_file(str(ckpt), config=str(base), subfolder="transformer",
                                                        torch_dtype=torch.bfloat16, local_files_only=True)
    model.to(torch.bfloat16).save_pretrained(str(out), max_shard_size="5GB")


PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "")
SHARD_BYTES = 5 * 2**30


def copy_diffusers_named(ckpt: Path, base_transformer: Path, out: Path) -> None:
    """A checkpoint whose transformer already has the base's diffusers names (Qwen-Image: ComfyUI and diffusers
    files alike), under any of ``PREFIXES``; the VAE and text encoders of a full checkpoint are left out. Weights are
    written as bf16 in 5 GB shards, one tensor at a time (memory stays around the largest tensor). fp8 files are
    accepted, ComfyUI's "scaled" ones too (each ``weight`` times its ``scale_weight``)."""
    from safetensors import safe_open

    names = _base_names(base_transformer)
    with safe_open(str(ckpt), "pt") as f:
        keys = set(f.keys())
        prefix = max(PREFIXES, key=lambda p: sum(1 for n in names if p + n in keys))
        _check_complete(names, {k[len(prefix):] for k in keys if k.startswith(prefix)})
        _write_shards(((n, _dequantized(f, keys, prefix + n)) for n in names), out)
    shutil.copy(base_transformer / "config.json", out / "config.json")


def convert_wan(ckpt: Path, base_transformer: Path, out: Path) -> None:
    """One Wan 2.2 expert: the official / ComfyUI names (``blocks.N.self_attn.q``…, KJ's scaled fp8 included) are
    renamed to diffusers' by diffusers' own converter; diffusers-named files are copied. bf16 output."""
    from safetensors import safe_open

    names = _base_names(base_transformer)
    with safe_open(str(ckpt), "pt") as f:
        keys = set(f.keys())
        prefix = max(PREFIXES, key=lambda p: sum(1 for n in names if p + n in keys))
        diffusers_named = sum(1 for n in names if prefix + n in keys) > len(names) // 2
        if not diffusers_named:
            prefix = next((p for p in PREFIXES if any(k.startswith(p + "blocks.") for k in keys)), "")
            state = {k[len(prefix):]: _dequantized(f, keys, k) for k in keys if k.startswith(prefix)
                     and not k.endswith((".scale_weight", ".scale_input")) and k != "scaled_fp8"}
    if diffusers_named:
        return copy_diffusers_named(ckpt, base_transformer, out)
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    state = convert_wan_transformer_to_diffusers(state)
    _check_complete(names, set(state))
    _write_shards(((n, state.pop(n)) for n in names), out)
    shutil.copy(base_transformer / "config.json", out / "config.json")


def _dequantized(f, keys: set, key: str):
    """One tensor as bf16; fp8 weights times their ``scale_weight`` when the file has one (ComfyUI "scaled" fp8)."""
    import torch

    t = f.get_tensor(key)
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        scale = key.rpartition(".")[0] + ".scale_weight"
        t = t.to(torch.float32) * (f.get_tensor(scale).to(torch.float32) if scale in keys else 1.0)
    return t.to(torch.bfloat16).contiguous()


def _check_complete(names: list[str], have: set) -> None:
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(f"this file isn't a full fine-tune of the base model ({len(missing)} of its "
                         f"{len(names)} weights are missing, e.g. {missing[0]}); a LoRA goes in the LoRAs section")


def _write_shards(tensors, out: Path) -> None:
    """(name, bf16 tensor) pairs → diffusers' sharded layout (5 GB shards and an index), written as they come."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    shard: dict = {}
    shard_bytes, files, weight_map = 0, [], {}

    def flush():
        nonlocal shard, shard_bytes
        if shard:
            name = f"part-{len(files) + 1:05d}.safetensors"
            save_file(shard, str(out / name), metadata={"format": "pt"})
            files.append(name)
            weight_map.update({k: name for k in shard})
            shard, shard_bytes = {}, 0

    for n, t in tensors:
        if shard_bytes + t.numel() * 2 > SHARD_BYTES:
            flush()
        shard[n] = t
        shard_bytes += t.numel() * 2
    flush()
    final = {}
    for i, name in enumerate(files, 1):
        new = f"diffusion_pytorch_model-{i:05d}-of-{len(files):05d}.safetensors"
        os.replace(out / name, out / new)
        final[name] = new
    (out / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: final[v] for k, v in weight_map.items()}}))


def _base_names(transformer: Path) -> list[str]:
    """The base transformer's weight names (from its shard index, else its safetensors headers)."""
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        return sorted(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    names = []
    for f in sorted(transformer.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            names += list(sf.keys())
    return sorted(names)


# MARK: running imports

class Imports:
    """Imports running on this machine (a GPU box, tests): one thread each, progress kept here. The API's view of
    variants: ``start``, ``listing``, ``remove`` (the Modal front desk has its own, see ``modal_app.py``)."""

    def __init__(self, models_root: Path, extra_hosts: tuple[str, ...] = ()):
        self.models_root, self.extra_hosts = models_root, extra_hosts
        self.states: dict[str, dict] = {}
        self._cancel: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def start(self, family: str, vid: str, name: str, files: list[dict] | None, repo: str | None) -> dict:
        variant_dir(self.models_root, family, vid)  # validates
        key = f"{family}/{vid}"
        with self.lock:
            if self.states.get(key, {}).get("state") in ("downloading", "converting", "starting"):
                return self.states[key]
            self.states[key] = {"state": "starting"}
            self._cancel[key] = threading.Event()

        def progress(p: dict):
            with self.lock:
                self.states[key] = {**self.states.get(key, {}), **p}

        def run():
            try:
                import_variant(self.models_root, family, vid, name, files, repo, progress,
                               extra_hosts=self.extra_hosts, cancelled=self._cancel[key].is_set)
            except BaseException as e:  # noqa: BLE001
                progress({"state": "failed", "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=run, daemon=True, name=f"import-{vid}").start()
        return self.states[key]

    def status(self) -> dict[str, dict]:
        with self.lock:
            return dict(self.states)

    def listing(self) -> dict:
        return {"families": list_variants(self.models_root),
                "imports": {k: v for k, v in self.status().items() if v.get("state") != "done"}}

    def remove(self, family: str, vid: str) -> bool:
        self.cancel(family, vid)
        with self.lock:
            self.states.pop(f"{family}/{vid}", None)
        return remove(self.models_root, family, vid)

    def cancel(self, family: str, vid: str) -> None:
        ev = self._cancel.get(f"{family}/{vid}")
        if ev:
            ev.set()
