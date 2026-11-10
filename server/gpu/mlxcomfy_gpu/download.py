"""Downloading models into the models folder (a Modal Volume, or a folder on a GPU box), once.

Wan 2.2's official diffusers repos store float32 weights (57 GB per transformer). The two transformers come from
bfloat16 copies instead (safetensors only; configs identical to the official ones, sizes exactly 2 bytes per
parameter — checked 2026-11-10); everything else comes from the official repo, with the 11 GB text encoder converted
to bfloat16 here. A cold start then reads ≈ 63 GB. Every repo is pinned to a commit. A folder counts as installed
once its ``.complete`` marker exists (written last).

The audio models: ACE-Step 1.5 XL in ACE-Step's own checkpoint layout (≈ 38 GB), Qwen3-TTS's three 1.7B checkpoints
(≈ 14 GB), and Stable Audio 3 medium, gated, converted here to the diffusers layout with diffusers' own conversion
script (fetched at the diffusers release the server pins, checked by its hash).
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TRANSFORMERS = ["transformer/*", "transformer_2/*"]
MODELS: dict[str, dict] = {
    "wan22-t2v": {"repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
                  "bf16": ["text_encoder"],
                  "mirror": ("ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "fe9c6303ddfdefcae7f06ec7fc7c244bada46cc3")},
    "wan22-i2v": {"repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "revision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
                  "bf16": ["text_encoder"],
                  "mirror": ("cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers", "1f424b0caf6617cfb63822bb0298331bcf44e0e1")},
    # The Lightning 4-step LoRAs (rank 64), laid out as <t2v|i2v>/{high,low}_noise_model.safetensors.
    "wan22-lightning": {"repo": "lightx2v/Wan2.2-Lightning", "revision": "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
                        "files": {
        "t2v/high_noise_model.safetensors": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
        "t2v/low_noise_model.safetensors": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
        "i2v/high_noise_model.safetensors": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "i2v/low_noise_model.safetensors": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    }},
    # Gated: needs a Hugging Face token whose account accepted the licence on the model page.
    "flux1-dev": {"repo": "black-forest-labs/FLUX.1-dev", "revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                  "gated": True, "allow": [
        "model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
        "transformer/*", "vae/*"]},
    # Qwen-Image (the original 20B, Apache 2.0): bf16 throughout (transformer ≈ 41 GB, Qwen2.5-VL text encoder ≈ 17 GB).
    "qwen-image": {"repo": "Qwen/Qwen-Image", "revision": "75e0b4be04f60ec59a75f475837eced720f823b6", "allow": [
        "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "LICENSE"]},
    # Its Lightning 8-step LoRA (V2.0, bf16): `fast` mode, ≈ 6× fewer GPU seconds than 50 steps with CFG.
    "qwen-image-lightning": {"repo": "lightx2v/Qwen-Image-Lightning",
                             "revision": "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a", "files": {
        "8steps.safetensors": "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"}},
    # ACE-Step looks for its main repo's four components in checkpoints/ before it loads the XL DiT and the 4B planner.
    "ace-step-xl": {"repo": "ACE-Step/Ace-Step1.5", "parts": [
        ("ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22", "checkpoints"),
        ("ACE-Step/acestep-v15-xl-turbo", "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee", "checkpoints/acestep-v15-xl-turbo"),
        ("ACE-Step/acestep-5Hz-lm-4B", "0a3ec94b557aea7d508da38b31cfe7341f6ff737", "checkpoints/acestep-5Hz-lm-4B"),
    ]},
    "qwen3-tts": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "parts": [
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3", "base"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe", "custom"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3", "design"),
    ]},
    # Gated (Stability AI Community licence); its T5Gemma text encoder ships inside the repo.
    "stable-audio-3": {"repo": "stabilityai/stable-audio-3-medium", "revision": "d4d3579395a8ac0df4ab13b5b1d7138ce7290f81",
                       "gated": True, "sa3": True},
}
# What a task name expands to.
GROUPS = {"wan22": ["wan22-t2v", "wan22-i2v", "wan22-lightning"], "flux": ["flux1-dev"],
          "qwen": ["qwen-image", "qwen-image-lightning"], "music": ["ace-step-xl"],
          "sound": ["stable-audio-3"], "speech": ["qwen3-tts"], "audio": ["ace-step-xl", "stable-audio-3", "qwen3-tts"]}
SA3_CONVERTER = ("https://raw.githubusercontent.com/huggingface/diffusers/v0.42.1/scripts/"
                 "convert_stable_audio_3_to_diffusers.py",
                 "ee782fa2dfacc637251d00add7cee93f6c3aafcbed6abf111ff209947d98267f")


def expand(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for m in GROUPS.get(n, [n]):
            if m not in MODELS:
                raise SystemExit(f"unknown model {m!r}; choose from {', '.join(list(GROUPS) + list(MODELS))}")
            if m not in out:
                out.append(m)
    return out


def download(names: list[str], root: Path, token: str | None = None, on_done=None) -> None:
    token = token or os.environ.get("HF_TOKEN")
    root.mkdir(parents=True, exist_ok=True)
    for name in expand(names):
        dest = root / name
        if (dest / ".complete").exists():
            print(f"✓ {name} already installed ({size_gb(dest):.1f} GB)")
            continue
        spec = MODELS[name]
        if spec.get("gated") and not token:
            raise SystemExit(f"{name} is gated: set HF_TOKEN (a token whose account accepted the licence at "
                             f"https://huggingface.co/{spec['repo']})")
        start = time.monotonic()
        print(f"↓ {name} from {spec['repo']}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        if "files" in spec:
            _files(spec, dest, token)
        elif "parts" in spec:
            for repo, revision, into in spec["parts"]:
                print(f"  {repo} → {name}/{into}", flush=True)
                _snapshot(repo, dest / into, token, revision=revision, ignore=[".gitattributes"])
        elif spec.get("sa3"):
            _convert_sa3(spec, dest, token)
        elif "bf16" in spec:
            _convert(spec, dest, token)
        else:
            _snapshot(spec["repo"], dest, token, allow=spec.get("allow"), revision=spec.get("revision"))
        (dest / ".complete").write_text(json.dumps({"repo": spec["repo"], "at": time.time()}))
        print(f"✓ {name}: {size_gb(dest):.1f} GB in {time.monotonic() - start:.0f} s", flush=True)
        if on_done:
            on_done(name)  # e.g. commit the Modal Volume so other containers see it


def _snapshot(repo: str, dest: Path, token: str | None, allow=None, ignore=None, revision=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_dir=dest, token=token, allow_patterns=allow, ignore_patterns=ignore,
                             revision=revision)


def _files(spec: dict, dest: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    for target, source in spec["files"].items():
        out = dest / target
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest) as tmp:
            got = hf_hub_download(spec["repo"], source, local_dir=tmp, token=token, revision=spec.get("revision"))
            shutil.move(got, out)


def _convert(spec: dict, dest: Path, token: str | None) -> None:
    """The transformers from the bfloat16 mirror; everything else from the official repo, the ``bf16`` components
    loaded in float32 and saved as bfloat16 (from a temporary cache, deleted right after)."""
    import torch

    repo, big, revision = spec["repo"], spec["bf16"], spec.get("revision")
    skip = [f"{c}/*" for c in big] + [".gitattributes", "*.md", "assets/*", "examples/*"]
    if "mirror" in spec:
        mirror, mirror_rev = spec["mirror"]
        print(f"  transformers (bfloat16) from {mirror}", flush=True)
        _snapshot(mirror, dest, token, allow=TRANSFORMERS, revision=mirror_rev)
        skip += TRANSFORMERS
    _snapshot(repo, dest, token, ignore=skip, revision=revision)
    index = json.loads((dest / "model_index.json").read_text())
    for component in big:
        if (dest / component / "config.json").exists():
            continue
        library, class_name = index[component]
        cls = getattr(importlib.import_module(library), class_name)
        with tempfile.TemporaryDirectory(dir=os.environ.get("MLXCOMFY_DOWNLOAD_TMP")) as cache:
            print(f"  converting {component} ({class_name}) to bfloat16", flush=True)
            model = cls.from_pretrained(repo, subfolder=component, dtype=torch.bfloat16, cache_dir=cache,
                                        token=token, revision=revision, low_cpu_mem_usage=True)
            model.save_pretrained(dest / component, max_shard_size="5GB", safe_serialization=True)
            del model
            gc.collect()


def _convert_sa3(spec: dict, dest: Path, token: str | None) -> None:
    """Stability's checkpoint (stable-audio-tools layout) → the diffusers layout ``StableAudio3Pipeline`` loads, with
    diffusers' own conversion script; the original is deleted afterwards."""
    source = dest / "source"
    _snapshot(spec["repo"], source, token, revision=spec["revision"],
              allow=["model.safetensors", "model_config.json", "t5gemma-b-b-ul2/*", "LICENSE*", "NOTICE"])
    url, sha = SA3_CONVERTER
    script = source / "convert_stable_audio_3_to_diffusers.py"
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"the conversion script at {url} changed (sha256 mismatch); not running it")
    script.write_bytes(data)
    print("  converting Stable Audio 3 to the diffusers layout", flush=True)
    subprocess.run([sys.executable, str(script), "--checkpoint_path", str(source / "model.safetensors"),
                    "--model_config_path", str(source / "model_config.json"), "--output_dir", str(dest),
                    "--text_encoder_repo", str(source / "t5gemma-b-b-ul2"), "--dtype", "float32",
                    "--skip_sanity_check"], check=True)
    for f in ("LICENSE.md", "LICENSE_GEMMA.md", "NOTICE"):
        if (source / f).exists():
            shutil.copy(source / f, dest / f)
    shutil.rmtree(source)


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
