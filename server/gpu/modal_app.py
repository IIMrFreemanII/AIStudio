"""MLXComfyUI's GPU server on Modal: pay only while a GPU works.

    pip install modal && modal setup                       # once, signs in
    modal secret create huggingface HF_TOKEN=hf_…          # for FLUX.1 dev (gated); skip for Wan only
    modal secret create civitai CIVITAI_TOKEN=…            # optional: LoRA links from Civitai (most need a key)
    modal run server/gpu/modal_download.py --models wan22,flux,qwen,audio   # the weights, on CPUs (modal_download.py)
    modal deploy server/gpu/modal_app.py                   # prints the URL of `api`; GPUs need a payment method
    # Modal dashboard › Settings › Proxy Auth Tokens → new token; in the app: Servers › + › GPU server,
    # the URL and the token as  wk-….ws-…

Layout:
- ``api``: the front desk (FastAPI on 0.25 CPU core, proxy-auth protected). Checking it, uploading, and the CPU test
  tasks never start a GPU. It scales to zero 5 min after the last request.
- ``WanT2V`` / ``WanI2V`` and ``QwenImage`` (H100), ``Flux`` (L40S): GPU workers, one per model family, each with
  the models loaded once per container. They scale to zero ``MLXCOMFY_IDLE`` seconds (default 120) after their last job.
- ``Music`` (ACE-Step 1.5 XL), ``Sound`` (Stable Audio 3) and ``Speech`` (Qwen3-TTS), on L40S: the same, for audio.
  ACE-Step and Qwen3-TTS pin their own torch and transformers versions, so each has its own container image.
- Volume ``mlxcomfy-gpu-models``: the weights, written once by ``modal_download.py``.
- Volume ``mlxcomfy-gpu-uploads``: the clients' uploads (start images, reference clips, LoRAs), kept across sessions
  up to ``MLXCOMFY_UPLOADS_GB`` (default 20; least recently used go first). Workers copy inputs straight from it.

Environment variables at deploy time: ``MLXCOMFY_IDLE``, ``MLXCOMFY_GPU_WAN`` (default H100),
``MLXCOMFY_GPU_FLUX`` (default L40S), ``MLXCOMFY_GPU_QWEN`` (default H100; on a 48 GB L40S its weights are stored in
fp8), ``MLXCOMFY_GPU_AUDIO`` (default L40S), ``MLXCOMFY_UPLOADS_GB`` (default 20).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

IDLE = int(os.environ.get("MLXCOMFY_IDLE", "120"))
GPU_WAN = os.environ.get("MLXCOMFY_GPU_WAN", "H100")
GPU_FLUX = os.environ.get("MLXCOMFY_GPU_FLUX", "L40S")
GPU_AUDIO = os.environ.get("MLXCOMFY_GPU_AUDIO", "L40S")
GPU_QWEN = os.environ.get("MLXCOMFY_GPU_QWEN", "H100")
MODELS = Path("/models")
UPLOADS = Path("/uploads")
UPLOADS_GB = float(os.environ.get("MLXCOMFY_UPLOADS_GB", "20"))


def _secret_if_exists(name: str) -> str:
    """The secret's name when it exists in the workspace (checked when deploying), else "" (deploy without it).
    Inside the containers the choice made at deploy time comes back through the environment."""
    if not name or not modal.is_local():
        return name
    try:
        modal.Secret.from_name(name).hydrate()
        return name
    except Exception:  # noqa: BLE001 — not created (yet)
        return ""


# Logins for LoRA links the front desk downloads: HF_TOKEN (gated repos), CIVITAI_TOKEN (most Civitai files).
HF_SECRET = _secret_if_exists(os.environ.get("MLXCOMFY_HF_SECRET", "huggingface"))
CIVITAI_SECRET = _secret_if_exists(os.environ.get("MLXCOMFY_CIVITAI_SECRET", "civitai"))

app = modal.App("mlxcomfy-gpu")
volume = modal.Volume.from_name("mlxcomfy-gpu-models", create_if_missing=True)
uploads = modal.Volume.from_name("mlxcomfy-gpu-uploads", create_if_missing=True)
events = modal.Queue.from_name("mlxcomfy-gpu-events", create_if_missing=True)
cancels = modal.Dict.from_name("mlxcomfy-gpu-cancels", create_if_missing=True)
# Fine-tune imports in progress ("family/id" → state), written by `import_model`, read by the front desk.
imports = modal.Dict.from_name("mlxcomfy-gpu-imports", create_if_missing=True)

SERVER_DEPS = ["fastapi==0.162.2", "httpx==0.28.1", "numpy==2.5.4", "pillow", "imageio-ffmpeg"]
TORCH_DEPS = ["torch==2.15.0", "diffusers==0.42.1", "transformers==5.22.2", "accelerate==1.15.0", "peft==0.21.2",
              "sentencepiece", "ftfy", "safetensors", "huggingface_hub"]
# Also inside the containers, so the module defines the same things there as here.
env = {"MLXCOMFY_GPU_WAN": GPU_WAN, "MLXCOMFY_GPU_FLUX": GPU_FLUX, "MLXCOMFY_GPU_AUDIO": GPU_AUDIO,
       "MLXCOMFY_GPU_QWEN": GPU_QWEN,
       "MLXCOMFY_IDLE": str(IDLE), "MLXCOMFY_UPLOADS_GB": str(UPLOADS_GB), "MLXCOMFY_HF_SECRET": HF_SECRET,
       "MLXCOMFY_CIVITAI_SECRET": CIVITAI_SECRET, "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1"}

front_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SERVER_DEPS).env(env)
               .add_local_python_source("mlxcomfy_gpu"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SERVER_DEPS, *TORCH_DEPS).env(env)
             .add_local_python_source("mlxcomfy_gpu"))

# ACE-Step 1.5's own code at a pinned commit, with the versions its uv.lock pins (torch 2.10 + CUDA 12.8, transformers
# 4.57) and its bundled nano-vllm for the planner; flash-attn from the prebuilt wheel its lock file names (py3.11).
ACESTEP_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
FLASH_ATTN = ("https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/"
              "flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl")
music_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands("git init -q /opt/acestep && cd /opt/acestep && git remote add origin "
                  f"https://github.com/ace-step/ACE-Step-1.5 && git fetch -q --depth 1 origin {ACESTEP_COMMIT} && "
                  "git checkout -q FETCH_HEAD")
    .uv_pip_install("torch==2.10.0", "torchaudio==2.10.0", "torchvision==0.25.0",
                    index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("transformers==4.57.6", "diffusers==0.37.1", "accelerate==1.12.0", "safetensors==0.7.0",
                    "vector-quantize-pytorch==1.27.20", "torchao==0.15.0", "numba==0.63.1", "numpy==2.3.5",
                    "scipy==1.17.0", "soundfile==0.13.1", "loguru==0.7.3", "einops==0.8.2", "peft==0.18.1",
                    "modelscope==1.34.0", "xxhash==3.6.0", "pytorch-wavelets==1.3.0", "pywavelets==1.9.0",
                    "toml==0.10.2", "diskcache==5.6.3", "pyyaml", "tqdm", "huggingface_hub", FLASH_ATTN,
                    "fastapi==0.162.2", "pillow", "imageio-ffmpeg")
    .run_commands("uv pip install --system --no-deps /opt/acestep/acestep/third_parts/nano-vllm")
    .env({**env, "PYTHONPATH": "/opt/acestep", "ACESTEP_CHECKPOINTS_DIR": f"{MODELS}/ace-step-xl/checkpoints"})
    .add_local_python_source("mlxcomfy_gpu"))

# Qwen3-TTS: its package pins transformers 4.57.3; installed without its demo UI (gradio).
speech_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsndfile1", "ffmpeg")
    .uv_pip_install("torch==2.10.0", "torchaudio==2.10.0", "transformers==4.57.3", "accelerate==1.12.0", "librosa",
                    "soundfile", "sox", "onnxruntime", "einops", "huggingface_hub", "safetensors",
                    "fastapi==0.162.2", "numpy", "pillow", "imageio-ffmpeg")
    .uv_pip_install("qwen-tts==0.1.1", extra_options="--no-deps")
    .env(env)
    .add_local_python_source("mlxcomfy_gpu"))


def _imports():
    # Captured in the memory snapshot, so a cold container skips importing torch and diffusers.
    import diffusers  # noqa: F401
    import torch  # noqa: F401

    import mlxcomfy_gpu.modal_runner  # noqa: F401


def _music_imports():
    import torch  # noqa: F401
    from acestep import handler, inference, llm_inference  # noqa: F401

    import mlxcomfy_gpu.modal_runner  # noqa: F401


def _speech_imports():
    import qwen_tts  # noqa: F401
    import torch  # noqa: F401

    import mlxcomfy_gpu.modal_runner  # noqa: F401


def _gpu_class(image=gpu_image, **kwargs):
    return app.cls(image=image, volumes={MODELS: volume, UPLOADS: uploads}, scaledown_window=IDLE, timeout=60 * 60,
                   enable_memory_snapshot=True, max_containers=1, **kwargs)


@_gpu_class(gpu=GPU_WAN, memory=(16384, 98304))
class WanT2V:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("wan22-t2v", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_WAN, memory=(16384, 98304))
class WanI2V:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("wan22-i2v", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_FLUX, memory=(8192, 65536))
class Flux:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("flux1-dev", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_QWEN, memory=(16384, 98304))
class QwenImage:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("qwen-image", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(image=music_image, gpu=GPU_AUDIO, memory=(16384, 65536))
class Music:
    @modal.enter(snap=True)
    def imports(self):
        _music_imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("music", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_AUDIO, memory=(8192, 32768))
class Sound:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("sound", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(image=speech_image, gpu=GPU_AUDIO, memory=(8192, 32768))
class Speech:
    @modal.enter(snap=True)
    def imports(self):
        _speech_imports()

    @modal.enter(snap=False)
    def load(self):
        from mlxcomfy_gpu.modal_runner import WorkerState

        self.state = WorkerState("speech", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from mlxcomfy_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@app.function(image=gpu_image, cpu=4, memory=(32768, 98304), volumes={MODELS: volume}, timeout=3 * 60 * 60,
              secrets=[modal.Secret.from_name(n) for n in (HF_SECRET, CIVITAI_SECRET) if n])
def import_model(family: str, vid: str, name: str, files: list | None, repo: str | None):
    """Installs a fine-tune from a link onto the models Volume (variants.py), on CPUs: download, convert, commit."""
    import time

    from mlxcomfy_gpu.variants import import_variant

    key = f"{family}/{vid}"
    last = {"t": 0.0, "cancel": False}

    def progress(p: dict):
        imports[key] = {**(imports.get(key) or {}), **p}

    def cancelled() -> bool:  # asked at most every 2 s
        if time.monotonic() - last["t"] > 2:
            last["t"] = time.monotonic()
            last["cancel"] = bool(imports.get(key + ":cancel"))
        return last["cancel"]

    try:
        import_variant(MODELS, family, vid, name, files, repo, progress, cancelled=cancelled)
        volume.commit()
    except BaseException as e:  # noqa: BLE001
        progress({"state": "failed", "error": f"{type(e).__name__}: {e}"})


class ModalVariants:
    """The front desk's view of fine-tunes: imports run as `import_model` calls, installed ones are on the Volume."""

    def __init__(self):
        self._reloaded = 0.0

    def start(self, family: str, vid: str, name: str, files, repo) -> dict:
        from mlxcomfy_gpu.variants import variant_dir

        variant_dir(MODELS, family, vid)  # validates
        key = f"{family}/{vid}"
        if (imports.get(key) or {}).get("state") in ("starting", "downloading", "converting"):
            return imports[key]
        imports.pop(key + ":cancel", None)
        imports[key] = {"state": "starting"}
        import_model.spawn(family, vid, name, files, repo)
        return imports[key]

    def listing(self) -> dict:
        import time

        from mlxcomfy_gpu.variants import list_variants

        if time.monotonic() - self._reloaded > 5:
            self._reloaded = time.monotonic()
            try:
                volume.reload()
            except Exception:  # noqa: BLE001
                pass
        running = {}
        for k, v in imports.items():
            if not k.endswith(":cancel") and isinstance(v, dict) and v.get("state") != "done":
                running[k] = v
        return {"families": list_variants(MODELS), "imports": running}

    def remove(self, family: str, vid: str) -> bool:
        from mlxcomfy_gpu.variants import remove

        key = f"{family}/{vid}"
        imports[key + ":cancel"] = True
        imports.pop(key, None)
        try:
            volume.reload()
        except Exception:  # noqa: BLE001
            pass
        removed = remove(MODELS, family, vid)
        if removed:
            volume.commit()
        return removed


@app.function(image=front_image, cpu=0.25, memory=1024, volumes={MODELS: volume, UPLOADS: uploads}, scaledown_window=300,
              max_containers=1, timeout=60 * 60,
              secrets=[modal.Secret.from_name(n) for n in (HF_SECRET, CIVITAI_SECRET) if n])
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from mlxcomfy_gpu.api import create_app
    from mlxcomfy_gpu.files import FileStore
    from mlxcomfy_gpu.jobs import JobManager
    from mlxcomfy_gpu.modal_runner import ModalRunner

    data = Path("/tmp/mlxcomfy-gpu")
    runner = ModalRunner({"wan22-t2v": WanT2V, "wan22-i2v": WanI2V, "flux1-dev": Flux, "qwen-image": QwenImage,
                          "music": Music,
                          "sound": Sound, "speech": Speech}, events, cancels, volume, MODELS, uploads_volume=uploads)
    # Uploads live on a Volume (kept across sessions, pruned past the cap); committed as each one completes, so the
    # GPU worker sees it.
    files = FileStore(UPLOADS, max_bytes=int(UPLOADS_GB * 1e9), on_stored=uploads.commit)
    files.prune()
    manager = JobManager(runner, files, data / "results", idle_seconds=IDLE)
    # No token check here: Modal's proxy auth rejects requests without a proxy token before they reach us.
    return create_app(manager, token=None, variants=ModalVariants())
