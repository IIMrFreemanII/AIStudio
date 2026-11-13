"""AIStudio's GPU server on Modal: pay only while a GPU works.

    pip install modal && modal setup                       # once, signs in
    modal secret create huggingface HF_TOKEN=hf_…          # for FLUX.1 dev (gated); skip for Wan only
    modal secret create civitai CIVITAI_TOKEN=…            # optional: LoRA links from Civitai (most need a key)
    modal run server/gpu/modal_download.py --models wan22,flux,qwen,audio   # the weights, on CPUs (modal_download.py)
    modal deploy server/gpu/modal_app.py                   # prints the URL of `api`; GPUs need a payment method
    # Modal dashboard › Settings › Proxy Auth Tokens → new token; in the app: Servers › + › GPU server,
    # the URL and the token as  wk-….ws-…

Layout:
- ``api``: the front desk (FastAPI on 1 CPU core, proxy-auth protected; it relays LLM streams too). Checking it, uploading, and the CPU test
  tasks never start a GPU. It scales to zero 5 min after the last request.
- ``WanT2V`` / ``WanI2V`` and ``QwenImage`` (H100), ``Flux`` (L40S): GPU workers, one per model family, each with
  the models loaded once per container. They scale to zero ``AISTUDIO_IDLE`` seconds (default 120) after their last job.
- ``Music`` (ACE-Step 1.5 XL), ``Sound`` (Stable Audio 3) and ``Speech`` (Qwen3-TTS), on L40S: the same, for audio.
  ACE-Step and Qwen3-TTS pin their own torch and transformers versions, so each has its own container image.
- Volume ``aistudio-gpu-models``: the weights, written once by ``modal_download.py``.
- ``LLM`` / ``LLMSnap`` / ``LLMLlamaCpp``: one container per installed language model (vLLM's OpenAI server, or llama.cpp's for
  GGUF, on the GPUs its manifest names: ``LLM.with_options(gpu="H200:2")(model=id)``). The app keeps it warm with keepalives while a chat is open;
  it scales to zero ``AISTUDIO_LLM_IDLE`` seconds (default 180) after the last chat or keepalive.
- Volume ``aistudio-gpu-uploads``: the clients' uploads (start images, reference clips, LoRAs), kept across sessions
  up to ``AISTUDIO_UPLOADS_GB`` (default 20; least recently used go first). Workers copy inputs straight from it.

Environment variables at deploy time: ``AISTUDIO_IDLE``, ``AISTUDIO_GPU_WAN`` (default H100),
``AISTUDIO_GPU_FLUX`` (default L40S), ``AISTUDIO_GPU_QWEN`` (default H100; on a 48 GB L40S its weights are stored in
fp8), ``AISTUDIO_GPU_AUDIO`` (default L40S), ``AISTUDIO_UPLOADS_GB`` (default 20), ``AISTUDIO_LLM_IDLE`` (default 180),
``AISTUDIO_VLLM`` (the vLLM version, default 0.33.0), ``AISTUDIO_LLAMA_IMAGE`` (llama.cpp's server image),
``AISTUDIO_LLM_MAX_GPUS`` (default 8), ``AISTUDIO_LLM_VOLUME`` (a separate v2 Volume for language models; unset,
they live on the models Volume).
"""


import os
from pathlib import Path

import modal

import aistudio_gpu  # noqa: F401  (maps MLXCOMFY_* variables from before the rename to their AISTUDIO_* names)

IDLE = int(os.environ.get("AISTUDIO_IDLE", "120"))
GPU_WAN = os.environ.get("AISTUDIO_GPU_WAN", "H100")
GPU_FLUX = os.environ.get("AISTUDIO_GPU_FLUX", "L40S")
GPU_AUDIO = os.environ.get("AISTUDIO_GPU_AUDIO", "L40S")
GPU_QWEN = os.environ.get("AISTUDIO_GPU_QWEN", "H100")
MODELS = Path("/models")
UPLOADS = Path("/uploads")
UPLOADS_GB = float(os.environ.get("AISTUDIO_UPLOADS_GB", "20"))
LLM_IDLE = int(os.environ.get("AISTUDIO_LLM_IDLE", "180"))
VLLM_VERSION = os.environ.get("AISTUDIO_VLLM", "0.33.0")
FLASHINFER_VERSION = os.environ.get("AISTUDIO_FLASHINFER", "0.7.2.post1")  # the one vLLM 0.33.0 depends on
FLASHINFER_CUDA = os.environ.get("AISTUDIO_FLASHINFER_CUDA", "cu129")  # matches the CUDA 12.9 base image
LLM_BOOT_TIMEOUT = int(os.environ.get("AISTUDIO_LLM_BOOT_TIMEOUT", "1800"))  # a start that takes longer is given up
CUDA_BASE = os.environ.get("AISTUDIO_CUDA_IMAGE", "nvidia/cuda:12.9.1-devel-ubuntu24.04")
LLAMA_IMAGE = os.environ.get("AISTUDIO_LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")


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
HF_SECRET = _secret_if_exists(os.environ.get("AISTUDIO_HF_SECRET", "huggingface"))
CIVITAI_SECRET = _secret_if_exists(os.environ.get("AISTUDIO_CIVITAI_SECRET", "civitai"))

app = modal.App("aistudio-gpu")
volume = modal.Volume.from_name("aistudio-gpu-models", create_if_missing=True)
uploads = modal.Volume.from_name("aistudio-gpu-uploads", create_if_missing=True)
# Language models can live on their own Volume (AISTUDIO_LLM_VOLUME, a v2 Volume: higher read throughput than the
# models Volume, which is v1), mounted at /llm-models; unset, they stay under /models/llm.
LLM_VOLUME = os.environ.get("AISTUDIO_LLM_VOLUME", "")
LLM_ROOT = Path("/llm-models") if LLM_VOLUME else MODELS
llm_volume = (modal.Volume.from_name(LLM_VOLUME, version=2, create_if_missing=True) if LLM_VOLUME else volume)
LLM_VOLUMES = {MODELS: volume, LLM_ROOT: llm_volume} if LLM_VOLUME else {MODELS: volume}
events = modal.Queue.from_name("aistudio-gpu-events", create_if_missing=True)
cancels = modal.Dict.from_name("aistudio-gpu-cancels", create_if_missing=True)
# Fine-tune imports in progress ("family/id" → state), written by `import_model`, read by the front desk.
imports = modal.Dict.from_name("aistudio-gpu-imports", create_if_missing=True)
# LLM engines (id → {state, phase, progress, started, beat, gpu, count, …}), written by the LLM containers.
llm_state = modal.Dict.from_name("aistudio-gpu-llm", create_if_missing=True)
# LLM installs in progress (id → state; "id:cancel" → True), written by `import_llm`.
llm_installs = modal.Dict.from_name("aistudio-gpu-llm-installs", create_if_missing=True)

SERVER_DEPS = ["fastapi==0.162.2", "httpx==0.28.1", "numpy==2.5.4", "pillow", "imageio-ffmpeg"]
TORCH_DEPS = ["torch==2.15.0", "diffusers==0.42.1", "transformers==5.22.2", "accelerate==1.15.0", "peft==0.21.2",
              "sentencepiece", "ftfy", "safetensors", "huggingface_hub"]
# Also inside the containers, so the module defines the same things there as here.
env = {"AISTUDIO_GPU_WAN": GPU_WAN, "AISTUDIO_GPU_FLUX": GPU_FLUX, "AISTUDIO_GPU_AUDIO": GPU_AUDIO,
       "AISTUDIO_GPU_QWEN": GPU_QWEN,
       "AISTUDIO_IDLE": str(IDLE), "AISTUDIO_UPLOADS_GB": str(UPLOADS_GB), "AISTUDIO_HF_SECRET": HF_SECRET,
       "AISTUDIO_CIVITAI_SECRET": CIVITAI_SECRET, "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1",
       "AISTUDIO_LLM_IDLE": str(LLM_IDLE), "AISTUDIO_VLLM": VLLM_VERSION,
       "AISTUDIO_LLM_MAX_GPUS": os.environ.get("AISTUDIO_LLM_MAX_GPUS", "8"), "AISTUDIO_LLM_VOLUME": LLM_VOLUME}

front_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SERVER_DEPS).env(env)
               .add_local_python_source("aistudio_gpu"))
# Downloads LLMs onto the volume (no torch needed).
llm_import_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SERVER_DEPS, "huggingface_hub")
                    .env(env).add_local_python_source("aistudio_gpu"))
# llama.cpp's server for GGUF models (its CUDA build; the binary lives in /app).
llama_image = (modal.Image.from_registry(LLAMA_IMAGE, add_python="3.12").entrypoint([])
               .uv_pip_install("httpx").env({**env, "PATH": "/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                             "LD_LIBRARY_PATH": "/app"})
               .add_local_python_source("aistudio_gpu"))
# vLLM's OpenAI server (with Run:ai's streamer, the optional fast loader); its compile, Triton and CUDA-graph caches
# live on the models volume, so restarts reuse them.
# FlashInfer's prebuilt kernels: without them vLLM builds FlashInfer's MoE and all-reduce kernels on the container's
# CPUs at every first start (over 25 min for Qwen3.8-Flash-Next on 8 CPUs). The version must match the
# flashinfer-python that vLLM pulls in, and the CUDA tag the base image.
vllm_base = (modal.Image.from_registry(CUDA_BASE, add_python="3.12").entrypoint([])
             .uv_pip_install(f"vllm[runai]=={VLLM_VERSION}", "httpx")
             .uv_pip_install(f"flashinfer-cubin=={FLASHINFER_VERSION}", extra_index_url="https://flashinfer.ai/whl")
             .uv_pip_install(f"flashinfer-jit-cache=={FLASHINFER_VERSION}",
                             extra_index_url=f"https://flashinfer.ai/whl/{FLASHINFER_CUDA}")
             .env({**env, "VLLM_CACHE_ROOT": f"{LLM_ROOT}/llm/.cache/vllm",
                   "TORCHINDUCTOR_CACHE_DIR": f"{LLM_ROOT}/llm/.cache/inductor",
                   "TRITON_CACHE_DIR": f"{LLM_ROOT}/llm/.cache/triton",
                   "FLASHINFER_WORKSPACE_BASE": f"{LLM_ROOT}/llm/.cache/flashinfer"})
             # AIStudio's vLLM plugin: the aistudio_processed load format (weights saved after vLLM's processing).
             .add_local_dir(Path(__file__).parent / "vllm_plugin", "/opt/aistudio_vllm_plugin", copy=True)
             .run_commands("python -m pip install --no-deps --no-build-isolation /opt/aistudio_vllm_plugin"))
vllm_image = vllm_base.add_local_python_source("aistudio_gpu")
# The same, for GPU memory snapshots: sleep/wake endpoints on, and inductor compiling in one thread (Modal's advice
# for snapshots that include torch.compile). Local code goes last: Modal refuses build steps after it.
vllm_snap_image = (vllm_base.env({"VLLM_SERVER_DEV_MODE": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"})
                   .add_local_python_source("aistudio_gpu"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SERVER_DEPS, *TORCH_DEPS).env(env)
             .add_local_python_source("aistudio_gpu"))

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
    .add_local_python_source("aistudio_gpu"))

# Qwen3-TTS: its package pins transformers 4.57.3; installed without its demo UI (gradio).
speech_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsndfile1", "ffmpeg")
    .uv_pip_install("torch==2.10.0", "torchaudio==2.10.0", "transformers==4.57.3", "accelerate==1.12.0", "librosa",
                    "soundfile", "sox", "onnxruntime", "einops", "huggingface_hub", "safetensors",
                    "fastapi==0.162.2", "numpy", "pillow", "imageio-ffmpeg")
    .uv_pip_install("qwen-tts==0.1.1", extra_options="--no-deps")
    .env(env)
    .add_local_python_source("aistudio_gpu"))


def _imports():
    # Captured in the memory snapshot, so a cold container skips importing torch and diffusers.
    import diffusers  # noqa: F401
    import torch  # noqa: F401

    import aistudio_gpu.modal_runner  # noqa: F401


def _music_imports():
    import torch  # noqa: F401
    from acestep import handler, inference, llm_inference  # noqa: F401

    import aistudio_gpu.modal_runner  # noqa: F401


def _speech_imports():
    import qwen_tts  # noqa: F401
    import torch  # noqa: F401

    import aistudio_gpu.modal_runner  # noqa: F401


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
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("wan22-t2v", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_WAN, memory=(16384, 98304))
class WanI2V:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("wan22-i2v", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_FLUX, memory=(8192, 65536))
class Flux:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("flux1-dev", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_QWEN, memory=(16384, 98304))
class QwenImage:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("qwen-image", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(image=music_image, gpu=GPU_AUDIO, memory=(16384, 65536))
class Music:
    @modal.enter(snap=True)
    def imports(self):
        _music_imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("music", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(gpu=GPU_AUDIO, memory=(8192, 32768))
class Sound:
    @modal.enter(snap=True)
    def imports(self):
        _imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("sound", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@_gpu_class(image=speech_image, gpu=GPU_AUDIO, memory=(8192, 32768))
class Speech:
    @modal.enter(snap=True)
    def imports(self):
        _speech_imports()

    @modal.enter(snap=False)
    def load(self):
        from aistudio_gpu.modal_runner import WorkerState

        self.state = WorkerState("speech", MODELS)

    @modal.method()
    def run(self, job_id: str, task: str, params: dict, inputs: dict) -> dict:
        from aistudio_gpu.modal_runner import run_job

        return run_job(self.state, events, cancels, job_id, task, params, inputs, uploads=(uploads, UPLOADS),
                       models=(volume, MODELS))


@app.function(image=gpu_image, cpu=4, memory=(32768, 98304), volumes={MODELS: volume}, timeout=3 * 60 * 60,
              secrets=[modal.Secret.from_name(n) for n in (HF_SECRET, CIVITAI_SECRET) if n])
def import_model(family: str, vid: str, name: str, files: list | None, repo: str | None):
    """Installs a fine-tune from a link onto the models Volume (variants.py), on CPUs: download, convert, commit."""
    import time

    from aistudio_gpu.variants import import_variant

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
        from aistudio_gpu.variants import variant_dir

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

        from aistudio_gpu.variants import list_variants

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
        from aistudio_gpu.variants import remove

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


# MARK: LLMs

def _llm_boot(self, snapshot: bool = False):
    """Starts the model's engine from its manifest. Its state goes to the ``llm_state`` Dict (phases while starting,
    a heartbeat every 15 s while up) for the front desk, and the boot's timing to ``<id>:coldstarts``. With
    ``snapshot`` (``LLMSnap``'s snapshot phase) the engine is warmed up and put to sleep instead, the heartbeat
    stops just before Modal takes the snapshot, and ``_llm_wake`` does the rest. Stop or a failure during a snapshot
    boot ends the container without a snapshot (``_llm_abandon``)."""
    import time

    from aistudio_gpu.llm import store
    from aistudio_gpu.llm.coldstart import ColdStartClock
    from aistudio_gpu.llm.engine import MEDIA_ENV, EngineProcess
    from aistudio_gpu.llm.service import variant

    self.error = None
    self.alive = True
    self.proc = None
    m = store.read(LLM_ROOT, self.model) or {}
    self.manifest = m
    _llm_clock(self, ColdStartClock)
    # "requested" stays, so the app's "starting for N s" still counts the wait for a GPU.
    # gpu, count, updated, snapshot, engine: which container this is, so requests reach it after the settings change.
    base = {"requested": self.clock.requested, "started": self.clock.started, **variant(m),
            "container": os.environ.get("MODAL_TASK_ID")}
    self.current = {**base, "state": "loading", "phase": "Starting engine", "progress": None}

    def on_status(phase, progress):
        _llm_publish(self, state="ready" if phase == "Ready" else "loading", phase=phase, progress=progress)
        self.clock.phase(phase)

    _llm_publish(self)
    _llm_heartbeat(self)  # in a snapshot boot too, until just before the snapshot (Stop works, the status stays live)
    if not m or not store.installed(LLM_ROOT, self.model):
        self.error = f"{self.model} isn't installed on the server"
        _llm_publish(self, state="failed", error=self.error)
        if snapshot:
            _llm_abandon(self, 1)
        _llm_release_floor(self)
        return
    self.proc = EngineProcess(m, LLM_ROOT, on_status, env={MEDIA_ENV: str(UPLOADS)})
    try:
        self.proc.start()
        self.proc.wait_ready(LLM_BOOT_TIMEOUT)
        if snapshot:
            on_status("Warming up", None)
            self.proc.warmup()
            self.proc.sleep()
    except Exception as e:  # noqa: BLE001
        if getattr(self, "stopping", False):  # Stop while it was starting (see _llm_heartbeat): not a failure
            self.alive = False
            _llm_publish(self, state="cold", phase=None, ended=time.time())
        else:
            self.error = str(e)[:4000]
            _llm_publish(self, state="failed", error=self.error)
        self.proc.stop()
        if snapshot:  # no snapshot of a stopped or broken engine
            _llm_abandon(self, 0 if getattr(self, "stopping", False) else 1)
        _llm_release_floor(self)
        modal.experimental.stop_fetching_inputs()
        return
    try:
        llm_volume.commit()  # the compile caches, for the next cold start
    except Exception:  # noqa: BLE001
        pass
    if snapshot:
        if getattr(self, "stopping", False) or \
                (llm_state.get(self.model + ":stop") or 0) > base["started"]:  # Stop pressed during the warmup
            self.proc.stop()
            _llm_publish(self, state="cold", phase=None, ended=time.time())
            _llm_abandon(self)
        _llm_heartbeat_stop(self)
        on_status("Saving a snapshot", None)  # the front desk allows this phase a longer silence
        self.boot_clock = self.clock  # if this same container wakes next, its start is timed from here
        return
    _llm_record(self, "cold")
    _llm_touch(self)  # the floor stays while it's in use: lowering it makes Modal end an idle container at once
    if (llm_state.get(self.model + ":stop") or 0) > base["started"]:
        _llm_stop(self)


def _llm_wake(self):
    """After an ``LLMSnap`` restore (or right after its snapshot was taken): wake the engine, time the start."""
    from aistudio_gpu.llm.coldstart import ColdStartClock

    if self.error or self.proc is None:
        _llm_heartbeat(self)
        return
    _llm_clock(self, ColdStartClock, how="snapshot")
    # A restore comes after the front desk asked again ("requested", from the Dict, is newer than the snapshot's
    # boot). Otherwise this is the container that just took the snapshot: its start (and its cost) began at its
    # boot, so the boot's clock and start time carry on.
    boot = getattr(self, "boot_clock", None)
    restored = boot is None or (self.clock.requested is not None and self.clock.requested > boot.started) \
        or self.clock.started - boot.started > LLM_BOOT_TIMEOUT + 15 * 60  # far longer than a snapshot takes
    if not restored:
        self.clock = boot
    self.current = {**self.current, "started": self.clock.started, "requested": self.clock.requested,
                    "container": os.environ.get("MODAL_TASK_ID"), "state": "loading", "phase": "Waking up",
                    "progress": None}
    self.clock.phase("Waking up")
    self.proc.on_status = lambda phase, progress: (
        _llm_publish(self, state="ready" if phase == "Ready" else "loading", phase=phase, progress=progress),
        self.clock.phase(phase))
    _llm_publish(self)
    _llm_heartbeat(self)
    try:
        self.proc.wake_up()
    except Exception as e:  # noqa: BLE001
        self.error = f"waking the engine after the snapshot failed: {str(e)[:3000]}. Turn GPU snapshots off for it."
        _llm_publish(self, state="failed", error=self.error)
        self.proc.stop()
        _llm_release_floor(self)
        modal.experimental.stop_fetching_inputs()
        return
    _llm_record(self, "snapshot" if restored else "cold")
    _llm_touch(self)  # the floor stays while it's in use (see _llm_heartbeat)


def _llm_clock(self, clock_cls, how: str = "cold"):
    """A new boot clock; the GPU wait counts from when the front desk asked for this container."""
    try:
        prev = llm_state.get(self.model) or {}
    except Exception:  # noqa: BLE001
        prev = {}
    self.clock = clock_cls(requested=prev.get("requested") if prev.get("state") == "starting" else None, how=how)


def _llm_publish(self, **kw):
    import time

    self.current = {**self.current, **kw, "beat": time.time()}
    try:
        llm_state[self.model] = self.current
    except Exception:  # noqa: BLE001
        pass


def _llm_touch(self):
    """A chat or a keepalive ping arrived (or the engine just got ready): the idle clock starts again."""
    import time

    self.last_input = time.time()


def _llm_idle_for(self) -> float:
    """Seconds with no chat in progress and nothing asked (0 while a chat streams)."""
    import time

    if getattr(self, "active", 0) > 0:
        return 0.0
    return time.time() - getattr(self, "last_input", time.time())


def _llm_heartbeat(self):
    """Publishes the state every 15 s until ``_llm_heartbeat_stop`` (or the container stops), and acts on a Stop
    pressed for this container: while starting it ends the engine; once ready it stops the container, so Stop
    works even when the stop request can't reach it (the container takes no new requests after a first Stop)."""
    import threading

    from aistudio_gpu.llm.service import stop_pressed

    self.beat_stop = threading.Event()

    def beat():
        while self.alive and not self.beat_stop.wait(15):
            if not self.alive:
                break
            state = self.current.get("state")
            if state in ("loading", "ready") and getattr(self, "proc", None) is not None:
                try:
                    stop = llm_state.get(self.model + ":stop")
                except Exception:  # noqa: BLE001
                    stop = None
                if stop_pressed(self.current, stop):
                    if state == "loading":
                        # Still starting: end the engine now (the boot then returns and the container stops),
                        # instead of paying for the rest of the start.
                        self.stopping = True
                        self.proc.stop()
                        break
                    _llm_stop(self)
            if state == "ready" and _llm_idle_for(self) >= LLM_IDLE:
                # Nothing asked for LLM_IDLE seconds (no chat, no keepalive ping): let Modal scale it down. With the
                # start's minimum of one lowered, Modal ends an idle container at once, without its scaledown
                # window, so this is the idle timeout.
                _llm_release_floor(self)
            _llm_publish(self)

    self.beat_thread = threading.Thread(target=beat, daemon=True)
    self.beat_thread.start()


def _llm_stop(self):
    """Stop: no new requests, and exit once the ones in progress finish, or after ``STOP_GRACE`` seconds whatever
    is still open (Modal's graceful stop alone waits for every open request, and a chat stream the client stopped
    reading never finishes: the container then ran on, billed, until stopped by hand). Safe to call again."""
    import threading
    import time

    from aistudio_gpu.llm.service import STOP_GRACE

    if getattr(self, "stop_requested", False):
        return
    self.stop_requested = True
    _llm_release_floor(self)
    try:
        modal.experimental.stop_fetching_inputs()
    except Exception as e:  # noqa: BLE001
        print(f"[aistudio] {self.model}: stop_fetching_inputs failed: {e}", flush=True)

    def force():
        time.sleep(STOP_GRACE)
        # Still here: either requests are still open, or the shutdown hook already ran and the container didn't
        # end anyway (seen on B200: engine stopped, container kept its GPU for minutes). Either way, exit.
        print(f"[aistudio] {self.model}: still running {STOP_GRACE:.0f} s after Stop; exiting now", flush=True)
        if getattr(self, "alive", False):
            _llm_shutdown(self)
        os._exit(0)

    threading.Thread(target=force, daemon=True).start()


def _llm_heartbeat_stop(self):
    """Ends the heartbeat (before a memory snapshot: no thread mid-request in it; the wake starts a new one)."""
    if getattr(self, "beat_stop", None) is not None:
        self.beat_stop.set()
        self.beat_thread.join(timeout=30)


def _llm_obj(m: dict):
    """The container class instance that serves the model ``m``: its engine, GPUs and snapshot setting."""
    n = int(m.get("count") or 1)
    gpu = m["gpu"] if n == 1 else f"{m['gpu']}:{n}"
    if m.get("engine") == "llamacpp":
        return LLMLlamaCpp.with_options(gpu=gpu)(model=m["id"])
    if m.get("snapshot") and n == 1:
        return LLMSnap.with_options(gpu=gpu)(model=m["id"], rev=str(m.get("updated") or ""))
    return LLM.with_options(gpu=gpu)(model=m["id"])


def _llm_release_floor(self):
    """Lets Modal scale this model down again. A start asks for one container with ``min_containers=1`` (no request:
    a queued request can't be cancelled and is retried on a new container when its container ends), and that floor
    stays until the engine failed or was stopped, or has been idle for ``LLM_IDLE`` seconds (lowering it makes Modal
    end an idle container at once, so it can't be lowered when the engine gets ready): otherwise an idle container
    would stay up, billed, and a stopped or failed one would be started again."""
    from aistudio_gpu.llm.service import live_manifest

    try:
        floor = llm_state.get(self.model + ":floor")
        if floor:
            # On the container the start asked for (the floor records it), even if the settings changed since.
            _llm_obj(live_manifest(self.manifest or {}, floor)).update_autoscaler(min_containers=0)
            llm_state.pop(self.model + ":floor", None)
    except Exception as e:  # noqa: BLE001 — the front desk releases it too when it sees the engine isn't starting
        print(f"[aistudio] {self.model}: releasing the start's container floor failed: {e}", flush=True)


def _llm_abandon(self, code: int = 0):
    """Ends a snapshot boot without a snapshot (Stop pressed, or the engine failed): Modal would otherwise snapshot a
    stopped engine and restore it from then on."""
    _llm_release_floor(self)
    os._exit(code)


def _llm_record(self, how: str):
    from aistudio_gpu.llm.coldstart import remember

    m = self.manifest or {}
    record = self.clock.finish(gpu=m.get("gpu"), count=m.get("count"), updated=m.get("updated"),
                               loadFormat=m.get("loadFormat") or "auto", snapshot=bool(m.get("snapshot")))
    key = self.model + ":coldstarts"
    try:
        llm_state[key] = remember(llm_state.get(key), record)
    except Exception:  # noqa: BLE001
        pass
    print(f"[aistudio] {self.model} ready in {record['seconds']} s ({how}): "
          + ", ".join(f"{p['name']} {p['seconds']} s" for p in record["phases"]), flush=True)


async def _llm_chat(self, body: dict):
    from aistudio_gpu.llm.engine import EngineError

    if self.error or self.proc is None:
        raise EngineError(self.error or "the engine isn't running")
    import asyncio

    _llm_touch(self)
    self.active = getattr(self, "active", 0) + 1
    try:
        await asyncio.to_thread(_llm_media_ready, body)
        async for chunk in self.proc.chat(body):
            yield chunk
    finally:
        self.active -= 1
        _llm_touch(self)


def _llm_media_ready(body: dict) -> None:
    """Chats' uploaded images and videos (``file:///uploads/…``) are on the uploads Volume, committed by the front
    desk when each upload completed: reload it when one isn't here yet (uploads never change, so only then). A
    reload fails while a file on it is open (another chat's video being read), so it's tried a few times."""
    import time

    from aistudio_gpu.llm.service import local_media

    paths = local_media(body.get("messages") or [], UPLOADS)
    for attempt in range(5):
        if all(p.exists() for p in paths):
            return
        try:
            uploads.reload()
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)


def _llm_shutdown(self):
    """The container's exit hook (Stop, idle scale-down): ends the engine and publishes "cold". The process is then
    made to end within 10 s: after this hook a container has been seen to keep running, and billing, its GPU."""
    import threading
    import time

    self.alive = False
    _llm_release_floor(self)
    if getattr(self, "proc", None) is not None:
        self.proc.stop()
    try:
        llm_state[self.model] = {**self.current, "state": "cold", "phase": None, "ended": time.time(),
                                 "beat": time.time()}
    except Exception:  # noqa: BLE001
        pass
    exit_timer = threading.Timer(10, lambda: os._exit(0))
    exit_timer.daemon = True  # doesn't hold up a container that ends normally
    exit_timer.start()


LLM_CLS = dict(volumes={**LLM_VOLUMES, UPLOADS: uploads}, scaledown_window=LLM_IDLE, timeout=24 * 60 * 60, max_containers=1,
               memory=(65536, 524288), cpu=8)


@app.cls(image=vllm_image, gpu="H100", **LLM_CLS)
@modal.concurrent(max_inputs=64)
class LLM:
    """One language model in safetensors: vLLM's OpenAI server on localhost (GPUs set per model with
    ``LLM.with_options(gpu="H200:2")``)."""

    model: str = modal.parameter()

    @modal.enter()
    def boot(self):
        _llm_boot(self)

    @modal.method()
    async def chat(self, body: dict):
        async for chunk in _llm_chat(self, body):
            yield chunk

    @modal.method()
    def ping(self) -> dict:
        _llm_touch(self)  # the app keeps a model warm while a chat with it is open
        return self.current

    @modal.method()
    def stop(self) -> None:
        _llm_stop(self)

    @modal.exit()
    def shutdown(self):
        _llm_shutdown(self)


@app.cls(image=vllm_snap_image, gpu="H100", enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True}, **LLM_CLS)
@modal.concurrent(max_inputs=64)
class LLMSnap:
    """A vLLM model on one GPU with a GPU memory snapshot (experimental, opt in per model): the first start loads,
    warms up and puts the engine to sleep, then Modal snapshots the container; later starts restore it and wake the
    engine. ``rev`` is the manifest's ``updated`` stamp, so changed settings get a new snapshot."""

    model: str = modal.parameter()
    rev: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def boot(self):
        _llm_boot(self, snapshot=True)

    @modal.enter(snap=False)
    def wake(self):
        _llm_wake(self)

    @modal.method()
    async def chat(self, body: dict):
        async for chunk in _llm_chat(self, body):
            yield chunk

    @modal.method()
    def ping(self) -> dict:
        _llm_touch(self)  # the app keeps a model warm while a chat with it is open
        return self.current

    @modal.method()
    def stop(self) -> None:
        _llm_stop(self)

    @modal.exit()
    def shutdown(self):
        _llm_shutdown(self)


@app.cls(image=llama_image, gpu="L40S", **LLM_CLS)
@modal.concurrent(max_inputs=16)
class LLMLlamaCpp:
    """One GGUF language model: llama.cpp's server on localhost, layers split across the GPUs."""

    model: str = modal.parameter()

    @modal.enter()
    def boot(self):
        _llm_boot(self)

    @modal.method()
    async def chat(self, body: dict):
        async for chunk in _llm_chat(self, body):
            yield chunk

    @modal.method()
    def ping(self) -> dict:
        _llm_touch(self)  # the app keeps a model warm while a chat with it is open
        return self.current

    @modal.method()
    def stop(self) -> None:
        _llm_stop(self)

    @modal.exit()
    def shutdown(self):
        _llm_shutdown(self)


@app.function(image=llm_import_image, cpu=4, memory=8192, volumes=LLM_VOLUMES, timeout=8 * 60 * 60,
              secrets=[modal.Secret.from_name(n) for n in (HF_SECRET, CIVITAI_SECRET) if n])
def import_llm(manifest: dict, files: list, tokens: dict):
    """Downloads an LLM (Hugging Face or Civitai) onto the models Volume (llm/install.py), on CPUs, then commits it."""
    import time

    from aistudio_gpu.llm.install import install

    key = manifest["id"]
    last = {"t": 0.0, "cancel": False}

    def progress(p: dict):
        llm_installs[key] = {**(llm_installs.get(key) or {}), **p}

    def cancelled() -> bool:
        if time.monotonic() - last["t"] > 2:
            last["t"] = time.monotonic()
            last["cancel"] = bool(llm_installs.get(key + ":cancel"))
        return last["cancel"]

    try:
        tokens = {"huggingface": tokens.get("huggingface") or os.environ.get("HF_TOKEN"),
                  "civitai": tokens.get("civitai") or os.environ.get("CIVITAI_TOKEN")}
        install(LLM_ROOT, manifest, files, tokens, progress, cancelled)
        llm_volume.commit()
    except BaseException as e:  # noqa: BLE001
        progress({"state": "cancelled" if isinstance(e, InterruptedError) else "failed",
                  "error": f"{type(e).__name__}: {e}"})
        try:
            llm_volume.commit()
        except Exception:  # noqa: BLE001
            pass


def _llm_service():
    import time

    from aistudio_gpu.llm.engine import EngineError
    from aistudio_gpu.llm.service import (
        ACTIVE,
        LLMService,
        live_manifest,
        stopping,
        variant,
    )

    class ModalLLMService(LLMService):
        """Each model in its own ``LLM`` container; installs as ``import_llm`` calls; state in Dicts."""

        def __init__(self):
            super().__init__(LLM_ROOT)
            self._reloaded = 0.0
            self._installs_cache: dict = {}

        def _obj(self, m: dict):
            return _llm_obj(m)

        def _live(self, m: dict, s: dict):
            """The instance of the container that is up (``s``, its state): with settings saved since it started,
            ``_obj(m)`` names another container, and a request to it would start a second GPU."""
            return _llm_obj(live_manifest(m, s))

        async def _release_floor(self, mid, m):
            """Back to no minimum container (see ``_llm_release_floor``, the container's own release), on the
            container the start asked for."""
            floor = await llm_state.get.aio(mid + ":floor")
            if floor:
                await _llm_obj(live_manifest(m, floor)).update_autoscaler.aio(min_containers=0)
                await llm_state.pop.aio(mid + ":floor", None)

        async def _coldstarts(self, mid):
            try:
                return await llm_state.get.aio(mid + ":coldstarts") or []
            except Exception:  # noqa: BLE001
                return []

        async def _reload(self):
            if time.monotonic() - self._reloaded > 5:
                self._reloaded = time.monotonic()
                try:
                    await llm_volume.reload.aio()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._installs_cache = {k: v async for k, v in llm_installs.items.aio()}
            except Exception:  # noqa: BLE001
                pass

        async def _commit(self):
            try:
                await llm_volume.commit.aio()
            except Exception:  # noqa: BLE001
                pass

        async def _engine_state(self, mid, m):
            s = dict(await llm_state.get.aio(mid) or {"state": "cold"})
            now = time.time()
            # No heartbeat for a minute: the container is gone. Except while Modal saves a GPU snapshot, when the
            # container is frozen and can't beat (a large one takes minutes).
            quiet = 10 * 60 if s.get("phase") == "Saving a snapshot" else 60
            if s.get("state") in ("loading", "ready") and now - (s.get("beat") or 0) > quiet:
                s.update(state="cold", ended=s.get("beat"))
            if s.get("state") == "starting" and now - (s.get("requested") or 0) > 20 * 60:
                s.update(state="cold")
            # Stop pressed for this container: "stopping" until it's gone (it finishes open requests first).
            if s.get("state") in ("loading", "ready"):
                s = stopping(s, await llm_state.get.aio(mid + ":stop"))
            # The start's minimum container outlives the container only if it couldn't lift it (it lifts it itself
            # when idle, failed or stopped): lift it here, so nothing is kept up or started again.
            # (Not while it's ready: lowering the minimum makes Modal end an idle container at once; the container
            # lowers it itself once idle for LLM_IDLE seconds.)
            if m and s.get("state") in ("cold", "failed"):
                try:
                    await self._release_floor(mid, m)
                except Exception as e:  # noqa: BLE001
                    print(f"[aistudio] {mid}: releasing the start's container floor failed: {e}", flush=True)
            return s

        async def _engine_warm(self, mid, m):
            s = await self._engine_state(mid, m)
            if s.get("state") in ACTIVE:
                return
            await llm_state.put.aio(mid, {"state": "starting", "phase": "Waiting for a GPU", "requested": time.time(),
                                          **variant(m)})
            # One container without a request (see _llm_release_floor): Stop cancels it by lowering the minimum
            # again, and nothing is left queued to start another one. The floor records which container it's on.
            await llm_state.put.aio(mid + ":floor", variant(m))
            await self._obj(m).update_autoscaler.aio(min_containers=1)

        async def _engine_ping(self, mid, m):
            # Only to a ready container. While it waits for a GPU or loads, a ping waits in Modal's queue, and when
            # that start is stopped the queued pings start a new container (each ping is a request it must answer).
            s = await self._engine_state(mid, m)
            if s.get("state") != "ready":
                return
            await self._live(m, s).ping.spawn.aio()

        async def _engine_stop(self, mid, m):
            s = await self._engine_state(mid, m)
            await llm_state.put.aio(mid + ":stop", time.time())
            # First: with the start's minimum still set, Modal would start a new container for the stopped one.
            await self._release_floor(mid, m)
            if s.get("state") == "ready":
                await self._live(m, s).stop.spawn.aio()
            elif s.get("state") == "starting":  # still waiting for a GPU: with no minimum, none is started
                await llm_state.put.aio(mid, {**s, "state": "cold", "phase": None, "ended": time.time()})
            # While loading, the container's heartbeat sees the stop and ends the engine.

        async def _engine_chat(self, mid, m, body):
            s = await self._engine_state(mid, m)
            if s.get("state") == "failed" and s.get("updated") == m.get("updated") and \
                    time.time() - (s.get("beat") or 0) < 30 * 60:
                raise EngineError(f"the model failed to start: {s.get('error')}. Change its settings or press "
                                  "Start again.")
            if s.get("state") != "ready":
                # Start with the cancellable ping and send the chat only once a container is ready: a chat call
                # queued while no container runs can't be cancelled (Modal can't cancel a generator call), outlives
                # a client that left or pressed Stop, and starts a GPU again each time a container ends without
                # answering it.
                await self._until_ready(mid, m)
                s = await self._engine_state(mid, m)
            # The running container, even with settings saved since it started (they apply from the next start).
            async for chunk in self._live(m, s).chat.remote_gen.aio(body):
                yield chunk

        async def warm(self, mid):
            s = await llm_state.get.aio(mid)
            if s and s.get("state") == "failed":
                await llm_state.put.aio(mid, {"state": "cold"})
            return await super().warm(mid)

        async def _start_install(self, manifest, files, tokens):
            mid = manifest["id"]
            await llm_installs.pop.aio(mid + ":cancel", None)
            state = {"state": "starting", "name": manifest["name"]}
            await llm_installs.put.aio(mid, state)
            self._installs_cache[mid] = state
            await import_llm.spawn.aio(manifest, files, tokens)
            return {"id": mid, **state}

        def _installs(self):
            return {k: v for k, v in self._installs_cache.items() if not k.endswith(":cancel")}

        def _cancel_install(self, mid):
            llm_installs[mid + ":cancel"] = True
            llm_installs.pop(mid, None)
            self._installs_cache.pop(mid, None)

    return ModalLLMService()


@app.function(image=front_image, cpu=1, memory=2048, volumes={**LLM_VOLUMES, UPLOADS: uploads}, scaledown_window=300,
              max_containers=1, timeout=60 * 60,
              secrets=[modal.Secret.from_name(n) for n in (HF_SECRET, CIVITAI_SECRET) if n])
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from aistudio_gpu.api import create_app
    from aistudio_gpu.files import FileStore
    from aistudio_gpu.jobs import JobManager
    from aistudio_gpu.modal_runner import ModalRunner

    data = Path("/tmp/aistudio-gpu")
    runner = ModalRunner({"wan22-t2v": WanT2V, "wan22-i2v": WanI2V, "flux1-dev": Flux, "qwen-image": QwenImage,
                          "music": Music,
                          "sound": Sound, "speech": Speech}, events, cancels, volume, MODELS, uploads_volume=uploads)
    # Uploads live on a Volume (kept across sessions, pruned past the cap); committed as each one completes, so the
    # GPU worker sees it.
    files = FileStore(UPLOADS, max_bytes=int(UPLOADS_GB * 1e9), on_stored=uploads.commit)
    files.prune()
    manager = JobManager(runner, files, data / "results", idle_seconds=IDLE)
    # No token check here: Modal's proxy auth rejects requests without a proxy token before they reach us.
    return create_app(manager, token=None, variants=ModalVariants(), llm=_llm_service())
