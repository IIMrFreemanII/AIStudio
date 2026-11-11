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
- ``LLM`` / ``LLMLlamaCpp``: one container per installed language model (vLLM's OpenAI server, or llama.cpp's for
  GGUF, on the GPUs its manifest names: ``LLM.with_options(gpu="H200:2")(model=id)``). The app keeps it warm with keepalives while a chat is open;
  it scales to zero ``AISTUDIO_LLM_IDLE`` seconds (default 180) after the last chat or keepalive.
- Volume ``aistudio-gpu-uploads``: the clients' uploads (start images, reference clips, LoRAs), kept across sessions
  up to ``AISTUDIO_UPLOADS_GB`` (default 20; least recently used go first). Workers copy inputs straight from it.

Environment variables at deploy time: ``AISTUDIO_IDLE``, ``AISTUDIO_GPU_WAN`` (default H100),
``AISTUDIO_GPU_FLUX`` (default L40S), ``AISTUDIO_GPU_QWEN`` (default H100; on a 48 GB L40S its weights are stored in
fp8), ``AISTUDIO_GPU_AUDIO`` (default L40S), ``AISTUDIO_UPLOADS_GB`` (default 20), ``AISTUDIO_LLM_IDLE`` (default 180),
``AISTUDIO_VLLM`` (the vLLM version, default 0.33.0), ``AISTUDIO_LLAMA_IMAGE`` (llama.cpp's server image),
``AISTUDIO_LLM_MAX_GPUS`` (default 8).
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
       "AISTUDIO_LLM_MAX_GPUS": os.environ.get("AISTUDIO_LLM_MAX_GPUS", "8")}

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
# vLLM's OpenAI server; its compile and CUDA-graph caches live on the models volume, so restarts reuse them.
vllm_image = (modal.Image.from_registry(CUDA_BASE, add_python="3.12").entrypoint([])
              .uv_pip_install(f"vllm=={VLLM_VERSION}", "httpx")
              .env({**env, "VLLM_CACHE_ROOT": f"{MODELS}/llm/.cache/vllm",
                    "TORCHINDUCTOR_CACHE_DIR": f"{MODELS}/llm/.cache/inductor"})
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

def _llm_boot(self):
    """Starts the model's engine from its manifest. Its state goes to the ``llm_state`` Dict (phases while starting,
    a heartbeat every 15 s while up) for the front desk."""
    import threading
    import time

    from aistudio_gpu.llm import store
    from aistudio_gpu.llm.engine import EngineProcess

    self.error = None
    self.alive = True
    self.proc = None
    m = store.read(MODELS, self.model) or {}
    base = {"started": time.time(), "gpu": m.get("gpu"), "count": m.get("count"), "updated": m.get("updated"),
            "container": os.environ.get("MODAL_TASK_ID")}
    self.current = {**base, "state": "loading", "phase": "Starting engine", "progress": None}

    def publish(**kw):
        self.current = {**self.current, **kw, "beat": time.time()}
        try:
            llm_state[self.model] = self.current
        except Exception:  # noqa: BLE001
            pass

    def on_status(phase, progress):
        publish(state="ready" if phase == "Ready" else "loading", phase=phase, progress=progress)

    def heartbeat():
        while self.alive:
            time.sleep(15)
            if self.alive:
                publish()

    publish()
    threading.Thread(target=heartbeat, daemon=True).start()
    if not m or not store.installed(MODELS, self.model):
        self.error = f"{self.model} isn't installed on the server"
        publish(state="failed", error=self.error)
        return
    self.proc = EngineProcess(m, MODELS, on_status)
    try:
        self.proc.start()
        self.proc.wait_ready()
    except Exception as e:  # noqa: BLE001
        self.error = str(e)[:4000]
        publish(state="failed", error=self.error)
        self.proc.stop()
        modal.experimental.stop_fetching_inputs()
        return
    try:
        volume.commit()  # the compile caches, for the next cold start
    except Exception:  # noqa: BLE001
        pass
    if (llm_state.get(self.model + ":stop") or 0) > base["started"]:
        modal.experimental.stop_fetching_inputs()


async def _llm_chat(self, body: dict):
    from aistudio_gpu.llm.engine import EngineError

    if self.error or self.proc is None:
        raise EngineError(self.error or "the engine isn't running")
    async for chunk in self.proc.chat(body):
        yield chunk


def _llm_shutdown(self):
    import time

    self.alive = False
    if getattr(self, "proc", None) is not None:
        self.proc.stop()
    try:
        llm_state[self.model] = {**self.current, "state": "cold", "phase": None, "ended": time.time(),
                                 "beat": time.time()}
    except Exception:  # noqa: BLE001
        pass


LLM_CLS = dict(volumes={MODELS: volume}, scaledown_window=LLM_IDLE, timeout=24 * 60 * 60, max_containers=1,
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
        return self.current

    @modal.method()
    def stop(self) -> None:
        modal.experimental.stop_fetching_inputs()

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
        return self.current

    @modal.method()
    def stop(self) -> None:
        modal.experimental.stop_fetching_inputs()

    @modal.exit()
    def shutdown(self):
        _llm_shutdown(self)


@app.function(image=llm_import_image, cpu=4, memory=8192, volumes={MODELS: volume}, timeout=8 * 60 * 60,
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
        install(MODELS, manifest, files, tokens, progress, cancelled)
        volume.commit()
    except BaseException as e:  # noqa: BLE001
        progress({"state": "cancelled" if isinstance(e, InterruptedError) else "failed",
                  "error": f"{type(e).__name__}: {e}"})
        try:
            volume.commit()
        except Exception:  # noqa: BLE001
            pass


def _llm_service():
    import time

    from aistudio_gpu.llm.engine import EngineError
    from aistudio_gpu.llm.service import ACTIVE, LLMService

    class ModalLLMService(LLMService):
        """Each model in its own ``LLM`` container; installs as ``import_llm`` calls; state in Dicts."""

        def __init__(self):
            super().__init__(MODELS)
            self._reloaded = 0.0
            self._installs_cache: dict = {}

        def _obj(self, m: dict):
            n = int(m.get("count") or 1)
            gpu = m["gpu"] if n == 1 else f"{m['gpu']}:{n}"
            cls = LLMLlamaCpp if m.get("engine") == "llamacpp" else LLM
            return cls.with_options(gpu=gpu)(model=m["id"])

        async def _reload(self):
            if time.monotonic() - self._reloaded > 5:
                self._reloaded = time.monotonic()
                try:
                    await volume.reload.aio()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._installs_cache = {k: v async for k, v in llm_installs.items.aio()}
            except Exception:  # noqa: BLE001
                pass

        async def _commit(self):
            try:
                await volume.commit.aio()
            except Exception:  # noqa: BLE001
                pass

        async def _engine_state(self, mid, m):
            s = dict(await llm_state.get.aio(mid) or {"state": "cold"})
            now = time.time()
            if s.get("state") in ("loading", "ready") and now - (s.get("beat") or 0) > 60:
                s.update(state="cold", ended=s.get("beat"))  # the container is gone
            if s.get("state") == "starting" and now - (s.get("requested") or 0) > 20 * 60:
                s.update(state="cold")
            return s

        async def _engine_warm(self, mid, m):
            s = await self._engine_state(mid, m)
            if s.get("state") in ACTIVE:
                return
            await llm_state.put.aio(mid, {"state": "starting", "phase": "Waiting for a GPU", "requested": time.time(),
                                          "gpu": m.get("gpu"), "count": m.get("count")})
            await self._obj(m).ping.spawn.aio()

        async def _engine_ping(self, mid, m):
            await self._obj(m).ping.spawn.aio()

        async def _engine_stop(self, mid, m):
            s = await self._engine_state(mid, m)
            await llm_state.put.aio(mid + ":stop", time.time())
            if s.get("state") == "ready":
                await self._obj(m).stop.spawn.aio()

        async def _engine_chat(self, mid, m, body):
            s = await self._engine_state(mid, m)
            if s.get("state") == "failed" and s.get("updated") == m.get("updated") and \
                    time.time() - (s.get("beat") or 0) < 30 * 60:
                raise EngineError(f"the model failed to start: {s.get('error')}. Change its settings or press "
                                  "Start again.")
            if s.get("state") not in ACTIVE:
                await llm_state.put.aio(mid, {"state": "starting", "phase": "Waiting for a GPU",
                                              "requested": time.time(), "gpu": m.get("gpu"), "count": m.get("count")})
            async for chunk in self._obj(m).chat.remote_gen.aio(body):
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


@app.function(image=front_image, cpu=1, memory=2048, volumes={MODELS: volume, UPLOADS: uploads}, scaledown_window=300,
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
