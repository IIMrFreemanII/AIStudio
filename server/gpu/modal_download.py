"""Downloads the GPU server's models into its Modal Volume, on CPUs only (no GPU, so it runs on any Modal account).

    modal run server/gpu/modal_download.py --models wan22,flux     # or: --models wan22 (no Hugging Face token)
    modal run --detach server/gpu/modal_download.py --models wan22  # keeps running if this Mac sleeps

FLUX.1 dev is gated: first `modal secret create huggingface HF_TOKEN=hf_…` with a token whose account accepted the
licence on huggingface.co/black-forest-labs/FLUX.1-dev. Finished models are skipped, so it is safe to re-run.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

MODELS = Path("/models")
HF_SECRET = os.environ.get("MLXCOMFY_HF_SECRET", "huggingface")

app = modal.App("mlxcomfy-gpu-download")
volume = modal.Volume.from_name("mlxcomfy-gpu-models", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.15.0", "diffusers==0.42.1", "transformers==5.22.2", "accelerate==1.15.0",
                         "sentencepiece", "safetensors", "huggingface_hub", "numpy==2.5.4", "pillow")
         # The same choice of secret inside the container as here (Modal objects must match on both sides).
         .env({"HF_HUB_DISABLE_TELEMETRY": "1", "MLXCOMFY_DOWNLOAD_TMP": "/tmp", "MLXCOMFY_HF_SECRET": HF_SECRET})
         .add_local_python_source("mlxcomfy_gpu"))


@app.function(image=image, cpu=8, memory=(32768, 65536), volumes={MODELS: volume}, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name(HF_SECRET)] if HF_SECRET else [])
def download(models: str):
    from mlxcomfy_gpu.download import download as fetch

    fetch([m.strip() for m in models.split(",") if m.strip()], MODELS, on_done=lambda _: volume.commit())


@app.local_entrypoint()
def main(models: str = "wan22,flux"):
    if "flux" in models and not HF_SECRET:
        raise SystemExit("FLUX.1 dev needs the huggingface secret (see the top of this file)")
    download.remote(models)
