# AIStudio's GPU server for a Linux NVIDIA machine (RunPod/Vast/Lambda pod, your own box). PyPI's torch wheels
# bring their own CUDA runtime; the host needs the NVIDIA driver and the container toolkit.
# Used by scripts/bootstrap-gpu-server.sh aistudio-gpu (behind Caddy's bearer-token gate).
# Serves Wan 2.2 and FLUX.1. The audio models (ACE-Step, Qwen3-TTS) pin older transformers versions and run on Modal
# (modal_app.py builds an image for each); /v1/info lists them as not installed here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 AISTUDIO_GPU_MODELS=/models AISTUDIO_GPU_DATA=/data
RUN pip install --no-cache-dir \
        "fastapi==0.162.2" "uvicorn==0.54.0" "websockets" "numpy==2.5.4" "pillow" "imageio-ffmpeg" \
        "torch==2.15.0" "diffusers==0.42.1" "transformers==5.22.2" "accelerate==1.15.0" "peft==0.21.2" \
        "sentencepiece" "ftfy" "safetensors" "huggingface_hub"
WORKDIR /srv
COPY aistudio_gpu ./aistudio_gpu
EXPOSE 8189
# No token here: the gate in front of it checks one. Models: `… run app python -m aistudio_gpu download`.
CMD ["python", "-m", "aistudio_gpu", "serve", "--host", "0.0.0.0", "--port", "8189", "--no-auth"]
