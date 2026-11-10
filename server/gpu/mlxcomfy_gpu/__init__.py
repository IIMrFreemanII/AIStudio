"""MLXComfyUI's GPU server: runs diffusers pipelines (Wan 2.2, FLUX.1) as jobs for the app's remote nodes.

The same FastAPI app runs on Modal (``modal_app.py``: a CPU front desk that hands jobs to GPU workers which scale to
zero) or on any machine (``python -m mlxcomfy_gpu serve``: an in-process worker).
"""

VERSION = "0.1.0"
# The job API's version; the app refuses servers with a different one.
API_VERSION = 1
