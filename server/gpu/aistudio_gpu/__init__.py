"""AIStudio's GPU server: runs diffusers pipelines (Wan 2.2, FLUX.1) as jobs for the app's remote nodes.

The same FastAPI app runs on Modal (``modal_app.py``: a CPU front desk that hands jobs to GPU workers which scale to
zero) or on any machine (``python -m aistudio_gpu serve``: an in-process worker).
"""

from . import legacy_env

# MLXCOMFY_* variables from before the rename count as their AISTUDIO_* names.
legacy_env.apply()

VERSION = "0.1.0"
# The job API's version; the app refuses servers with a different one.
API_VERSION = 1
