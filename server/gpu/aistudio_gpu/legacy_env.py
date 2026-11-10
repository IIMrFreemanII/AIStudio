"""Environment variables from before the project was renamed (MLXComfyUI → AIStudio).

``MLXCOMFY_X`` still works as ``AISTUDIO_X`` when the new name isn't set, so existing deploy commands, Docker
settings and shell profiles keep working.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

LEGACY_PREFIX = "MLXCOMFY_"
PREFIX = "AISTUDIO_"


def apply(environ: MutableMapping[str, str] = os.environ) -> None:
    for key in [k for k in environ if k.startswith(LEGACY_PREFIX)]:
        environ.setdefault(PREFIX + key[len(LEGACY_PREFIX):], environ[key])
