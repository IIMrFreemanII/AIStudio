"""`python -m mlxcomfy_gpu serve` — the GPU server without Modal (a GPU box, or CPU-only for the test tasks).

    python -m mlxcomfy_gpu serve [--host 127.0.0.1] [--port 8189] [--data DIR] [--models DIR] [--token-file F]
                                 [--cpu-only]
    python -m mlxcomfy_gpu download --models wan22-t2v,wan22-i2v,flux1-dev [--models-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlxcomfy_gpu")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the job API with an in-process worker")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8189)
    serve.add_argument("--data", type=Path, default=Path(os.environ.get("MLXCOMFY_GPU_DATA", "data")))
    serve.add_argument("--models", type=Path, default=Path(os.environ.get("MLXCOMFY_GPU_MODELS", "models")))
    serve.add_argument("--token-file", type=Path, help="require this bearer token (created if missing)")
    serve.add_argument("--no-auth", action="store_true", help="no token (only behind a gate that checks one)")
    serve.add_argument("--cpu-only", action="store_true", help="only the CPU test tasks (no torch needed)")

    dl = sub.add_parser("download", help="download and convert models into the models folder")
    dl.add_argument("--models", default="wan22-t2v,wan22-i2v,flux1-dev")
    dl.add_argument("--models-dir", type=Path, default=Path(os.environ.get("MLXCOMFY_GPU_MODELS", "models")))

    args = parser.parse_args(argv)
    if args.command == "download":
        from .download import download
        download([m.strip() for m in args.models.split(",") if m.strip()], args.models_dir)
        return 0

    token: str | None = os.environ.get("MLXCOMFY_GPU_TOKEN")
    if token is None and args.token_file:
        if not args.token_file.exists():
            args.token_file.parent.mkdir(parents=True, exist_ok=True)
            args.token_file.write_text(secrets.token_hex(32) + "\n")
            args.token_file.chmod(0o600)
            print(f"created token in {args.token_file}", file=sys.stderr)
        token = args.token_file.read_text().split()[0]
    if token is None and not args.no_auth:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            parser.error("listening beyond this machine needs --token-file (or --no-auth behind a token gate)")

    import uvicorn

    from .api import create_app
    from .files import FileStore
    from .jobs import JobManager
    from .local import LocalRunner

    runner = LocalRunner(args.models.resolve(), allow_gpu_tasks=not args.cpu_only)
    # Uploads are kept across restarts, up to MLXCOMFY_UPLOADS_GB (least recently used go first).
    files = FileStore(args.data / "uploads", max_bytes=int(float(os.environ.get("MLXCOMFY_UPLOADS_GB", "20")) * 1e9))
    files.prune()
    manager = JobManager(runner, files, args.data / "results")
    uvicorn.run(create_app(manager, token), host=args.host, port=args.port, log_level="info",
                ws_max_size=16 << 20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
