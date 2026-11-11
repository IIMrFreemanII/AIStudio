"""The engine process: vLLM's OpenAI-compatible server on localhost, started from an installed model's manifest, with
its startup log turned into phases ("Loading weights 41%") for the app. The same class runs inside a Modal container
and on a GPU box. ``FakeEngine`` stands in for it in tests and with ``AISTUDIO_FAKE_LLM=1``."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx

from . import store

Status = Callable[[str, "float | None"], None]  # (phase, progress 0…1 or None)

PHASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Loading (safetensors )?checkpoint shards|Loading weights|load_weights", re.I), "Loading weights"),
    (re.compile(r"torch\.compile|Compiling a graph|Dynamo bytecode", re.I), "Compiling"),
    (re.compile(r"Capturing CUDA graph|CUDA graphs", re.I), "Capturing CUDA graphs"),
    (re.compile(r"KV cache|Available KV cache memory|GPU blocks", re.I), "Allocating KV cache"),
    (re.compile(r"Starting vLLM API server|Application startup complete|Uvicorn running", re.I), "Starting server"),
    (re.compile(r"Initializing|init engine|Initializing a V\d+ LLM engine|world_size", re.I), "Starting engine"),
]
PERCENT = re.compile(r"(\d{1,3})%")


def phase_of(line: str) -> tuple[str, float | None] | None:
    for pattern, name in PHASES:
        if pattern.search(line):
            m = PERCENT.search(line)
            return name, (min(100, int(m.group(1))) / 100 if m else None)
    return None


def vllm_command(manifest: dict, weights: Path, port: int, host: str = "127.0.0.1") -> list[str]:
    n = int(manifest.get("count") or 1)
    cmd = ["vllm", "serve", str(weights), "--served-model-name", manifest["id"], "--host", host, "--port", str(port),
           "--tensor-parallel-size", str(n), "--max-model-len", str(int(manifest["context"])),
           "--gpu-memory-utilization", "0.90", "--enable-prefix-caching"]
    if manifest.get("toolParser"):
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", manifest["toolParser"]]
    if manifest.get("reasoningParser"):
        cmd += ["--reasoning-parser", manifest["reasoningParser"]]
    if n > 1 and manifest.get("experts"):
        cmd.append("--enable-expert-parallel")
    return cmd + list(manifest.get("engineArgs") or [])


class EngineError(RuntimeError):
    pass


class VLLMProcess:
    """One running vLLM server. ``start`` returns at once; ``wait_ready`` blocks until it answers /health."""

    def __init__(self, manifest: dict, models_root: Path, on_status: Status, port: int = 8000,
                 env: dict | None = None):
        self.manifest, self.models_root, self.on_status, self.port = manifest, models_root, on_status, port
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque(maxlen=200)
        self.url = f"http://127.0.0.1:{port}"

    def start(self) -> None:
        weights = store.weights_dir(self.models_root, self.manifest["id"])
        cmd = vllm_command(self.manifest, weights, self.port)
        self.on_status("Starting engine", None)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=self.env, start_new_session=True)
        threading.Thread(target=self._read, daemon=True, name="vllm-log").start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            self.log.append(line)
            print(line, flush=True)  # stays in the container's / service's log
            if (p := phase_of(line)) is not None:
                self.on_status(*p)

    def wait_ready(self, timeout: float = 3600) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                tail = "\n".join(list(self.log)[-25:])
                raise EngineError(f"vLLM exited while starting:\n{tail}" if tail else "vLLM exited while starting")
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    self.on_status("Ready", 1.0)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise EngineError("vLLM didn't start in time")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def chat(self, body: dict) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None)) as client:
            async with client.stream("POST", self.url + "/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:2000]
                    raise EngineError(_error_text(r.status_code, detail))
                async for chunk in r.aiter_raw():
                    yield chunk

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _error_text(status: int, detail: str) -> str:
    try:
        d = json.loads(detail)
        msg = d.get("message") or (d.get("error") or {}).get("message") or d.get("detail") or detail
    except ValueError:
        msg = detail
    return f"the model server answered HTTP {status}: {msg}"


class FakeEngine:
    """Answers like vLLM with a scripted reply: thinking, then text that repeats the last user message; a tool call
    (split over two chunks) when that message says "use a tool"; an error partway when it says "fail"."""

    def __init__(self, manifest: dict, models_root: Path, on_status: Status, boot_seconds: float | None = None):
        self.manifest, self.on_status = manifest, on_status
        self.boot = float(os.environ.get("AISTUDIO_FAKE_LLM_BOOT", "0.3")) if boot_seconds is None else boot_seconds
        self.running = False

    def start(self) -> None:
        self.running = True
        self.on_status("Starting engine", None)

    def wait_ready(self, timeout: float = 60) -> None:
        steps = 4
        for i in range(steps):
            time.sleep(self.boot / steps)
            self.on_status("Loading weights", (i + 1) / steps)
        self.on_status("Ready", 1.0)

    def alive(self) -> bool:
        return self.running

    async def chat(self, body: dict) -> AsyncIterator[bytes]:
        mid = self.manifest["id"]
        last = ""
        for m in reversed(body.get("messages") or []):
            if m.get("role") == "user":
                c = m.get("content")
                last = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or []
                                                             if isinstance(p, dict) and p.get("type") == "text")
                break
        created = int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> bytes:
            d = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": created, "model": mid,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(d)}\n\n".encode()

        if not body.get("stream"):
            reply = {"id": "chatcmpl-fake", "object": "chat.completion", "created": created, "model": mid,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": f"You said: {last}"},
                                  "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            yield json.dumps(reply).encode()
            return
        yield chunk({"role": "assistant", "content": ""})
        for piece in ("Thinking ", "about it."):
            await asyncio.sleep(0.01)
            yield chunk({"reasoning": piece, "reasoning_content": piece})
        if "fail" in last.lower():
            yield chunk({"content": "Partly"})
            raise EngineError("the fake engine failed on purpose")
        words = f"You said: {last}".split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(0.005)
            yield chunk({"content": w if i == 0 else " " + w})
        finish = "stop"
        if "use a tool" in last.lower():
            yield chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                         "function": {"name": "read_file", "arguments": "{\"path\": "}}]})
            yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"README.md\"}"}}]})
            finish = "tool_calls"
        yield chunk({}, finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": created, "model": mid,
                     "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": len(words) + 2,
                                              "total_tokens": 14 + len(words)}}
            yield f"data: {json.dumps(usage)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def stop(self) -> None:
        self.running = False


def fake_enabled() -> bool:
    return os.environ.get("AISTUDIO_FAKE_LLM", "") not in ("", "0")
