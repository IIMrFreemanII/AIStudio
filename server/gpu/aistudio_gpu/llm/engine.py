"""The engine process: vLLM's OpenAI-compatible server (llama.cpp's ``llama-server`` for GGUF models) on localhost,
started from an installed model's manifest, with its startup log turned into phases ("Loading weights 41%") for the
app. The same class runs inside a Modal container
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
    (re.compile(r"Loading (safetensors )?checkpoint shards|Loading safetensors using|Loading weights|load_weights|"
                r"load_tensors|llama_model_load", re.I), "Loading weights"),
    (re.compile(r"server is listening|main: model loaded", re.I), "Starting server"),
    (re.compile(r"llama_context|llama_kv_cache", re.I), "Allocating KV cache"),
    (re.compile(r"flashinfer\.jit: Building|Building JIT module", re.I), "Building GPU kernels"),
    (re.compile(r"torch\.compile|Compiling a graph|Dynamo bytecode", re.I), "Compiling"),
    (re.compile(r"Capturing CUDA graph|CUDA graphs", re.I), "Capturing CUDA graphs"),
    # Not just "KV cache": vLLM's config lines ("Using fp8 data type to store kv cache") come before the weights.
    (re.compile(r"Available KV cache memory|GPU KV cache size|GPU blocks|KV cache layout", re.I),
     "Allocating KV cache"),
    (re.compile(r"Starting vLLM API server|Application startup complete|Uvicorn running", re.I), "Starting server"),
    (re.compile(r"Initializing|init engine|Initializing a V\d+ LLM engine|world_size", re.I), "Starting engine"),
]
PERCENT = re.compile(r"(\d{1,3})%")
RUNAI_CONCURRENCY = int(os.environ.get("AISTUDIO_RUNAI_CONCURRENCY", "16"))
MEDIA_ENV = "AISTUDIO_LLM_MEDIA"  # the uploads folder chats' images and videos are read from (vllm_command)
MAX_SEQS = int(os.environ.get("AISTUDIO_VLLM_MAX_SEQS", "64"))  # requests at once; 0 keeps vLLM's default


def phase_of(line: str) -> tuple[str, float | None] | None:
    for pattern, name in PHASES:
        if pattern.search(line):
            m = PERCENT.search(line)
            return name, (min(100, int(m.group(1))) / 100 if m else None)
    return None


def vllm_command(manifest: dict, weights: Path, port: int, host: str = "127.0.0.1",
                 media: str | None = None) -> list[str]:
    """vLLM's OpenAI server for a safetensors model. ``media``: the folder of uploads that chats may point at
    (``file://`` image and video URLs, see ``LLMService.resolve_uploads``), for models that take images."""
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
    if manifest.get("loadFormat") == "runai":
        cmd += ["--load-format", "runai_streamer", "--model-loader-extra-config",
                json.dumps({"concurrency": RUNAI_CONCURRENCY})]
    if manifest.get("snapshot"):
        cmd.append("--enable-sleep-mode")  # weights go to CPU memory before the snapshot, back after the restore
    if media and manifest.get("vision"):
        cmd += ["--allowed-local-media-path", media]
    extra = list(manifest.get("engineArgs") or [])
    if MAX_SEQS and "--max-num-seqs" not in extra:
        # vLLM's default (1024 requests at once) is for shared servers. Here one person chats: a small cap means
        # fewer CUDA graphs to capture at every start, and models with linear-attention (Mamba / Gated DeltaNet)
        # layers need one state slot per request, which on one GPU may not fit 1024 (Qwen3.8-Flash-Next on a B200
        # had room for 761).
        cmd += ["--max-num-seqs", str(MAX_SEQS)]
    return cmd + extra


def llama_command(manifest: dict, weights: Path, port: int, host: str = "127.0.0.1") -> list[str]:
    """llama.cpp's server for a GGUF model: every layer on the GPUs (split by layer across several), one slot with
    the whole context, tool calls through the model's own chat template."""
    n = int(manifest.get("count") or 1)
    cmd = ["llama-server", "-m", str(weights / manifest["ggufFile"]), "--alias", manifest["id"], "--host", host,
           "--port", str(port), "-c", str(int(manifest["context"])), "-np", "1", "-ngl", "999", "--jinja",
           "--no-webui"]
    if manifest.get("reasoningParser"):
        cmd += ["--reasoning-format", "deepseek"]
    if manifest.get("mmproj"):
        cmd += ["--mmproj", str(weights / manifest["mmproj"])]
    if n > 1:
        cmd += ["--split-mode", "layer"]
    return cmd + list(manifest.get("engineArgs") or [])


def command(manifest: dict, weights: Path, port: int, media: str | None = None) -> list[str]:
    if manifest.get("engine") == "llamacpp":
        return llama_command(manifest, weights, port)
    return vllm_command(manifest, weights, port, media=media)


class EngineError(RuntimeError):
    pass


class EngineProcess:
    """One running engine (vLLM or llama.cpp, by the manifest). ``start`` returns at once; ``wait_ready`` blocks until
    it answers /health."""

    def __init__(self, manifest: dict, models_root: Path, on_status: Status, port: int = 8000,
                 env: dict | None = None):
        self.manifest, self.models_root, self.on_status, self.port = manifest, models_root, on_status, port
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque(maxlen=200)
        self.url = f"http://127.0.0.1:{port}"

    def start(self) -> None:
        weights = store.weights_dir(self.models_root, self.manifest["id"])
        cmd = command(self.manifest, weights, self.port, self.env.get(MEDIA_ENV))
        self.on_status("Starting engine", None)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=self.env, start_new_session=True)
        threading.Thread(target=self._read, daemon=True, name="vllm-log").start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        later = False  # past "Starting engine": its patterns also match lines from later on ("Initializing model")
        for line in self.proc.stdout:
            line = line.rstrip()
            self.log.append(line)
            print(line, flush=True)  # stays in the container's / service's log
            if (p := phase_of(line)) is not None:
                if p[0] == "Starting engine" and later:
                    continue
                later = later or p[0] != "Starting engine"
                self.on_status(*p)

    def wait_ready(self, timeout: float = 3600) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                tail = "\n".join(list(self.log)[-25:])
                name = "llama.cpp" if self.manifest.get("engine") == "llamacpp" else "vLLM"
                raise EngineError(f"{name} exited while starting:\n{tail}" if tail else f"{name} exited while starting")
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    self.on_status("Ready", 1.0)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise EngineError(f"the engine didn't start in {round(timeout / 60)} min:\n" + "\n".join(list(self.log)[-25:]))

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # Memory snapshots (vLLM with --enable-sleep-mode): warm up, sleep (weights to CPU memory, KV cache freed),
    # snapshot; after the restore, wake up.

    def warmup(self, requests: int = 2) -> None:
        body = {"model": self.manifest["id"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
        for _ in range(requests):
            httpx.post(self.url + "/v1/chat/completions", json=body, timeout=300).raise_for_status()

    def sleep(self, level: int = 1) -> None:
        httpx.post(self.url + "/sleep", params={"level": level}, timeout=600).raise_for_status()

    def wake_up(self) -> None:
        self.on_status("Waking up", None)
        httpx.post(self.url + "/wake_up", timeout=600).raise_for_status()
        self.wait_ready(timeout=600)

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


VLLMProcess = EngineProcess  # the name modal_app.py and older code use


def _error_text(status: int, detail: str) -> str:
    try:
        d = json.loads(detail)
        msg = d.get("message") or (d.get("error") or {}).get("message") or d.get("detail") or detail
    except ValueError:
        msg = detail
    return f"the model server answered HTTP {status}: {msg}"


# The fake engine's widget: a slider that reports its value, and a button that sends a message.
FAKE_WIDGET = """<label>Strength <input id="s" type="range" min="0" max="1" step="0.1" value="0.8"></label>
<span id="v">0.8</span> <button id="b">Explain this</button>
<script>
  const s = document.getElementById("s"), v = document.getElementById("v");
  s.oninput = () => { v.textContent = s.value; aistudio.reportState({strength: Number(s.value)}); };
  document.getElementById("b").onclick = () => aistudio.sendPrompt("Explain strength " + s.value);
  aistudio.reportState({strength: 0.8});
</script>"""


class FakeEngine:
    """Answers like vLLM with a scripted reply: thinking, then text that repeats the last user message; a tool call
    (split over two chunks) when that message says "use a tool"; an error partway when it says "fail".

    Videos in the last message that it could read from disk add " [N video(s)]" to the reply. With the chat tools in
    the request, "ask me" calls ask_user and "show a widget" calls show_widget (then "Done: " and the result);
    "suggest" ends the reply with a <followups> block.

    With tools in the request and "create hello.txt" in the message, it acts as a coding agent: write_file, then
    run_command `cat hello.txt`, then "Done: " and the last tool result. With "add an upscale" it acts as the graph
    helper: get_graph, then edit_graph putting an ImageScale between `decode` and `save` (the text-to-image
    template's ids), then "Done: " and the first line of the result."""

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

    def warmup(self, requests: int = 2) -> None:
        pass

    def sleep(self, level: int = 1) -> None:
        self.on_status("Asleep", None)

    def wake_up(self) -> None:
        self.on_status("Waking up", None)
        self.on_status("Ready", 1.0)

    async def chat(self, body: dict) -> AsyncIterator[bytes]:
        mid = self.manifest["id"]
        last = ""
        messages = body.get("messages") or []
        tool_results = []
        for m in reversed(messages):
            if m.get("role") == "tool":
                tool_results.insert(0, str(m.get("content") or ""))
            if m.get("role") == "user":
                c = m.get("content")
                last = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or []
                                                             if isinstance(p, dict) and p.get("type") == "text")
                # Videos it could read (uploads resolved to files, as vLLM gets them) are counted in the reply.
                videos = [p for p in (c if isinstance(c, list) else []) if isinstance(p, dict)
                          and p.get("type") == "video_url" and str((p.get("video_url") or {}).get("url", ""))
                          .startswith("file://") and Path(p["video_url"]["url"][7:]).is_file()]
                if videos:
                    last += f" [{len(videos)} video{'s' if len(videos) > 1 else ''}]"
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
        script = []
        if "create hello.txt" in last.lower():
            script = [("write_file", {"path": "hello.txt", "content": "hi from the agent\n"}),
                      ("run_command", {"command": "cat hello.txt"})]
        elif "add an upscale" in last.lower():
            script = [("get_graph", {}),
                      ("edit_graph", {"operations": [
                          {"op": "add", "id": "upscale", "type": "ImageScale",
                           "inputs": {"image": {"node": "decode", "output": "image"}, "factor": 2}},
                          {"op": "connect", "from": "upscale", "output": "image", "to": "save", "input": "image"}]})]
        elif "ask me" in last.lower():
            script = [("ask_user", {"questions": [{"question": "Which database should I use?", "header": "Database",
                                                   "options": [{"label": "SQLite (Recommended)",
                                                                "description": "One file, no server"},
                                                               {"label": "Postgres", "description": "A server"}]}]})]
        elif "show a widget" in last.lower():
            script = [("show_widget", {"title": "Strength", "html": FAKE_WIDGET})]
        tool_names = {(t.get("function") or {}).get("name") for t in body.get("tools") or []}
        if script and script[0][0] in ("ask_user", "show_widget") and script[0][0] not in tool_names:
            script = []  # the chat tools' scripts only when the chat offers them
        if body.get("tools") and script:
            step = len(tool_results)
            if step < len(script):
                name, args = script[step]
                yield chunk({"content": "Working on it."})
                yield chunk({"tool_calls": [{"index": 0, "id": f"call_{step}", "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args)}}]})
                yield chunk({}, "tool_calls")
                yield b"data: [DONE]\n\n"
                return
            last_result = (tool_results[-1].strip().splitlines() or [""])[0] if "upscale" in last.lower() \
                else tool_results[-1].strip().replace("\n", " ")
            words = f"Done: {last_result}".split(" ")
        else:
            words = f"You said: {last}".split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(0.005)
            yield chunk({"content": w if i == 0 else " " + w})
        if "suggest" in last.lower():
            yield chunk({"content": "\n\n<followups>\n- Tell me more\n- Show an example\n</follow"})
            yield chunk({"content": "ups>"})
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
