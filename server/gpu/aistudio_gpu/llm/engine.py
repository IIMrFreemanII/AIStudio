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
    # AIStudio's vLLM plugin (loadFormat "processed").
    (re.compile(r"Saving processed weights to", re.I), "Saving fast-start weights"),
    (re.compile(r"Loading processed weights from|No usable processed weights", re.I), "Loading weights"),
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
# Batch sizes vLLM records CUDA graphs for ("" keeps vLLM's: 19 sizes up to 128, recorded twice per start, once to
# measure their memory). One person's chats rarely run more than a few requests at once; bigger batches run without
# a graph (slower per step, still correct). On a B200 with Qwen3.8-Flash-Next: 3 s instead of 10–18 s.
GRAPH_SIZES = [int(s) for s in os.environ.get("AISTUDIO_VLLM_GRAPH_SIZES", "1,2,4,8").split(",") if s.strip()]
# vLLM's advice after measuring free memory: "Replace gpu_memory_utilization config with `--kv-cache-memory=N`
# (… GiB) to fit into requested memory". Passing it on later starts skips the measuring.
KV_ADVICE = re.compile(r"--kv-cache-memory=(\d+)`? \([\d.]+ GiB\) to fit into requested memory")
# vLLM's startup lines on what the KV cache holds: "GPU KV cache size: 1,234,560 tokens" and "Maximum concurrency for
# 131,072 tokens per request: 9.42x" (wording and commas vary by version; the "x" is optional).
KV_TOKENS = re.compile(r"KV cache size:\s*([\d,_]+)\s*tokens", re.I)
KV_CONCURRENCY = re.compile(r"Maximum concurrency for [\d,_]+ tokens per request:\s*([\d,]*\.?\d+)\s*x?", re.I)
REMEMBER_KV = os.environ.get("AISTUDIO_VLLM_REMEMBER_KV", "1") != "0"
# Multi-token prediction: models that ship MTP layers (Qwen3-Next and later, DeepSeek V3, GLM-4.5…) draft this many
# tokens per step with them, which the model then checks in one pass (same output, fewer steps). Qwen3.8-Flash-Next on
# a B200: replies 1.56× faster (191 → 298 tokens/s), starts ~40 s slower (more kernel warm-up). 0 turns it off.
MTP_TOKENS = int(os.environ.get("AISTUDIO_VLLM_MTP_TOKENS", "2"))


def mtp_on(manifest: dict) -> bool:
    """Whether a start drafts with the model's MTP layers: it has them, the model's "mtp" setting isn't off, and its
    engine arguments don't set their own speculative decoding."""
    extra = list(manifest.get("engineArgs") or [])
    return bool(MTP_TOKENS and manifest.get("mtpLayers") and manifest.get("mtp", True) is not False
                and manifest.get("engine") != "llamacpp" and not _has(extra, "--speculative-config", "-sc"))


def _has(extra: list[str], *flags: str) -> bool:
    """Whether the user's engine arguments set one of ``flags`` ("--x", "--x=…", "-O3", "-cc.y=…" forms)."""
    return any(a == f or a.startswith(f + "=") or (f in ("-O", "-cc") and a.startswith(f)) for a in extra
               for f in flags)


def kv_setup(manifest: dict) -> str:
    """The key the measured KV cache size is remembered under (store.kv_setup, plus this server's defaults)."""
    return store.kv_setup(manifest, MAX_SEQS, GRAPH_SIZES, MTP_TOKENS if mtp_on(manifest) else 0)


def parse_kv_capacity(line: str) -> dict:
    """What a vLLM log line says about the KV cache: ``kv_tokens`` (int) and/or ``max_concurrency`` (float)."""
    out: dict = {}
    if m := KV_TOKENS.search(line):
        out["kv_tokens"] = int(re.sub(r"[,_]", "", m.group(1)))
    if m := KV_CONCURRENCY.search(line):
        out["max_concurrency"] = float(m.group(1).replace(",", ""))
    return out


def phase_of(line: str) -> tuple[str, float | None] | None:
    for pattern, name in PHASES:
        if pattern.search(line):
            m = PERCENT.search(line)
            return name, (min(100, int(m.group(1))) / 100 if m else None)
    return None


def vllm_command(manifest: dict, weights: Path, port: int, host: str = "127.0.0.1",
                 media: str | None = None, kv_bytes: int | None = None) -> list[str]:
    """vLLM's OpenAI server for a safetensors model. ``media``: the folder of uploads that chats may point at
    (``file://`` image and video URLs, see ``LLMService.resolve_uploads``), for models that take images.
    ``kv_bytes``: the KV cache size an earlier start of the same setup measured (skips measuring it again)."""
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
    elif manifest.get("loadFormat") == "processed":
        # AIStudio's vLLM plugin: loads the copy saved after vLLM's processing when it matches this setup, else the
        # checkpoint (then saves a copy for the next start). One copy per GPU kind: they differ (kernel layouts).
        cmd += ["--load-format", "aistudio_processed", "--model-loader-extra-config",
                json.dumps({"dir": str(weights.parent / "processed" / str(manifest.get("gpu") or "gpu"))})]
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
    if GRAPH_SIZES and not _has(extra, "--compilation-config", "-cc", "-O", "--enforce-eager"):
        cmd += ["--compilation-config", json.dumps({"cudagraph_capture_sizes": GRAPH_SIZES})]
    if "mtpLayers" not in manifest:
        manifest = {**manifest, "mtpLayers": store.mtp_layers(weights)}
    if mtp_on(manifest):
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP_TOKENS})]
    if kv_bytes and not _has(extra, "--kv-cache-memory", "--gpu-memory-utilization"):
        cmd += ["--kv-cache-memory", str(int(kv_bytes))]
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


def command(manifest: dict, weights: Path, port: int, media: str | None = None,
            kv_bytes: int | None = None) -> list[str]:
    if manifest.get("engine") == "llamacpp":
        return llama_command(manifest, weights, port)
    return vllm_command(manifest, weights, port, media=media, kv_bytes=kv_bytes)


class EngineError(RuntimeError):
    pass


class EngineProcess:
    """One running engine (vLLM or llama.cpp, by the manifest). ``start`` returns at once; ``wait_ready`` blocks until
    it answers /health."""

    def __init__(self, manifest: dict, models_root: Path, on_status: Status, port: int = 8000,
                 env: dict | None = None):
        if manifest.get("engine") != "llamacpp" and "mtpLayers" not in manifest:  # installed before the field
            manifest = {**manifest, "mtpLayers": store.mtp_layers(store.weights_dir(models_root, manifest["id"]))}
        self.manifest, self.models_root, self.on_status, self.port = manifest, models_root, on_status, port
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque(maxlen=200)
        self.url = f"http://127.0.0.1:{port}"
        self.kv_tokens = 0  # requests' worth of KV cache vLLM reported at startup (0: not known; never for llama.cpp)
        self.max_concurrency = 0.0

    def start(self) -> None:
        weights = store.weights_dir(self.models_root, self.manifest["id"])
        self.kv_bytes = self._remembered_kv()
        cmd = command(self.manifest, weights, self.port, self.env.get(MEDIA_ENV), kv_bytes=self.kv_bytes)
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
            if (m := KV_ADVICE.search(line)) and not getattr(self, "kv_bytes", None):
                self._remember_kv(int(m.group(1)))
            if getattr(self, "manifest", {}).get("engine") != "llamacpp":
                for k, v in parse_kv_capacity(line).items():
                    setattr(self, k, v)
            if (p := phase_of(line)) is not None:
                if p[0] == "Starting engine" and later:
                    continue
                later = later or p[0] != "Starting engine"
                self.on_status(*p)

    def wait_ready(self, timeout: float = 3600) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                if getattr(self, "kv_bytes", None):  # maybe the remembered size no longer fits: measure next time
                    self._remember_kv(None)
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

    # The KV cache size vLLM measured on the first start of a setup, passed as --kv-cache-memory on the later ones
    # (saves measuring free memory and recording the CUDA graphs once more to size them). Forgotten when a start
    # with it fails. Committed to the Volume with the rest after the start (modal_app.py).

    def _remembered_kv(self) -> int | None:
        if not REMEMBER_KV or self.manifest.get("engine") == "llamacpp":
            return None
        return store.kv_cache_memory(self.models_root, self.manifest["id"], kv_setup(self.manifest))

    def _remember_kv(self, nbytes: int | None) -> None:
        if not REMEMBER_KV:
            return
        try:
            store.save_kv_cache_memory(self.models_root, self.manifest["id"], kv_setup(self.manifest), nbytes)
        except OSError as e:
            print(f"[aistudio] couldn't save the KV cache size: {e}", flush=True)

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

# The fake engine's canned drafts for the "draft a commit message" and "draft a pull request" buttons.
FAKE_COMMIT = "Add hello.txt\n\nCreates hello.txt with a greeting from the agent."
FAKE_PR = "Add hello.txt\n\n## Summary\n- Adds hello.txt with a greeting\n\n## Test plan\n- cat hello.txt"


# Scripts for "run a flow" / "run a slow flow" (the flow tool's JS), see FakeEngine.
FAKE_FLOW = """export const meta = { name: 'smoke-flow', description: 'Two reviewers, verify each finding, one edit', phases: [{ title: 'Review' }, { title: 'Verify' }, { title: 'Edit' }] }
const FINDINGS = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' } }, required: ['title'] } } }, required: ['findings'] }
const found = await parallel(['bugs', 'style'].map(d => () => agent(`Review the project for ${d}. sleep 1`, { label: `review:${d}`, phase: 'Review', schema: FINDINGS })))
const checked = await verifyEach(found.flatMap(r => r.findings), { phase: 'Verify' })
const made = await agent('write flow.txt', { label: 'edit', phase: 'Edit', mode: 'edit' })
return { findings: checked.length, real: checked.filter(f => f.verdict.real).length, branch: made.branch }"""

FAKE_SLOW_FLOW = """export const meta = { name: 'slow-flow', description: 'Three slow agents', phases: [{ title: 'Wait' }] }
await parallel([1, 2, 3].map(i => () => agent(`Wait a while. sleep 20`, { label: `slow:${i}`, phase: 'Wait' })))
return 'finished'"""

SCHEMA_MARKER = "Reply with only JSON matching this schema"
SCHEMA_RETRY = "That reply wasn't valid JSON for the schema"
SLEEP = re.compile(r"\bsleep (\d+(?:\.\d+)?)", re.I)


def schema_instance(schema) -> object:
    """A value that fits a JSON schema: objects get every property, arrays one item, "x", 1, true, the first enum."""
    if not isinstance(schema, dict):
        return "x"
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    if kind == "object" or "properties" in schema:
        return {k: schema_instance(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [schema_instance(schema.get("items") or {})]
    if kind in ("number", "integer"):
        return 1
    if kind == "boolean":
        return True
    return "x"


def schema_reply(text: str) -> str | None:
    """The JSON instance for a message that holds the schema marker (everything after the first newline after it)."""
    i = text.find(SCHEMA_MARKER)
    if i < 0:
        return None
    rest = text[i:].partition("\n")[2]
    try:
        schema = json.JSONDecoder().raw_decode(rest.lstrip())[0]
    except ValueError:
        return None
    return json.dumps(schema_instance(schema))


class FakeEngine:
    """Answers like vLLM with a scripted reply: thinking, then text that repeats the last user message; a tool call
    (split over two chunks) when that message says "use a tool"; an error partway when it says "fail".

    Videos in the last message that it could read from disk add " [N video(s)]" to the reply. With the chat tools in
    the request, "ask me" calls ask_user and "show a widget" calls show_widget (then "Done: " and the result);
    "suggest" ends the reply with a <followups> block.

    With tools in the request and "create hello.txt" in the message, it acts as a coding agent: write_file, then
    run_command `cat hello.txt`, then "Done: " and the last tool result. With "add an upscale" it acts as the graph
    helper: get_graph, then edit_graph putting an ImageScale between `decode` and `save` (the text-to-image
    template's ids), then "Done: " and the first line of the result. With start_session offered and "start a session"
    in the message, it calls start_session (title "Hello file", brief "create hello.txt and show it", worktree on), then
    "Done: " and the result. "create hello.txt with a todo list" calls todo_write (two steps, the first in progress)
    before the hello.txt script. "Write the commit message now." / "Write the pull request now." answer with FAKE_COMMIT /
    FAKE_PR."""

    def __init__(self, manifest: dict, models_root: Path, on_status: Status, boot_seconds: float | None = None):
        self.manifest, self.on_status = manifest, on_status
        self.boot = float(os.environ.get("AISTUDIO_FAKE_LLM_BOOT", "0.3")) if boot_seconds is None else boot_seconds
        self.running = False
        self.kv_tokens, self.max_concurrency = 262144, 2.0

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
        user_texts = []  # every user message, latest first (the schema retry looks back for the marker)
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content")
                user_texts.insert(0, c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text"))
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

        schema_json = schema_reply(last)
        if schema_json is None and SCHEMA_RETRY in last:
            schema_json = next((r for t in user_texts if (r := schema_reply(t)) is not None), None)
        if (m := SLEEP.search(last)) is not None:
            await asyncio.sleep(float(m.group(1)))  # cancelled with the request when the client goes away
        if not body.get("stream"):
            reply = {"id": "chatcmpl-fake", "object": "chat.completion", "created": created, "model": mid,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": schema_json or f"You said: {last}"},
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
        tool_names = {(t.get("function") or {}).get("name") for t in body.get("tools") or []}
        if "start a session" in last.lower() and "start_session" in tool_names:
            script = [("start_session", {"title": "Hello file", "prompt": "create hello.txt and show it",
                                         "worktree": True})]
        elif "run a slow flow" in last.lower() and "flow" in tool_names:
            script = [("flow", {"script": FAKE_SLOW_FLOW})]
        elif "run a flow" in last.lower() and "flow" in tool_names:
            script = [("flow", {"script": FAKE_FLOW})]
        elif "write flow.txt" in last.lower() and "write_file" in tool_names:
            script = [("write_file", {"path": "flow.txt", "content": "made by a flow\n"})]
        elif "create hello.txt with a todo list" in last.lower():
            script = [("todo_write", {"todos": [
                          {"content": "Write hello.txt", "status": "in_progress", "activeForm": "Writing hello.txt"},
                          {"content": "Show it", "status": "pending", "activeForm": "Showing it"}]}),
                      ("write_file", {"path": "hello.txt", "content": "hi from the agent\n"}),
                      ("run_command", {"command": "cat hello.txt"})]
        elif "create hello.txt" in last.lower():
            script = [("write_file", {"path": "hello.txt", "content": "hi from the agent\n"}),
                      ("run_command", {"command": "cat hello.txt"})]
        elif "add an upscale" in last.lower():
            script = [("get_graph", {}),
                      ("edit_graph", {"operations": [
                          {"op": "add", "id": "upscale", "type": "ImageScale",
                           "inputs": {"image": {"node": "decode", "output": "image"}, "factor": 2}},
                          {"op": "connect", "from": "upscale", "output": "image", "to": "save", "input": "image"}]})]
        elif "map the project" in last.lower() and "code_graph" in tool_names:
            # The read goes to the first location the graph returned (step 0's result), README.md when it has none.
            m = re.search(r"([\w./-]+\.\w+):(\d+)", tool_results[0]) if tool_results else None
            read_args = {"path": m.group(1), "offset": int(m.group(2)), "limit": 20} if m else {"path": "README.md"}
            script = [("code_graph", {"action": "query", "target": "greet"}), ("read_file", read_args)]
        elif "ask me" in last.lower():
            script = [("ask_user", {"questions": [{"question": "Which database should I use?", "header": "Database",
                                                   "options": [{"label": "SQLite (Recommended)",
                                                                "description": "One file, no server"},
                                                               {"label": "Postgres", "description": "A server"}]}]})]
        elif "show a widget" in last.lower():
            script = [("show_widget", {"title": "Strength", "html": FAKE_WIDGET})]
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
        elif schema_json is not None:
            words = schema_json.split(" ")
        elif "write the commit message now." in last.lower():
            words = FAKE_COMMIT.split(" ")
        elif "write the pull request now." in last.lower():
            words = FAKE_PR.split(" ")
        else:
            words = f"You said: {last}".split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(0.005)
            yield chunk({"content": w if i == 0 else " " + w})
        if "suggest" in last.lower() and schema_json is None:
            yield chunk({"content": "\n\n<followups>\n- Tell me more\n- Show an example\n</follow"})
            yield chunk({"content": "ups>"})
        finish = "stop"
        if "use a tool" in last.lower() and schema_json is None:
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
