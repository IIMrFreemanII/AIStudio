"""The front desk's LLM service: inspecting links, installs, warmth, and chats. ``LLMService`` holds what every
deployment shares; ``LocalLLMService`` runs the engine in this machine's processes (a GPU box, the tests), and
``modal_app.ModalLLMService`` runs each model in its own Modal container.

Warmth: an engine stays up while the app sends keepalives (every minute while a chat using the model is open and
in use), and stops ``AISTUDIO_LLM_IDLE`` seconds (default 180) after the last chat or keepalive. Keepalives stop
counting ``AISTUDIO_LLM_LEASE`` seconds (default 1800) after the last chat request, so a forgotten window can't keep
a GPU running for long.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx

from ..tasks import GPU_PRICES
from . import facts as facts_mod
from . import install as install_mod
from . import coldstart, profiles, proxy, sizing, store
from .engine import EngineError, EngineProcess, FakeEngine, fake_enabled

IDLE = float(os.environ.get("AISTUDIO_LLM_IDLE", "180"))
LEASE = float(os.environ.get("AISTUDIO_LLM_LEASE", "1800"))
ACTIVE = ("starting", "loading", "ready")


class LLMService:
    billed = True  # the Modal prices apply (a GPU box is paid for by the hour, not by these)

    def __init__(self, models_root: Path, http: Callable[[], httpx.AsyncClient] | None = None,
                 idle: float = IDLE, lease: float = LEASE, sync_http: Callable[[], httpx.Client] | None = None):
        self.models_root, self._http, self.idle, self.lease = models_root, http, idle, lease
        self._sync_http = sync_http
        self.last_chat: dict[str, float] = {}
        self.warmed: dict[str, float] = {}

    # MARK: hooks a deployment implements

    async def _engine_state(self, mid: str, m: dict) -> dict:
        raise NotImplementedError

    async def _engine_warm(self, mid: str, m: dict) -> None:
        raise NotImplementedError

    async def _engine_ping(self, mid: str, m: dict) -> None:
        raise NotImplementedError

    async def _engine_stop(self, mid: str, m: dict) -> None:
        raise NotImplementedError

    def _engine_chat(self, mid: str, m: dict, body: dict) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def _start_install(self, manifest: dict, files: list[dict], tokens: dict) -> dict:
        raise NotImplementedError

    def _installs(self) -> dict[str, dict]:
        raise NotImplementedError

    def _cancel_install(self, mid: str) -> None:
        raise NotImplementedError

    async def _coldstarts(self, mid: str) -> list:  # the model's last boots, oldest first (coldstart.py)
        return []

    async def _reload(self) -> None:  # the volume, where writes from other containers show up
        pass

    async def _commit(self) -> None:
        pass

    # MARK: links

    async def _facts(self, link: str, tokens: dict, quant: str | None = None) -> facts_mod.Facts:
        if fake_enabled() and str(link).startswith("fake/"):
            return fake_facts(str(link))
        ref = facts_mod.parse_link(link)
        client = self._http() if self._http else None
        try:
            return await facts_mod.inspect(ref, tokens, client, self._sync_http, quant)
        finally:
            if client is not None:
                await client.aclose()

    @staticmethod
    def _tokens(body: dict) -> dict:
        """Logins for this request: the app's (used once, never stored), else the server's own."""
        return {"huggingface": body.get("hfToken") or os.environ.get("HF_TOKEN") or None,
                "civitai": body.get("civitaiToken") or os.environ.get("CIVITAI_TOKEN") or None}

    async def inspect(self, body: dict) -> dict:
        f = await self._facts(str(body["url"]), self._tokens(body), body.get("quant"))
        profile = profiles.guess(f)
        plan = sizing.plan(f, body.get("context"), profiles.kv_dtype_bytes(profile["engineArgs"]))
        await self._reload()
        return {"facts": f.public(), "profile": profile, "plan": plan, "suggestedId": self._free_id(f.repo),
                "name": f.repo.split("/")[-1]}

    def _free_id(self, repo: str) -> str:
        base = store.suggest_id(repo)
        mid, n = base, 2
        while store.read(self.models_root, mid) is not None or mid in self._installs():
            mid, n = f"{base[:60]}-{n}", n + 1
        return mid

    async def install(self, body: dict) -> dict:
        tokens = self._tokens(body)
        f = await self._facts(str(body["url"]), tokens, body.get("quant"))
        await self._reload()
        mid = str(body.get("id") or self._free_id(f.repo))
        store.model_dir(self.models_root, mid)  # validates
        running = self._installs().get(mid, {})
        if store.read(self.models_root, mid) is not None and running.get("state") != "failed":
            raise FileExistsError(f"a model named {mid} already exists")
        profile = profiles.guess(f)
        for k in ("toolParser", "reasoningParser", "engineArgs"):
            if k in body:
                profile[k] = body[k]
        plan = sizing.plan(f, body.get("context"), profiles.kv_dtype_bytes(profile["engineArgs"]))
        gpu, count = body.get("gpu"), body.get("count")
        option = None
        if gpu:
            count = int(count or 1)
            option = next((o for o in plan["options"] if o["gpu"] == gpu and o["count"] == count), None)
            if option is None or not option["fits"]:
                raise ValueError(f"{f.repo} doesn't fit on {count}× {gpu}"
                                 + (f" ({option['note']})" if option and option["note"] else ""))
        else:
            option = plan["recommended"]
            if option is None:
                raise ValueError(f"{f.repo} doesn't fit on {sizing.max_gpus()} GPUs of any kind")
            gpu, count = option["gpu"], option["count"]
        context = int(body.get("context") or min(plan["context"], option["maxContext"]))
        if option["maxContext"] and context > option["maxContext"]:
            raise ValueError(f"{context} tokens of context don't fit on {count}× {gpu} (at most "
                             f"{option['maxContext']})")
        manifest = store.new_manifest(mid, str(body.get("name") or f.repo.split("/")[-1]), str(body["url"]), f,
                                      profile, gpu, count, context)
        return await self._start_install(manifest, f.download, tokens)

    # MARK: models

    async def listing(self) -> dict:
        await self._reload()
        models = []
        for m in store.listing(self.models_root):
            models.append({**m, "status": await self.status(m["id"], m)})
        return {"models": models, "installs": {k: v for k, v in self._installs().items() if v.get("state") != "done"}}

    def manifest(self, mid: str) -> dict:
        m = store.read(self.models_root, mid)
        if m is None:
            raise LookupError(f"no model {mid}")
        return m

    async def plan(self, mid: str, context: int | None = None) -> dict:
        """The GPU options for an installed model (from the facts kept at install), and its current choice."""
        await self._reload()
        m = self.manifest(mid)
        if not m.get("facts"):
            raise ValueError(f"{mid} was installed before GPU plans were kept; remove and add it again to change it")
        f = facts_mod.Facts(**m["facts"])
        p = sizing.plan(f, context or m.get("context"), profiles.kv_dtype_bytes(m.get("engineArgs") or []))
        return {**p, "current": {"gpu": m["gpu"], "count": m["count"], "context": m["context"]}}

    async def update(self, mid: str, changes: dict) -> dict:
        await self._reload()
        m = store.patch(self.manifest(mid), changes)
        if m.get("facts") and {"gpu", "count", "context"} & set(changes):
            p = sizing.plan(facts_mod.Facts(**m["facts"]), m["context"],
                            profiles.kv_dtype_bytes(m.get("engineArgs") or []))
            o = next((o for o in p["options"] if o["gpu"] == m["gpu"] and o["count"] == m["count"]), None)
            if o is None or not o["fits"]:
                raise ValueError(f"{mid} doesn't fit on {m['count']}× {m['gpu']}")
            if m["context"] > o["maxContext"]:
                raise ValueError(f"{m['context']} tokens of context don't fit on {m['count']}× {m['gpu']} "
                                 f"(at most {o['maxContext']})")
        store.write(self.models_root, m)
        await self._commit()
        return m

    async def remove(self, mid: str) -> bool:
        await self._reload()
        self._cancel_install(mid)
        m = store.read(self.models_root, mid)
        if m is None:
            return False
        if (await self._engine_state(mid, m)).get("state") in ACTIVE:
            await self._engine_stop(mid, m)
        removed = store.remove(self.models_root, mid)
        await self._commit()
        return removed

    # MARK: warmth

    def usd_per_second(self, m: dict) -> float:
        if not self.billed:
            return 0.0
        return (GPU_PRICES.get(m.get("gpu") or "", 0.0) or 0.0) * int(m.get("count") or 1)

    async def status(self, mid: str, m: dict | None = None) -> dict:
        m = m or self.manifest(mid)
        s = await self._engine_state(mid, m)
        now = time.time()
        started = s.get("started")
        uptime = max(0.0, (s.get("ended") or now) - started) if started else 0.0
        price = self.usd_per_second({**m, "gpu": s.get("gpu") or m.get("gpu"), "count": s.get("count") or m.get("count")})
        return {"id": mid, "state": s.get("state", "cold"), "phase": s.get("phase"), "progress": s.get("progress"),
                "error": s.get("error"), "gpu": s.get("gpu") or m.get("gpu"), "count": s.get("count") or m.get("count"),
                "uptime": round(uptime, 1), "usdPerSecond": price, "costUSD": round(uptime * price, 4),
                "idleSeconds": self.idle, "leaseSeconds": self.lease,
                "lastChat": self.last_chat.get(mid),
                # Seconds since the start was asked for, while starting (the app compares it with the typical start).
                "bootSeconds": round(now - (s.get("requested") or started or now), 1)
                if s.get("state") in ("starting", "loading") else None,
                "coldStart": coldstart.summary(await self._coldstarts(mid), m.get("updated"))}

    async def warm(self, mid: str) -> dict:
        m = self._installed(mid)
        self.warmed[mid] = time.time()
        await self._engine_warm(mid, m)
        return await self.status(mid, m)

    async def keepalive(self, mid: str) -> dict:
        m = self._installed(mid)
        last = max(self.last_chat.get(mid, 0.0), self.warmed.get(mid, 0.0))
        s = await self.status(mid, m)
        if time.time() - last > self.lease:
            return {**s, "ignored": True}
        if s["state"] in ACTIVE:
            await self._engine_ping(mid, m)
        return s

    async def stop(self, mid: str) -> dict:
        m = self.manifest(mid)
        self.warmed.pop(mid, None)
        await self._engine_stop(mid, m)
        return await self.status(mid, m)

    def _installed(self, mid: str) -> dict:
        m = self.manifest(mid)
        if not store.installed(self.models_root, mid):
            raise LookupError(f"{mid} is still installing")
        return m

    # MARK: chat

    def chat(self, mid: str, body: dict) -> tuple[bool, AsyncIterator[bytes]]:
        """(streaming?, the bytes to send). Raises before anything is sent when the model or body is wrong."""
        m = self._installed(mid)
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise ValueError("expected messages")
        self.last_chat[mid] = time.time()
        stream = bool(body.get("stream"))
        body = {**body, "model": mid}
        if stream:
            body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        upstream = self._engine_chat(mid, m, body)
        if not stream:
            return False, upstream

        async def status():
            s = await self.status(mid, m)
            return {k: s[k] for k in ("state", "phase", "progress", "gpu", "count")}

        return True, proxy.relay(upstream, status, self.usd_per_second(m))


def fake_facts(link: str) -> facts_mod.Facts:
    """A tiny made-up model for ``fake/<name>`` links (AISTUDIO_FAKE_LLM=1): app tests without Hugging Face."""
    name = link.split("/", 1)[1] or "tiny"
    return facts_mod.Facts(
        repo=f"fake/{name}", revision="main", sha="0" * 40, architecture="FakeForCausalLM", model_type="qwen3",
        weight_bytes=16_000_000_000, files=1, quant={}, dtype="bfloat16", native_context=32768, vision="vl" in name,
        layers=32, attention_layers=32, attention_heads=32, kv_heads=8, head_dim=128, kv_lora_rank=0,
        rope_head_dim=0, sliding_window=0, experts=0, gated=False, chat_template="<think><tool_call>",
        download=[{"name": "config.json", "size": 2}, {"name": "model.safetensors", "size": 8}])


def fake_fetch(repo: str, revision: str, name: str, dest: Path, token: str | None) -> None:
    (dest / name).write_bytes(b"{}" if name.endswith(".json") else b"\0" * 8)


class LocalLLMService(LLMService):
    """Engines as processes on this machine, one model at a time (the GPUs are this machine's). Installs run in
    threads. With ``AISTUDIO_FAKE_LLM=1`` the engine is ``FakeEngine`` and ``fake/…`` links install instantly."""

    billed = False

    def __init__(self, models_root: Path, http=None, fetch: install_mod.FetchFile | None = None,
                 engine_factory=None, idle: float = IDLE, lease: float = LEASE, billed: bool | None = None,
                 sync_http=None, fetch_url: install_mod.FetchURL | None = None):
        super().__init__(models_root, http, idle, lease, sync_http)
        if billed is not None:
            self.billed = billed
        self.fetch, self.fetch_url = fetch, fetch_url
        self.engine_factory = engine_factory or (FakeEngine if fake_enabled() else EngineProcess)
        self.states: dict[str, dict] = {}
        self.engine: tuple[str, object] | None = None
        self.activity: dict[str, float] = {}
        self.streams: dict[str, int] = {}
        self.lock = asyncio.Lock()
        self.install_states: dict[str, dict] = {}
        self.install_cancels: dict[str, threading.Event] = {}
        self.reaper: asyncio.Task | None = None
        self.history: dict[str, list] = {}

    async def _coldstarts(self, mid):
        return self.history.get(mid, [])

    # installs

    async def _start_install(self, manifest, files, tokens):
        mid = manifest["id"]
        cancel = threading.Event()
        self.install_cancels[mid] = cancel
        self.install_states[mid] = {"state": "starting", "name": manifest["name"]}
        fetch = self.fetch or (fake_fetch if manifest["repo"].startswith("fake/") else None)

        def progress(p):
            self.install_states[mid] = {**self.install_states.get(mid, {}), **p}

        def run():
            try:
                install_mod.install(self.models_root, manifest, files, tokens, progress, cancel.is_set, fetch,
                                    self.fetch_url)
            except BaseException as e:  # noqa: BLE001
                progress({"state": "cancelled" if isinstance(e, InterruptedError) else "failed",
                          "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=run, daemon=True, name=f"llm-install-{mid}").start()
        return {"id": mid, **self.install_states[mid]}

    def _installs(self):
        return dict(self.install_states)

    def _cancel_install(self, mid):
        if (ev := self.install_cancels.get(mid)) is not None:
            ev.set()
        self.install_states.pop(mid, None)

    # engines

    async def _engine_state(self, mid, m):
        s = dict(self.states.get(mid) or {"state": "cold"})
        if s.get("state") in ACTIVE and (self.engine is None or self.engine[0] != mid):
            s["state"] = "cold"
        return s

    def _set(self, mid, **kw):
        self.states[mid] = {**self.states.get(mid, {}), **kw}

    async def _engine_warm(self, mid, m):
        async with self.lock:
            if self.engine and self.engine[0] == mid and self.states.get(mid, {}).get("state") in ACTIVE:
                self.activity[mid] = time.monotonic()
                return
            if self.engine:
                other, proc = self.engine
                self.engine = None
                await asyncio.to_thread(proc.stop)
                self._set(other, state="cold", ended=time.time())

            clock = coldstart.ColdStartClock()

            def on_status(phase, progress):
                state = "ready" if phase == "Ready" else ("loading" if phase != "Starting engine" else "starting")
                self._set(mid, state=state, phase=phase, progress=progress)
                clock.phase(phase)

            proc = self.engine_factory(m, self.models_root, on_status)
            self.engine = (mid, proc)
            self.states[mid] = {"state": "starting", "phase": "Starting engine", "progress": None,
                                "started": time.time(), "gpu": m.get("gpu"), "count": m.get("count")}
            self.activity[mid] = time.monotonic()

            def boot():
                try:
                    proc.start()
                    proc.wait_ready()
                    record = clock.finish(gpu=m.get("gpu"), count=m.get("count"), updated=m.get("updated"),
                                          loadFormat=m.get("loadFormat") or "auto")
                    self.history[mid] = coldstart.remember(self.history.get(mid), record)
                except BaseException as e:  # noqa: BLE001
                    self._set(mid, state="failed", error=str(e)[:4000], ended=time.time())
                    if self.engine and self.engine[1] is proc:
                        self.engine = None
                    proc.stop()

            threading.Thread(target=boot, daemon=True, name=f"llm-boot-{mid}").start()
            if self.reaper is None or self.reaper.done():
                self.reaper = asyncio.create_task(self._reap())

    async def _engine_ping(self, mid, m):
        self.activity[mid] = time.monotonic()

    async def _engine_stop(self, mid, m):
        async with self.lock:
            if self.engine and self.engine[0] == mid:
                _, proc = self.engine
                self.engine = None
                await asyncio.to_thread(proc.stop)
                self._set(mid, state="cold", phase=None, progress=None, ended=time.time())

    async def _reap(self):
        while self.engine is not None:
            await asyncio.sleep(min(1.0, self.idle / 4))
            if self.engine is None:
                break
            mid = self.engine[0]
            if self.streams.get(mid):
                continue
            if time.monotonic() - self.activity.get(mid, 0) > self.idle:
                await self._engine_stop(mid, None)

    async def _engine_chat(self, mid, m, body):
        await self._engine_warm(mid, m)
        self.streams[mid] = self.streams.get(mid, 0) + 1
        try:
            while True:
                s = self.states.get(mid, {})
                if s.get("state") == "ready":
                    break
                if s.get("state") == "failed":
                    raise EngineError(s.get("error") or "the model failed to start")
                if self.engine is None or self.engine[0] != mid:
                    raise EngineError("the model was stopped")
                await asyncio.sleep(0.2)
            proc = self.engine[1]
            async for chunk in proc.chat(body):
                self.activity[mid] = time.monotonic()
                yield chunk
        finally:
            self.streams[mid] -= 1
            self.activity[mid] = time.monotonic()
