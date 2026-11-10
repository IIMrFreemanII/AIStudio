"""Jobs on the front desk: their state, the event stream (a replay ring + live subscribers), and the sink a worker
reports through. Independent of where the work runs (``local.LocalRunner`` or ``modal_runner.ModalRunner``)."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import API_VERSION, VERSION
from .files import FileStore
from .variants import list_variants
from .media import jpeg_preview
from .tasks import TASKS, JobCancelled, Output, TaskSpec
from .wire import Kind, encode, finished_payload, output_payload

TERMINAL = {"finished", "failed", "cancelled"}
RING = 1024  # records kept per job for replay (previews: only the latest)
PREVIEW_INTERVAL = 0.25  # at most 4 previews a second


@dataclass
class Result:
    path: Path
    kind: str
    content_type: str
    size: int
    etag: str


@dataclass
class Job:
    id: str
    run: int
    spec: TaskSpec
    params: dict
    inputs: dict[str, str]
    created: float = field(default_factory=time.time)
    state: str = "queued"
    started: float | None = None
    ended: float | None = None
    seq: int = 0
    ring: deque = field(default_factory=lambda: deque(maxlen=RING))  # (seq, kind, record bytes)
    subscribers: set = field(default_factory=set)
    result: Result | None = None
    error: str | None = None
    status: str | None = None
    gpu_seconds: float = 0.0
    load_seconds: float = 0.0
    cost: float | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    handle: Any = None  # the runner's (a Modal FunctionCall, a Future…)

    def summary(self) -> dict:
        return {
            "id": self.id, "run": self.run, "task": self.spec.id, "state": self.state, "status": self.status,
            "params": self.params, "created": self.created, "started": self.started, "ended": self.ended,
            "gpuSeconds": round(self.gpu_seconds, 3), "loadSeconds": round(self.load_seconds, 3),
            "costUSD": None if self.cost is None else round(self.cost, 5), "error": self.error,
            "result": None if self.result is None else {
                "kind": self.result.kind, "contentType": self.result.content_type, "bytes": self.result.size},
        }


class Runner(Protocol):
    name: str

    def installed(self, spec: TaskSpec) -> bool: ...
    async def start(self, manager: "JobManager", job: Job) -> None: ...
    async def cancel(self, manager: "JobManager", job: Job) -> None: ...


class JobManager:
    def __init__(self, runner: Runner, files: FileStore, results_dir: Path, keep_jobs: int = 200,
                 idle_seconds: int | None = None):
        self.runner = runner
        self.files = files
        self.results_dir = results_dir
        self.keep_jobs = keep_jobs
        self.idle_seconds = idle_seconds
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._next_run = secrets.randbelow(1 << 20) + 1
        results_dir.mkdir(parents=True, exist_ok=True)

    # MARK: info and submission

    def info(self) -> dict:
        return {
            "server": "aistudio-gpu", "version": VERSION, "api": API_VERSION, "runner": self.runner.name,
            "idleSeconds": self.idle_seconds,
            "tasks": [t.describe(self.runner.installed(t)) for t in TASKS.values()],
            # Fine-tunes installed from links, per family ({"flux.1": [{"id", "name", "bytes"}]}): a task's `model`.
            "variants": list_variants(self.runner.models_root) if getattr(self.runner, "models_root", None) else {},
            "jobs": {s: sum(1 for j in self.jobs.values() if j.state == s) for s in ("queued", "running")},
        }

    async def submit(self, task: str, params: dict, inputs: dict[str, str]) -> Job:
        self.loop = asyncio.get_running_loop()
        spec = TASKS.get(task)
        if spec is None:
            raise LookupError(f"unknown task {task!r} (this server has: {', '.join(TASKS)})")
        normalized = spec.normalize(params or {})
        inputs = inputs or {}
        for name in spec.inputs:
            if name not in inputs:
                raise ValueError(f"{task} needs the input '{name}'")
        for name, file in inputs.items():
            if name not in spec.inputs and name not in spec.optional_inputs:
                raise ValueError(f"{task} takes no input '{name}'")
            self.files.touch(file)
            if not self.files.exists(file):
                raise FileNotFoundError(file)
        if not self.runner.installed(spec):
            raise ModelsMissing(spec)
        job = Job(id=secrets.token_hex(16), run=self._next_run, spec=spec, params=normalized, inputs=dict(inputs))
        self._next_run = (self._next_run + 1) & 0xFFFFFFFF or 1
        self.jobs[job.id] = job
        self._prune()
        self.emit(job, Kind.STARTED)
        self.set_status(job, "Queued")
        await self.runner.start(self, job)
        return job

    def _prune(self) -> None:
        while len(self.jobs) > self.keep_jobs:
            old_id, old = next(iter(self.jobs.items()))
            if old.state not in TERMINAL:
                break
            self.jobs.pop(old_id)
            if old.result:
                old.result.path.unlink(missing_ok=True)

    # MARK: events (event-loop thread only)

    def emit(self, job: Job, kind: int, step: int = 0, total: int = 0, payload: bytes = b"") -> None:
        job.seq += 1
        record = encode(kind, job.run, job.seq, step, total, payload)
        if kind == Kind.PREVIEW:  # latest-wins: a reconnecting client only needs the newest preview
            for i in range(len(job.ring) - 1, -1, -1):
                if job.ring[i][1] == Kind.PREVIEW:
                    del job.ring[i]
                    break
        job.ring.append((job.seq, kind, record))
        for q in list(job.subscribers):
            q.put_nowait((kind, record))

    def set_status(self, job: Job, text: str | None) -> None:
        if text == job.status:
            return
        job.status = text
        self.emit(job, Kind.STATUS, payload=(text or "").encode())

    def subscribe(self, job: Job, after: int) -> tuple[list[bytes], asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue()
        backlog = [rec for seq, _, rec in job.ring if seq > after]
        if job.state in TERMINAL:
            q.put_nowait(None)
        else:
            job.subscribers.add(q)
        return backlog, q

    def unsubscribe(self, job: Job, q: asyncio.Queue) -> None:
        job.subscribers.discard(q)

    def _close(self, job: Job) -> None:
        for q in list(job.subscribers):
            q.put_nowait(None)
        job.subscribers.clear()

    def threadsafe(self, fn, *args) -> None:
        assert self.loop is not None
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # the server is shutting down: stop the job's worker at its next step
            for a in args:
                if isinstance(a, Job):
                    a.cancel_requested.set()

    # MARK: lifecycle (event-loop thread only)

    def mark_running(self, job: Job) -> None:
        if job.state == "queued":
            job.state = "running"
            job.started = time.time()

    def finish(self, job: Job, output: Output, gpu_seconds: float) -> None:
        if job.state in TERMINAL:
            return
        size = output.path.stat().st_size
        final = self.results_dir / f"{job.id}{output.path.suffix}"
        if output.path != final:
            output.path.replace(final)
        etag = hashlib.sha256(f"{job.id}:{size}".encode()).hexdigest()[:16]
        job.result = Result(final, output.kind, output.content_type, size, etag)
        job.gpu_seconds = gpu_seconds
        job.load_seconds = output.load_seconds
        price = job.spec.price_per_second
        # Billed GPU time we can see: this job plus the model loading of a cold container. (The idle window after
        # the last job is billed too, but isn't part of any one job.)
        job.cost = None if price is None else (gpu_seconds + output.load_seconds) * price
        job.state = "finished"
        job.ended = time.time()
        self.set_status(job, None)
        self.emit(job, Kind.OUTPUT_READY, payload=output_payload(final.name, output.kind, size))
        self.emit(job, Kind.FINISHED, payload=finished_payload(job.ended - job.created, gpu_seconds, job.cost))
        self._close(job)

    def fail(self, job: Job, message: str) -> None:
        if job.state in TERMINAL:
            return
        job.state, job.error, job.ended = "failed", message, time.time()
        self.set_status(job, None)
        self.emit(job, Kind.RUN_FAILED, payload=message.encode())
        self._close(job)

    def mark_cancelled(self, job: Job) -> None:
        if job.state in TERMINAL:
            return
        job.state, job.ended = "cancelled", time.time()
        self.set_status(job, None)
        self.emit(job, Kind.CANCELLED)
        self._close(job)

    async def cancel(self, job: Job) -> None:
        if job.state in TERMINAL:
            return
        job.cancel_requested.set()
        await self.runner.cancel(self, job)
        self.mark_cancelled(job)


class ModelsMissing(Exception):
    def __init__(self, spec: TaskSpec):
        super().__init__(f"{spec.title} isn't installed on this server: run the download step for "
                         f"{', '.join(spec.models)} (see docs/remote-server.md)")
        self.spec = spec


class PreviewLimiter:
    """At most one preview per PREVIEW_INTERVAL (the last step's always passes)."""

    def __init__(self):
        self._last = 0.0

    def allow(self, step: int, total: int) -> bool:
        now = time.monotonic()
        if now - self._last < PREVIEW_INTERVAL and step < total:
            return False
        self._last = now
        return True


class ThreadSink:
    """A worker's sink when the worker runs in a thread of this process (local runner)."""

    def __init__(self, manager: JobManager, job: Job):
        self.manager, self.job = manager, job
        self.previews = PreviewLimiter()

    def status(self, text: str | None) -> None:
        self.manager.threadsafe(self.manager.set_status, self.job, text)

    def progress(self, step: int, total: int) -> None:
        self.manager.threadsafe(self.manager.emit, self.job, Kind.PROGRESS, step, total)

    def preview(self, image: Any, step: int, total: int) -> None:
        if not self.previews.allow(step, total):
            return
        data = jpeg_preview(image)
        self.manager.threadsafe(self.manager.emit, self.job, Kind.PREVIEW, step, total, data)

    def check(self) -> None:
        if self.job.cancel_requested.is_set():
            raise JobCancelled()
