"""Running jobs on Modal: the front desk spawns a GPU worker per job and relays its events.

Worker → front desk: a ``modal.Queue`` partition per job (status, progress, previews), the result as the
function's return value. Front desk → worker: cancel flags in a ``modal.Dict`` the worker checks between steps — so
a cancel stops the work but keeps the warm container (and its 60 GB of loaded weights) for the next job. A job whose
worker hasn't started yet is cancelled with ``FunctionCall.cancel()`` instead.
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .jobs import TERMINAL, Job, JobManager, PreviewLimiter
from .local import LocalRunner, describe_error
from .media import jpeg_preview
from .tasks import TASKS, JobCancelled, Output, TaskSpec, models_installed
from .wire import Kind

MARKER_REFRESH = 30.0  # seconds between re-reading the models Volume (new downloads)


class ModalRunner:
    name = "modal"

    def __init__(self, workers: dict, events, cancels, models_volume, models_root: Path, uploads_volume=None):
        self.workers = workers  # task family → modal Cls
        # With an uploads Volume (the front desk's file store) workers read inputs from it; otherwise they get bytes.
        self.uploads_volume = uploads_volume
        self.events, self.cancels = events, cancels
        self.models_volume, self.models_root = models_volume, models_root
        self._refreshed = 0.0
        # The CPU test tasks run right here on the front desk: a deployment check that spends no GPU time.
        self.cpu = LocalRunner(models_root, allow_gpu_tasks=False)

    def installed(self, spec: TaskSpec) -> bool:
        if spec.cpu_only:
            return True
        now = time.monotonic()
        if now - self._refreshed > MARKER_REFRESH:
            self._refreshed = now
            try:
                self.models_volume.reload()
            except Exception:  # noqa: BLE001 — a busy Volume: use what we saw last time
                pass
        return spec.family in self.workers and models_installed(spec, self.models_root)

    async def start(self, manager: JobManager, job: Job) -> None:
        if job.spec.cpu_only:
            await self.cpu.start(manager, job)
            return
        inputs = {name: (file, None if self.uploads_volume is not None else manager.files.path(file).read_bytes())
                  for name, file in job.inputs.items()}
        worker = self.workers[job.spec.family]()
        job.handle = await worker.run.spawn.aio(job.id, job.spec.id, job.params, inputs)
        asyncio.create_task(self._pump(manager, job))

    async def cancel(self, manager: JobManager, job: Job) -> None:
        if job.spec.cpu_only:
            return
        if job.state == "running":
            await self.cancels.put.aio(job.id, True)  # the worker stops at its next step
        elif job.handle is not None:
            await job.handle.cancel.aio()

    async def _pump(self, manager: JobManager, job: Job) -> None:
        call = job.handle
        result = asyncio.create_task(call.get.aio())
        start, started = time.monotonic(), False
        try:
            while True:
                try:
                    items = await self.events.get_many.aio(64, partition=job.id, timeout=1.0)
                except Exception:  # noqa: BLE001
                    items = []
                for kind, step, total, payload in items:
                    if job.state in TERMINAL:  # cancelled here while the worker winds down
                        break
                    if not started:
                        started = True
                        manager.mark_running(job)
                    if kind == "status":
                        manager.set_status(job, payload)
                    elif kind == "progress":
                        manager.emit(job, Kind.PROGRESS, step, total)
                    elif kind == "preview":
                        manager.emit(job, Kind.PREVIEW, step, total, payload)
                if not started and job.state not in TERMINAL:
                    m, s = divmod(int(time.monotonic() - start), 60)
                    what = "Starting GPU" if m == 0 and s < 20 else "Starting GPU and loading models"
                    manager.set_status(job, f"{what} {m}:{s:02}")
                if result.done() and not items:
                    break
            try:
                res = result.result()
            except Exception as e:  # noqa: BLE001 — cancelled before starting, crashed, timed out
                if job.cancel_requested.is_set():
                    manager.mark_cancelled(job)
                else:
                    manager.fail(job, describe_error(e))
                return
            if res.get("cancelled"):
                manager.mark_cancelled(job)
            elif "error" in res:
                manager.fail(job, res["error"])
            else:
                path = manager.results_dir / f"{job.id}{res['suffix']}"
                path.write_bytes(res["data"])
                manager.finish(job, Output(path, res["kind"], res["contentType"], load_seconds=res["loadSeconds"]),
                               res["gpuSeconds"])
        finally:
            if not result.done():
                result.cancel()
            for cleanup in (self.events.clear.aio(partition=job.id), self.cancels.pop.aio(job.id, None)):
                try:
                    await cleanup
                except Exception:  # noqa: BLE001
                    pass


class QueueSink:
    """A GPU worker's sink on Modal: events go out in batches from a background thread, so a slow network never
    holds up the GPU; the cancel flag is checked at most twice a second."""

    def __init__(self, events, cancels, job_id: str):
        self.events, self.cancels, self.job_id = events, cancels, job_id
        self.buffer: queue_mod.Queue = queue_mod.Queue()
        self.previews = PreviewLimiter()
        self._checked = 0.0
        self.thread = threading.Thread(target=self._send, daemon=True)
        self.thread.start()

    def status(self, text):
        self.buffer.put(("status", 0, 0, text))

    def progress(self, step, total):
        self.buffer.put(("progress", step, total, b""))

    def preview(self, image, step, total):
        if self.previews.allow(step, total):
            self.buffer.put(("preview", step, total, jpeg_preview(image)))

    def check(self):
        now = time.monotonic()
        if now - self._checked < 0.5:
            return
        self._checked = now
        try:
            if self.cancels.get(self.job_id, None):
                raise JobCancelled()
        except JobCancelled:
            raise
        except Exception:  # noqa: BLE001 — the Dict briefly unreachable: keep working
            pass

    def close(self):
        self.buffer.put(None)
        self.thread.join(timeout=10)

    def _send(self):
        done = False
        while not done:
            items = [self.buffer.get()]
            while True:
                try:
                    items.append(self.buffer.get_nowait())
                except queue_mod.Empty:
                    break
            if None in items:
                done = True
                items = [i for i in items if i is not None]
            if items:
                try:
                    self.events.put_many(items, partition=self.job_id)
                except Exception:  # noqa: BLE001 — events are best effort; the result is the return value
                    pass


class WorkerState:
    """What a GPU container keeps between jobs: the loaded worker and how long loading took."""

    def __init__(self, family: str, models_root: Path):
        from .tasks import make_worker

        start = time.monotonic()
        self.worker = make_worker(family)
        self.worker.load(models_root, lambda s: print(s, flush=True))
        self.load_seconds = time.monotonic() - start
        self.first_job = True
        print(f"{family} loaded in {self.load_seconds:.1f} s", flush=True)


def _see_new_variant(spec: TaskSpec, params: dict, models: tuple | None) -> None:
    """A fine-tune installed after this container started: re-read the models Volume once to see it."""
    from . import variants

    family = variants.TASK_FAMILIES.get(spec.family)
    vid = params.get("model")
    if models is None or not family or vid in (None, "", "base"):
        return
    volume, root = models
    try:
        if not variants.installed(Path(root), family, vid):
            volume.reload()
    except Exception:  # noqa: BLE001 — a bad id or a busy Volume: the worker's own error says what's missing
        pass


def fetch_upload(uploads: tuple | None, file: str) -> Path:
    """An uploaded file on the uploads Volume, re-reading the Volume once if this container hasn't seen it yet."""
    if uploads is None:
        raise RuntimeError(f"input {file} was sent by reference, but this worker has no uploads Volume")
    volume, root = uploads
    path = Path(root) / file
    if not path.exists():
        try:
            volume.reload()
        except Exception:  # noqa: BLE001 — open files: fall through to the check below
            pass
    if not path.exists():
        raise FileNotFoundError(f"the upload {file} isn't on the server any more: run again to re-send it")
    return path


def run_job(state: WorkerState, events, cancels, job_id: str, task_id: str, params: dict,
            inputs: dict[str, tuple[str, bytes | None]], uploads: tuple | None = None,
            models: tuple | None = None) -> dict:
    """The body of every GPU worker's ``run`` method; returns the result (or the error) to the front desk.

    An input is ``(name, bytes)``, or ``(name, None)`` to copy from the uploads Volume (``uploads`` = the Volume and
    where it's mounted): big files such as LoRAs then never pass through the front desk's memory."""
    spec = TASKS[task_id]
    _see_new_variant(spec, params, models)
    sink = QueueSink(events, cancels, job_id)
    load = state.load_seconds if state.first_job else 0.0
    state.first_job = False
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        try:
            for name, (file, data) in inputs.items():
                p = Path(tmp) / file
                if data is None:
                    shutil.copyfile(fetch_upload(uploads, file), p)
                else:
                    p.write_bytes(data)
                paths[name] = p
        except Exception as e:  # noqa: BLE001
            sink.close()
            return {"error": describe_error(e)}
        start = time.monotonic()
        try:
            out = state.worker.run(spec, params, paths, sink, Path(tmp))
        except JobCancelled:
            return {"cancelled": True}
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return {"error": describe_error(e)}
        finally:
            sink.close()
        return {"data": out.path.read_bytes(), "suffix": out.path.suffix, "kind": out.kind,
                "contentType": out.content_type, "gpuSeconds": time.monotonic() - start, "loadSeconds": load,
                "extra": out.extra}
