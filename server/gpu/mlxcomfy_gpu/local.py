"""Running jobs in this process: one worker thread (the GPU does one job at a time), models loaded once per task
family and kept until a job of another family needs the memory. Used by the tests (CPU test tasks) and on a GPU
box (`python -m mlxcomfy_gpu serve`)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .jobs import Job, JobManager, ThreadSink
from .tasks import JobCancelled, TaskSpec, make_worker, models_installed


class LocalRunner:
    name = "local"

    def __init__(self, models_root: Path, allow_gpu_tasks: bool = True):
        self.models_root = models_root
        self.allow_gpu_tasks = allow_gpu_tasks
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")
        self.workers: dict[str, object] = {}
        self.load_seconds: dict[str, float] = {}

    def installed(self, spec: TaskSpec) -> bool:
        if spec.cpu_only:
            return True
        return self.allow_gpu_tasks and models_installed(spec, self.models_root)

    async def start(self, manager: JobManager, job: Job) -> None:
        job.handle = asyncio.get_running_loop().run_in_executor(self.executor, self._run, manager, job)

    async def cancel(self, manager: JobManager, job: Job) -> None:
        # The worker sees `cancel_requested` at its next step (or skips the job if it hasn't started).
        pass

    def _worker(self, family: str, status):
        if family not in self.workers:
            # One GPU family resident at a time: free the others first.
            for other in [f for f in self.workers if f != family and f != "test"]:
                self.workers.pop(other)
                _free_gpu_memory()
            start = time.monotonic()
            status("Loading models")
            worker = make_worker(family)
            worker.load(self.models_root, status)
            self.workers[family] = worker
            self.load_seconds[family] = time.monotonic() - start
        return self.workers[family]

    def _run(self, manager: JobManager, job: Job) -> None:
        sink = ThreadSink(manager, job)
        if job.cancel_requested.is_set():
            return
        manager.threadsafe(manager.mark_running, job)
        work_dir = Path(tempfile.mkdtemp(prefix=f"job-{job.id[:8]}-", dir=manager.results_dir))
        try:
            cold = job.spec.family not in self.workers
            worker = self._worker(job.spec.family, sink.status)
            inputs = {name: manager.files.path(file) for name, file in job.inputs.items()}
            start = time.monotonic()
            output = worker.run(job.spec, job.params, inputs, sink, work_dir)
            output.load_seconds = self.load_seconds.get(job.spec.family, 0.0) if cold else 0.0
            # Out of the work folder before it is deleted below (finish runs later, on the event loop).
            final = manager.results_dir / f"{job.id}{output.path.suffix}"
            output.path.replace(final)
            output.path = final
            manager.threadsafe(manager.finish, job, output, time.monotonic() - start)
        except JobCancelled:
            manager.threadsafe(manager.mark_cancelled, job)
        except Exception as e:  # noqa: BLE001 — every failure goes back to the client
            traceback.print_exc()
            manager.threadsafe(manager.fail, job, describe_error(e))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def describe_error(e: BaseException) -> str:
    text = str(e).strip() or type(e).__name__
    if "out of memory" in text.lower():
        return f"The GPU ran out of memory ({text.splitlines()[0]}). Try a smaller size or fewer frames."
    return text.splitlines()[0][:500]


def _free_gpu_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
