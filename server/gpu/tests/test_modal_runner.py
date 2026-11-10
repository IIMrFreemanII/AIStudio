"""The Modal runner's logic — spawn, relaying worker events, "Starting GPU" while the worker boots, results, errors,
cancelling a running job by flag (container kept) or a queued one by FunctionCall.cancel — with in-memory stand-ins
for Modal's Queue, Dict and FunctionCall, through the real API and the real worker-side ``run_job``."""

import asyncio
import json
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlxcomfy_gpu.api import create_app
from mlxcomfy_gpu.files import FileStore
from mlxcomfy_gpu.jobs import JobManager
from mlxcomfy_gpu.media import write_png
from mlxcomfy_gpu.modal_runner import ModalRunner, run_job
from mlxcomfy_gpu.tasks import Output
from mlxcomfy_gpu.wire import Kind, decode_all


class Aio:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *a, **k):
        return self.fn(*a, **k)

    async def aio(self, *a, **k):
        return await asyncio.to_thread(self.fn, *a, **k)


class FakeQueue:
    def __init__(self):
        self.parts, self.lock = defaultdict(list), threading.Condition()
        self.put_many = Aio(self._put_many)
        self.get_many = Aio(self._get_many)
        self.clear = Aio(lambda partition=None: self.parts.pop(partition, None))

    def _put_many(self, items, partition=None):
        with self.lock:
            self.parts[partition] += items
            self.lock.notify_all()

    def _get_many(self, n, partition=None, timeout=None):
        with self.lock:
            self.lock.wait_for(lambda: self.parts[partition], timeout=timeout)
            items, self.parts[partition] = self.parts[partition][:n], self.parts[partition][n:]
            return items


class FakeDict:
    def __init__(self):
        self.d = {}
        self.put = Aio(lambda k, v: self.d.__setitem__(k, v))
        self.pop = Aio(lambda k, default=None: self.d.pop(k, default))

    def get(self, k, default=None):
        return self.d.get(k, default)


class FakeCall:
    def __init__(self, fn, boot_seconds):
        self.cancelled = False
        self.done = threading.Event()
        self.value, self.error = None, None

        def go():
            time.sleep(boot_seconds)  # the container starting
            if self.cancelled:
                self.error = RuntimeError("input cancelled")
            else:
                try:
                    self.value = fn()
                except Exception as e:  # noqa: BLE001
                    self.error = e
            self.done.set()

        threading.Thread(target=go, daemon=True).start()
        self.get = Aio(self._get)
        self.cancel = Aio(self._cancel)

    def _get(self):
        self.done.wait()
        if self.error:
            raise self.error
        return self.value

    def _cancel(self):
        self.cancelled = True


class StepWorker:
    """A stand-in model: `steps` steps of `delay` s, a preview each, a PNG result; fails if prompt == "boom"."""

    def run(self, task, params, inputs, sink, out_dir):
        sink.status("Sampling")
        for i in range(1, params["steps"] + 1):
            sink.check()
            time.sleep(params["delay"])
            if params["prompt"] == "boom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            sink.progress(i, params["steps"])
            sink.preview(np.zeros((8, 8, 3), np.uint8), i, params["steps"])
        path = out_dir / "result.png"
        write_png(np.full((16, 16, 3), 200, np.uint8), path)
        return Output(path, "image", "image/png")


class State:
    worker = StepWorker()
    load_seconds = 7.5
    first_job = True


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    from mlxcomfy_gpu import tasks
    from mlxcomfy_gpu.tasks import Param, TaskSpec

    spec = TaskSpec("fake-gpu", "Fake", "image", "fake", "H100",
                    {"prompt": Param("string", ""), "steps": Param("int", 3), "delay": Param("float", 0.02)},
                    models=("fake",))
    monkeypatch.setitem(tasks.TASKS, "fake-gpu", spec)
    (tmp_path / "models" / "fake").mkdir(parents=True)
    (tmp_path / "models" / "fake" / ".complete").write_text("{}")
    events, cancels, calls = FakeQueue(), FakeDict(), []
    state = State()
    boot = {"seconds": 1.2}

    class Cls:
        def __init__(self):
            self.run = self

        @property
        def spawn(self):
            def spawn(job_id, task, params, inputs):
                call = FakeCall(lambda: run_job(state, events, cancels, job_id, task, params, inputs),
                                boot["seconds"])
                calls.append(call)
                return call
            return Aio(spawn)

    class Volume:
        def reload(self):
            pass

    runner = ModalRunner({"fake": Cls}, events, cancels, Volume(), tmp_path / "models")
    manager = JobManager(runner, FileStore(tmp_path / "uploads"), tmp_path / "results")
    with TestClient(create_app(manager)) as client:
        yield client, calls, cancels, boot


def events_of(client, job_id):
    out = []
    with client.websocket_connect(f"/v1/jobs/{job_id}/events") as ws:
        while True:
            try:
                out += decode_all(ws.receive_bytes())
            except Exception:
                break
    return out


def test_relays_events_and_returns_the_result(env):
    client, calls, _, _ = env
    job = client.post("/v1/jobs", json={"task": "fake-gpu", "params": {"steps": 3}}).json()
    evs = events_of(client, job["id"])
    statuses = [e.payload.decode() for e in evs if e.kind == Kind.STATUS]
    assert any(s.startswith("Starting GPU 0:0") for s in statuses) and "Sampling" in statuses
    assert [e.step for e in evs if e.kind == Kind.PROGRESS] == [1, 2, 3]
    assert any(e.kind == Kind.PREVIEW for e in evs)
    assert json.loads(next(e for e in evs if e.kind == Kind.OUTPUT_READY).payload)["kind"] == "image"
    seconds, gpu, cost = struct.unpack("<ddd", evs[-1].payload)
    assert evs[-1].kind == Kind.FINISHED and seconds >= 1.2 and 0 < gpu < 1
    # Cost: (GPU seconds + the cold container's load time) × the H100 price.
    assert cost == pytest.approx((gpu + 7.5) * 0.001097, rel=1e-6)
    assert client.get(f"/v1/jobs/{job['id']}/result").content[:4] == b"\x89PNG"
    # The next job on the warm container doesn't pay the load again.
    job2 = client.post("/v1/jobs", json={"task": "fake-gpu"}).json()
    _, gpu2, cost2 = struct.unpack("<ddd", events_of(client, job2["id"])[-1].payload)
    assert cost2 == pytest.approx(gpu2 * 0.001097, rel=1e-6)


def test_worker_errors_come_back_readable(env):
    client, _, _, _ = env
    job = client.post("/v1/jobs", json={"task": "fake-gpu", "params": {"prompt": "boom"}}).json()
    evs = events_of(client, job["id"])
    assert evs[-1].kind == Kind.RUN_FAILED and b"ran out of memory" in evs[-1].payload


def test_cancel_while_running_uses_the_flag_and_keeps_the_container(env):
    client, calls, cancels, boot = env
    boot["seconds"] = 0.0
    job = client.post("/v1/jobs", json={"task": "fake-gpu", "params": {"steps": 40, "delay": 0.05}}).json()
    deadline = time.time() + 5
    while client.get(f"/v1/jobs/{job['id']}").json()["state"] != "running" and time.time() < deadline:
        time.sleep(0.05)
    assert client.post(f"/v1/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
    calls[-1].done.wait(5)
    assert calls[-1].value == {"cancelled": True} and not calls[-1].cancelled  # stopped by flag, not killed
    assert events_of(client, job["id"])[-1].kind == Kind.CANCELLED


def test_cancel_while_starting_cancels_the_call(env):
    client, calls, _, boot = env
    boot["seconds"] = 1.0
    job = client.post("/v1/jobs", json={"task": "fake-gpu"}).json()
    assert client.post(f"/v1/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
    assert calls[-1].cancelled


class FakeVolume:
    def __init__(self):
        self.reloads = 0

    def reload(self):
        self.reloads += 1


def test_inputs_by_reference_come_from_the_uploads_volume(tmp_path):
    """With an uploads Volume the runner sends (name, None) and the worker copies the file from the Volume,
    re-reading it once when this container hasn't seen the upload yet; a missing upload is a job error."""
    from mlxcomfy_gpu.modal_runner import run_job

    seen = {}

    class Reader:
        def run(self, task, params, inputs, sink, out_dir):
            seen.update({k: p.read_bytes() for k, p in inputs.items()})
            path = out_dir / "result.png"
            write_png(np.zeros((4, 4, 3), np.uint8), path)
            return Output(path, "image", "image/png")

    class S:
        worker, load_seconds, first_job = Reader(), 0.0, False

    root, volume = tmp_path / "uploads", FakeVolume()
    root.mkdir()
    name = "a" * 64 + ".lora"
    (root / name).write_bytes(b"lora bytes")
    res = run_job(S, FakeQueue(), FakeDict(), "j1", "test-image", {}, {"lora1": (name, None)},
                  uploads=(volume, root))
    assert "data" in res and seen == {"lora1": b"lora bytes"} and volume.reloads == 0
    res = run_job(S, FakeQueue(), FakeDict(), "j2", "test-image", {}, {"lora1": ("b" * 64, None)},
                  uploads=(volume, root))
    assert "isn't on the server any more" in res["error"] and volume.reloads == 1
