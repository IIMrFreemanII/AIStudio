"""The job API end to end with the local runner and the CPU test tasks (no torch, no models)."""

import hashlib
import json
import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlxcomfy_gpu.api import create_app
from mlxcomfy_gpu.files import FileStore
from mlxcomfy_gpu.jobs import JobManager
from mlxcomfy_gpu.local import LocalRunner
from mlxcomfy_gpu.wire import Kind, decode_all

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture()
def client(tmp_path: Path):
    runner = LocalRunner(tmp_path / "models", allow_gpu_tasks=False)
    manager = JobManager(runner, FileStore(tmp_path / "uploads"), tmp_path / "results", idle_seconds=120)
    with TestClient(create_app(manager, TOKEN)) as c:
        yield c


def events(client, job_id, after=0):
    out = []
    with client.websocket_connect(f"/v1/jobs/{job_id}/events?after={after}", headers=AUTH) as ws:
        while True:
            try:
                out += decode_all(ws.receive_bytes())
            except Exception:  # closed after the terminal record
                break
    return out


def test_auth_and_info(client):
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/info", headers={"Authorization": "Bearer nope"}).status_code == 401
    info = client.get("/v1/info", headers=AUTH).json()
    assert info["api"] == 1 and info["runner"] == "local"
    tasks = {t["id"]: t for t in info["tasks"]}
    assert tasks["test-video"]["installed"] and not tasks["wan22-t2v"]["installed"]
    assert tasks["wan22-t2v"]["gpu"] == "H100" and tasks["wan22-t2v"]["pricePerSecond"] > 0
    assert tasks["wan22-i2v"]["inputs"] == ["image"]


def test_video_job_streams_status_progress_previews_and_result(client):
    r = client.post("/v1/jobs", json={"task": "test-video", "params": {"steps": 5, "length": 8}}, headers=AUTH)
    assert r.status_code == 201, r.text
    job = r.json()
    evs = events(client, job["id"])
    kinds = [e.kind for e in evs]
    assert kinds[0] == Kind.STARTED and kinds[-1] == Kind.FINISHED
    assert [e.seq for e in evs] == sorted(e.seq for e in evs)
    assert [e.step for e in evs if e.kind == Kind.PROGRESS] == [1, 2, 3, 4, 5]
    assert any(e.kind == Kind.PREVIEW and e.payload[:2] == b"\xff\xd8" for e in evs)
    statuses = [e.payload.decode() for e in evs if e.kind == Kind.STATUS]
    assert "Queued" in statuses and "Sampling" in statuses and statuses[-1] == ""
    out = json.loads(next(e for e in evs if e.kind == Kind.OUTPUT_READY).payload)
    assert out["kind"] == "video" and out["bytes"] > 0
    seconds, gpu, cost = struct.unpack("<ddd", evs[-1].payload)
    assert seconds > 0 and gpu > 0 and cost == -1.0  # CPU task: no price

    res = client.get(f"/v1/jobs/{job['id']}/result", headers=AUTH)
    assert res.status_code == 200 and res.headers["content-type"] == "video/mp4"
    assert res.content[4:8] == b"ftyp" and len(res.content) == out["bytes"]
    part = client.get(f"/v1/jobs/{job['id']}/result", headers={**AUTH, "Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == res.content[10:20]
    assert part.headers["content-range"] == f"bytes 10-19/{len(res.content)}"
    tail = client.get(f"/v1/jobs/{job['id']}/result", headers={**AUTH, "Range": "bytes=-5"})
    assert tail.content == res.content[-5:]
    assert client.get(f"/v1/jobs/{job['id']}/result",
                      headers={**AUTH, "Range": f"bytes={len(res.content)}-"}).status_code == 416

    # Replay after a reconnect: only what came after `after`, and the job summary.
    later = events(client, job["id"], after=evs[-3].seq)
    assert [e.seq for e in later] == [e.seq for e in evs[-2:]]
    summary = client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()
    assert summary["state"] == "finished" and summary["result"]["kind"] == "video"


def test_image_job(client):
    job = client.post("/v1/jobs", json={"task": "test-image", "params": {"width": 64, "height": 48}},
                      headers=AUTH).json()
    events(client, job["id"])
    res = client.get(f"/v1/jobs/{job['id']}/result", headers=AUTH)
    assert res.headers["content-type"] == "image/png" and res.content[:4] == b"\x89PNG"


def test_failure_and_cancel(client):
    job = client.post("/v1/jobs", json={"task": "test-video", "params": {"steps": 6, "fail_at": 3}},
                      headers=AUTH).json()
    evs = events(client, job["id"])
    assert evs[-1].kind == Kind.RUN_FAILED and b"step 3" in evs[-1].payload

    slow = client.post("/v1/jobs", json={"task": "test-video", "params": {"steps": 50, "step_seconds": 0.1}},
                       headers=AUTH).json()
    time.sleep(0.3)
    assert client.post(f"/v1/jobs/{slow['id']}/cancel", headers=AUTH).json()["state"] == "cancelled"
    evs = events(client, slow["id"])
    assert evs[-1].kind == Kind.CANCELLED
    assert client.get(f"/v1/jobs/{slow['id']}/result", headers=AUTH).status_code == 404


def test_validation(client):
    def post(body):
        return client.post("/v1/jobs", json=body, headers=AUTH)

    assert post({"task": "nope"}).status_code == 400
    assert "at most" in post({"task": "test-video", "params": {"steps": 1000}}).json()["error"]
    assert "unknown parameter" in post({"task": "test-video", "params": {"bogus": 1}}).json()["error"]
    # GPU tasks need their models; the local CPU-only runner has none.
    r = post({"task": "wan22-i2v", "inputs": {"image": "a" * 64 + ".png"}})
    assert r.status_code == 404  # the input isn't uploaded
    r = post({"task": "flux1-dev", "params": {"prompt": "x"}})
    assert r.status_code == 503 and "download" in r.json()["error"]
    # Sizes snap to multiples of 16.
    job = post({"task": "test-image", "params": {"width": 70, "height": 33}}).json()
    assert (job["params"]["width"], job["params"]["height"]) == (64, 32)


def test_uploads_are_verified_resumable_and_deduplicated(client):
    data = bytes(range(256)) * 400
    name = hashlib.sha256(data).hexdigest() + ".png"
    head = client.head(f"/v1/files/{name}", headers=AUTH)
    assert head.status_code == 404 and head.headers["x-upload-offset"] == "0"
    # First half, as if the connection dropped after it.
    half = len(data) // 2
    r = client.put(f"/v1/files/{name}", content=data[:half], headers={**AUTH,
                   "Content-Range": f"bytes 0-{half - 1}/{len(data)}"})
    assert r.status_code == 202
    assert client.head(f"/v1/files/{name}", headers=AUTH).headers["x-upload-offset"] == str(half)
    # Wrong offset → 409 with where to continue.
    r = client.put(f"/v1/files/{name}", content=data[10:], headers={**AUTH,
                   "Content-Range": f"bytes 10-{len(data) - 1}/{len(data)}"})
    assert r.status_code == 409 and r.headers["x-upload-offset"] == str(half)
    r = client.put(f"/v1/files/{name}", content=data[half:], headers={**AUTH,
                   "Content-Range": f"bytes {half}-{len(data) - 1}/{len(data)}"})
    assert r.status_code == 201
    assert client.head(f"/v1/files/{name}", headers=AUTH).status_code == 200
    # Content that doesn't match the name is refused.
    bad = "b" * 64 + ".png"
    assert client.put(f"/v1/files/{bad}", content=b"hello", headers=AUTH).status_code == 400
    assert client.head("/v1/files/../etc", headers=AUTH).status_code in (400, 404)


def _wav_info(data: bytes):
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def test_audio_tasks_are_listed_with_choices_and_optional_inputs(client):
    tasks = {t["id"]: t for t in client.get("/v1/info", headers=AUTH).json()["tasks"]}
    for task in ("ace-step-xl", "stable-audio-3", "qwen3-tts"):
        assert tasks[task]["kind"] == "audio" and tasks[task]["gpu"] == "L40S" and not tasks[task]["installed"]
    speech = tasks["qwen3-tts"]
    assert speech["inputs"] == [] and speech["optionalInputs"] == ["reference"]
    assert "Ryan" in speech["params"]["voice"]["choices"] and "designed" in speech["params"]["voice"]["choices"]
    assert "C# minor" in tasks["ace-step-xl"]["params"]["key"]["choices"]
    # Choices are checked (case-insensitively) at submit.
    r = client.post("/v1/jobs", json={"task": "qwen3-tts", "params": {"voice": "nobody"}}, headers=AUTH)
    assert r.status_code == 400 and "voice must be one of" in r.text


def test_audio_job_returns_a_wav(client):
    params = {"seconds": 1.5, "sample_rate": 22050, "channels": 2, "steps": 3}
    job = client.post("/v1/jobs", json={"task": "test-audio", "params": params}, headers=AUTH).json()
    evs = events(client, job["id"])
    kinds = [e.kind for e in evs]
    assert kinds[-1] == Kind.FINISHED and Kind.PREVIEW not in kinds
    assert [e.step for e in evs if e.kind == Kind.PROGRESS] == [1, 2, 3]
    out = json.loads(next(e.payload for e in evs if e.kind == Kind.OUTPUT_READY))
    assert out["kind"] == "audio"
    r = client.get(f"/v1/jobs/{job['id']}/result", headers=AUTH)
    assert r.headers["content-type"] == "audio/wav"
    assert _wav_info(r.content) == (22050, 2, round(1.5 * 22050))


def test_optional_audio_input_is_uploaded_and_used(client):
    from mlxcomfy_gpu.media import tone, write_wav

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.wav"
        write_wav(tone(0.5, 24000, 1, 3), 24000, ref)
        data = ref.read_bytes()
    name = hashlib.sha256(data).hexdigest() + ".wav"
    assert client.put(f"/v1/files/{name}", content=data, headers={**AUTH, "Upload-Offset": "0"}).status_code in (200, 201)
    job = client.post("/v1/jobs", json={"task": "test-audio", "params": {"seconds": 1.0},
                                        "inputs": {"reference": name}}, headers=AUTH).json()
    events(client, job["id"])
    rate, channels, frames = _wav_info(client.get(f"/v1/jobs/{job['id']}/result", headers=AUTH).content)
    assert (rate, channels, frames) == (24000, 1, 12000 + 24000)  # the reference clip, then the tone
    # An input the task doesn't take is refused.
    r = client.post("/v1/jobs", json={"task": "test-video", "inputs": {"reference": name}}, headers=AUTH)
    assert r.status_code == 400


def test_fake_models_give_each_audio_model_its_format(tmp_path, monkeypatch):
    from mlxcomfy_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("MLXCOMFY_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): raise AssertionError("audio has no image previews")
        def check(self): pass

    expected = {"ace-step-xl": (48000, 2, 10 * 48000), "stable-audio-3": (44100, 2, 3 * 44100),
                "qwen3-tts": (24000, 1, None)}
    for task, (rate, channels, frames) in expected.items():
        spec = TASKS[task]
        params = spec.normalize({"seconds": 10} if task == "ace-step-xl" else
                                {"seconds": 3} if task == "stable-audio-3" else {"text": "Hello there, world."})
        out = make_worker(spec.family).run(spec, params, {}, Sink(), tmp_path)
        assert out.kind == "audio" and out.content_type == "audio/wav"
        got = _wav_info(out.path.read_bytes())
        assert got[:2] == (rate, channels) and (frames is None or got[2] == frames), (task, got)


def test_file_store_prunes_the_least_recently_used_past_its_cap(tmp_path):
    import asyncio
    import os

    from mlxcomfy_gpu.files import FileStore

    stored = []
    store = FileStore(tmp_path, max_bytes=250, on_stored=lambda: stored.append(1))

    async def body(data):
        yield data

    names = []
    for i, data in enumerate([b"a" * 100, b"b" * 100]):
        name = hashlib.sha256(data).hexdigest() + ".lora"
        assert asyncio.run(store.receive(name, 0, len(data), body(data)))
        os.utime(store.path(name), (1000 + i, 1000 + i))
        names.append(name)
    store.touch(names[0])  # used by a job just now: the second is now the least recently used
    stale = tmp_path / ("c" * 64 + ".partial")
    stale.write_bytes(b"x")
    os.utime(stale, (1, 1))
    data = b"d" * 100
    third = hashlib.sha256(data).hexdigest()
    assert asyncio.run(store.receive(third, 0, len(data), body(data)))
    assert store.exists(names[0]) and not store.exists(names[1]) and store.exists(third)
    assert not stale.exists() and stored == [1, 1, 1]
