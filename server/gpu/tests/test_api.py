"""The job API end to end with the local runner and the CPU test tasks (no torch, no models)."""

import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aistudio_gpu.api import create_app
from aistudio_gpu.files import FileStore
from aistudio_gpu.jobs import JobManager
from aistudio_gpu.local import LocalRunner
from aistudio_gpu.wire import Kind, decode_all

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
    assert info["parallelJobs"] == 1  # one GPU thread here (Modal reports its containers per model)
    tasks = {t["id"]: t for t in info["tasks"]}
    assert tasks["test-video"]["installed"] and not tasks["wan22-t2v"]["installed"]
    assert tasks["test-video"]["activeJobs"] == 0 and tasks["wan22-i2v"]["activeJobs"] == 0  # nothing queued or running
    assert tasks["wan22-t2v"]["gpu"] == "H100" and tasks["wan22-t2v"]["pricePerSecond"] > 0
    assert tasks["wan22-i2v"]["inputs"] == ["image"] and tasks["wan22-i2v"]["optionalInputs"][0] == "last_image"
    assert tasks["qwen-image-edit"]["inputs"] == ["image1"] and tasks["qwen-image-edit"]["gpu"] == "H100"
    assert tasks["qwen-image-edit"]["optionalInputs"][:2] == ["image2", "image3"]
    assert tasks["wan22-s2v"]["inputs"] == ["image", "audio"] and tasks["wan22-s2v"]["gpu"] == "H100"
    assert tasks["wan22-s2v"]["params"]["max_seconds"]["default"] == 15.0
    assert tasks["longcat-avatar"]["inputs"] == ["image", "audio"] and tasks["longcat-avatar"]["gpu"] == "H100"
    assert tasks["longcat-avatar"]["optionalInputs"] == ["audio2"]
    assert tasks["longcat-avatar"]["params"]["resolution"]["choices"] == ["480p", "720p"]
    assert tasks["infinitetalk"]["inputs"] == ["video", "audio"] and tasks["infinitetalk"]["gpu"] == "H100"
    assert not tasks["longcat-avatar"]["installed"] and not tasks["infinitetalk"]["installed"]
    animate = tasks["wan-animate2"]
    assert animate["inputs"] == ["image", "video"] and animate["gpu"] == "H200" and animate["fallbackGpu"] == "B200"
    assert animate["params"]["steps"]["default"] == 10 and animate["params"]["keep_audio"]["default"] is True
    assert not animate["installed"]


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
    for task in ("ace-step-xl", "stable-audio-3", "qwen3-tts", "breeze-tts"):
        assert tasks[task]["kind"] == "audio" and tasks[task]["gpu"] == "L40S" and not tasks[task]["installed"]
    breeze = tasks["breeze-tts"]
    assert "non-commercial" in breeze["title"] and breeze["optionalInputs"] == ["reference"]
    assert set(breeze["params"]) == {"text", "instruction", "reference_text", "cfg_scale", "seed"}
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
    from aistudio_gpu.media import tone, write_wav

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
    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): raise AssertionError("audio has no image previews")
        def check(self): pass

    expected = {"ace-step-xl": (48000, 2, 10 * 48000), "stable-audio-3": (44100, 2, 3 * 44100),
                "qwen3-tts": (24000, 1, None), "breeze-tts": (24000, 1, None)}
    for task, (rate, channels, frames) in expected.items():
        spec = TASKS[task]
        params = spec.normalize({"seconds": 10} if task == "ace-step-xl" else
                                {"seconds": 3} if task == "stable-audio-3" else {"text": "Hello there, world."})
        out = make_worker(spec.family).run(spec, params, {}, Sink(), tmp_path)
        assert out.kind == "audio" and out.content_type == "audio/wav"
        got = _wav_info(out.path.read_bytes())
        assert got[:2] == (rate, channels) and (frames is None or got[2] == frames), (task, got)


def test_fake_models_edit_returns_picture_one_tinted(tmp_path, monkeypatch):
    """The movie studio's keyframes against the fake server: picture 1 at the asked size, tinted red, its bottom row
    the number of pictures / 4."""
    import numpy as np
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    spec = TASKS["qwen-image-edit"]
    pics = {}
    for i, color in enumerate([(0, 0, 255), (0, 255, 0)], 1):
        pics[f"image{i}"] = tmp_path / f"p{i}.png"
        Image.new("RGB", (40, 20), color).save(pics[f"image{i}"])
    out = make_worker(spec.family).run(spec, spec.normalize({"width": 80, "height": 48}), pics, Sink(), tmp_path)
    image = np.asarray(Image.open(out.path)).astype(float) / 255
    assert image.shape == (48, 80, 3)
    assert abs(image[10, 10, 0] - 0.5) < 0.02 and abs(image[10, 10, 2] - 0.25) < 0.02  # blue, tinted
    assert abs(image[-1, 10, 0] - 0.5) < 0.02  # two pictures
    out = make_worker(spec.family).run(spec, spec.normalize({}), {"image1": pics["image1"]}, Sink(), tmp_path)
    assert Image.open(out.path).size == (40, 20)  # 0 × 0: picture 1's own size


def test_fake_models_sdxl_accepts_a_face_picture(tmp_path, monkeypatch):
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    face = tmp_path / "face.png"
    Image.new("RGB", (16, 16), (200, 150, 120)).save(face)
    spec = TASKS["sdxl"]
    out = make_worker(spec.family).run(spec, spec.normalize({"width": 512, "height": 512}), {"face": face}, Sink(),
                                       tmp_path)
    assert Image.open(out.path).size == (512, 512)


def test_sd15_is_removed():
    # SD 1.5 was removed on 2026-11-27: no task, no container, no download, no fine-tune family.
    from aistudio_gpu import download, variants
    from aistudio_gpu.tasks import CONTAINERS, TASKS
    assert "sd15" not in TASKS and "sd15" not in CONTAINERS and "sd15" not in download.MODELS
    assert "sd15" not in download.expand(["sd"]) and "sd1.5" not in variants.FAMILIES


def test_ltx23_task_wiring(tmp_path, monkeypatch):
    from aistudio_gpu import download, variants
    from aistudio_gpu.tasks import CONTAINERS, GPU_PRICES, TASKS, container_price, make_worker

    spec = TASKS["ltx23-video"]
    assert spec.family == "ltx23" and spec.kind == "video" and spec.models == ("ltx23",) and spec.gpu == "B200"
    assert CONTAINERS["ltx23"] == (131072, 196608, 8) and spec.fallback_gpus == ()  # resident: no smaller card
    assert spec.optional_inputs == ("image", "lora1", "lora2", "lora3", "lora4") and spec.inputs == ()
    p = spec.normalize({})
    assert (p["seconds"], p["width"], p["height"], p["fast"], p["steps"], p["guidance"], p["model"], p["negative"]) == (
        5.0, 768, 512, True, 0, 3.0, "base", "")
    assert spec.normalize({"width": 770})["width"] == 768 and "lora4_strength" in p
    with pytest.raises(ValueError, match="at most 10"):
        spec.normalize({"seconds": 11})
    with pytest.raises(ValueError, match="at least 1"):
        spec.normalize({"guidance": 0.5})
    d = spec.describe(installed=True)
    assert "fallbackGpu" not in d and d["pricePerSecond"] == GPU_PRICES["B200"] + container_price("ltx23")
    assert variants.FAMILIES["ltx-2.3"] == "ltx23" and variants.TASK_FAMILIES["ltx23"] == "ltx-2.3"
    assert variants.components("ltx-2.3") == ("transformer",)
    models = download.MODELS
    allow = models["ltx23"]["allow"]
    assert models["ltx23"]["repo"] == "diffusers/LTX-2.3-Diffusers" and "connectors/diffusion_pytorch_model.safetensors" not in allow
    assert "connectors/diffusion_pytorch_model-*.safetensors" in allow and "text_encoder/*" in allow
    assert models["ltx23-distilled"]["allow"] == ["transformer/*"]
    assert download.expand(["ltx23"]) == ["ltx23", "ltx23-distilled"]
    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    out = make_worker(spec.family).run(spec, spec.normalize({"width": 256, "height": 256}), {}, Sink(), tmp_path)
    assert out.kind == "video" and out.path.exists()


def test_anima_task_wiring(tmp_path, monkeypatch):
    from aistudio_gpu import download, variants
    from aistudio_gpu.tasks import CONTAINERS, TASKS, make_worker

    spec = TASKS["anima"]
    assert spec.family == "anima" and spec.models == ("anima",) and spec.gpu == "L40S" and "anima" in CONTAINERS
    p = spec.normalize({})
    assert (p["width"], p["height"], p["steps"], p["guidance"], p["model"]) == (1024, 1024, 30, 4.0, "base")
    assert p["negative"].startswith("worst quality, low quality") and "lora4" in spec.optional_inputs
    assert variants.FAMILIES["anima"] == "anima" and variants.TASK_FAMILIES["anima"] == "anima"
    assert variants.components("anima") == ("transformer", "text_conditioner")
    assert variants.SINGLE_FILE["anima"] == "Anima" and variants.SINGLE_FILE_CONVERTERS["anima"].__name__ == "convert_anima"
    assert download.MODELS["anima"]["revision"] == "073c3a9db359c31ad0e8aa268d15775473c2176c"
    assert download.expand(["anima"]) == ["anima"] and "anima" not in download.expand(["sd"])
    assert make_worker.__name__ == "make_worker"
    monkeypatch.setattr("aistudio_gpu.tasks.fake_models", lambda: False)
    worker = make_worker("anima")
    assert (worker.family, worker.title, worker.components) == ("anima", "Anima", ("transformer",))
    assert worker.variant_components == ("transformer", "text_conditioner")


def test_fake_models_chroma_accepts_a_face_picture(tmp_path, monkeypatch):
    """Character photos on Chroma against the fake server: a face picture is accepted and the size kept."""
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    face = tmp_path / "face.png"
    Image.new("RGB", (16, 16), (200, 150, 120)).save(face)
    spec = TASKS["chroma"]
    out = make_worker(spec.family).run(spec, spec.normalize({"width": 512, "height": 768}), {"face": face}, Sink(),
                                       tmp_path)
    assert Image.open(out.path).size == (512, 768)


def test_fake_models_image_to_video_starts_and_ends_on_its_pictures(tmp_path, monkeypatch):
    """The movie studio's shots against the fake server: the first frame is the start picture, the last frame the
    end picture (``last_image``) when there is one."""
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    spec = TASKS["wan22-i2v"]
    start, end = tmp_path / "start.png", tmp_path / "end.png"
    Image.new("RGB", (64, 48), (0, 0, 255)).save(start)
    Image.new("RGB", (64, 48), (0, 255, 0)).save(end)
    # Past the parameter limits: a tiny clip is quicker.
    params = {**spec.normalize({}), "width": 64, "height": 48, "length": 9}
    out = make_worker(spec.family).run(spec, params, {"image": start, "last_image": end}, Sink(), tmp_path)
    reader = imageio_ffmpeg.read_frames(str(out.path))
    meta = next(reader)
    frames = [np.frombuffer(f, np.uint8).reshape(meta["size"][1], meta["size"][0], 3) for f in reader]
    assert len(frames) == 9
    assert frames[0][24, 32, 2] > 200 and frames[-1][24, 32, 1] > 200  # blue first, green last


def _s2v_run(tmp_path, monkeypatch, seconds, **params):
    """The fake speech-to-video worker with a blue picture and a tone of the given length."""
    from PIL import Image

    from aistudio_gpu.media import tone, write_wav
    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")

    class Sink:
        def status(self, t): pass
        def progress(self, s, t): pass
        def preview(self, *a): pass
        def check(self): pass

    spec = TASKS["wan22-s2v"]
    pic, wav = tmp_path / "face.png", tmp_path / "speech.wav"
    Image.new("RGB", (64, 64), (0, 0, 255)).save(pic)
    write_wav(tone(seconds, 16000, 1, 3), 16000, wav)
    # Past the size limits: a tiny clip is quicker.
    p = {**spec.normalize({}), "width": 64, "height": 64, **params}
    return make_worker(spec.family).run(spec, p, {"image": pic, "audio": wav}, Sink(), tmp_path)


def test_fake_models_speech_to_video_follows_the_audio(tmp_path, monkeypatch):
    """A talking shot against the fake server: 16 frames per second of audio, the first one the picture, and the
    audio in the MP4."""
    import imageio_ffmpeg
    import numpy as np

    out = _s2v_run(tmp_path, monkeypatch, 1.5)
    assert out.kind == "video" and out.content_type == "video/mp4"
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(out.path))
    assert frames == 24  # ceil(1.5 * 16)
    assert abs(secs - 1.5) < 0.1
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(out.path)], capture_output=True, text=True)
    assert "Audio:" in info.stderr and "aac" in info.stderr and "Video: h264" in info.stderr
    reader = imageio_ffmpeg.read_frames(str(out.path))
    meta = next(reader)
    first = np.frombuffer(next(reader), np.uint8).reshape(meta["size"][1], meta["size"][0], 3)
    assert first[32, 32, 2] > 200 and first[32, 32, 0] < 60  # the blue picture
    assert out.extra["frames"] == 24 and out.extra["audioSeconds"] == 1.5


def test_fake_models_speech_to_video_refuses_audio_past_max_seconds(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"the audio is 2\.5 s; at most 2 s per clip"):
        _s2v_run(tmp_path, monkeypatch, 2.5, max_seconds=2.0)


class _QuietSink:
    def status(self, t): pass
    def progress(self, s, t): pass
    def preview(self, *a): pass
    def check(self): pass


def _frames(path):
    import imageio_ffmpeg
    import numpy as np

    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    return meta, [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader]


def test_fake_models_longcat_avatar_follows_the_longer_speaker_at_25_fps(tmp_path, monkeypatch):
    """Two speakers: the clip is as long as the longer voice, at LongCat's 25 fps, starting on the picture."""
    from PIL import Image

    from aistudio_gpu.media import tone, write_wav
    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    spec = TASKS["longcat-avatar"]
    pic, a, b = tmp_path / "two.png", tmp_path / "a.wav", tmp_path / "b.wav"
    Image.new("RGB", (64, 64), (0, 0, 255)).save(pic)
    write_wav(tone(0.8, 16000, 1, 1), 16000, a)
    write_wav(tone(1.2, 16000, 1, 2), 16000, b)
    out = make_worker(spec.family).run(spec, spec.normalize({}), {"image": pic, "audio": a, "audio2": b},
                                       _QuietSink(), tmp_path)
    meta, frames = _frames(out.path)
    assert len(frames) == 30 and meta["fps"] == 25  # ceil(1.2 * 25)
    assert frames[0][10, 10, 2] > 200 and frames[0][10, 10, 0] < 60  # the blue picture
    assert out.extra["speakers"] == 2 and out.extra["audioSeconds"] == 1.2


def test_fake_models_infinitetalk_dubs_the_clip_at_25_fps_for_the_speech(tmp_path, monkeypatch):
    """A dub: the uploaded 16 fps clip re-timed to 25 fps, as long as the speech (its last frame held), the speech
    muxed in; speech past max_seconds is refused."""
    import imageio_ffmpeg
    import numpy as np

    from aistudio_gpu.media import tone, write_mp4, write_wav
    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    spec = TASKS["infinitetalk"]
    clip, wav = tmp_path / "take.mp4", tmp_path / "line.wav"
    take = np.zeros((16, 48, 64, 3), np.float32)  # 1 s at 16 fps: red, then green from frame 8
    take[:8, :, :, 0] = 1.0
    take[8:, :, :, 1] = 1.0
    write_mp4(take, clip, fps=16)
    write_wav(tone(1.6, 16000, 1, 4), 16000, wav)
    out = make_worker(spec.family).run(spec, spec.normalize({}), {"video": clip, "audio": wav}, _QuietSink(),
                                       tmp_path)
    meta, frames = _frames(out.path)
    assert len(frames) == 40 and meta["fps"] == 25 and meta["size"] == (64, 48)  # ceil(1.6 * 25)
    assert frames[0][20, 20, 0] > 200 and frames[-1][20, 20, 1] > 200  # starts red, holds the green last frame
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(out.path)], capture_output=True, text=True)
    assert "Audio:" in info.stderr
    with pytest.raises(ValueError, match=r"the audio is 1\.6 s; at most 1 s per clip"):
        make_worker(spec.family).run(spec, spec.normalize({"max_seconds": 1.0}), {"video": clip, "audio": wav},
                                     _QuietSink(), tmp_path)


def test_lip_sync_downloads_pick_their_files():
    """Each part of the lip-sync models has an allow list or named files (no 54 GB DiT or quantised copies)."""
    from aistudio_gpu.download import MODELS, expand

    assert expand(["lipsync"]) == ["wan22-s2v", "longcat-avatar", "infinitetalk"]
    for name in ("longcat-avatar", "infinitetalk"):
        for repo, revision, into, pick in MODELS[name]["parts"]:
            assert len(revision) == 40 and pick
    base = dict((p[2], p[3]) for p in MODELS["longcat-avatar"]["parts"])["base"]
    assert "dit/*" not in base and "text_encoder/*" in base
    it = dict((p[2], p[3]) for p in MODELS["infinitetalk"]["parts"])
    assert it["infinitetalk"] == ["single/*"] and "lightx2v.safetensors" in it["lightx2v"]


def test_file_store_prunes_the_least_recently_used_past_its_cap(tmp_path):
    import asyncio
    import os

    from aistudio_gpu.files import FileStore

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


def test_two_uploads_of_the_same_bytes_at_once_both_succeed(tmp_path):
    # Two jobs sharing an input (both lines clone one voice sample) upload it together: the second must not write
    # over the first's partial file, and both see it stored.
    import asyncio
    import hashlib

    from aistudio_gpu.files import FileStore

    data = bytes(range(256)) * 4096  # 1 MB
    name = hashlib.sha256(data).hexdigest() + ".wav"
    store = FileStore(tmp_path)

    async def body():
        for i in range(0, len(data), 65536):
            await asyncio.sleep(0)  # yield between chunks, so the two uploads interleave without the lock
            yield data[i:i + 65536]

    async def both():
        return await asyncio.gather(store.receive(name, 0, len(data), body()), store.receive(name, 0, len(data), body()))

    assert asyncio.run(both()) == [True, True]
    assert store.path(name).read_bytes() == data and not store._receiving


def test_breeze_picks_its_mode_and_needs_a_transcript_to_clone(tmp_path, monkeypatch):
    """Breeze's mode comes from what it's given; guidance 4 with an instruction (unless set); a reference without its
    transcript is refused, by the fake worker too."""
    import pytest

    from aistudio_gpu.tasks import TASKS, make_worker
    from aistudio_gpu.tasks.breeze import request

    spec = TASKS["breeze-tts"]
    ref = {"reference": tmp_path / "voice.wav"}
    cases = [({"text": "Hi."}, {}, "plain", 1.0),
             ({"text": "Hi.", "instruction": "a calm older man"}, {}, "design", 4.0),
             ({"text": "Hi.", "reference_text": "I remember."}, ref, "clone", 1.0),
             ({"text": "Hi.", "reference_text": "I remember.", "instruction": "quietly"}, ref, "direction", 4.0),
             ({"text": "Hi.", "instruction": "quietly", "cfg_scale": 2.5}, {}, "design", 2.5)]
    for params, inputs, mode, cfg in cases:
        req, got_mode, got_cfg = request(spec.normalize(params), inputs)
        assert (got_mode, got_cfg) == (mode, cfg), params
        assert ("ref_audio_path" in req) == bool(inputs) and ("instruction" in req) == ("instruction" in params)
    with pytest.raises(ValueError, match="exact transcript"):
        request(spec.normalize({"text": "Hi."}), ref)
    with pytest.raises(ValueError, match="no text"):
        request(spec.normalize({"text": "  "}), {})

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    (tmp_path / "voice.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="exact transcript"):
        make_worker(spec.family).run(spec, spec.normalize({"text": "Hi."}), ref, _QuietSink(), tmp_path)


def test_breeze_download_skips_the_readme_assets():
    from aistudio_gpu.download import GROUPS, MODELS

    spec = MODELS["breeze-tts"]
    assert spec["revision"] == "3e28c5151381a722f1d8661b4118c298caa77aa4" and "assets/*" not in spec["allow"]
    assert "audio_tokenizer/*" in spec["allow"] and GROUPS["breeze-tts"] == ["breeze-tts"]
    assert "breeze-tts" not in GROUPS["audio"]


def test_resample_keeps_a_tone_and_its_length():
    import numpy as np
    from aistudio_gpu.media import resample
    t = np.arange(24000) / 24000
    x = np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 440 * t)]).astype(np.float32)
    y = resample(x, 24000, 32000)
    assert y.shape == (2, 32000) and y.dtype == np.float32
    u = np.arange(32000) / 32000
    assert np.abs(y[0, 100:-100] - np.sin(2 * np.pi * 440 * u[100:-100])).max() < 1e-3
    assert abs(np.abs(y[1]).max() - 0.5) < 1e-2
    assert resample(x, 24000, 24000) is x
    assert resample(x, 48000, 16000).shape == (2, 8000)


def test_prices_count_the_containers_memory_and_cpu_and_keep_h3_on_one_b200():
    from aistudio_gpu.tasks import CPU_PRICE, GPU_PRICES, MEMORY_PRICE, TASKS
    h3 = TASKS["h3-video"]
    container = 160 * MEMORY_PRICE + 8 * CPU_PRICE  # 160 GiB and 8 cores requested
    assert abs(h3.price_per_second - (GPU_PRICES["B200"] + container)) < 1e-12
    d = h3.describe(installed=True)
    assert "fallbackGpu" not in d  # everything resident on a B200: no smaller card that would swap
    sdxl = TASKS["sdxl"].describe(installed=True)  # 8 GiB, Modal's minimum 0.125 cores; no fallback
    assert abs(sdxl["pricePerSecond"] - (GPU_PRICES["L40S"] + 8 * MEMORY_PRICE + 0.125 * CPU_PRICE)) < 1e-12
    assert "fallbackGpu" not in sdxl
    assert TASKS["test-video"].price_per_second is None


def test_h3_offload_margin_never_keeps_both_big_models_on_the_gpu():
    from aistudio_gpu.tasks.minimax_h3 import offload_margin
    gib = 2**30
    transformer, conditioner, vae = 61.7 * gib, 62.1 * gib, 9.7 * gib
    for total in (139.8 * gib, 178.4 * gib):  # H200, B200
        margin = offload_margin(int(total))
        # the transformer comes on with the conditioner resident only if both fit under the margin: they must not
        assert total - conditioner - vae - margin < transformer
        # the transformer and the VAE do fit together
        assert total - vae - margin >= transformer
    assert offload_margin(40 * gib) == 12 * gib


def test_h3_snaps_frames_sizes_the_canvas_and_folds_lora_alphas():
    import numpy as np
    import pytest

    from aistudio_gpu.tasks import TASKS
    from aistudio_gpu.tasks.minimax_h3 import canvas, num_frames, prepare_lora, target_size

    # 24 fps, 17·n + 5 frames, within 5–15 s.
    assert [num_frames(s) for s in (5, 5.2, 10, 15)] == [124, 141, 243, 345]
    assert all(n % 17 == 5 and 5 <= n / 24 <= 15 for n in (num_frames(s / 4) for s in range(20, 61)))
    # The checkpoint's canvases; 544p is the faster draft size.
    assert canvas(16, 9) == (768, 1344) and canvas(9, 16) == (1344, 768) and canvas(1, 1) == (768, 768)
    assert canvas(16, 9, 544) == (544, 960)
    spec = TASKS["h3-video"]
    assert target_size(spec.normalize({"prompt": "x"}), (1344, 768)) == (768, 1344)  # auto: the picture's
    assert target_size(spec.normalize({"prompt": "x"}), None) == (768, 1344)  # auto, no picture: 16:9
    assert target_size(spec.normalize({"prompt": "x", "aspect": "9:16", "size": "544p"}), None) == (960, 544)
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        canvas(5, 1)

    # LightX2V's FL2VA file: peft names, a per-layer alpha (and a metadata one, which loses); rank 4 here.
    a, b = np.ones((4, 8), np.float32), np.ones((8, 4), np.float32)
    sd = {"blocks.0.to_q.lora_A.default.weight": a, "blocks.0.to_q.lora_B.default.weight": b,
          "blocks.0.to_q.alpha": np.float32(8.0),
          "norm_out.linear.lora_A.default.weight": a, "norm_out.linear.lora_B.default.weight": b}
    out = prepare_lora(sd, {"alpha": "2"})
    assert set(out) == {"transformer.blocks.0.to_q.lora_A.weight", "transformer.blocks.0.to_q.lora_B.weight",
                        "transformer.norm_out.linear.lora_A.weight", "transformer.norm_out.linear.lora_B.weight"}
    assert out["transformer.blocks.0.to_q.lora_B.weight"].max() == 2.0  # its own alpha 8 / rank 4
    assert out["transformer.norm_out.linear.lora_B.weight"].max() == 0.5  # the metadata's 2 / rank 4
    assert out["transformer.blocks.0.to_q.lora_A.weight"].max() == 1.0  # A untouched
    # The Ref2VA file: metadata alpha only; without any alpha, scale 1.
    assert prepare_lora({"x.lora_A.weight": a, "x.lora_B.weight": b}, {"alpha": "8"})["transformer.x.lora_B.weight"].max() == 2.0
    assert prepare_lora({"x.lora_A.weight": a, "x.lora_B.weight": b}, None)["transformer.x.lora_B.weight"].max() == 1.0


def test_fake_h3_makes_24fps_video_with_stereo_sound_on_its_canvas(tmp_path, monkeypatch):
    import imageio_ffmpeg
    import numpy as np
    import pytest
    from PIL import Image

    from aistudio_gpu.media import write_wav
    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    info = {t: TASKS[t] for t in ("h3-video", "h3-references")}
    assert info["h3-video"].gpu == "B200"
    assert info["h3-video"].optional_inputs == ("image", "last_image", "lora1", "lora2", "lora3", "lora4")
    assert "lora4" in info["h3-references"].optional_inputs and "lora1_strength" in info["h3-references"].params
    assert info["h3-references"].inputs == ("image1",) and "audio3" in info["h3-references"].optional_inputs
    assert "US/EU/UK/Korea" in info["h3-video"].title

    start = tmp_path / "start.png"
    Image.new("RGB", (1344, 768), (0, 0, 255)).save(start)
    spec = info["h3-video"]
    out_dir = tmp_path / "v"
    out_dir.mkdir()
    out = make_worker(spec.family).run(spec, spec.normalize({"prompt": "a lighthouse", "seconds": 5, "size": "544p"}),
                                       {"image": start}, _QuietSink(), out_dir)
    assert out.extra["frames"] == 124 and (out.extra["width"], out.extra["height"]) == (960, 544)
    reader = imageio_ffmpeg.read_frames(str(out.path))
    meta = reader.__next__()
    assert meta["fps"] == 24 and tuple(meta["size"]) == (960, 544)
    first = np.frombuffer(next(reader), np.uint8).reshape(544, 960, 3)
    assert first[..., 2].mean() > 200 and first[..., 0].mean() < 60  # it starts on the start picture
    import subprocess
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(out.path)], capture_output=True, text=True)
    assert "32000 Hz, stereo" in probe.stderr

    # References: picture 1 is required; an audio reference has to be 2–15 s.
    spec = info["h3-references"]
    voice = tmp_path / "voice.wav"
    write_wav(np.zeros((1, 16000), np.float32), 16000, voice)  # 1 s
    with pytest.raises(ValueError, match="2–15 s"):
        make_worker(spec.family).run(spec, spec.normalize({"prompt": "she speaks"}),
                                     {"image1": start, "audio1": voice}, _QuietSink(), tmp_path)
    with pytest.raises(ValueError, match="prompt"):
        make_worker(spec.family).run(spec, spec.normalize({"prompt": " "}), {"image1": start}, _QuietSink(), tmp_path)


def test_h3_download_takes_the_diffusers_layout_and_both_turbo_loras():
    from aistudio_gpu.download import GROUPS, MODELS

    (repo, revision, into, allow), (lora_repo, _, lora_into, files) = MODELS["minimax-h3"]["parts"]
    assert revision == "0919fb684102ef115126612f3fbb84d1027e7317" and into == "."
    assert "transformer/*" in allow and "transformer_ref/*" in allow and "modular_model_index.json" in allow
    assert not any(p.startswith(("FL2VA", "Ref2VA")) for p in allow)
    assert lora_into == "turbo" and set(files) == {"fl2v.safetensors", "ref2v.safetensors"}
    assert "8step_v1.2_768p" in files["fl2v.safetensors"] and "ref2v_turbo_8step_v1.1_768p" in files["ref2v.safetensors"]
    assert GROUPS["minimax-h3"] == ["minimax-h3", "minimax-h3-pruned-affine"]  # + the pruned affine maps (KB)


def test_h3_tasks_take_a_fine_tune_from_their_partitions_library_family():
    from aistudio_gpu import variants
    from aistudio_gpu.tasks import TASKS

    for task in ("h3-video", "h3-references"):
        p = TASKS[task].normalize({"prompt": "a lighthouse"})
        assert p["model"] == "base"  # the full layout stays the default
        assert TASKS[task].normalize({"prompt": "x", "model": "eros"})["model"] == "eros"
    assert variants.TASK_FAMILIES["h3-video"] == "minimax-h3"
    assert variants.TASK_FAMILIES["h3-references"] == "minimax-h3-ref"
    assert variants.components("minimax-h3") == ("transformer",)
    assert variants.components("minimax-h3-ref") == ("transformer_ref",)
    assert variants.FAMILIES["minimax-h3"] == variants.FAMILIES["minimax-h3-ref"] == "minimax-h3"


def test_h3_image_spec_download_and_fake_run(tmp_path, monkeypatch):
    import pytest
    from PIL import Image

    from aistudio_gpu.download import GROUPS, MODELS
    from aistudio_gpu.tasks import CONTAINERS, TASKS, make_worker
    from aistudio_gpu.tasks.minimax_h3_image import output_size

    spec = TASKS["h3-image"]
    assert spec.kind == "image" and spec.gpu == "B200" and "US/EU/UK/Korea" in spec.title
    assert spec.fallback_gpus == ()  # no smaller card by default: there the models would swap for every picture
    from aistudio_gpu.tasks.minimax_h3_image import fits_resident
    gib = 2**30
    assert fits_resident(125 * gib, 178 * gib)  # a B200 holds transformer + text encoder + VAE
    assert not fits_resident(125 * gib, 131 * gib)  # an H200 doesn't (with room to run)
    assert spec.models == ("minimax-h3-image", "minimax-h3")  # the text encoder is H3's
    assert spec.optional_inputs[:4] == ("image1", "image2", "image3", "image4") and not spec.inputs
    assert {"lora1", "lora4"} <= set(spec.optional_inputs) and spec.normalize({"prompt": "x"})["lora1_strength"] == 1.0
    assert {"steps", "cfg", "guidance", "negative"}.isdisjoint(spec.params)  # six fixed steps, guidance baked in
    with pytest.raises(ValueError, match="unknown parameter"):
        spec.normalize({"prompt": "x", "negative": "blurry"})
    p = spec.normalize({"prompt": "a lighthouse", "width": 832, "height": 1216})
    assert output_size(p) == (832, 1216) and p["canvas_mp"] == 4.0
    assert output_size(spec.normalize({"prompt": "x", "width": 1000, "height": 1000})) == (992, 992)
    assert CONTAINERS["h3-image"][0] >= 163840  # ≈ 130 GB of weights in host RAM

    entry = MODELS["minimax-h3-image"]
    assert entry["revision"] == "8ced60068623def5b74b4d10558117cb61d6482d"
    assert "transformer_diffusers/*" in entry["allow"] and "vae_diffusers/*" in entry["allow"]
    assert "transformer/*" not in entry["allow"] and not any(a.startswith("text_encoder") for a in entry["allow"])
    assert GROUPS["minimax-h3-image"] == ["minimax-h3-image"]

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    worker = make_worker(spec.family)
    out = worker.run(spec, p, {}, _QuietSink(), tmp_path)
    assert Image.open(out.path).size == (832, 1216)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 512), (0, 0, 255)).save(ref)
    out = worker.run(spec, p, {"image1": ref, "image2": ref}, _QuietSink(), tmp_path)
    assert Image.open(out.path).size == (832, 1216)
    with pytest.raises(ValueError, match="in order"):
        worker.run(spec, p, {"image2": ref}, _QuietSink(), tmp_path)
    with pytest.raises(ValueError, match="prompt"):
        worker.run(spec, spec.normalize({"prompt": " "}), {}, _QuietSink(), tmp_path)



def test_big_models_stay_resident_on_their_gpu():
    """The user's rule: every part on the GPU at once (residency.py), with room left for a job's activations."""
    from aistudio_gpu.tasks import TASKS
    from aistudio_gpu.tasks import ltx23, minimax_h3, minimax_h3_image
    from aistudio_gpu.tasks.residency import fits, modules_of, weight_bytes

    gib = 2**30
    totals = {"B200": 178.4 * gib, "H200": 139.8 * gib, "H100": 79.2 * gib}
    h3 = (61.7 + 62.1 + 9.7) * gib  # one transformer partition, the conditioner, the VAEs
    assert TASKS["h3-video"].gpu == TASKS["h3-references"].gpu == "B200"
    assert fits(h3, totals["B200"], minimax_h3.RESIDENT_MARGIN_GIB)
    assert not fits(h3, totals["H200"], minimax_h3.RESIDENT_MARGIN_GIB)
    ltx = (49 + 38 + 38 + 3) * 1e9  # text encoder, base and distilled transformers (both kept), VAEs
    assert TASKS["ltx23-video"].gpu == "B200" and fits(ltx, totals["B200"], ltx23.RESIDENT_MARGIN_GIB)
    assert not fits(ltx, totals["H200"], ltx23.RESIDENT_MARGIN_GIB)
    # The image models whose fast mode is a second transformer keep both on an H100 (adapters.WeightPark).
    assert TASKS["chroma"].gpu == TASKS["sd35-large"].gpu == "H100" and TASKS["sdxl"].gpu == "L40S"
    from aistudio_gpu.tasks import chroma, sd35  # noqa: F401  (import without torch)
    assert chroma.ChromaWorker.parked_variants == ("base", "@flash")
    assert ltx23.LTX23Worker.parked_variants == ("base", "@distilled")
    image = (66.3 + 66.7 + 2.3) * 1e9
    assert TASKS["h3-image"].gpu == "B200" and minimax_h3_image.fits_resident(int(image), int(totals["B200"]))

    class Fake:
        def __init__(self, n):
            import numpy as np
            self.p = [np.zeros(n, dtype=np.float32)]

        def parameters(self):
            return iter([_T(x) for x in self.p])

        def buffers(self):
            return iter([])

        def to(self, _):
            return self

    class _T:
        def __init__(self, a):
            self.a = a

        def numel(self):
            return self.a.size

        def element_size(self):
            return self.a.itemsize

    assert weight_bytes(Fake(10)) == 40 and weight_bytes("tokenizer") == 0
    assert set(modules_of({"transformer": Fake(1), "tokenizer": "tok", "scheduler": None})) == {"transformer"}


def test_h3_image_lora_keeps_only_layers_it_has():
    # A video LoRA's audio-branch tensors are dropped; the image transformer's layers stay.
    from aistudio_gpu.tasks.minimax_h3_image import keep_existing, lora_module

    assert lora_module("transformer.blocks.0.attn.to_q.lora_A.weight") == "blocks.0.attn.to_q"
    assert lora_module("transformer.blocks.0.attn.to_q.alpha") == "blocks.0.attn.to_q"
    assert lora_module("transformer.blocks.0.attn.to_q.weight") is None
    state = {"transformer.blocks.0.attn.to_q.lora_A.weight": 1, "transformer.blocks.0.attn.to_q.lora_B.weight": 2,
             "transformer.audio_blocks.0.to_q.lora_A.weight": 3, "transformer.audio_blocks.0.to_q.lora_B.weight": 4}
    kept, dropped = keep_existing(state, {"blocks.0.attn.to_q"})
    assert dropped == 2 and set(kept) == {"transformer.blocks.0.attn.to_q.lora_A.weight",
                                          "transformer.blocks.0.attn.to_q.lora_B.weight"}


def test_train_sdxl_lora_task_wiring(tmp_path, monkeypatch):
    # A training job takes one zip; the fake worker checks it as the real one does and returns a small file.
    import io
    import zipfile

    from PIL import Image

    from aistudio_gpu.tasks import CONTAINERS, TASKS, make_worker

    spec = TASKS["train-sdxl-lora"]
    assert spec.kind == "train" and spec.inputs == ("dataset",) and spec.models == ("sdxl",)
    assert "sdxl-train" in CONTAINERS and spec.normalize({})["steps"] == 0 and spec.normalize({})["rank"] == 16
    data = tmp_path / "d.zip"
    with zipfile.ZipFile(data, "w") as z:
        for i in range(4):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32)).save(buf, "PNG")
            z.writestr(f"{i}.png", buf.getvalue())
    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    out = make_worker(spec.family).run(spec, spec.normalize({}), {"dataset": data}, _QuietSink(), tmp_path)
    assert out.kind == "file" and out.extra["pictures"] == 4



def test_train_h3_image_lora_task_wiring():
    from aistudio_gpu.tasks import CONTAINERS, TASKS

    spec = TASKS["train-h3-image-lora"]
    assert spec.kind == "train" and spec.gpu == "B200" and spec.inputs == ("dataset",)
    assert spec.models == ("minimax-h3-image", "minimax-h3") and CONTAINERS["h3-image-train"] == CONTAINERS["h3-image"]
    p = spec.normalize({})
    assert p["rank"] == 16 and p["steps"] == 0 and p["resolution"] == 1024


def test_a_file_result_is_kept_in_the_uploads(tmp_path):
    # A trained LoRA (a job's "file" result) is stored like an upload under its SHA-256: the next job finds it.
    import hashlib

    from aistudio_gpu.jobs import keep_file

    store = FileStore(tmp_path / "uploads")
    result = tmp_path / "result.safetensors"
    result.write_bytes(b"lora bytes")
    name = keep_file(store, result)
    assert name == hashlib.sha256(b"lora bytes").hexdigest() + ".lora"
    assert store.path(name).read_bytes() == b"lora bytes" and result.exists()
    assert keep_file(store, result) == name  # again: already there
    assert not any(p.name.endswith(".keeping") for p in (tmp_path / "uploads").iterdir())
