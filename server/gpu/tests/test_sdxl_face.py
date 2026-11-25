"""The SDXL worker's optional face picture (IP-Adapter), against a stub pipeline: no models."""

import types

import pytest
from PIL import Image

from aistudio_gpu import download
from aistudio_gpu.tasks import TASKS
from aistudio_gpu.tasks.sdxl import SDXLWorker


class Sink:
    def status(self, t): pass
    def progress(self, s, t): pass
    def preview(self, *a): pass
    def check(self): pass


class StubPipe:
    def __init__(self):
        self.calls = []
        self.call_kwargs = []
        self.feature_extractor = None

    def load_ip_adapter(self, *a, **kw): self.calls.append(("load_ip_adapter", kw))
    def register_modules(self, **kw): self.calls.append(("register_modules", sorted(kw)))
    def set_ip_adapter_scale(self, scale): self.calls.append(("set_ip_adapter_scale", scale))
    def unload_ip_adapter(self): self.calls.append(("unload_ip_adapter",))

    def __call__(self, **kw):
        self.call_kwargs.append(kw)
        return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    def names(self):
        return [c[0] for c in self.calls]


class StubEncoder:
    def to(self, device):
        return self


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    import torch
    import transformers

    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection, "from_pretrained",
                        lambda *a, **kw: StubEncoder())
    monkeypatch.setattr(transformers, "CLIPImageProcessor", lambda: object())
    w = SDXLWorker()
    w.pipe = StubPipe()
    w.schedulers = {"dpmpp-2m-karras": object(), "euler-a": object(), "euler": object()}
    w.models_root = tmp_path / "models"
    w.device, w.dtype = "cpu", torch.float32
    monkeypatch.setattr(w, "_use_loras", lambda *a, **kw: 0)
    return w


def face_file(tmp_path):
    path = tmp_path / "face.png"
    Image.new("RGB", (32, 32), (200, 150, 120)).save(path)
    return path


def run(worker, tmp_path, inputs=None, **params):
    spec = TASKS["sdxl"]
    return worker.run(spec, spec.normalize({"prompt": "a portrait", "steps": 2, **params}), inputs or {}, Sink(),
                      tmp_path)


def install_face(worker):
    root = worker.models_root / "sdxl-face"
    root.mkdir(parents=True)
    (root / ".complete").write_text("{}")


def test_face_job_loads_the_adapter_once_and_passes_the_picture(worker, tmp_path):
    install_face(worker)
    face = face_file(tmp_path)
    out = run(worker, tmp_path, {"face": face}, face_strength=0.8)
    assert worker.pipe.names() == ["load_ip_adapter", "register_modules", "register_modules", "set_ip_adapter_scale"]
    assert worker.pipe.calls[0][1]["weight_name"] == "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    assert worker.pipe.calls[3] == ("set_ip_adapter_scale", 0.8)
    assert worker.pipe.call_kwargs[0]["ip_adapter_image"].mode == "RGB"
    assert out.extra["face"] is True
    worker.pipe.calls.clear()
    run(worker, tmp_path, {"face": face})
    assert worker.pipe.names() == ["set_ip_adapter_scale"]  # not loaded again
    assert worker.pipe.calls[0] == ("set_ip_adapter_scale", 0.6)
    assert "ip_adapter_image" in worker.pipe.call_kwargs[1]


def test_a_job_without_a_face_unloads_the_adapter_and_passes_no_picture(worker, tmp_path):
    install_face(worker)
    run(worker, tmp_path, {"face": face_file(tmp_path)})
    worker.pipe.calls.clear()
    out = run(worker, tmp_path)
    assert worker.pipe.names() == ["unload_ip_adapter"] and not worker.face_loaded
    assert "ip_adapter_image" not in worker.pipe.call_kwargs[-1]
    assert out.extra["face"] is False
    worker.pipe.calls.clear()
    run(worker, tmp_path)  # nothing loaded: nothing to unload
    assert worker.pipe.calls == []


def test_a_model_change_unloads_the_face_first(worker, tmp_path, monkeypatch):
    install_face(worker)
    run(worker, tmp_path, {"face": face_file(tmp_path)})
    order = []
    monkeypatch.setattr(worker, "_use_variant", lambda vid, sink: order.append(("variant", worker.face_loaded)))
    run(worker, tmp_path, model="other")
    assert order == [("variant", False)]


def test_a_face_without_the_model_says_what_to_download(worker, tmp_path):
    with pytest.raises(ValueError, match="sdxl-face"):
        run(worker, tmp_path, {"face": face_file(tmp_path)})
    assert worker.pipe.calls == []


def test_spec_and_downloads():
    spec = TASKS["sdxl"]
    assert "face" in spec.optional_inputs
    assert spec.normalize({})["face_strength"] == 0.6
    assert spec.models == ("sdxl",)
    assert download.expand(["sdxl"]) == ["sdxl", "sdxl-face"]
    assert "sdxl-face" in download.expand(["sd"])
