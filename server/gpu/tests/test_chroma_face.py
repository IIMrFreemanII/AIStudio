"""The Chroma worker's optional face picture (XLabs' FLUX.1 IP-Adapter v2), against a stub pipeline: no models."""

import types

import pytest
from PIL import Image

from aistudio_gpu import download
from aistudio_gpu.tasks import TASKS
from aistudio_gpu.tasks.chroma import ChromaWorker


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

    def load_ip_adapter(self, *a, **kw): self.calls.append(("load_ip_adapter", a, kw))
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
    monkeypatch.setattr(transformers.CLIPImageProcessor, "from_pretrained", lambda *a, **kw: object())
    w = ChromaWorker()
    w.pipe = StubPipe()
    w.models_root = tmp_path / "models"
    w.device, w.dtype = "cpu", torch.float32
    monkeypatch.setattr(w, "_use_loras", lambda *a, **kw: 0)
    return w


def face_file(tmp_path):
    path = tmp_path / "face.png"
    Image.new("RGB", (32, 32), (200, 150, 120)).save(path)
    return path


def run(worker, tmp_path, inputs=None, **params):
    spec = TASKS["chroma"]
    return worker.run(spec, spec.normalize({"prompt": "a portrait", "steps": 2, **params}), inputs or {}, Sink(),
                      tmp_path)


def install_face(worker):
    root = worker.models_root / "chroma-face"
    root.mkdir(parents=True)
    (root / ".complete").write_text("{}")


def test_face_job_loads_the_adapter_once_and_passes_the_picture(worker, tmp_path):
    install_face(worker)
    face = face_file(tmp_path)
    out = run(worker, tmp_path, {"face": face}, face_strength=0.8)
    assert worker.pipe.names() == ["load_ip_adapter", "register_modules", "register_modules", "set_ip_adapter_scale"]
    _, args, kw = worker.pipe.calls[0]
    assert args[0].endswith("chroma-face/adapter") and kw["weight_name"] == "ip_adapter.safetensors"
    assert kw["image_encoder_pretrained_model_name_or_path"] is None
    assert worker.pipe.calls[3] == ("set_ip_adapter_scale", 0.8)
    assert worker.pipe.call_kwargs[0]["ip_adapter_image"].mode == "RGB"
    assert out.extra["face"] is True
    worker.pipe.calls.clear()
    run(worker, tmp_path, {"face": face})
    assert worker.pipe.names() == ["set_ip_adapter_scale"]  # not loaded again


def test_a_job_without_a_face_unloads_the_adapter(worker, tmp_path):
    install_face(worker)
    run(worker, tmp_path, {"face": face_file(tmp_path)})
    worker.pipe.calls.clear()
    out = run(worker, tmp_path)
    assert worker.pipe.names() == ["unload_ip_adapter"] and not worker.face_loaded
    assert "ip_adapter_image" not in worker.pipe.call_kwargs[-1] and out.extra["face"] is False


def test_a_transformer_change_unloads_the_face_first(worker, tmp_path, monkeypatch):
    install_face(worker)
    worker.has_fast = True
    worker.variant = "@flash"
    run(worker, tmp_path, {"face": face_file(tmp_path)})
    order = []
    monkeypatch.setattr(worker, "_use_variant", lambda vid, sink: order.append((vid, worker.face_loaded)))
    run(worker, tmp_path, {"face": face_file(tmp_path)}, fast=False)
    assert order[0] == ("base", False)


def test_the_negative_prompt_goes_only_with_guidance(worker, tmp_path):
    worker.has_fast = True
    worker.variant = "@flash"
    run(worker, tmp_path, negative="blurry")  # fast: guidance 1, no negative
    run(worker, tmp_path, negative="blurry", fast=False)
    assert "negative_prompt" not in worker.pipe.call_kwargs[0] and worker.pipe.call_kwargs[0]["guidance_scale"] == 1.0
    assert worker.pipe.call_kwargs[1]["negative_prompt"] == "blurry"
    assert worker.pipe.call_kwargs[1]["guidance_scale"] == 3.0


def test_a_face_without_the_model_says_what_to_download(worker, tmp_path):
    with pytest.raises(ValueError, match="chroma-face"):
        run(worker, tmp_path, {"face": face_file(tmp_path)})
    assert worker.pipe.calls == []


def test_spec_and_downloads():
    spec = TASKS["chroma"]
    assert "face" in spec.optional_inputs and "lora1" in spec.optional_inputs
    assert spec.normalize({})["face_strength"] == 0.6 and spec.normalize({})["fast"] is True
    assert spec.models == ("chroma",)
    assert download.expand(["chroma"]) == ["chroma", "chroma-flash", "chroma-face"]
    assert download.MODELS["chroma"]["revision"] and download.MODELS["chroma-flash"]["allow"] == ["transformer/*"]
