"""The MiniMax H3 worker's community LoRAs, against a stub pipeline: no models."""

import pytest

from aistudio_gpu.tasks import TASKS
from aistudio_gpu.tasks.minimax_h3 import H3Worker


class Sink:
    def status(self, t): pass
    def progress(self, s, t): pass
    def preview(self, *a): pass
    def check(self): pass


class StubPipe:
    """Holds one partition, as a worker's pipeline does; ``load_lora_weights`` registers an fp32 LoRA factor unless
    the file is "foreign" (diffusers then only warns and loads nothing)."""

    def __init__(self, component: str):
        import torch

        self.component = component
        setattr(self, component, torch.nn.Module())
        self.adapters = ["turbo"]
        self.calls = []

    def load_lora_weights(self, state, adapter_name=None, **kw):
        import torch

        self.calls.append(("load", adapter_name, kw))
        if "foreign" in state:
            return
        module = getattr(self, self.component)
        setattr(module, f"x_{adapter_name}", torch.nn.Module())
        getattr(module, f"x_{adapter_name}").register_parameter(
            "lora_A", torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32)))
        self.adapters.append(adapter_name)

    def get_list_adapters(self):
        return {self.component: list(self.adapters)}

    def set_adapters(self, names, weights): self.calls.append(("set", list(names), list(weights)))
    def enable_lora(self): self.calls.append(("enable",))
    def disable_lora(self): self.calls.append(("disable",))

    def delete_adapters(self, name):
        self.adapters.remove(name)


def lora_file(tmp_path, name, foreign=False):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / f"{name}.lora"
    save_file({"foreign" if foreign else "blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(2, 2)}, str(path))
    return path


def worker(workflow, tmp_path):
    w = H3Worker(workflow)
    w.pipe = StubPipe("transformer_ref" if workflow == "ref2va" else "transformer")
    w.models_root = tmp_path
    return w


def params(task="h3-video", **kw):
    return TASKS[task].normalize({"prompt": "a lighthouse", **kw})


def test_h3_loras_stack_on_turbo_and_go_to_the_workers_partition(tmp_path):
    import torch

    w = worker("t2va", tmp_path)
    f = lora_file(tmp_path, "a" * 64)
    assert w._use_loras(params(lora1_strength=0.8), {"lora1": f}, Sink(), pinned={"turbo": 1.0}) == 1
    name = "lora_" + "a" * 16
    assert ("set", ["turbo", name], [1.0, 0.8]) in w.pipe.calls
    # fp32 factors (most community files) run in bf16.
    assert all(p.dtype == torch.bfloat16 for n, p in w.pipe.transformer.named_parameters() if "lora_" in n)
    # Off fast and without LoRAs: no adapter at all (not even Turbo).
    w.pipe.calls.clear()
    assert w._use_loras(params(fast=False), {}, Sink(), pinned=None) == 0
    assert w.pipe.calls == [("disable",)]

    # The references worker holds transformer_ref; diffusers routes the converted keys there by itself.
    r = worker("ref2va", tmp_path)
    assert r.components == ("transformer_ref",)
    assert r._use_loras(params("h3-references", lora2_strength=1.0), {"lora2": f}, Sink(), pinned=None) == 1
    assert r.pipe.calls[0] == ("load", name, {})


def test_h3_refuses_a_lora_with_no_h3_layers(tmp_path):
    w = worker("t2va", tmp_path)
    f = lora_file(tmp_path, "b" * 64, foreign=True)
    with pytest.raises(ValueError, match="doesn't fit MiniMax H3"):
        w._use_loras(params(), {"lora1": f}, Sink(), pinned={"turbo": 1.0})
    assert w.pipe.adapters == ["turbo"]
