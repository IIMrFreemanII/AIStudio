"""MiniMax H3 fine-tunes from links: ConvRot int8 dequantization, ComfyUI (full or AdaLN-pruned) → diffusers
conversion on tiny models, and the worker's layout swap against a stub pipeline. No downloads, no real models."""

import json

import pytest

torch = pytest.importorskip("torch")

from aistudio_gpu import variants  # noqa: E402
from aistudio_gpu.tasks import h3_pruned  # noqa: E402
from aistudio_gpu.tasks.minimax_h3 import H3Worker  # noqa: E402

TINY = dict(num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=2, num_refiner_layers=1,
            ffn_dim=64, in_channels=4, audio_in_channels=8, patch_size=(1, 2, 2), text_dim=24, freq_dim=8,
            time_embed_hidden_dim=16, time_embed_dim=24, rope_freq_dim=4)
TINY_PRUNED = {**TINY, "adaln_rank": 4, "time_table_size": 17}

# diffusers' names → ComfyUI's (the converter's rename dict, reversed; top-level prefixes first)
TOP = [("audio_proj_in.", "audio_patch_proj."), ("proj_in.", "video_patch_proj."), ("context_embedder.", "condition_proj."),
       ("norm_out.norm.", "final_layer.norm."), ("norm_out.linear.", "final_layer.adaln_proj.linear."),
       ("audio_proj_out.", "final_layer.audio_out."), ("proj_out.", "final_layer.video_out."),
       ("time_embedder.linear_1.", "time_embedder.proj_in."), ("time_embedder.linear_2.", "time_embedder.proj_out."),
       ("token_refiner.refiner_blocks.", "token_refiner.blocks."), ("transformer_blocks.", "blocks.")]
INNER = [(".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."), (".attn.to_out.0.", ".attn.out_proj."),
         (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2.")]


def random_state(model) -> dict:
    """Every persistent tensor of ``model``, random but bf16-exact (so a bf16 round trip is lossless)."""
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(v.shape, generator=g).bfloat16().float() for k, v in model.state_dict().items()}


def comfy_names(state: dict, pruned: bool) -> dict:
    """A diffusers state dict as ComfyUI files store it: qkv fused ``[q; k; v]``, fc1 halves ``[gate; value]``."""
    out = {}
    for k, v in state.items():
        if k == "time_embedder.table":
            out["adaln_t_table"] = v
            continue
        if pruned and k.endswith("adaln_proj.folded_bias"):
            out[k.replace("transformer_blocks.", "blocks.").replace("folded_bias", "linear.bias")] = v
            continue
        if pruned and k == "norm_out.folded_bias":
            out["final_layer.adaln_proj.linear.bias"] = v
            continue
        if k.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        if k.endswith(".attn.to_q.weight"):
            v = torch.cat([v, state[k.replace("to_q", "to_k")], state[k.replace("to_q", "to_v")]])
            k = k.replace("to_q", "qkv_proj")
        if k.endswith(".ff.net.0.proj.weight"):
            value, gate = v.chunk(2)
            v = torch.cat([gate, value])
        for a, b in TOP:
            if k.startswith(a):
                k = b + k[len(a):]
                break
        for a, b in INNER:
            k = k.replace(a, b)
        out[k] = v
    return out


def quantize_convrot(w, group: int):
    """``W @ H`` per group, int8 per row; returns (int8 weight, float32 scale [N, 1], comfy_quant bytes)."""
    n, k = w.shape
    h = h3_pruned._hadamard(group, "cpu", torch.float32)
    rotated = (w.float().reshape(n, k // group, group) @ h).reshape(n, k)
    scale = (rotated.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
    q = torch.round(rotated / scale).clamp(-127, 127).to(torch.int8)
    note = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}).encode()
    return q, scale, torch.tensor(list(note), dtype=torch.uint8)


def save(path, tensors: dict):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def read(folder):
    from safetensors.torch import load_file

    out = {}
    for f in sorted(folder.glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


# MARK: ConvRot

def test_convrot_int8_dequantizes_back_to_the_weight(tmp_path):
    from safetensors import safe_open

    w = torch.randn(64, 512)
    q, scale, note = quantize_convrot(w, 256)
    save(tmp_path / "f.safetensors", {"x.weight": q, "x.weight_scale": scale, "x.comfy_quant": note})
    with safe_open(str(tmp_path / "f.safetensors"), "pt") as f:
        keys = set(f.keys())
        quant = variants._h3_quant_info(f, keys)
        assert quant == {"x.weight": ("int8_tensorwise", True, 256)}
        back = variants._h3_tensor(f, keys, "x.weight", quant)
    assert back.dtype == torch.bfloat16
    assert (back.float() - w).abs().max() < 0.05 * w.abs().max()  # int8 step ≈ amax / 127, plus bf16 rounding


def test_a_quantization_other_than_int8_or_fp8_is_refused(tmp_path):
    from safetensors import safe_open

    note = torch.tensor(list(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8)
    save(tmp_path / "f.safetensors", {"x.weight": torch.zeros(4, 4), "x.comfy_quant": note})
    with safe_open(str(tmp_path / "f.safetensors"), "pt") as f:
        with pytest.raises(ValueError, match=r"this quantization \(nvfp4\) isn't supported; pick the bf16, fp8 or int8"):
            variants._h3_quant_info(f, set(f.keys()))


# MARK: conversion

def build_pruned(monkeypatch):
    monkeypatch.setattr(h3_pruned, "PRUNED_CONFIG", TINY_PRUNED)
    return h3_pruned.MiniMaxH3PrunedTransformer3DModel(**TINY_PRUNED)


def test_the_real_pruned_config_has_the_published_weight_names():
    with torch.device("meta"):
        names = list(h3_pruned.MiniMaxH3PrunedTransformer3DModel(**h3_pruned.PRUNED_CONFIG).state_dict())
    assert len(names) == 637  # the pruned repo's index
    assert {"time_embedder.table", "adaln_basis", "adaln_mean", "norm_out.folded_bias",
            "transformer_blocks.49.adaln_proj.linear.weight"} <= set(names)
    assert not any(n.startswith("time_embedder.linear") for n in names)


def test_a_pruned_comfy_file_converts_and_loads_with_the_pruned_class(tmp_path, monkeypatch):
    model = build_pruned(monkeypatch)
    state = random_state(model)
    comfy = comfy_names(state, pruned=True)
    # Two layers quantized as ComfyUI's ConvRot int8 files do (group 16 for the tiny widths), the file prefixed.
    qkv, fc2 = "blocks.1.attn.qkv_proj.weight", "blocks.0.mlp.fc2.weight"
    for name in (qkv, fc2):
        q, scale, note = quantize_convrot(comfy[name], 16)
        stem = name[: -len(".weight")]
        comfy[name] = q
        comfy[stem + ".weight_scale"], comfy[stem + ".comfy_quant"] = scale, note
        comfy[stem + ".input_scale"] = torch.ones(1)
    save(tmp_path / "pruned.safetensors", {f"model.diffusion_model.{k}": v for k, v in comfy.items()})
    out = tmp_path / "out"
    variants.convert_minimax_h3(tmp_path / "pruned.safetensors", tmp_path / "unused", out)

    assert variants.h3_layout(out) == "pruned"
    loaded = h3_pruned.MiniMaxH3PrunedTransformer3DModel.from_pretrained(str(out), torch_dtype=torch.bfloat16)
    got = loaded.state_dict()
    assert set(got) == set(state)
    q_names = {"transformer_blocks.1.attn.to_q.weight", "transformer_blocks.1.attn.to_k.weight",
               "transformer_blocks.1.attn.to_v.weight", "transformer_blocks.0.ff.net.2.weight"}
    for k, v in state.items():
        if k in q_names:
            assert (got[k].float() - v).abs().max() < 0.2, k  # int8 + bf16 error on unit-variance weights
        else:
            assert torch.equal(got[k].float(), v), k
    assert got["time_embedder.table"].dtype == torch.float32  # the class's float32 modules stay so
    assert got["adaln_basis"].dtype == torch.float32 and got["norm_out.folded_bias"].dtype == torch.float32
    assert got["transformer_blocks.0.adaln_proj.folded_bias"].dtype == torch.float32
    assert got["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16
    assert got["transformer_blocks.0.adaln_proj.linear.weight"].dtype == torch.bfloat16


def test_a_pruned_file_missing_a_weight_is_refused(tmp_path, monkeypatch):
    model = build_pruned(monkeypatch)
    comfy = comfy_names(random_state(model), pruned=True)
    del comfy["blocks.1.adaln_proj.linear.weight"]
    save(tmp_path / "p.safetensors", comfy)
    with pytest.raises(ValueError, match="1 of its .* weights are missing, e.g. transformer_blocks.1.adaln_proj.linear"):
        variants.convert_minimax_h3(tmp_path / "p.safetensors", tmp_path / "unused", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_pruned_file_without_the_affine_map_borrows_the_reference_one(tmp_path, monkeypatch):
    """DaSiWa's pruned files leave out adaln_basis / adaln_mean: the reference checkpoint's map for that partition
    (``minimax-h3-pruned-affine``) fills them in; without it installed the import says so."""
    model = build_pruned(monkeypatch)
    state = random_state(model)
    comfy = comfy_names(state, pruned=True)
    basis, mean = comfy.pop("adaln_basis"), comfy.pop("adaln_mean")
    save(tmp_path / "p.safetensors", comfy)
    base = tmp_path / "minimax-h3" / "transformer_ref"
    with pytest.raises(RuntimeError, match="install minimax-h3-pruned-affine"):
        variants.convert_minimax_h3(tmp_path / "p.safetensors", base, tmp_path / "out")
    (tmp_path / variants.PRUNED_AFFINE / "transformer_ref").mkdir(parents=True)
    save(tmp_path / variants.PRUNED_AFFINE / "transformer_ref" / "adaln_affine.safetensors",
         {"adaln_basis": basis, "adaln_mean": mean})
    variants.convert_minimax_h3(tmp_path / "p.safetensors", base, tmp_path / "out")
    got = h3_pruned.MiniMaxH3PrunedTransformer3DModel.from_pretrained(str(tmp_path / "out")).state_dict()
    assert torch.equal(got["adaln_basis"].float(), basis.float()) and torch.equal(got["adaln_mean"].float(), mean.float())


def test_a_full_layout_comfy_file_converts_to_the_bases_names(tmp_path):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel(**TINY)
    state = random_state(model)
    base = tmp_path / "base" / "transformer"
    model.save_pretrained(str(base))
    comfy = comfy_names(state, pruned=False)
    save(tmp_path / "full.safetensors", comfy)
    out = tmp_path / "out"
    variants.convert_minimax_h3(tmp_path / "full.safetensors", base, out)
    assert variants.h3_layout(out) == "full"
    assert json.loads((out / "config.json").read_text()) == json.loads((base / "config.json").read_text())
    loaded = MiniMaxH3Transformer3DModel.from_pretrained(str(out), torch_dtype=torch.bfloat16).state_dict()
    assert set(loaded) == set(state)
    assert all(torch.equal(loaded[k].float(), v) for k, v in state.items())
    # A pruned file is not a full fine-tune of this base: its AdaLN tensors don't exist in the base.
    del comfy["blocks.0.mlp.fc1.weight"]
    save(tmp_path / "cut.safetensors", comfy)
    with pytest.raises(ValueError, match="weights are missing"):
        variants.convert_minimax_h3(tmp_path / "cut.safetensors", base, tmp_path / "out2")


def test_a_file_without_h3_blocks_is_refused(tmp_path):
    save(tmp_path / "x.safetensors", {"foo.weight": torch.zeros(2)})
    with pytest.raises(ValueError, match="isn't a MiniMax H3 checkpoint"):
        variants.convert_minimax_h3(tmp_path / "x.safetensors", tmp_path, tmp_path / "out")


# MARK: the worker

class Manager:
    def __init__(self):
        self.calls = []

    def disable_auto_cpu_offload(self): self.calls.append("disable")


class RebuildPipe:
    def __init__(self, component):
        self.component = component
        self.calls = []
        setattr(self, component, "old")

    def unload_components(self, name):
        self.calls.append(("unload", name))
        setattr(self, name, None)

    def update_components(self, **kw):
        self.calls.append(("update", {k: type(v).__name__ for k, v in kw.items()}))
        for k, v in kw.items():
            setattr(self, k, v)


def install(root, family, vid, layout):
    d = variants.variant_dir(root, family, vid)
    (d / "transformer").mkdir(parents=True)
    (d / ".complete").write_text("{}")
    (d / "variant.json").write_text(json.dumps({"id": vid, "layout": layout}))


class Sink:
    def __init__(self): self.messages = []
    def status(self, t): self.messages.append(t)


@pytest.fixture
def h3(tmp_path, monkeypatch):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    w = H3Worker("t2va")
    w.models_root, w.dtype = tmp_path, torch.bfloat16
    w.pipe, w.manager = RebuildPipe("transformer"), Manager()
    w.events = []
    monkeypatch.setattr(w, "_load_turbo", lambda status: w.events.append("turbo"), raising=False)
    monkeypatch.setattr(w, "_enable_offload", lambda: w.events.append("offload"), raising=False)
    monkeypatch.setattr("aistudio_gpu.tasks.adapters._copy_weights",
                        lambda module, src, vid: w.events.append(("copy", vid)))
    loaded = []

    def fake_load(cls, path, **kw):
        loaded.append((cls.__name__, path, kw["torch_dtype"]))
        return cls.__new__(cls)

    for cls in (MiniMaxH3Transformer3DModel, h3_pruned.MiniMaxH3PrunedTransformer3DModel):
        monkeypatch.setattr(cls, "from_pretrained", classmethod(fake_load))
    w.loaded_models = loaded
    for vid, layout in (("eros", "pruned"), ("gxs", "pruned"), ("plain", "full")):
        install(tmp_path, "minimax-h3", vid, layout)
    return w


def test_workers_use_their_partitions_library_family():
    assert H3Worker("t2va").family == "minimax-h3" and H3Worker("ref2va").family == "minimax-h3-ref"
    assert variants.components("minimax-h3-ref") == H3Worker("ref2va").components


def test_a_base_job_swaps_nothing(h3):
    h3._use_h3_variant("base", Sink())
    h3._use_h3_variant("", Sink())
    assert h3.events == [] and h3.pipe.calls == [] and h3.loaded_models == []


def test_a_pruned_fine_tune_rebuilds_the_partition_and_back_to_base_rebuilds_the_full_one(h3, tmp_path):
    h3.loras["lora_x"] = None
    sink = Sink()
    h3._use_h3_variant("eros", sink)
    assert sink.messages == ["Loading eros (pruned layout)"]
    assert h3.manager.calls == ["disable"]
    assert h3.pipe.calls == [("unload", "transformer"), ("update", {"transformer": "MiniMaxH3PrunedTransformer3DModel"})]
    assert h3.loaded_models == [("MiniMaxH3PrunedTransformer3DModel",
                                 str(variants.variant_dir(tmp_path, "minimax-h3", "eros") / "transformer"),
                                 torch.bfloat16)]
    assert h3.events == ["turbo", "offload"]  # Turbo re-attached, offload on again with the worker's margin
    assert (h3.variant, h3.layout) == ("eros", "pruned") and not h3.loras

    h3.events.clear()
    h3._use_h3_variant("base", sink)
    assert h3.loaded_models[-1][:2] == ("MiniMaxH3Transformer3DModel", str(tmp_path / "minimax-h3" / "transformer"))
    assert h3.events == ["turbo", "offload"] and (h3.variant, h3.layout) == ("base", "full")
    assert sink.messages[-1] == "Loading the base model (full layout)"


def test_a_same_layout_fine_tune_copies_weights_in_place(h3):
    h3._use_h3_variant("eros", Sink())
    rebuilds = len(h3.loaded_models)
    h3.events.clear()
    h3._use_h3_variant("gxs", Sink())  # pruned → pruned
    assert h3.events == [("copy", "gxs")] and len(h3.loaded_models) == rebuilds and h3.variant == "gxs"
    h3._use_h3_variant("base", Sink())  # → full: rebuild
    h3.events.clear()
    h3._use_h3_variant("plain", Sink())  # full → full
    assert h3.events == [("copy", "plain")] and h3.variant == "plain"


def test_an_uninstalled_fine_tune_is_refused_and_a_failed_rebuild_is_retried(h3, monkeypatch):
    with pytest.raises(ValueError, match="isn't installed"):
        h3._use_h3_variant("nope", Sink())
    from aistudio_gpu.tasks.h3_pruned import MiniMaxH3PrunedTransformer3DModel as Pruned

    def boom(cls, *a, **k):
        raise OSError("disk")

    monkeypatch.setattr(Pruned, "from_pretrained", classmethod(boom))
    with pytest.raises(OSError):
        h3._use_h3_variant("eros", Sink())
    assert h3.variant is None and h3.layout is None  # nothing loaded: the next job rebuilds
    monkeypatch.undo()
