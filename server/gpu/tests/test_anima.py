"""Anima: kohya LoRA conversion, LoRAs attaching to a real (tiny) Cosmos transformer, the ComfyUI single-file
conversion, and the worker against a stub pipeline. Tiny models built here; nothing is downloaded."""

import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aistudio_gpu import variants
from aistudio_gpu.tasks.anima import AnimaWorker, kohya_to_diffusion_model

HIDDEN, TEXT, SOURCE = 32, 16, 8


class Sink:
    def __init__(self):
        self.statuses, self.steps, self.checks = [], [], 0

    def status(self, t): self.statuses.append(t)
    def progress(self, s, t): self.steps.append((s, t))
    def preview(self, *a): pass

    def check(self): self.checks += 1


def tiny_transformer(seed=0):
    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(seed)
    return CosmosTransformer3DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=16, num_layers=2, mlp_ratio=2.0,
        text_embed_dim=TEXT, adaln_lora_dim=8, max_size=(1, 16, 16), patch_size=(1, 2, 2), rope_scale=(1.0, 4.0, 4.0),
        concat_padding_mask=True, extra_pos_embed_type=None)


def tiny_conditioner(seed=0):
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner

    torch.manual_seed(seed)
    return AnimaTextConditioner(source_dim=SOURCE, target_dim=TEXT, model_dim=16, num_layers=1,
                                num_attention_heads=2, target_vocab_size=32)


# kohya module → diffusers module, for a transformer block
KOHYA_TO_DIFFUSERS = {
    "self_attn_q_proj": "attn1.to_q", "self_attn_k_proj": "attn1.to_k", "self_attn_v_proj": "attn1.to_v",
    "self_attn_output_proj": "attn1.to_out.0",
    "cross_attn_q_proj": "attn2.to_q", "cross_attn_k_proj": "attn2.to_k", "cross_attn_v_proj": "attn2.to_v",
    "cross_attn_output_proj": "attn2.to_out.0",
    "mlp_layer1": "ff.net.0.proj", "mlp_layer2": "ff.net.2",
    "adaln_modulation_self_attn_1": "norm1.linear_1", "adaln_modulation_self_attn_2": "norm1.linear_2",
    "adaln_modulation_cross_attn_1": "norm2.linear_1", "adaln_modulation_cross_attn_2": "norm2.linear_2",
    "adaln_modulation_mlp_1": "norm3.linear_1", "adaln_modulation_mlp_2": "norm3.linear_2",
}


def fake_kohya(model, rank=4, alpha=None, dtype=torch.float16, blocks=(0, 1)):
    """A kohya Anima LoRA for ``model``'s blocks: every module kind, ``lora_down`` [rank, in], ``lora_up`` [out, rank];
    ``alpha`` (a number, or None for no alpha key) on every module."""
    torch.manual_seed(1)
    state = {}
    for b in blocks:
        for kohya, path in KOHYA_TO_DIFFUSERS.items():
            out_f, in_f = model.get_submodule(f"transformer_blocks.{b}.{path}").weight.shape
            name = f"lora_unet_blocks_{b}_{kohya}"
            state[f"{name}.lora_down.weight"] = torch.randn(rank, in_f).to(dtype)
            state[f"{name}.lora_up.weight"] = torch.randn(out_f, rank).to(dtype)
            if alpha is not None:
                state[f"{name}.alpha"] = torch.tensor(alpha, dtype=dtype)
    return state


# MARK: kohya conversion

def test_kohya_names_and_alpha_scaling():
    model = tiny_transformer()
    state = fake_kohya(model, rank=4, alpha=2.0, blocks=(0, 1))
    out = kohya_to_diffusion_model(state)
    assert len(out) == 2 * len(KOHYA_TO_DIFFUSERS) * 2 and all(k.startswith("diffusion_model.") for k in out)
    assert {"diffusion_model.blocks.1.self_attn.output_proj.lora_A.weight",
            "diffusion_model.blocks.0.cross_attn.k_proj.lora_B.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_A.weight",
            "diffusion_model.blocks.1.mlp.layer2.lora_B.weight",
            "diffusion_model.blocks.0.adaln_modulation_self_attn.1.lora_A.weight",
            "diffusion_model.blocks.1.adaln_modulation_mlp.2.lora_B.weight"} <= set(out)
    down = state["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"]
    up = state["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"]
    assert torch.equal(out["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"], down * 0.5)  # alpha 2 / rank 4
    assert torch.equal(out["diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight"], up)


def test_kohya_without_alpha_keeps_lora_down():
    model = tiny_transformer()
    state = fake_kohya(model, alpha=None, blocks=(0,))
    out = kohya_to_diffusion_model(state)
    key = "lora_unet_blocks_0_mlp_layer2"
    assert torch.equal(out["diffusion_model.blocks.0.mlp.layer2.lora_A.weight"], state[f"{key}.lora_down.weight"])
    assert len(out) == 2 * len(KOHYA_TO_DIFFUSERS)


def test_other_layouts_pass_through_unchanged():
    state = {"diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.zeros(1, 1)}
    assert kohya_to_diffusion_model(state) is state


def test_kohya_unknown_layer_and_text_encoder_keys_raise():
    good = {"lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.zeros(2, 4),
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.zeros(4, 2)}
    with pytest.raises(ValueError, match="unknown layer final_layer_linear"):
        kohya_to_diffusion_model({**good, "lora_unet_final_layer_linear.lora_down.weight": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="unknown layer blocks_0_self_attn_x_proj"):
        kohya_to_diffusion_model({"lora_unet_blocks_0_self_attn_x_proj.lora_down.weight": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="text-encoder LoRAs aren't supported"):
        kohya_to_diffusion_model({**good, "lora_te_text_model_encoder.lora_down.weight": torch.zeros(2, 4)})


# MARK: attaching to a real model

def Pipe(transformer, text_conditioner=None):
    """The loader mixin without a full pipeline: a transformer and (here) maybe no text conditioner."""
    from diffusers.loaders import AnimaLoraLoaderMixin

    class P(AnimaLoraLoaderMixin):
        hf_device_map = None  # what a DiffusionPipeline has

        @property
        def components(self):
            return {"transformer": self.transformer, "text_conditioner": self.text_conditioner}

    pipe = P()
    pipe.transformer, pipe.text_conditioner = transformer, text_conditioner
    return pipe


def worker_for(pipe) -> AnimaWorker:
    worker = AnimaWorker()
    worker.pipe, worker.dtype = pipe, torch.bfloat16
    return worker


def test_converted_kohya_lora_attaches_to_a_cosmos_transformer():
    model = tiny_transformer()
    worker = worker_for(Pipe(model))
    state = fake_kohya(model, rank=4, alpha=4.0)
    worker.pipe.load_lora_weights(worker._prepare_lora(state), adapter_name="x")
    assert "x" in worker.pipe.get_list_adapters()["transformer"]
    lora = {n for n, _ in model.named_parameters() if ".lora_" in n}
    assert len(lora) == 2 * len(KOHYA_TO_DIFFUSERS) * 2
    for b in (0, 1):
        for path in ("attn1.to_q", "attn1.to_out.0", "attn2.to_k", "ff.net.0.proj", "ff.net.2", "norm3.linear_2"):
            assert f"transformer_blocks.{b}.{path}.lora_A.x.weight" in lora
            assert f"transformer_blocks.{b}.{path}.lora_B.x.weight" in lora
    # alpha == rank: lora_A is the file's lora_down as it is
    a = model.get_submodule("transformer_blocks.0.attn1.to_q").lora_A["x"].weight
    assert torch.allclose(a.float(), state["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"].float())


def test_load_lora_through_the_worker_casts_to_bf16_and_scales(tmp_path):
    model = tiny_transformer()
    worker = worker_for(Pipe(model))
    state = fake_kohya(model, rank=4, alpha=2.0, dtype=torch.float32)
    path = tmp_path / "abc.safetensors"
    save_file(state, str(path))
    worker._load_lora("lora_abc", [(path, "transformer")], 1)
    assert "lora_abc" in worker.pipe.get_list_adapters()["transformer"]
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert params and all(p.dtype == torch.bfloat16 for p in params.values())
    a = model.get_submodule("transformer_blocks.1.ff.net.2").lora_A["lora_abc"].weight
    expected = state["lora_unet_blocks_1_mlp_layer2.lora_down.weight"] * 0.5
    assert torch.allclose(a.float(), expected.to(torch.bfloat16).float())


def test_a_lora_for_another_model_is_refused_and_leaves_nothing(tmp_path):
    model = tiny_transformer()
    worker = worker_for(Pipe(model))
    bad = tmp_path / "bad.safetensors"
    save_file({"lora_unet_input_blocks_1_0_proj_in.lora_down.weight": torch.zeros(2, 4),
               "lora_unet_input_blocks_1_0_proj_in.lora_up.weight": torch.zeros(4, 2)}, str(bad))
    with pytest.raises(ValueError, match=r"LoRA 2 doesn't fit Anima: unknown layer"):
        worker._load_lora("lora_bad", [(bad, "transformer")], 2)
    # a diffusers-style file naming no Anima module: diffusers only warns; the worker says so
    foreign = tmp_path / "foreign.safetensors"
    save_file({"transformer.something_else.lora_A.weight": torch.zeros(2, 4),
               "transformer.something_else.lora_B.weight": torch.zeros(4, 2)}, str(foreign))
    with pytest.raises(ValueError, match=r"LoRA 3 doesn't fit Anima"):
        worker._load_lora("lora_foreign", [(foreign, "transformer")], 3)
    assert not any(".lora_" in n for n, _ in model.named_parameters())
    assert not any(worker.pipe.get_list_adapters().values())


# MARK: the single-file conversion

def to_comfy_name(key: str) -> str:
    """A diffusers Cosmos name → the original Cosmos-2.0 / ComfyUI one (diffusers' rename dict, reversed; each
    pattern is unique in the names, so the order only matters where one contains another)."""
    for new, old in [
        ("time_embed.t_embedder", "t_embedder.1"), ("time_embed.norm", "t_embedding_norm"),
        ("patch_embed.proj", "x_embedder.proj.1"),
        ("norm1.linear_1", "adaln_modulation_self_attn.1"), ("norm1.linear_2", "adaln_modulation_self_attn.2"),
        ("norm2.linear_1", "adaln_modulation_cross_attn.1"), ("norm2.linear_2", "adaln_modulation_cross_attn.2"),
        ("norm3.linear_1", "adaln_modulation_mlp.1"), ("norm3.linear_2", "adaln_modulation_mlp.2"),
        ("norm_out.linear_1", "final_layer.adaln_modulation.1"), ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
        ("proj_out", "final_layer.linear"),
        ("transformer_blocks", "blocks"), ("attn1", "self_attn"), ("attn2", "cross_attn"),
        ("to_q", "q_proj"), ("to_k", "k_proj"), ("to_v", "v_proj"), ("to_out.0", "output_proj"),
        ("norm_q", "q_norm"), ("norm_k", "k_norm"),
        ("ff.net.0.proj", "mlp.layer1"), ("ff.net.2", "mlp.layer2")]:
        key = key.replace(new, old)
    return key


def bf16(state: dict) -> dict:
    return {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}


def write_base(root, transformer, conditioner):
    (root / "transformer").parent.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(str(root / "transformer"))
    conditioner.save_pretrained(str(root / "text_conditioner"))
    (root / ".complete").write_text("{}")
    return root


def read_folder(folder) -> dict:
    out = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            out.update({k: sf.get_tensor(k) for k in sf.keys()})
    return out


def checkpoint(path, transformer, conditioner=None, prefix="net.", drop=()):
    state = {prefix + to_comfy_name(k): v for k, v in bf16(transformer.state_dict()).items()}
    if conditioner is not None:
        state.update({prefix + "llm_adapter." + k: v for k, v in bf16(conditioner.state_dict()).items()})
    state["vae.decoder.conv.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)  # outside the prefix: ignored
    state[prefix + "pos_embedder.seq"] = torch.arange(4, dtype=torch.bfloat16)  # the converter drops it
    for k in drop:
        state.pop(prefix + k)
    save_file(state, str(path))
    return path


def test_comfy_names_round_trip_through_diffusers_converter():
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    model = tiny_transformer()
    original = {"net." + to_comfy_name(k): v for k, v in model.state_dict().items()}
    assert "net.blocks.0.self_attn.q_proj.weight" in original and "net.final_layer.linear.weight" in original
    assert set(convert_cosmos_transformer_checkpoint_to_diffusers(dict(original))) == set(model.state_dict())


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model.", ""])
def test_convert_anima_writes_both_components(tmp_path, prefix):
    model, conditioner = tiny_transformer(0), tiny_conditioner(0)
    base = write_base(tmp_path / "base", model, conditioner)
    tuned, tuned_cond = tiny_transformer(7), tiny_conditioner(7)
    ckpt = checkpoint(tmp_path / "ft.safetensors", tuned, tuned_cond, prefix=prefix)
    out = tmp_path / "out"
    variants.convert_anima(ckpt, base, out)
    for component, want in (("transformer", tuned), ("text_conditioner", tuned_cond)):
        got = read_folder(out / component)
        assert set(got) == set(want.state_dict())  # the base's names, nothing else
        for k, v in got.items():
            assert v.dtype == torch.bfloat16 and torch.equal(v, want.state_dict()[k].to(torch.bfloat16)), k
        assert (out / component / "config.json").read_text() == (base / component / "config.json").read_text()


def test_convert_anima_without_llm_adapter_writes_only_the_transformer(tmp_path):
    base = write_base(tmp_path / "base", tiny_transformer(0), tiny_conditioner(0))
    ckpt = checkpoint(tmp_path / "ft.safetensors", tiny_transformer(3), None)
    out = tmp_path / "out"
    variants.convert_anima(ckpt, base, out)
    assert (out / "transformer").is_dir() and not (out / "text_conditioner").exists()


def test_convert_anima_refuses_incomplete_and_foreign_files(tmp_path):
    model, conditioner = tiny_transformer(0), tiny_conditioner(0)
    base = write_base(tmp_path / "base", model, conditioner)
    missing = checkpoint(tmp_path / "missing.safetensors", model, conditioner, drop=("blocks.1.mlp.layer2.weight",))
    with pytest.raises(ValueError, match=r"1 of its \d+ weights are missing, e.g. transformer_blocks.1.ff.net.2.weight"):
        variants.convert_anima(missing, base, tmp_path / "o1")
    assert not (tmp_path / "o1").exists()
    partial = checkpoint(tmp_path / "partial.safetensors", model, conditioner, drop=("llm_adapter.out_proj.bias",))
    with pytest.raises(ValueError, match="e.g. out_proj.bias"):
        variants.convert_anima(partial, base, tmp_path / "o2")
    other = tmp_path / "other.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.weight": torch.zeros(1)}, str(other))
    with pytest.raises(ValueError, match="this file isn't an Anima checkpoint"):
        variants.convert_anima(other, base, tmp_path / "o3")


def test_import_variant_converts_and_the_worker_swaps_it_in(tmp_path, monkeypatch):
    model, conditioner = tiny_transformer(0), tiny_conditioner(0)
    models = tmp_path / "models"
    write_base(models / "anima", model, conditioner)
    tuned = tiny_transformer(9)
    ckpt = checkpoint(tmp_path / "src.safetensors", tuned, None)  # no llm_adapter: the base's conditioner stays

    def fake_download(url, dest, sha256, tokens, progress, extra_hosts=(), cancelled=lambda: False):
        dest.write_bytes(ckpt.read_bytes())

    monkeypatch.setattr(variants, "download_file", fake_download)
    dest = variants.import_variant(models, "anima", "tiny-ft", "Tiny", files=[{"url": "https://civitai.com/x"}])
    assert (dest / "transformer").is_dir() and not (dest / "text_conditioner").exists() and not (dest / ".work").exists()
    assert variants.list_variants(models)["anima"][0]["id"] == "tiny-ft"
    with pytest.raises(ValueError, match="an Anima fine-tune is one checkpoint file"):
        variants.import_variant(models, "anima", "two", "Two", files=[{"url": "https://civitai.com/a"},
                                                                      {"url": "https://civitai.com/b"}])

    worker = worker_for(Pipe(tiny_transformer(0), tiny_conditioner(0)))
    worker.models_root = models
    before = {k: v.clone() for k, v in worker.pipe.text_conditioner.state_dict().items()}
    sink = Sink()
    worker._use_variant("tiny-ft", sink)
    assert "Loading tiny-ft" in sink.statuses
    for k, v in worker.pipe.transformer.state_dict().items():
        assert torch.equal(v, tuned.state_dict()[k].to(torch.bfloat16).to(v.dtype)), k
    assert all(torch.equal(v, before[k]) for k, v in worker.pipe.text_conditioner.state_dict().items())
    worker._use_variant("base", sink)
    for k, v in worker.pipe.transformer.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k


# MARK: the worker's run, against a stub pipeline

def test_run_sets_guidance_reports_steps_and_restores_the_progress_bar(tmp_path):
    import numpy as np
    from diffusers.modular_pipelines.anima import denoise
    from PIL import Image

    from aistudio_gpu.tasks import TASKS

    original = denoise.AnimaDenoiseLoopWrapper.progress_bar
    seen = {}

    class Guider:
        guidance_scale = 4.0

    class StubPipe:
        guider = Guider()

        def get_list_adapters(self):
            return {}

        def __call__(self, **kwargs):
            seen.update(kwargs, guidance=self.guider.guidance_scale)
            with denoise.AnimaDenoiseLoopWrapper.progress_bar(None, total=kwargs["num_inference_steps"]) as bar:
                for _ in range(kwargs["num_inference_steps"]):
                    bar.update()
            return [Image.fromarray(np.full((kwargs["height"], kwargs["width"], 3), 200, dtype=np.uint8))]

    spec = TASKS["anima"]
    worker = worker_for(StubPipe())
    params = spec.normalize({"prompt": "1girl", "negative": "  ", "width": 512, "height": 528, "steps": 3,
                             "guidance": 1.0, "seed": 5})
    sink = Sink()
    out = worker.run(spec, params, {}, sink, tmp_path)
    assert seen["guidance"] == 1.0 and seen["negative_prompt"] is None and seen["prompt"] == "1girl"
    assert (seen["width"], seen["height"], seen["num_inference_steps"]) == (512, 528, 3)
    assert seen["output"] == "images" and seen["output_type"] == "pil"
    assert seen["generator"].initial_seed() == 5
    assert sink.steps == [(1, 3), (2, 3), (3, 3)] and sink.checks == 3
    assert denoise.AnimaDenoiseLoopWrapper.progress_bar is original
    assert Image.open(out.path).size == (512, 528) and out.kind == "image"
    assert out.extra["steps"] == 3 and out.extra["guidance"] == 1.0 and out.extra["loras"] == 0
    assert re.fullmatch(r"result\.png", out.path.name)

    # a cancel raised from sink.check() leaves the progress bar restored too
    class Cancelling(Sink):
        def check(self): raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        worker.run(spec, params, {}, Cancelling(), tmp_path)
    assert denoise.AnimaDenoiseLoopWrapper.progress_bar is original
