"""LTX-2.3: frame snapping, the ComfyUI single-file conversion, and the worker against a tiny real pipeline (random
weights, the real Gemma tokenizer; nothing else is downloaded): text and image to video, fast and full, the distilled
pseudo-variant, a fine-tune, a LoRA, progress and cancel."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from aistudio_gpu import variants
from aistudio_gpu.tasks import TASKS
from aistudio_gpu.tasks.ltx23 import DISTILLED, DISTILLED_DIR, LTX23Worker, num_frames

LTX23_REVISION = "8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
SIZE = 256  # the smallest the task takes (a multiple of 32)


class Sink:
    def __init__(self, cancel_at=None):
        self.statuses, self.steps, self.checks, self.cancel_at = [], [], 0, cancel_at

    def status(self, t): self.statuses.append(t)
    def progress(self, s, t): self.steps.append((s, t))
    def preview(self, *a): pass

    def check(self):
        self.checks += 1
        if self.cancel_at and len(self.steps) >= self.cancel_at:
            raise RuntimeError("cancelled")


# MARK: frame snapping

def test_frames_snap_to_the_nearest_8k_plus_1_and_stay_at_least_9():
    assert num_frames(5.0) == 121  # 120 frames → 121
    assert num_frames(1.0) == 25 and num_frames(10.0) == 241
    assert num_frames(2.0) == 49 and num_frames(3.0) == 73
    assert num_frames(0.1) == 9 and num_frames(0.0) == 9
    assert num_frames(4.3) == 105  # 103 → 105 (nearer than 97)
    assert all((num_frames(s / 10) - 1) % 8 == 0 for s in range(1, 101))


# MARK: a tiny LTX-2.3

def tiny_transformer(seed=0):
    from diffusers import LTX2VideoTransformer3DModel

    torch.manual_seed(seed)
    return LTX2VideoTransformer3DModel(
        in_channels=8, out_channels=8, num_attention_heads=2, attention_head_dim=8, cross_attention_dim=16,
        audio_in_channels=8, audio_out_channels=8, audio_num_attention_heads=2, audio_attention_head_dim=8,
        audio_cross_attention_dim=16, num_layers=2, caption_channels=16, gated_attn=True, cross_attn_mod=True,
        audio_gated_attn=True, audio_cross_attn_mod=True, rope_double_precision=False, use_prompt_embeddings=False,
        perturbed_attn=True, rope_type="split")


def tiny_pipeline():
    """The real pipeline class over 2-block random weights: LTX-2.3's flags (gated attention, per-modality
    projections, prompt AdaLN), the real scheduler settings and the real tokenizer (fetched once)."""
    from diffusers import (AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler,
                           LTX2Pipeline)
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma3Config, Gemma3ForConditionalGeneration

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(snapshot_download("diffusers/LTX-2.3-Diffusers", revision=LTX23_REVISION,
                               allow_patterns=["tokenizer/*"])) / "tokenizer")
    text_encoder = Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, vocab_size=262208, sliding_window=16,
                         layer_types=["sliding_attention", "full_attention"]),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        mm_tokens_per_image=4))
    vae = AutoencoderKLLTX2Video(
        block_out_channels=(8, 16, 32, 32), decoder_block_out_channels=(8, 16, 16, 32), layers_per_block=(1,) * 5,
        decoder_layers_per_block=(1,) * 5, latent_channels=8, decoder_spatial_padding_mode="zeros",
        decoder_causal=False, decoder_spatio_temporal_scaling=(True,) * 4, decoder_inject_noise=(False,) * 5,
        upsample_factor=(2, 2, 1, 2), upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
        upsample_residual=(False,) * 4, spatial_compression_ratio=32, temporal_compression_ratio=8)
    audio_vae = AutoencoderKLLTX2Audio(base_channels=8, ch_mult=(1, 2), num_res_blocks=1, latent_channels=2,
                                       mel_bins=16, resolution=64)
    vocoder = LTX2Vocoder(in_channels=32, hidden_channels=8, out_channels=2, upsample_kernel_sizes=[4, 4],
                          upsample_factors=[2, 2], resnet_kernel_sizes=[3], resnet_dilations=[[1, 3]],
                          output_sampling_rate=24000)
    connectors = LTX2TextConnectors(
        caption_channels=16, text_proj_in_factor=3, video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8, video_connector_num_layers=1, video_connector_num_learnable_registers=8,
        video_gated_attn=True, audio_connector_num_attention_heads=2, audio_connector_attention_head_dim=8,
        audio_connector_num_layers=1, audio_connector_num_learnable_registers=8, audio_gated_attn=True,
        per_modality_projections=True, video_hidden_dim=16, audio_hidden_dim=16, proj_bias=True,
        rope_double_precision=False, rope_type="split")
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, shift_terminal=0.1, base_shift=0.95, max_shift=2.05, base_image_seq_len=1024,
        max_image_seq_len=4096, time_shift_type="exponential")
    return LTX2Pipeline(scheduler=scheduler, vae=vae, audio_vae=audio_vae, text_encoder=text_encoder,
                        tokenizer=tokenizer, connectors=connectors, transformer=tiny_transformer(0),
                        vocoder=vocoder)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """``ltx23`` (the base), ``ltx23-distilled`` (another transformer) and a fine-tune ``tune``, in the server's
    folder layout."""
    root = tmp_path_factory.mktemp("models")
    tiny_pipeline().save_pretrained(root / "ltx23")
    (root / "ltx23" / ".complete").write_text("{}")
    tiny_transformer(1).save_pretrained(root / DISTILLED_DIR / "transformer")
    (root / DISTILLED_DIR / ".complete").write_text("{}")
    tune = root / "variants" / "ltx-2.3" / "tune"
    tiny_transformer(2).save_pretrained(tune / "transformer")
    (tune / "variant.json").write_text("{}")
    (tune / ".complete").write_text("{}")
    return root



def base_weights(models):
    return load_file(str(models / "ltx23" / "transformer" / "diffusion_pytorch_model.safetensors"))


def transformer_matches(worker, folder: Path) -> bool:
    want = load_file(str(next(folder.glob("*.safetensors"))))
    have = worker.pipe.transformer.state_dict()
    return all(torch.equal(have[k].float().cpu(), v.float()) for k, v in want.items() if k in have)


@pytest.fixture
def worker(models, monkeypatch):
    monkeypatch.setenv("AISTUDIO_DEVICE", "cpu")
    w = LTX23Worker()
    w.load(models, lambda t: None)
    # What each run asks the pipelines, and how many transformer passes a step costs.
    w.passes, w.negatives, w.called = [], [], []
    for name, pipe in (("t2v", w.pipe), ("i2v", w.i2v)):
        original = pipe.encode_prompt

        def spy(*a, _name=name, _original=original, **kw):
            w.called.append(_name)
            w.negatives.append(kw.get("negative_prompt"))
            return _original(*a, **kw)

        pipe.encode_prompt = spy
    w.pipe.transformer.register_forward_hook(lambda *a: w.passes.append(1))
    return w


def run(worker, tmp_path, inputs=None, sink=None, **params):
    spec = TASKS["ltx23-video"]
    p = spec.normalize({"prompt": "a fox in the snow", "width": SIZE, "height": SIZE, "seconds": 1.0, **params})
    return worker.run(spec, p, inputs or {}, sink or Sink(), tmp_path)


def probe(path: Path) -> str:
    import imageio_ffmpeg

    return subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr


# MARK: the worker

def test_load_starts_on_the_distilled_transformer_and_shares_the_modules(worker, models):
    assert worker.has_fast and worker.variant == DISTILLED
    assert transformer_matches(worker, models / DISTILLED_DIR / "transformer")
    assert not transformer_matches(worker, models / "ltx23" / "transformer")
    for name in ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder", "tokenizer"):
        assert getattr(worker.i2v, name) is getattr(worker.pipe, name)
    assert type(worker.i2v).__name__ == "LTX2ImageToVideoPipeline"


def test_fast_runs_the_distilled_schedule_with_one_pass_per_step(worker, tmp_path):
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    sink = Sink()
    out = run(worker, tmp_path, sink=sink, seed=3)
    assert out.kind == "video" and out.content_type == "video/mp4" and out.path.name == "result.mp4"
    assert out.extra["fast"] is True and out.extra["steps"] == 8 and out.extra["loras"] == 0
    assert out.extra["frames"] == 25 and (out.extra["width"], out.extra["height"]) == (SIZE, SIZE)
    assert out.extra["sampleRate"] == 24000 and out.extra["seconds"] == round(25 / 24, 3)
    assert len(worker.passes) == 8  # one transformer pass per step: no guidance, no STG, no modality pass
    assert sink.steps == [(i, 8) for i in range(1, 9)] and sink.checks == 8
    assert worker.pipe.num_timesteps == 8
    # the distilled sigmas as given (the base scheduler's dynamic shift and terminal stretch would have moved them)
    assert np.allclose(worker.pipe.scheduler.sigmas.numpy()[:-1], DISTILLED_SIGMA_VALUES, atol=1e-6)
    assert worker.variant == DISTILLED and worker.called == ["t2v"]
    info = probe(out.path)
    assert re.search(r"Video: h264.*256x256.*24 fps", info) and "Audio:" in info
    assert (out.path.parent / "sound.wav").exists()
    frames = subprocess.run([__import__("imageio_ffmpeg").get_ffmpeg_exe(), "-i", str(out.path), "-f", "null", "-"],
                            capture_output=True, text=True).stderr
    assert re.search(r"frame=\s*25\b", frames)


def test_full_runs_the_base_with_every_guidance_pass_and_the_default_negative(worker, models, tmp_path):
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

    sink = Sink()
    out = run(worker, tmp_path, sink=sink, fast=False, steps=3, guidance=4.0, negative="  ")
    assert out.extra["fast"] is False and out.extra["steps"] == 3
    assert worker.variant == "base" and transformer_matches(worker, models / "ltx23" / "transformer")
    assert len(worker.passes) == 3 * 3  # CFG (batched), spatio-temporal, modality isolation
    assert sink.steps == [(1, 3), (2, 3), (3, 3)]
    assert worker.pipe.guidance_scale == 4.0 and worker.pipe._stg_scale == 1.0 and worker.pipe._modality_scale == 3.0
    assert worker.negatives == [DEFAULT_NEGATIVE_PROMPT]
    # the base scheduler's own settings: dynamic shift, so the sigmas aren't the plain linspace
    assert not np.allclose(worker.pipe.scheduler.sigmas.numpy()[:-1], np.linspace(1.0, 1 / 3, 3), atol=1e-3)
    # an own negative prompt is passed as typed; steps 0 → 30
    out = run(worker, tmp_path, fast=False, steps=2, negative=" no fog ")
    assert worker.negatives[-1] == "no fog"
    # switching back to fast loads the distilled weights again
    run(worker, tmp_path)
    assert worker.variant == DISTILLED and transformer_matches(worker, models / DISTILLED_DIR / "transformer")


def test_full_without_steps_runs_30(worker, tmp_path):
    out = run(worker, tmp_path, fast=False, guidance=1.0)  # guidance 1: a single pass (STG and modality still run)
    assert out.extra["steps"] == 30


def test_a_fine_tune_keeps_its_weights_in_fast_mode_and_runs_the_distilled_schedule(worker, models, tmp_path):
    out = run(worker, tmp_path, model="tune")  # fast is the default
    assert out.extra["fast"] is True and out.extra["steps"] == 8 and len(worker.passes) == 8
    assert worker.variant == "tune"
    assert transformer_matches(worker, models / "variants" / "ltx-2.3" / "tune" / "transformer")
    assert worker.pipe.scheduler.sigmas[-2] > 0.4  # the distilled schedule: its last sigma is 0.421875
    worker.passes.clear()
    run(worker, tmp_path, model="tune", fast=False, steps=2)
    assert worker.variant == "tune" and len(worker.passes) == 2 * 3  # still the fine-tune; the full schedule
    with pytest.raises(ValueError, match="isn't installed"):
        run(worker, tmp_path, model="nope")


def test_without_the_distilled_model_a_base_job_runs_the_full_schedule(models, monkeypatch, tmp_path):
    monkeypatch.setenv("AISTUDIO_DEVICE", "cpu")
    import shutil

    shutil.copytree(models, tmp_path / "m", ignore=shutil.ignore_patterns("ltx23-distilled"))
    w = LTX23Worker()
    w.load(tmp_path / "m", lambda t: None)
    assert not w.has_fast and w.variant == "base"
    out = run(w, tmp_path, steps=2)
    assert out.extra["fast"] is False and out.extra["steps"] == 2


def test_an_image_input_runs_the_image_to_video_pipeline_on_the_shared_modules(worker, tmp_path):
    from PIL import Image

    picture = tmp_path / "start.png"
    Image.new("RGB", (200, 120), (255, 0, 0)).save(picture)
    out = run(worker, tmp_path, {"image": picture})
    assert worker.called == ["i2v"] and out.extra["frames"] == 25
    assert "Audio:" in probe(out.path)
    run(worker, tmp_path)
    assert worker.called == ["i2v", "t2v"]


def test_a_lora_loads_through_the_text_pipeline_and_reaches_the_image_pipeline(worker, tmp_path):
    from PIL import Image

    torch.manual_seed(5)
    state = {}
    for block in (0, 1):
        for module in ("attn1.to_q", "attn1.to_k", "audio_attn1.to_v"):
            state[f"transformer.transformer_blocks.{block}.{module}.lora_A.weight"] = torch.randn(4, 16)
            state[f"transformer.transformer_blocks.{block}.{module}.lora_B.weight"] = torch.randn(16, 4)
    lora = tmp_path / ("c" * 64 + ".safetensors")
    save_file(state, str(lora))
    picture = tmp_path / "start.png"
    Image.new("RGB", (64, 64), (0, 255, 0)).save(picture)

    plain = run(worker, tmp_path, seed=1)
    with_lora = run(worker, tmp_path, {"lora1": lora}, seed=1, lora1_strength=0.8)
    assert plain.extra["loras"] == 0 and with_lora.extra["loras"] == 1
    name = "lora_" + "c" * 16
    assert name in worker.pipe.get_list_adapters()["transformer"] and list(worker.loras) == [name]
    # the image pipeline sees the same adapter (the module is shared) and runs with it
    assert worker.i2v.get_list_adapters() == worker.pipe.get_list_adapters()
    out = run(worker, tmp_path, {"lora1": lora, "image": picture}, seed=1)
    assert out.extra["loras"] == 1 and worker.called[-1] == "i2v"
    assert worker.pipe.transformer.peft_config[name].r == 4
    # a job without it switches the adapter off
    assert run(worker, tmp_path).extra["loras"] == 0
    # a file that fits nothing is refused
    foreign = tmp_path / ("d" * 64 + ".safetensors")
    save_file({"transformer.blocks.0.nope.lora_A.weight": torch.randn(4, 16),
               "transformer.blocks.0.nope.lora_B.weight": torch.randn(16, 4)}, str(foreign))
    with pytest.raises(ValueError, match="LoRA 1"):
        run(worker, tmp_path, {"lora1": foreign})


def test_cancel_stops_between_steps_and_the_next_job_runs(worker, tmp_path):
    with pytest.raises(RuntimeError, match="cancelled"):
        run(worker, tmp_path, sink=Sink(cancel_at=2))
    assert len(worker.passes) == 2
    worker.passes.clear()
    assert run(worker, tmp_path).extra["steps"] == 8 and len(worker.passes) == 8


# MARK: the ComfyUI single-file conversion

def to_original_name(name: str) -> str:
    """The inverse of diffusers' ``convert_ltx2_transformer_to_diffusers`` (plus LTX-2.3's prompt AdaLN names)."""
    for new, old in (("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
                     ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
                     ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
                     ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
                     ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
                     ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
                     ("norm_q", "q_norm"), ("norm_k", "k_norm")):
        name = name.replace(new, old)
    for new, old in (("audio_proj_in.", "audio_patchify_proj."), ("proj_in.", "patchify_proj."),
                     ("audio_time_embed.", "audio_adaln_single."), ("time_embed.", "adaln_single."),
                     ("audio_prompt_adaln.", "audio_prompt_adaln_single."), ("prompt_adaln.", "prompt_adaln_single.")):
        if name.startswith(new):
            name = old + name[len(new):]
    return name


def comfy_file(base: Path, path: Path, drop: str | None = None, fp8: bool = True) -> dict:
    """A ComfyUI-style LTX-2.3 file for the base's weights: original names under ``model.diffusion_model.``, 2-D block
    weights in fp8 with a ``weight_scale``, the rest bf16, next to other components. Returns each weight's expected
    value after the conversion (bf16)."""
    state = load_file(str(base / "diffusion_pytorch_model.safetensors"))
    # the diffusers names are what the real converter maps *to*; the inverse must round-trip them
    file, expected = {}, {}
    for name, w in state.items():
        if name == drop:
            continue
        key = "model.diffusion_model." + to_original_name(name)
        if fp8 and w.ndim == 2 and name.startswith("transformer_blocks.") and name.endswith(".weight"):
            scale = w.abs().max() / 448.0
            q = (w / scale).to(torch.float8_e4m3fn)
            file[key], file[key + "_scale"] = q, scale.reshape(())
            expected[name] = (q.float() * scale).to(torch.bfloat16)
        else:
            file[key] = w.to(torch.bfloat16)
            expected[name] = w.to(torch.bfloat16)
    file["model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"] = torch.zeros(4, 4)
    file["model.diffusion_model.audio_embeddings_connector.learnable_registers"] = torch.zeros(4, 4)
    file["text_embedding_projection.video_aggregate_embed.weight"] = torch.zeros(4, 4)
    file["vae.encoder.conv_in.weight"] = torch.zeros(2, 2)
    file["audio_vae.decoder.conv_in.weight"] = torch.zeros(2, 2)
    file["vocoder.conv_pre.weight"] = torch.zeros(2, 2)
    some = next(k for k in file if k.endswith("attn1.to_k.bias"))
    file[some[: -len("bias")] + "comfy_quant"] = torch.zeros(3, dtype=torch.uint8)
    save_file(file, str(path))
    return expected


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    tiny_transformer(7).save_pretrained(base / "transformer")
    return base


def test_convert_ltx23_reads_a_comfyui_fp8_file_back_into_the_base_layout(tiny_base, tmp_path):
    expected = comfy_file(tiny_base / "transformer", tmp_path / "dragon.safetensors")
    assert any(k.endswith("_scale") for k in load_file(str(tmp_path / "dragon.safetensors")))
    variants.convert_checkpoint("ltx-2.3", tmp_path / "dragon.safetensors", tiny_base, tmp_path / "out")
    names = variants._base_names(tiny_base / "transformer")
    index = __import__("json").loads((tmp_path / "out" / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert sorted(index["weight_map"]) == names  # exactly the base's names: no connectors, VAEs or scales
    got = {}
    for f in (tmp_path / "out").glob("*.safetensors"):
        got.update(load_file(str(f)))
    assert set(got) == set(names)
    for name in names:
        assert got[name].dtype == torch.bfloat16 and torch.equal(got[name], expected[name]), name
    # fp8 is lossy, but close: within its 3-bit mantissa
    original = load_file(str(tiny_base / "transformer" / "diffusion_pytorch_model.safetensors"))
    w = "transformer_blocks.0.attn1.to_q.weight"
    assert torch.allclose(got[w].float(), original[w].float(), atol=0.08 * original[w].abs().max().item())
    assert (tmp_path / "out" / "config.json").read_text() == (tiny_base / "transformer" / "config.json").read_text()
    # the folder loads as the model
    from diffusers import LTX2VideoTransformer3DModel

    model = LTX2VideoTransformer3DModel.from_pretrained(tmp_path / "out")
    assert sum(p.numel() for p in model.parameters()) == sum(v.numel() for v in original.values())


def test_convert_ltx23_takes_a_bf16_file_and_other_prefixes(tiny_base, tmp_path):
    expected = comfy_file(tiny_base / "transformer", tmp_path / "bf16.safetensors", fp8=False)
    file = load_file(str(tmp_path / "bf16.safetensors"))
    save_file({k.replace("model.diffusion_model.", "diffusion_model.", 1): v for k, v in file.items()},
              str(tmp_path / "short.safetensors"))
    variants.convert_ltx23(tmp_path / "short.safetensors", tiny_base / "transformer", tmp_path / "out")
    got = load_file(str(next((tmp_path / "out").glob("diffusion_pytorch_model-*.safetensors"))))
    assert all(torch.equal(got[k], expected[k]) for k in expected)


def test_convert_ltx23_refuses_a_missing_weight_and_a_foreign_file(tiny_base, tmp_path):
    gone = "transformer_blocks.1.audio_to_video_attn.to_out.0.weight"
    comfy_file(tiny_base / "transformer", tmp_path / "short.safetensors", drop=gone)
    with pytest.raises(ValueError, match=re.escape(f"e.g. {gone}")):
        variants.convert_ltx23(tmp_path / "short.safetensors", tiny_base / "transformer", tmp_path / "o1")
    save_file({"model.diffusion_model.blocks.0.attn.weight": torch.zeros(2, 2)}, str(tmp_path / "wan.safetensors"))
    with pytest.raises(ValueError, match="isn't an LTX-2.3 checkpoint"):
        variants.convert_ltx23(tmp_path / "wan.safetensors", tiny_base / "transformer", tmp_path / "o2")


def test_dequantizing_keeps_the_scale_weight_rule_and_adds_weight_scale(tmp_path):
    q = (torch.arange(8, dtype=torch.float32).reshape(2, 4) / 4).to(torch.float8_e4m3fn)
    save_file({"a.weight": q.clone(), "a.scale_weight": torch.tensor(2.0), "b.weight": q.clone(),
               "b.weight_scale": torch.tensor(4.0), "c.weight": q.clone()}, str(tmp_path / "f.safetensors"))
    with safe_open(str(tmp_path / "f.safetensors"), "pt") as f:
        keys = set(f.keys())
        a, b, c = (variants._dequantized(f, keys, f"{n}.weight") for n in "abc")
    base = q.float()
    assert torch.equal(a.float(), (base * 2).to(torch.bfloat16).float())
    assert torch.equal(b.float(), (base * 4).to(torch.bfloat16).float())
    assert torch.equal(c.float(), base.to(torch.bfloat16).float())  # no scale: as stored
