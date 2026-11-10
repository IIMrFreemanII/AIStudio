"""The real Wan 2.2 and FLUX workers on tiny random-weight models (a few MB), on whatever device this machine has
(MPS on a Mac, CUDA, or the CPU). Checks the wiring — loading the folder layout the download step writes, both
Wan transformers, the Lightning LoRA files in their original key format, fp8 storage, progress, previews, cancel and
the encoded result — not image quality. Skipped without torch/diffusers. Downloads a tiny T5 (≈ 1 MB) once."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from mlxcomfy_gpu.tasks import TASKS, JobCancelled, make_worker  # noqa: E402

TINY_T5 = "hf-internal-testing/tiny-random-t5"


class Sink:
    def __init__(self, cancel_at=None):
        self.statuses, self.steps, self.previews, self.cancel_at = [], [], [], cancel_at

    def status(self, text):
        self.statuses.append(text)

    def progress(self, step, total):
        self.steps.append((step, total))

    def preview(self, image, step, total):
        self.previews.append((np.asarray(image).shape, step))

    def check(self):
        if self.cancel_at is not None and len(self.steps) >= self.cancel_at:
            raise JobCancelled()


def tiny_wan(root: Path, family: str) -> None:
    from diffusers import (AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline,
                           WanTransformer3DModel)
    from transformers import AutoTokenizer, T5EncoderModel

    torch.manual_seed(0)
    i2v = family == "wan22-i2v"

    def transformer():
        return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                     in_channels=36 if i2v else 16, out_channels=16, text_dim=32, freq_dim=256,
                                     ffn_dim=32, num_layers=2, cross_attn_norm=True,
                                     qk_norm="rms_norm_across_heads", rope_max_seq_len=32)

    parts = dict(
        transformer=transformer(), transformer_2=transformer(), boundary_ratio=0.875,
        vae=AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                             temperal_downsample=[False, True, True]),
        scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0),
        text_encoder=T5EncoderModel.from_pretrained(TINY_T5), tokenizer=AutoTokenizer.from_pretrained(TINY_T5))
    pipe = (WanImageToVideoPipeline(image_encoder=None, image_processor=None, **parts) if i2v
            else WanPipeline(**parts))
    pipe.save_pretrained(root / family)
    (root / family / ".complete").write_text("{}")


def tiny_lightning(root: Path) -> None:
    """LoRA files in the original Wan key format, like lightx2v/Wan2.2-Lightning's."""
    from safetensors.torch import save_file

    rank, inner, ffn = 4, 24, 32
    shapes = {**{f"{a}.{p}": (inner, inner) for a in ("self_attn", "cross_attn") for p in "qkvo"},
              "ffn.0": (ffn, inner), "ffn.2": (inner, ffn)}
    for kind in ("t2v", "i2v"):
        for part in ("high", "low"):
            tensors = {}
            for block in range(2):
                for name, (out_f, in_f) in shapes.items():
                    key = f"diffusion_model.blocks.{block}.{name}"
                    tensors[f"{key}.lora_down.weight"] = torch.randn(rank, in_f) * 0.01
                    tensors[f"{key}.lora_up.weight"] = torch.randn(out_f, rank) * 0.01
                    tensors[f"{key}.alpha"] = torch.tensor(float(rank))
            out = root / "wan22-lightning" / kind / f"{part}_noise_model.safetensors"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, out)
    (root / "wan22-lightning" / ".complete").write_text("{}")


def tiny_flux(root: Path) -> None:
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

    torch.manual_seed(0)
    clip = CLIPTextModel(CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37,
                                        layer_norm_eps=1e-05, num_attention_heads=4, num_hidden_layers=5,
                                        pad_token_id=1, vocab_size=1000, hidden_act="gelu", projection_dim=32))
    pipe = FluxPipeline(
        transformer=FluxTransformer2DModel(patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
                                           attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                           pooled_projection_dim=32, axes_dims_rope=[4, 4, 8]),
        vae=AutoencoderKL(sample_size=32, in_channels=3, out_channels=3, block_out_channels=(4,), layers_per_block=1,
                          latent_channels=1, norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False,
                          shift_factor=0.0609, scaling_factor=1.5035),
        scheduler=FlowMatchEulerDiscreteScheduler(), text_encoder=clip,
        tokenizer=CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip"),
        text_encoder_2=T5EncoderModel.from_pretrained(TINY_T5), tokenizer_2=AutoTokenizer.from_pretrained(TINY_T5))
    pipe.save_pretrained(root / "flux1-dev")
    (root / "flux1-dev" / ".complete").write_text("{}")


QWEN_IMAGE = ("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6")


def tiny_qwen(root: Path) -> None:
    """A random Qwen-Image (two blocks, a 2-layer Qwen2.5-VL) with the real tokenizer (≈ 11 MB, fetched once: the
    tiny test tokenizers don't load in this transformers version), plus a Lightning-style LoRA."""
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline,
                           QwenImageTransformer2DModel)
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    torch.manual_seed(0)
    repo, revision = QWEN_IMAGE
    tokenizer = Path(snapshot_download(repo, revision=revision, allow_patterns=["tokenizer/*"])) / "tokenizer"
    config = Qwen2_5_VLConfig(
        text_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "rope_theta": 1000000.0, "vocab_size": 152064,
                     "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"}},
        vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 16},
        hidden_size=16, vocab_size=152064, vision_end_token_id=151653, vision_start_token_id=151652,
        vision_token_id=151654)
    transformer = QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=2,
                                              attention_head_dim=16, num_attention_heads=3, joint_attention_dim=16,
                                              guidance_embeds=False, axes_dims_rope=(8, 4, 4))
    pipe = QwenImagePipeline(
        transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=AutoencoderKLQwenImage(base_dim=24, z_dim=4, dim_mult=[1, 2, 4], num_res_blocks=1,
                                   temperal_downsample=[False, True], latents_mean=[0.0] * 4, latents_std=[1.0] * 4),
        text_encoder=Qwen2_5_VLForConditionalGeneration(config), tokenizer=Qwen2Tokenizer.from_pretrained(tokenizer))
    pipe.save_pretrained(root / "qwen-image")
    (root / "qwen-image" / ".complete").write_text("{}")
    out = root / "qwen-image-lightning" / "8steps.safetensors"
    out.parent.mkdir(parents=True)
    save_file(qwen_lora(transformer, 7, scale=0.05), str(out))
    (root / "qwen-image-lightning" / ".complete").write_text("{}")


def qwen_lora(transformer, seed: int, scale: float = 1.0) -> dict:
    """A diffusers-format LoRA on the image stream's attention projections."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in transformer.named_modules():
        if name.endswith(("attn.to_q", "attn.to_k", "attn.to_v")):
            tensors[f"transformer.{name}.lora_A.weight"] = torch.randn(4, module.in_features, generator=g) * scale
            tensors[f"transformer.{name}.lora_B.weight"] = torch.randn(module.out_features, 4, generator=g) * scale
    return tensors


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    tiny_wan(root, "wan22-t2v")
    tiny_wan(root, "wan22-i2v")
    tiny_lightning(root)
    tiny_flux(root)
    tiny_qwen(root)
    return root


def run(models, task, params, inputs=None, sink=None, tmp=None):
    spec = TASKS[task]
    worker = make_worker(spec.family)
    worker.load(models, lambda s: None)
    sink = sink or Sink()
    # The defaults, then the test's values without the limits (tiny models want tiny sizes).
    out = worker.run(spec, {**spec.normalize({}), **params}, inputs or {}, sink, tmp)
    return worker, sink, out


def read_mp4_frames(path: Path) -> int:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    n = sum(1 for _ in reader)
    return n, meta["size"]


@pytest.mark.parametrize("fast", [True, False])
def test_wan_text_to_video(models, tmp_path, fast):
    worker, sink, out = run(models, "wan22-t2v", {"prompt": "a fox", "width": 64, "height": 48, "length": 9,
                                                  "fast": fast, "steps": 0 if fast else 3}, tmp=tmp_path)
    assert worker.has_fast
    steps = 4 if fast else 3
    assert sink.steps == [(i, steps) for i in range(1, steps + 1)]
    assert sink.previews and sink.previews[0][0] == (48, 64, 3)
    assert out.kind == "video" and out.extra["fast"] is fast
    frames, size = read_mp4_frames(out.path)
    assert frames == 9 and size == (64, 48)
    # Fast swaps in the Lightning scheduler; full mode restores the original.
    assert type(worker.pipe.scheduler).__name__ == ("FlowMatchEulerDiscreteScheduler" if fast
                                                     else "UniPCMultistepScheduler")


def test_wan_image_to_video(models, tmp_path):
    from PIL import Image

    img = tmp_path / "start.png"
    Image.fromarray((np.random.rand(90, 120, 3) * 255).astype(np.uint8)).save(img)
    _, sink, out = run(models, "wan22-i2v", {"prompt": "waves", "width": 64, "height": 48, "length": 5},
                       {"image": img}, tmp=tmp_path)
    assert sink.steps[-1] == (4, 4)
    assert read_mp4_frames(out.path) == (5, (64, 48))


def test_wan_cancel(models, tmp_path):
    with pytest.raises(JobCancelled):
        run(models, "wan22-t2v", {"width": 64, "height": 48, "length": 5}, sink=Sink(cancel_at=2), tmp=tmp_path)


def test_wan_fp8_storage_with_loras(models, tmp_path, monkeypatch):
    """fp8 weights + bf16 LoRAs (the CUDA path), forced onto the CPU."""
    monkeypatch.setenv("MLXCOMFY_DEVICE", "cpu")
    monkeypatch.setenv("MLXCOMFY_FP8", "force")
    worker, sink, out = run(models, "wan22-t2v", {"width": 64, "height": 48, "length": 5}, tmp=tmp_path)
    blocks = worker.pipe.transformer.blocks[0]
    assert blocks.attn1.to_q.base_layer.weight.dtype == torch.float8_e4m3fn
    assert blocks.attn1.to_q.lora_A["fast"].weight.dtype != torch.float8_e4m3fn
    assert read_mp4_frames(out.path)[0] == 5


def test_nag_is_wan_attention_without_a_different_negative():
    """The NAG processor leaves a prompt alone when there is no negative half, or when the negative is the prompt
    itself (nothing to steer away from), and changes the output when the negative differs."""
    from diffusers import WanTransformer3DModel

    from mlxcomfy_gpu.tasks import nag

    torch.manual_seed(0)
    t = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                              out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2,
                              cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32).eval()
    x, step = torch.randn(1, 16, 2, 8, 8), torch.tensor([500])
    pos, neg = torch.randn(1, nag.TEXT_LEN, 32), torch.randn(1, nag.TEXT_LEN, 32)

    def forward(context):
        with torch.no_grad():
            return t(hidden_states=x, timestep=step, encoder_hidden_states=context, return_dict=False)[0]

    plain = forward(pos)
    nag.install([t])
    assert torch.equal(forward(pos), plain)
    assert torch.allclose(forward(torch.cat([pos, pos], dim=1)), plain, atol=1e-5)
    assert not torch.allclose(forward(torch.cat([pos, neg], dim=1)), plain, atol=1e-3)


def test_nag_guide_clips_the_norm():
    from mlxcomfy_gpu.tasks.nag import guide

    pos, neg = torch.ones(1, 1, 4), -torch.ones(1, 1, 4)  # extrapolated 21× away: clipped to 2.5× the norm
    out = guide(pos, neg, scale=11, tau=2.5, alpha=1.0)
    assert torch.allclose(out.abs().sum(), pos.abs().sum() * 2.5, rtol=1e-4)
    assert torch.allclose(guide(pos, neg, 11, 2.5, alpha=0.0), pos)


@pytest.mark.parametrize("fast, negative, how", [(True, "", "none"), (True, "blurry, static", "nag"),
                                                 (False, "blurry, static", "cfg")])
def test_wan_negative_prompt(models, tmp_path, monkeypatch, fast, negative, how):
    """Fast mode (no guidance) applies a negative prompt through NAG; the full model through classifier-free
    guidance."""
    from mlxcomfy_gpu.tasks import nag

    calls = []
    real = nag.guide
    monkeypatch.setattr(nag, "guide", lambda *a: calls.append(1) or real(*a))
    _, sink, out = run(models, "wan22-t2v", {"prompt": "a fox", "negative": negative, "width": 64, "height": 48,
                                             "length": 5, "fast": fast, "steps": 0 if fast else 2}, tmp=tmp_path)
    assert out.extra["negative"] == how
    # Two blocks in each of the two transformers, once per step.
    assert len(calls) == (2 * 4 if how == "nag" else 0)
    assert read_mp4_frames(out.path)[0] == 5


def test_flux(models, tmp_path):
    _, sink, out = run(models, "flux1-dev", {"prompt": "a lighthouse", "width": 64, "height": 64, "steps": 8},
                       tmp=tmp_path)
    assert sink.steps[-1] == (8, 8) and [s for _, s in sink.previews] == [4]
    from PIL import Image

    assert Image.open(out.path).size == (64, 64)


def tiny_flux_lora(worker, path: Path, seed: int) -> Path:
    """A diffusers-format LoRA on the tiny FLUX transformer's attention projections."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in worker.pipe.transformer.named_modules():
        if name.endswith(("attn.to_q", "attn.to_k", "attn.to_v")):
            tensors[f"transformer.{name}.lora_A.weight"] = torch.randn(4, module.in_features, generator=g)
            tensors[f"transformer.{name}.lora_B.weight"] = torch.randn(module.out_features, 4, generator=g)
    save_file(tensors, str(path))
    return path


def test_flux_loras(models, tmp_path, monkeypatch):
    """LoRAs change the image, strength 0 or no LoRA gives the plain image back, a loaded LoRA is reused by the next
    job, old ones are dropped past KEEP_LOADED, and a file that isn't a FLUX LoRA fails with a clear message."""
    from PIL import Image

    from mlxcomfy_gpu.tasks import adapters

    spec = TASKS["flux1-dev"]
    worker = make_worker(spec.family)
    worker.load(models, lambda s: None)
    a = tiny_flux_lora(worker, tmp_path / ("a" * 64 + ".safetensors"), 1)
    b = tiny_flux_lora(worker, tmp_path / ("b" * 64 + ".safetensors"), 2)
    n = iter(range(100))

    def image(inputs=None, **params):
        sink, out_dir = Sink(), tmp_path / f"out{next(n)}"
        out_dir.mkdir()
        out = worker.run(spec, {**spec.normalize({}), "prompt": "a lighthouse", "width": 64, "height": 64,
                                "steps": 2, **params}, inputs or {}, sink, out_dir)
        return np.asarray(Image.open(out.path)).astype(int), sink, out

    plain, _, out = image()
    assert out.extra["loras"] == 0
    with_a, sink, out = image({"lora1": a})
    assert out.extra["loras"] == 1 and "Loading LoRA 1" in sink.statuses
    assert np.abs(with_a - plain).max() > 0
    again, sink, _ = image({"lora1": a})
    assert np.array_equal(again, with_a) and not any(s.startswith("Loading LoRA") for s in sink.statuses)
    off, _, out = image({"lora1": a}, lora1_strength=0.0)
    assert np.array_equal(off, plain) and out.extra["loras"] == 0
    assert np.array_equal(image()[0], plain)  # the adapters are switched off again
    both, sink, out = image({"lora1": a, "lora2": b}, lora2_strength=0.5)
    assert out.extra["loras"] == 2 and sink.statuses.count("Loading LoRA 2") == 1
    assert np.abs(both - with_a).max() > 0
    monkeypatch.setattr(adapters, "KEEP_LOADED", 1)
    image({"lora1": b})
    assert list(worker.loras) == ["lora_" + "b" * 16]

    bad = tmp_path / ("c" * 64 + ".safetensors")
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="isn't a .safetensors file"):
        image({"lora1": bad})
    from safetensors.torch import save_file

    other = tmp_path / ("d" * 64 + ".safetensors")
    save_file({"lora_unet_some_other_model.lora_down.weight": torch.zeros(4, 7),
               "lora_unet_some_other_model.lora_up.weight": torch.zeros(7, 4)}, str(other))
    with pytest.raises(ValueError, match="doesn't fit FLUX.1 dev"):
        image({"lora1": other})
    assert np.array_equal(image()[0], plain)  # a failed LoRA leaves nothing behind


def test_flux_fine_tune_from_a_single_file_and_swapping_back(models, tmp_path):
    """A fine-tune checkpoint (diffusers names, one file) becomes a variant; a job with `model` loads its
    transformer weights in place, and `base` restores the original exactly."""
    from PIL import Image
    from safetensors.torch import save_file

    from mlxcomfy_gpu import variants

    spec = TASKS["flux1-dev"]
    worker = make_worker(spec.family)
    worker.load(models, lambda s: None)
    torch.manual_seed(3)
    ckpt = tmp_path / "fine-tune.safetensors"
    save_file({"model." + k if False else k: (v + torch.randn_like(v) * 0.05).contiguous()
               for k, v in worker.pipe.transformer.state_dict().items()}, str(ckpt))
    dest = variants.variant_dir(models, "flux.1", "tiny-ft")
    variants.convert_checkpoint("flux.1", ckpt, models / "flux1-dev", dest / "transformer")
    (dest / ".complete").write_text("{}")
    assert variants.list_variants(models)["flux.1"][0]["id"] == "tiny-ft"
    n = iter(range(100))

    def image(**params):
        out_dir = tmp_path / f"o{next(n)}"
        out_dir.mkdir()
        sink = Sink()
        out = worker.run(spec, {**spec.normalize({}), "prompt": "a lighthouse", "width": 64, "height": 64, "steps": 2,
                                **params}, {}, sink, out_dir)
        return np.asarray(Image.open(out.path)).astype(int), sink

    base, _ = image()
    tuned, sink = image(model="tiny-ft")
    assert "Loading tiny-ft" in sink.statuses and np.abs(tuned - base).max() > 0
    again, sink = image(model="tiny-ft")
    assert np.array_equal(again, tuned) and not any(s.startswith("Loading") for s in sink.statuses)
    assert np.array_equal(image(model="base")[0], base)
    with pytest.raises(ValueError, match="isn't installed on this server"):
        image(model="nope")


def wan_lora(transformer, seed: int, scale: float = 0.3) -> dict:
    """A diffusers-format LoRA on a tiny Wan transformer's self-attention projections."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in transformer.named_modules():
        if name.endswith(("attn1.to_q", "attn1.to_k", "attn1.to_v")):
            tensors[f"transformer.{name}.lora_A.weight"] = torch.randn(4, module.in_features, generator=g) * scale
            tensors[f"transformer.{name}.lora_B.weight"] = torch.randn(module.out_features, 4, generator=g) * scale
    return tensors


def wan_frames(worker, tmp_path, inputs=None, **params):
    import imageio_ffmpeg

    spec = TASKS["wan22-t2v"]
    out_dir = tmp_path / f"w{len(list(tmp_path.iterdir()))}"
    out_dir.mkdir()
    sink = Sink()
    out = worker.run(spec, {**spec.normalize({}), "prompt": "a fox", "width": 64, "height": 48, "length": 5,
                            **params}, inputs or {}, sink, out_dir)
    reader = imageio_ffmpeg.read_frames(str(out.path))
    meta = next(reader)
    first = np.frombuffer(next(reader), dtype=np.uint8).reshape(meta["size"][1], meta["size"][0], 3).astype(int)
    reader.close()
    return first, sink, out


def test_wan_loras_per_expert(models, tmp_path):
    """A Wan LoRA goes to both experts, to one, or comes as a pair (loraN + loraN_low); each is its own adapter in
    the transformers it applies to, and works with the Lightning LoRA (fast) and without it."""
    from safetensors.torch import save_file

    worker = make_worker("wan22-t2v")
    worker.load(models, lambda s: None)
    t = worker.pipe.transformer
    a, b = tmp_path / ("a" * 64 + ".safetensors"), tmp_path / ("b" * 64 + ".safetensors")
    save_file(wan_lora(t, 1), str(a))
    save_file(wan_lora(t, 2), str(b))

    def adapters():
        return {c: sorted(getattr(worker.pipe, c).peft_config) for c in ("transformer", "transformer_2")}

    plain, _, out = wan_frames(worker, tmp_path, fast=False, steps=2)
    assert out.extra["loras"] == 0
    seen = {}
    for expert, inputs in [("both", {"lora1": a}), ("high", {"lora1": a}), ("low", {"lora1": a}),
                           ("both", {"lora1": a, "lora1_low": b})]:
        frame, sink, out = wan_frames(worker, tmp_path, inputs, fast=False, steps=2, lora1_expert=expert)
        assert out.extra["loras"] == 1 and "Loading LoRA 1" in sink.statuses
        assert np.abs(frame - plain).max() > 0, expert
        seen[expert if len(inputs) == 1 else "pair"] = frame
    ad = adapters()
    assert "lora_" + "a" * 16 + "_h" in ad["transformer"] and "lora_" + "a" * 16 + "_h" not in ad["transformer_2"]
    assert "lora_" + "a" * 16 + "_l" in ad["transformer_2"] and "lora_" + "a" * 16 + "_l" not in ad["transformer"]
    pair = "lora_" + "a" * 16 + "_" + "b" * 8
    assert pair in ad["transformer"] and pair in ad["transformer_2"]
    assert len({f.tobytes() for f in seen.values()}) == 4  # all four apply differently
    # Again: no reload; with the Lightning LoRA too; and none gives the plain clip back.
    again, sink, _ = wan_frames(worker, tmp_path, {"lora1": a}, fast=False, steps=2, lora1_expert="high")
    assert np.array_equal(again, seen["high"]) and not any(x.startswith("Loading LoRA") for x in sink.statuses)
    fast_plain, _, _ = wan_frames(worker, tmp_path)
    fast_lora, _, out = wan_frames(worker, tmp_path, {"lora1": a}, lora1_expert="low")
    assert out.extra["fast"] and np.abs(fast_lora - fast_plain).max() > 0
    assert np.array_equal(wan_frames(worker, tmp_path, fast=False, steps=2)[0], plain)


def test_wan_loras_and_fine_tunes_on_fp8_storage(models, tmp_path, monkeypatch):
    """On the GPU the Wan transformers keep fp8 weights: a LoRA loaded afterwards, and a fine-tune swapped in, work
    there too (forced on the CPU here)."""
    from safetensors.torch import save_file

    monkeypatch.setenv("MLXCOMFY_FP8", "force")
    monkeypatch.setenv("MLXCOMFY_DEVICE", "cpu")
    worker = make_worker("wan22-t2v")
    worker.load(models, lambda s: None)
    assert worker.pipe.transformer.blocks[0].attn1.to_q.base_layer.weight.dtype == torch.float8_e4m3fn
    a = tmp_path / ("c" * 64 + ".safetensors")
    save_file(wan_lora(worker.pipe.transformer, 3), str(a))
    plain, _, _ = wan_frames(worker, tmp_path, fast=False, steps=2)
    with_lora, _, out = wan_frames(worker, tmp_path, {"lora1": a}, fast=False, steps=2)
    assert out.extra["loras"] == 1 and np.abs(with_lora - plain).max() > 0
    fast, _, _ = wan_frames(worker, tmp_path, {"lora1": a})
    assert fast.shape == plain.shape


def official_wan_names(state: dict) -> dict:
    """diffusers → the official Wan checkpoint names (the inverse of diffusers' converter), as ComfyUI saves them."""
    renames = [("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
               ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
               ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
               ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
               ("condition_embedder.time_proj", "time_projection.1"), ("attn2", "cross_attn"), ("attn1", "self_attn"),
               (".to_out.0.", ".o."), (".to_q.", ".q."), (".to_k.", ".k."), (".to_v.", ".v."),
               ("ffn.net.0.proj", "ffn.0"), ("ffn.net.2", "ffn.2"), ("proj_out", "head.head"),
               ("norm2", "norm__x"), ("norm3", "norm2"), ("norm__x", "norm3")]
    out = {}
    for k, v in state.items():
        n = k
        for a, b in renames:
            n = n.replace(a, b)
        n = "head.modulation" if n == "scale_shift_table" else n.replace(".scale_shift_table", ".modulation")
        out[n] = v
    return out


def test_wan_fine_tune_of_one_expert_from_an_official_fp8_file(models, tmp_path):
    """A low-noise expert fine-tune in the official names, scaled fp8 (as Kijai's files): converted by diffusers'
    converter; a job with it swaps only transformer_2, and `base` restores the original exactly."""
    from diffusers import WanTransformer3DModel
    from safetensors.torch import save_file

    from mlxcomfy_gpu import variants

    base_low = WanTransformer3DModel.from_pretrained(models / "wan22-t2v" / "transformer_2")
    torch.manual_seed(9)
    tuned, ckpt = {}, {}
    for k, v in base_low.state_dict().items():
        w = v.float() + torch.randn_like(v.float()) * 0.05
        tuned[k] = w
    for k, w in official_wan_names(tuned).items():
        if w.ndim == 2:
            scale = w.abs().max() / 448.0
            ckpt[f"model.diffusion_model.{k}"] = (w / scale).to(torch.float8_e4m3fn)
            ckpt[f"model.diffusion_model.{k[:-len('.weight')]}.scale_weight"] = scale.reshape(1)
        else:
            ckpt[f"model.diffusion_model.{k}"] = w.to(torch.bfloat16)
    ckpt["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
    path = tmp_path / "wan-low.safetensors"
    save_file(ckpt, str(path))
    dest = variants.variant_dir(models, "wan-2.2-t2v", "tiny-wan-low")
    variants.convert_checkpoint("wan-2.2-t2v", path, models / "wan22-t2v", dest / "transformer_2", "transformer_2")
    (dest / ".complete").write_text("{}")
    converted = WanTransformer3DModel.from_pretrained(dest / "transformer_2").state_dict()
    for k in ("blocks.0.attn1.to_q.weight", "blocks.1.ffn.net.2.weight", "condition_embedder.time_proj.weight"):
        assert torch.allclose(converted[k].float(), tuned[k], atol=2e-2), k
    assert variants.transformer_dir(models, "wan-2.2-t2v", "tiny-wan-low", "transformer") == \
        models / "wan22-t2v" / "transformer"

    worker = make_worker("wan22-t2v")
    worker.load(models, lambda s: None)
    high_before = {k: v.clone() for k, v in worker.pipe.transformer.state_dict().items()}
    plain, _, _ = wan_frames(worker, tmp_path, fast=False, steps=2)
    ft, sink, _ = wan_frames(worker, tmp_path, fast=False, steps=2, model="tiny-wan-low")
    assert "Loading tiny-wan-low" in sink.statuses and np.abs(ft - plain).max() > 0
    assert all(torch.equal(v, high_before[k]) for k, v in worker.pipe.transformer.state_dict().items())
    assert np.array_equal(wan_frames(worker, tmp_path, fast=False, steps=2, model="base")[0], plain)


def qwen_image(worker, tmp_path, inputs=None, **params):
    from PIL import Image

    spec = TASKS["qwen-image"]
    out_dir = tmp_path / f"q{len(list(tmp_path.iterdir()))}"
    out_dir.mkdir()
    sink = Sink()
    out = worker.run(spec, {**spec.normalize({}), "prompt": "a sign that says hello", "width": 64, "height": 64,
                            **params}, inputs or {}, sink, out_dir)
    return np.asarray(Image.open(out.path)).astype(int), sink, out


def test_qwen_image_fast_and_full(models, tmp_path):
    """Fast: the Lightning LoRA, 8 steps, no guidance (a negative prompt is ignored). Full: true CFG with the
    negative prompt."""
    spec = TASKS["qwen-image"]
    worker = make_worker(spec.family)
    worker.load(models, lambda s: None)
    assert worker.has_fast and not worker.fp8
    fast, sink, out = qwen_image(worker, tmp_path, negative="blurry")
    assert out.extra["negative"] == "ignored" and out.extra["steps"] == 8 and out.extra["cfg"] == 1.0
    assert sink.steps[-1] == (8, 8) and [s for _, s in sink.previews] == [4]
    assert sink.previews[0][0] == (64, 64, 3) and "Sampling (fast)" in sink.statuses
    _, _, out = qwen_image(worker, tmp_path, negative="blurry", fast=False, steps=2)
    assert out.extra["negative"] == "cfg" and out.extra["cfg"] == 4.0 and not out.extra["fast"]
    _, _, out = qwen_image(worker, tmp_path, fast=False, steps=2)
    assert out.extra["negative"] == "none"
    # The Lightning adapter is really switched off for the full model, and on again.
    off, _, _ = qwen_image(worker, tmp_path, fast=False, steps=8, cfg=1.0)
    assert np.abs(off - fast).max() > 0
    assert np.array_equal(qwen_image(worker, tmp_path, negative="blurry")[0], fast)


def test_qwen_image_loras_and_a_comfyui_fp8_fine_tune(models, tmp_path):
    """A user LoRA on top of Lightning; a ComfyUI-style checkpoint (prefixed, scaled fp8, with a VAE inside)
    becomes a variant, swaps in with the LoRAs still loaded, and `base` restores the original exactly."""
    from diffusers import QwenImageTransformer2DModel
    from safetensors import safe_open
    from safetensors.torch import save_file

    from mlxcomfy_gpu import variants

    spec = TASKS["qwen-image"]
    worker = make_worker(spec.family)
    worker.load(models, lambda s: None)
    lora = tmp_path / ("e" * 64 + ".safetensors")
    save_file(qwen_lora(worker.pipe.transformer, 11, scale=0.3), str(lora))

    plain, _, out = qwen_image(worker, tmp_path)
    with_lora, sink, out = qwen_image(worker, tmp_path, {"lora1": lora})
    assert out.extra["loras"] == 1 and "Loading LoRA 1" in sink.statuses and np.abs(with_lora - plain).max() > 0

    # The fine-tune, as ComfyUI saves one: model.diffusion_model.* names, 2-D weights in fp8 with a scale each.
    base_t = QwenImageTransformer2DModel.from_pretrained(models / "qwen-image" / "transformer")
    torch.manual_seed(5)
    ckpt, tuned = {}, {}
    for k, v in base_t.state_dict().items():
        w = (v.float() + torch.randn_like(v.float()) * 0.05)
        if w.ndim == 2:
            scale = w.abs().max() / 448.0
            ckpt[f"model.diffusion_model.{k}"] = (w / scale).to(torch.float8_e4m3fn)
            ckpt[f"model.diffusion_model.{k[:-len('.weight')]}.scale_weight"] = scale.reshape(1)
            w = ckpt[f"model.diffusion_model.{k}"].float() * scale
        else:
            ckpt[f"model.diffusion_model.{k}"] = w.to(torch.bfloat16)
        tuned[k] = w
    ckpt["vae.decoder.conv_in.weight"] = torch.zeros(3, 3)
    ckpt["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
    path = tmp_path / "qwen-fine-tune.safetensors"
    save_file(ckpt, str(path))
    dest = variants.variant_dir(models, "qwen-image", "tiny-qwen-ft")
    variants.convert_checkpoint("qwen-image", path, models / "qwen-image", dest / "transformer")
    (dest / ".complete").write_text("{}")
    shard = next((dest / "transformer").glob("*.safetensors"))
    with safe_open(str(shard), "pt") as f:
        key = next(k for k in f.keys() if k.endswith("to_q.weight"))
        assert f.get_tensor(key).dtype == torch.bfloat16
        assert torch.allclose(f.get_tensor(key).float(), tuned[key], atol=1e-2)
        assert not any(k.startswith(("vae", "model.")) for k in f.keys())
    assert (dest / "transformer" / "diffusion_pytorch_model.safetensors.index.json").exists()
    assert [v["id"] for v in variants.list_variants(models)["qwen-image"]] == ["tiny-qwen-ft"]
    # It loads as a model on its own too.
    QwenImageTransformer2DModel.from_pretrained(dest / "transformer")

    ft, sink, _ = qwen_image(worker, tmp_path, {"lora1": lora}, model="tiny-qwen-ft")
    assert "Loading tiny-qwen-ft" in sink.statuses and np.abs(ft - with_lora).max() > 0
    assert not any(s.startswith("Loading LoRA") for s in sink.statuses)  # the LoRA stayed loaded
    assert np.array_equal(qwen_image(worker, tmp_path, {"lora1": lora}, model="base")[0], with_lora)
    assert np.array_equal(qwen_image(worker, tmp_path)[0], plain)

    # A LoRA (or anything that isn't the whole transformer) is refused with a hint.
    with pytest.raises(ValueError, match="isn't a full fine-tune"):
        variants.convert_checkpoint("qwen-image", lora, models / "qwen-image", tmp_path / "nope")


def test_qwen_image_fp8_storage(models, tmp_path, monkeypatch):
    """On a card under 70 GB the transformer's weights are stored in fp8 (here: forced on the CPU)."""
    monkeypatch.setenv("MLXCOMFY_FP8", "force")
    monkeypatch.setenv("MLXCOMFY_DEVICE", "cpu")
    worker = make_worker("qwen-image")
    worker.load(models, lambda s: None)
    assert worker.fp8
    image, _, out = qwen_image(worker, tmp_path, steps=2)
    assert out.extra["fp8"] and image.shape == (64, 64, 3)
    from safetensors.torch import save_file

    lora = tmp_path / ("f" * 64 + ".safetensors")
    save_file(qwen_lora(worker.pipe.transformer, 13, scale=0.3), str(lora))
    with_lora, _, out = qwen_image(worker, tmp_path, {"lora1": lora}, steps=2)
    assert out.extra["loras"] == 1 and np.abs(with_lora - image).max() > 0


def test_download_takes_bf16_transformers_from_the_mirror_and_converts_the_rest(models, tmp_path, monkeypatch):
    """The download step with local folders for the Hub: an "official" repo with float32 weights and a "mirror"
    with bfloat16 transformers."""
    from diffusers import WanTransformer3DModel
    from safetensors import safe_open
    from transformers import T5EncoderModel

    from mlxcomfy_gpu import download as dl

    repo, mirror = tmp_path / "repo", tmp_path / "mirror"
    shutil.copytree(models / "wan22-t2v", repo)
    shutil.copytree(models / "wan22-t2v", mirror)
    for c in ("transformer", "transformer_2"):
        WanTransformer3DModel.from_pretrained(repo / c).to(torch.float32).save_pretrained(repo / c)
        WanTransformer3DModel.from_pretrained(mirror / c).to(torch.bfloat16).save_pretrained(mirror / c)
    T5EncoderModel.from_pretrained(repo / "text_encoder").to(torch.float32).save_pretrained(repo / "text_encoder")
    fetched = []

    def fake_snapshot(src, dest, token, allow=None, ignore=None, revision=None):
        import fnmatch

        for f in Path(src).rglob("*"):
            rel = f.relative_to(src).as_posix()
            if not f.is_file() or any(fnmatch.fnmatch(rel, p) for p in (ignore or [])):
                continue
            if allow is None or any(fnmatch.fnmatch(rel, p) for p in allow):
                fetched.append((Path(src).name, rel.split("/")[0]))
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / rel)

    monkeypatch.setattr(dl, "_snapshot", fake_snapshot)
    monkeypatch.setitem(dl.MODELS, "wan22-t2v", {"repo": str(repo), "bf16": ["text_encoder"],
                                                 "mirror": (str(mirror), None)})
    out_root = tmp_path / "out"
    dl.download(["wan22-t2v"], out_root)
    assert (out_root / "wan22-t2v" / ".complete").exists()
    # Transformers only from the mirror; nothing big from the official repo.
    assert {top for src, top in fetched if src == "repo"}.isdisjoint({"transformer", "transformer_2"})
    assert {top for src, top in fetched if src == "mirror"} == {"transformer", "transformer_2"}
    for component in ("transformer", "transformer_2", "text_encoder"):
        shard = next((out_root / "wan22-t2v" / component).glob("*.safetensors"))
        with safe_open(shard, "pt") as f:
            assert f.get_tensor(next(iter(f.keys()))).dtype == torch.bfloat16, component
    # The result loads as a pipeline, and installed folders are skipped on the next run.
    from diffusers import WanPipeline

    WanPipeline.from_pretrained(out_root / "wan22-t2v", torch_dtype=torch.bfloat16)
    fetched.clear()
    dl.download(["wan22-t2v"], out_root)
    assert fetched == []
