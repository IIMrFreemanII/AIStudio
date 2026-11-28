"""fp8.py: the big models' Linear layers in fp8 with a scale per output channel, and the loaders that write into them."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from aistudio_gpu.tasks import adapters, fp8  # noqa: E402



class Sink:
    def status(self, s):
        pass


class Block(torch.nn.Module):
    """Names like a DiT block: the AdaLN projection, the embedder and the tiny layer must stay bf16."""

    def __init__(self):
        super().__init__()
        self.x_embedder = torch.nn.Linear(512, 512)
        self.adaln_proj = torch.nn.Linear(512, 1024)
        self.ff = torch.nn.Linear(512, 1024)
        self.out = torch.nn.Linear(1024, 512)
        self.tiny = torch.nn.Linear(512, 64)

    def forward(self, x):
        return self.out(torch.relu(self.ff(self.x_embedder(x))))


def _block():
    torch.manual_seed(0)
    return Block().to(torch.bfloat16)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_quantize_weight_round_trips_within_fp8_precision():
    w = torch.randn(256, 512) * torch.logspace(-3, 1, 256)[:, None]  # rows of very different sizes
    q, s = fp8.quantize_weight(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (256,) and s.dtype == torch.float32
    back = q.float() * s[:, None]
    assert _rel(back, w) < 0.05  # each row keeps its own range


def test_convert_keeps_steering_layers_and_matches_bf16(monkeypatch):
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    ref = model(x)
    stats = fp8.convert(model, device="cpu")
    assert stats["layers"] == 2  # ff and out
    assert model.ff.weight.dtype == torch.float8_e4m3fn and model.ff.w_scale.shape == (1024,)
    for kept in (model.x_embedder, model.adaln_proj, model.tiny):
        assert kept.weight.dtype == torch.bfloat16 and not hasattr(kept, "w_scale")
    assert model.dtype == torch.bfloat16 if hasattr(model, "dtype") else True
    assert _rel(model(x), ref) < 0.08
    assert fp8.convert(model, device="cpu")["layers"] == 0  # converting twice changes nothing


def test_pinned_dtype_reports_bf16_for_a_model_whose_first_parameter_is_fp8():
    class M(torch.nn.Module):
        @property
        def dtype(self):
            return next(self.parameters()).dtype

    m = M()
    m.ff = torch.nn.Linear(512, 512).to(torch.bfloat16)
    fp8.convert(m, device="cpu")
    assert next(m.parameters()).dtype == torch.float8_e4m3fn and m.dtype == torch.bfloat16


def _save(state: dict, folder: Path) -> Path:
    folder.mkdir(parents=True)
    safetensors_torch.save_file(state, str(folder / "diffusion_pytorch_model.safetensors"))
    return folder


def test_park_loads_a_mode_into_fp8_layers_and_swaps_weights_with_scales(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "_room_for", lambda params: True)
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    base = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(1)
    fast = {k: torch.randn_like(v.float()).to(torch.bfloat16) for k, v in base.items()}
    fast_dir = _save(fast, tmp_path / "fast")
    fp8.convert(model, device="cpu")
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    base_out = model(x)

    reference = _block()
    reference.load_state_dict(fast)
    fp8.convert(reference, device="cpu")
    park = adapters.WeightPark()
    assert park.switch(model, "base", "@fast", fast_dir, Sink())
    assert model.ff.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(model.ff.w_scale, reference.ff.w_scale) and torch.equal(model(x), reference(x))
    assert park.switch(model, "@fast", "base", tmp_path / "unused", Sink())  # parked: pointers back, scales too
    assert torch.equal(model(x), base_out)


def test_copy_weights_quantizes_a_fine_tune_into_fp8_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    fp8.convert(model, device="cpu")
    torch.manual_seed(2)
    tuned = {k: torch.randn(v.shape).to(torch.bfloat16) for k, v in _block().state_dict().items()}
    adapters._copy_weights(model, _save(tuned, tmp_path / "tuned"), "tuned")
    q, s = fp8.quantize_weight(tuned["ff.weight"])
    assert torch.equal(model.ff.weight.view(torch.uint8), q.view(torch.uint8)) and torch.equal(model.ff.w_scale, s)
    assert torch.equal(model.adaln_proj.weight, tuned["adaln_proj.weight"])  # a bf16 layer is copied as is


def test_a_lora_factor_given_fp8_by_peft_goes_back_to_bf16():
    model = _block()
    fp8.convert(model, device="cpu")
    model.ff.lora_A = torch.nn.Linear(512, 4, bias=False).to(torch.float8_e4m3fn)  # what PEFT does on an fp8 base
    adapters._lora_to_compute_dtype(model, torch.bfloat16)
    assert model.ff.lora_A.weight.dtype == torch.bfloat16 and model.ff.weight.dtype == torch.float8_e4m3fn


def test_a_peft_lora_wraps_an_fp8_layer_and_still_adds_its_delta(monkeypatch):
    peft = pytest.importorskip("peft")
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    fp8.convert(model, device="cpu")
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    plain = model(x)
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=4, target_modules=["ff"], init_lora_weights=False), model)
    adapters._lora_to_compute_dtype(model, torch.bfloat16)
    assert type(model.ff.base_layer).__name__ == "FP8Linear" and model.ff.lora_A["default"].weight.dtype == torch.bfloat16
    assert not torch.equal(model(x), plain)  # the LoRA's delta is added on top of the fp8 layer
    params = adapters._base_params(model)
    assert "ff.weight" in params and "ff.w_scale" in params  # what the park swaps (LoRA factors left out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 tensor-core matmuls need a CUDA GPU")
def test_scaled_matmul_matches_dequantized():
    model = _block().cuda()
    fp8.convert(model)
    x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    fp8.MATMUL = "dequant"
    slow = model(x)
    fp8.MATMUL = "scaled"
    assert _rel(model(x), slow) < 0.05


def test_a_peft_lora_on_an_fp8_layer_keeps_its_factors_exact():
    """PEFT gives an adapter its base layer's dtype before loading the values: fp8 would round them. The layer's
    ``compute_dtype`` makes it bf16, so the file's factors arrive exactly."""
    peft = pytest.importorskip("peft")
    from peft.utils import set_peft_model_state_dict

    model = _block()
    fp8.convert(model, device="cpu")
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=4, target_modules=["ff"]), model)
    a = (torch.randn(4, 512) * 0.0123).to(torch.bfloat16)
    set_peft_model_state_dict(model, {"ff.lora_A.weight": a, "ff.lora_B.weight": torch.zeros(1024, 4,
                                                                                            dtype=torch.bfloat16)})
    assert model.ff.lora_A["default"].weight.dtype == torch.bfloat16
    assert torch.equal(model.ff.lora_A["default"].weight, a)


def test_a_lora_on_wans_fp8_cast_layers_keeps_its_factors_exact():
    """The same for the layer-wise fp8 casting Wan and Qwen-Image use (``device.fp8_cast``), through diffusers'
    LoRA loader as the workers call it."""
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("peft")

    from aistudio_gpu.tasks.device import fp8_cast

    torch.manual_seed(0)
    t = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=256, ffn_dim=32, num_layers=1, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32).to(torch.bfloat16)
    fp8_cast(t, torch.bfloat16)
    a = (torch.randn(4, 24) * 0.0123).to(torch.bfloat16)
    b = (torch.randn(24, 4) * 0.0123).to(torch.bfloat16)
    t.load_lora_adapter({"blocks.0.attn1.to_q.lora_A.weight": a, "blocks.0.attn1.to_q.lora_B.weight": b},
                        adapter_name="x", prefix=None)
    layer = t.blocks[0].attn1.to_q
    assert layer.base_layer.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(layer.lora_A["x"].weight, a) and torch.equal(layer.lora_B["x"].weight, b)


def test_an_fp8_layer_takes_an_input_cast_to_its_weight_dtype(monkeypatch):
    """T5 casts its feed-forward input to that layer's weight dtype: an fp8 input runs in bf16 instead of failing."""
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    fp8.convert(model, device="cpu")
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    want = model.ff(x.to(torch.float8_e4m3fn).to(torch.bfloat16))
    assert torch.equal(model.ff(x.to(torch.float8_e4m3fn)), want)


def test_t5s_feed_forward_output_stays_bf16():
    class FF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wi_0 = torch.nn.Linear(512, 1024, bias=False)
            self.wo = torch.nn.Linear(1024, 512, bias=False)

    class T5ish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.DenseReluDense = FF()

    m = T5ish().to(torch.bfloat16)
    fp8.convert_parts({"text_encoder": m}, device="cpu")
    assert m.DenseReluDense.wi_0.weight.dtype == torch.float8_e4m3fn
    assert m.DenseReluDense.wo.weight.dtype == torch.bfloat16


def test_a_left_behind_plain_forward_is_dropped(monkeypatch):
    """LongCat's LoRA unhooking leaves ``module.forward = <the Linear's own bound forward>`` on the instance."""
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _block()
    model.ff.forward = model.ff.forward  # what `module.forward = module.org_forward` leaves behind
    fp8.convert(model, device="cpu")
    assert "forward" not in model.ff.__dict__
    model(torch.randn(4, 512, dtype=torch.bfloat16))  # runs the fp8 layer's own forward


# MARK: fp8 copies on disk


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k].contiguous().view(torch.uint8), sb[k].contiguous().view(torch.uint8)), k
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    for k in ba:
        assert torch.equal(ba[k], bb[k]), k


def _tiny_wan():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128, in_channels=16, out_channels=16,
        text_dim=256, freq_dim=256, ffn_dim=512, num_layers=1, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32).to(torch.bfloat16)


def test_a_diffusers_copy_loads_back_exactly_with_a_lora_left_out(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    model = _tiny_wan()
    src = tmp_path / "transformer"
    src.mkdir()
    fp8.convert(model, device="cpu")
    assert fp8.convert(model, device="cpu")["layers"] == 0
    reference = _tiny_wan()
    fp8.convert(reference, device="cpu")
    model.load_lora_adapter({"blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 256, dtype=torch.bfloat16),
                             "blocks.0.attn1.to_q.lora_B.weight": torch.randn(256, 4, dtype=torch.bfloat16)},
                            adapter_name="turbo", prefix=None)
    assert not fp8.copy_ready(src)
    assert fp8.save_copy(model, src) and fp8.copy_ready(src) and not fp8.copy_ready(src, text_encoder=True)
    assert not fp8.save_copy(model, src)  # current: kept
    loaded = fp8.load_copy(src, device="cpu")
    assert type(loaded.blocks[0].attn1.to_q).__name__ == "FP8Linear" and loaded.dtype == torch.bfloat16
    _same(reference, loaded)
    x = dict(hidden_states=torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16), timestep=torch.tensor([500]),
             encoder_hidden_states=torch.randn(1, 8, 256, dtype=torch.bfloat16))
    with torch.no_grad():
        assert torch.equal(reference(**x).sample, loaded(**x).sample)


def test_a_transformers_copy_keeps_tied_weights_tied(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    torch.manual_seed(0)
    config = transformers.T5Config(vocab_size=320, d_model=256, d_kv=32, d_ff=512, num_layers=1, num_heads=4)
    model = transformers.T5EncoderModel(config).to(torch.bfloat16).eval()
    src = tmp_path / "text_encoder"
    fp8.convert_parts({"text_encoder": model}, device="cpu")
    assert model.encoder.block[0].layer[1].DenseReluDense.wi.weight.dtype == torch.float8_e4m3fn
    fp8.save_copy(model, src, text_encoder=True)
    loaded = fp8.load_copy(src, device="cpu")
    _same(model, loaded)
    assert loaded.shared.weight.data_ptr() == loaded.encoder.embed_tokens.weight.data_ptr()
    ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).last_hidden_state, loaded(input_ids=ids).last_hidden_state)


def test_a_stale_copy_is_made_again(tmp_path, monkeypatch):
    model = _tiny_wan()
    fp8.convert(model, device="cpu")
    src = tmp_path / "transformer"
    fp8.save_copy(model, src)
    monkeypatch.setattr(fp8, "COPY_FORMAT", fp8.COPY_FORMAT + 1)
    assert not fp8.copy_ready(src)
    assert fp8.save_copy(model, src) and fp8.copy_ready(src)


def test_copy_weights_takes_an_fp8_copy_as_stored(tmp_path, monkeypatch):
    """A folder that is itself an fp8 copy (weights with their scales) is copied in, not quantized again."""
    monkeypatch.setattr(fp8, "MATMUL", "dequant")
    tuned = _block()
    torch.manual_seed(3)
    for p in tuned.parameters():
        p.data = torch.randn_like(p.float()).to(torch.bfloat16)
    fp8.convert(tuned, device="cpu")
    src = tmp_path / "tuned"
    src.mkdir()
    folder = fp8.copy_dir(src)
    folder.mkdir()
    safetensors_torch.save_file({k: v.contiguous() for k, v in tuned.state_dict().items()},
                                str(folder / "diffusion_pytorch_model.safetensors"))
    model = _block()
    fp8.convert(model, device="cpu")
    adapters._copy_weights(model, folder, "tuned")
    _same(tuned, model)


@pytest.mark.parametrize("version,cap,ok", [("2.6.0+cu124", (9, 0), False), ("2.15.0", (10, 0), True),
                                            ("2.15.0", (8, 0), False)])
def test_scaled_matmul_needs_new_torch_and_gpu(monkeypatch, version, cap, ok):
    # torch 2.6's row-scaled _scaled_mm was twice as slow as bf16 on an H100: dequantize there instead.
    monkeypatch.setattr(fp8, "_capability_ok", None)
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "GPU")
    assert fp8._scaled_ok() is ok
