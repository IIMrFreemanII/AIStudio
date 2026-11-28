"""fp4.py (experimental): NVFP4 weights for the big models' transformers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from aistudio_gpu.tasks import adapters, fp4, fp8  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = torch.nn.Linear(512, 512)
        self.ff = torch.nn.Linear(512, 1024)
        self.out = torch.nn.Linear(1024, 512)
        self.tiny = torch.nn.Linear(512, 64)

    def forward(self, x):
        return self.tiny(self.out(torch.relu(self.ff(self.x_embedder(x)))))


def _block():
    torch.manual_seed(0)
    return Block().to(torch.bfloat16).eval()


def test_values_round_trip_and_ties_go_to_even():
    v = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6, 6]], dtype=torch.float32)
    q, bs, ts = fp4.quantize(v)
    assert q.dtype == torch.float4_e2m1fn_x2 and q.shape == (1, 8) and bs.shape == (1, 1)
    assert torch.allclose(fp4.dequantize(q, bs, ts, torch.float32), v, rtol=1e-3, atol=1e-3)
    assert fp4.codes(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert fp4.codes(torch.tensor([-0.75])).tolist() == [8 | 2]


def test_weight_error_is_about_a_tenth():
    torch.manual_seed(0)
    w = torch.randn(256, 1024) * 0.02
    q, bs, ts = fp4.quantize(w)
    err = ((fp4.dequantize(q, bs, ts, torch.float32) - w).norm() / w.norm()).item()
    assert 0.05 < err < 0.12


def test_blocked_layout_pads_to_whole_tiles():
    s = torch.arange(130 * 6, dtype=torch.float32).view(130, 6)
    b = fp4.blocked(s)
    assert b.numel() == 256 * 8  # 2 row tiles of 128, 2 column tiles of 4
    assert b[0].item() == 0 and b[1].item() == 1 and b[4].item() == 32 * 6  # row 32 sits after row 0's four


def test_convert_keeps_the_steering_and_small_layers_and_runs_dequantized():
    m = _block()
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    ref = m(x).float()
    info = fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4w")
    assert info["layers"] == 2 and type(m.ff).__name__ == "FP4Linear" and type(m.out).__name__ == "FP4Linear"
    assert type(m.x_embedder) is torch.nn.Linear and type(m.tiny) is torch.nn.Linear
    assert m.ff.weight.dtype == torch.float4_e2m1fn_x2 and m.ff.weight.shape == (1024, 256)
    assert m.dtype == torch.bfloat16 and fp4.has_fp4(m)
    err = ((m(x).float() - ref).norm() / ref.norm()).item()
    assert err < 0.2


def test_convert_parts_puts_transformers_in_fp4_and_text_encoders_in_fp8():
    parts = {"transformer": _block(), "text_encoder": _block()}
    fp8.convert_parts(parts, device="cpu", mode="fp4w")
    assert type(parts["transformer"].ff).__name__ == "FP4Linear"
    assert type(parts["text_encoder"].ff).__name__ == "FP8Linear"


def test_precision_setting(monkeypatch):
    for value, want, quantized in [("fp8", "fp8", True), ("bf16", "bf16", False), ("FP4", "fp4", True),
                                   ("fp4w", "fp4w", True), ("nonsense", "fp8", True)]:
        monkeypatch.setenv("AISTUDIO_PRECISION_LTX", value)
        assert fp8.precision("AISTUDIO_PRECISION_LTX") == want
        assert fp8.wanted("AISTUDIO_PRECISION_LTX") is quantized


def _tiny_wan():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128, in_channels=16, out_channels=16,
        text_dim=256, freq_dim=256, ffn_dim=512, num_layers=1, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32).to(torch.bfloat16).eval()


def _bytes(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t


def test_an_fp4_copy_loads_back_exactly_in_either_mode(tmp_path):
    model = _tiny_wan()
    assert fp4.convert(model, fp8.TRANSFORMER_KEEP, "cpu", "fp4w")["layers"] > 0
    src = tmp_path / "transformer"
    assert not fp8.copy_ready(src, kind="fp4")
    assert fp8.save_copy(model, src) and fp8.copy_ready(src, kind="fp4")
    assert not fp8.copy_ready(src) and not fp8.copy_dir(src).exists()  # its own folder, not the fp8 one
    assert not fp8.save_copy(model, src)  # current: kept
    assert not any("w_blocked" in k for f in fp8.copy_dir(src, "fp4").glob("model-*.safetensors")
                   for k in __import__("safetensors.torch").torch.load_file(str(f)))
    for mode in ("fp4w", "fp4"):
        loaded = fp8.load_copy(src, device="cpu", kind="fp4", mode=mode)
        layer = loaded.blocks[0].attn1.to_q
        assert type(layer).__name__ == "FP4Linear" and layer.fp4_mode == mode and loaded.dtype == torch.bfloat16
        assert layer.weight.dtype == torch.float4_e2m1fn_x2
        a, b = model.state_dict(), loaded.state_dict()
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(_bytes(a[k]), _bytes(b[k])), k
    x = dict(hidden_states=torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16), timestep=torch.tensor([500]),
             encoder_hidden_states=torch.randn(1, 8, 256, dtype=torch.bfloat16))
    with torch.no_grad():
        assert torch.equal(model(**x).sample, loaded(**x).sample)  # fp4 falls back to dequantized on the CPU


def test_fp4_modes_read_the_fp4_transformer_copy_and_the_fp8_text_encoder_copy(tmp_path, monkeypatch):
    transformer, text = tmp_path / "transformer", tmp_path / "text_encoder"
    m = _block()
    fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4")
    fp8.save_copy(m, transformer)
    t = _block()
    fp8.convert(t, fp8.TEXT_KEEP, "cpu")
    fp8.save_copy(t, text, text_encoder=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: type("P", (), {"total_memory": 180 * 2**30})())
    sources = {"transformer": transformer, "text_encoder": text}
    assert fp8.copy_kind("transformer", "fp4") == "fp4" and fp8.copy_kind("text_encoder", "fp4w") == "fp8"
    for mode, usable in (("fp4", True), ("fp4w", True), ("fp8", False), ("bf16", False)):
        monkeypatch.setenv("AISTUDIO_PRECISION_LTX", mode)
        assert fp8.copies_usable(sources, "AISTUDIO_PRECISION_LTX") is usable, mode  # fp8 lacks its transformer


def test_a_stale_fp4_copy_is_made_again(tmp_path, monkeypatch):
    m = _block()
    fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4")
    src = tmp_path / "transformer"
    fp8.save_copy(m, src)
    monkeypatch.setattr(fp4, "COPY_FORMAT", fp4.COPY_FORMAT + 1)
    assert not fp8.copy_ready(src, kind="fp4")
    assert fp8.save_copy(m, src) and fp8.copy_ready(src, kind="fp4")


def test_modes_cannot_be_swapped_into_fp4_layers(tmp_path):
    m = _block()
    fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4w")
    with pytest.raises(NotImplementedError, match="fp4"):
        adapters._copy_weights(m, tmp_path, "full")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel and fp4 matmuls need a CUDA GPU")


@cuda
@pytest.mark.parametrize("m,k", [(3840, 4096), (130, 2048), (1, 4096), (257, 16384), (33, 96)])
def test_triton_kernel_matches_the_reference_bit_for_bit(m, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a[:, : k // 50] *= 40
    a[: max(1, m // 7), 16:32] = 0
    q1, bs, t1 = fp4.quantize(a)
    q2, s2, t2 = fp4.quantize_rows_fast(a)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(t1.reshape(1), t2)
    assert torch.equal(fp4.blocked(bs).view(torch.uint8), s2.view(torch.uint8))


@cuda
def test_fp4_layer_on_the_gpu_is_close_to_bf16():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 640, 4096, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    fp4.convert(torch.nn.Sequential(lin), (), "cuda", "fp4")
    err = ((lin(x).float() - ref).norm() / ref.norm()).item()
    assert err < 0.2
