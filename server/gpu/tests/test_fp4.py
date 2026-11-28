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


def test_fp4_parts_store_no_copies_and_never_load_fp8_ones(tmp_path, monkeypatch):
    m = _block()
    fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4")
    src = tmp_path / "transformer"
    src.mkdir()
    assert fp8.save_copy(m, src) is False and not fp8.copy_dir(src).exists()
    monkeypatch.setenv("AISTUDIO_PRECISION_LTX", "fp4")
    assert fp8.copies_usable({"transformer": src}, "AISTUDIO_PRECISION_LTX") is False


def test_modes_cannot_be_swapped_into_fp4_layers(tmp_path):
    m = _block()
    fp4.convert(m, fp8.TRANSFORMER_KEEP, "cpu", "fp4w")
    with pytest.raises(NotImplementedError, match="fp4"):
        adapters._copy_weights(m, tmp_path, "full")
