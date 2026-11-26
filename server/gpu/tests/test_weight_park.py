"""adapters.WeightPark: a mode change points the transformer at the other mode's tensors kept on the device."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from aistudio_gpu.tasks import adapters  # noqa: E402


class Sink:
    def __init__(self):
        self.statuses = []

    def status(self, s):
        self.statuses.append(s)


def _save(module_like: dict, folder: Path) -> Path:
    folder.mkdir(parents=True)
    safetensors_torch.save_file(module_like, str(folder / "diffusion_pytorch_model.safetensors"))
    return folder


def test_switch_loads_once_then_swaps_pointers(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "_room_for", lambda params: True)
    model = torch.nn.Linear(4, 4)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    fast = {k: torch.full_like(v, 7.0) for k, v in base.items()}
    fast_dir = _save(fast, tmp_path / "fast")
    base_dir = _save(base, tmp_path / "base")
    park, sink = adapters.WeightPark(), Sink()

    assert park.switch(model, "base", "@fast", fast_dir, sink)  # read once, beside the base weights
    assert torch.equal(model.weight, fast["weight"]) and set(park.parked) == {"base"}
    fast_tensor = model.weight.data
    assert park.switch(model, "@fast", "base", base_dir, Sink())  # parked: no read
    assert torch.equal(model.weight, base["weight"]) and set(park.parked) == {"@fast"}
    reads = Sink()
    assert park.switch(model, "base", "@fast", fast_dir, reads)
    assert model.weight.data.data_ptr() == fast_tensor.data_ptr() and reads.statuses == []  # the same tensor again


def test_no_room_leaves_the_module_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "_room_for", lambda params: False)
    model = torch.nn.Linear(2, 2)
    before = model.weight.data.clone()
    folder = _save({k: torch.zeros_like(v) for k, v in model.state_dict().items()}, tmp_path / "x")
    assert not adapters.WeightPark().switch(model, "base", "@fast", folder, Sink())
    assert torch.equal(model.weight, before)


def test_a_file_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "_room_for", lambda params: True)
    model = torch.nn.Linear(2, 2)
    folder = _save({"weight": torch.zeros(2, 2)}, tmp_path / "x")  # no bias
    with pytest.raises(ValueError, match="missing"):
        adapters.WeightPark().switch(model, "base", "@fast", folder, Sink())
    folder = _save({"weight": torch.zeros(2, 2), "bias": torch.zeros(2), "extra": torch.zeros(1)}, tmp_path / "y")
    with pytest.raises(ValueError, match="unexpected"):
        adapters.WeightPark().switch(model, "base", "@fast", folder, Sink())
