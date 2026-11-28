"""graphs.py: per-job CUDA graphs for a transformer's steps (the plumbing; real capture needs a GPU)."""

import pytest

torch = pytest.importorskip("torch")

from aistudio_gpu.tasks import graphs  # noqa: E402


class Step(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.calls = 0

    def forward(self, hidden_states, timestep, num_frames=1, return_dict=False):
        self.calls += 1
        return (self.lin(hidden_states) * timestep[:, None] + num_frames,)


def _fake_recording(monkeypatch):
    """A stand-in graph: replay reruns the step on the recorded buffers and writes into the recorded outputs."""
    from torch.utils._pytree import tree_flatten, tree_unflatten

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    recorded = []

    def record(self, leaves, spec):
        static = [x.clone() if isinstance(x, torch.Tensor) else None for x in leaves]
        args, kwargs = tree_unflatten([s if s is not None else x for s, x in zip(static, leaves)], spec)
        outputs, out_spec = tree_flatten(self.forward(*args, **kwargs))
        forward = self.forward

        class Graph:
            def replay(self):
                new, _ = tree_flatten(forward(*args, **kwargs))
                for o, n in zip(outputs, new):
                    o.copy_(n)

        recorded.append(len(leaves))
        return Graph(), static, outputs, out_spec

    monkeypatch.setattr(graphs.StepGraphs, "_record", record)
    return recorded


def test_first_call_runs_as_usual_then_replays_with_new_inputs(monkeypatch):
    recorded = _fake_recording(monkeypatch)
    m = Step()
    g = graphs.StepGraphs(m, "test")
    x = torch.randn(3, 8)
    outs = []
    for t in (0.1, 0.5, 0.9, 1.3):
        ts = torch.full((3,), t)
        outs.append(m(hidden_states=x, timestep=ts, num_frames=2, return_dict=False)[0])
        assert torch.allclose(outs[-1], m.lin(x) * t + 2)  # each step's own inputs, not the recorded ones
    assert len(recorded) == 1 and g.replays == 3
    outs[1].add_(100)  # a returned tensor is a copy: changing it doesn't touch the graph's buffers
    again = m(hidden_states=x, timestep=torch.full((3,), 0.9), num_frames=2, return_dict=False)[0]
    assert torch.allclose(again, outs[2])
    g.remove()
    assert "forward" not in m.__dict__ and not g.graphs


def test_a_new_shape_or_argument_value_gets_its_own_recording(monkeypatch):
    recorded = _fake_recording(monkeypatch)
    m = Step()
    graphs.StepGraphs(m)
    for _ in range(3):
        m(torch.randn(3, 8), torch.ones(3), num_frames=2)
    for _ in range(3):
        m(torch.randn(5, 8), torch.ones(5), num_frames=2)  # another shape
    for _ in range(3):
        m(torch.randn(5, 8), torch.ones(5), num_frames=4)  # another plain argument
    assert len(recorded) == 3


def test_a_step_that_cannot_be_recorded_runs_as_usual_for_the_rest_of_the_job(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def broken(self, leaves, spec):
        self.broken = True
        return None

    monkeypatch.setattr(graphs.StepGraphs, "_record", broken)
    m = Step()
    g = graphs.StepGraphs(m)
    x, t = torch.randn(2, 8), torch.ones(2)
    for _ in range(4):
        assert torch.allclose(m(x, t)[0], m.lin(x) + 1)
    assert g.broken and m.calls == 4


def test_install_needs_a_gpu_and_the_switch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert graphs.install(Step()) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("AISTUDIO_CUDA_GRAPHS", "0")
    assert graphs.install(Step()) is None
