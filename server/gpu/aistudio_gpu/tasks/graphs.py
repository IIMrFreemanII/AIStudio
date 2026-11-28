"""CUDA graphs for a transformer's denoising steps: the step is recorded once and replayed, so the CPU no longer
launches its thousands of kernels one by one. A short LTX clip spent most of its sampling time there (B200 profile,
2026-11-28: the GPU busy 1.8 s of 6.5 s, 109k launches), more so in fp8/fp4, whose layers launch more kernels.

Per job (``StepGraphs`` is installed for one pipeline call and removed after it), so the weights, LoRAs and modes
a graph reads can't change under it. Within the job, the first call with a given signature (tensor shapes, dtypes,
strides; every other argument's value) runs as usual; the second records that call and replays it; later calls copy
their inputs into the graph's buffers and replay. A call that can't be recorded (a CPU sync inside, say) turns the
graphs off for the rest of the job. ``AISTUDIO_CUDA_GRAPHS=0`` turns them off."""

from __future__ import annotations

import inspect
import os
import time

_stream = None  # the stream every capture runs on (its library handles warmed up once per process)
_warm = False


def enabled() -> bool:
    return os.environ.get("AISTUDIO_CUDA_GRAPHS", "1") != "0"


def _signature(x):
    import torch

    if isinstance(x, torch.Tensor):
        return ("tensor", tuple(x.shape), x.dtype, x.stride(), str(x.device))
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


class StepGraphs:
    """Replaces ``module.forward`` with recorded replays until ``remove``."""

    def __init__(self, module, label: str = "step"):
        self.module, self.label = module, label
        self.forward = module.forward
        try:  # some pipelines pick the arguments they pass by the forward's signature (H3's denoisers)
            self.__signature__ = inspect.signature(self.forward)
        except (TypeError, ValueError):
            pass
        self.seen: dict = {}
        self.graphs: dict = {}
        self.broken = False
        self.replays = 0
        self.capture_seconds = 0.0
        self.pool = None
        module.forward = self

    def remove(self) -> None:
        import torch

        if self.module.__dict__.get("forward") is self:
            del self.module.__dict__["forward"]
        if self.graphs:
            torch.cuda.synchronize()
            print(f"graphs: {self.label}: {len(self.graphs)} recorded in {self.capture_seconds:.1f} s, "
                  f"{self.replays} replays", flush=True)
        self.graphs.clear()
        self.pool = None

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten

        if self.broken or torch.cuda.is_current_stream_capturing():
            return self.forward(*args, **kwargs)
        leaves, spec = tree_flatten((args, kwargs))
        key = (str(spec), tuple(_signature(x) for x in leaves))
        entry = self.graphs.get(key)
        if entry is None:
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.seen[key] == 1:
                return self.forward(*args, **kwargs)
            entry = self._record(leaves, spec)
            if entry is None:
                return self.forward(*args, **kwargs)
            self.graphs[key] = entry
        graph, static, outputs, out_spec = entry
        for dst, src in zip(static, leaves):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        self.replays += 1
        return tree_unflatten([o.clone() if isinstance(o, torch.Tensor) else o for o in outputs], out_spec)

    def _record(self, leaves, spec):
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten

        global _stream, _warm
        start = time.monotonic()
        static = [x.clone() if isinstance(x, torch.Tensor) else None for x in leaves]
        args, kwargs = tree_unflatten([s if s is not None else x for s, x in zip(static, leaves)], spec)
        try:
            if _stream is None:
                _stream = torch.cuda.Stream()
            _stream.wait_stream(torch.cuda.current_stream())
            if not _warm:  # cuBLAS/cuDNN handles for the capture stream, made once outside any capture
                with torch.cuda.stream(_stream):
                    self.forward(*args, **kwargs)
                _warm = True
            torch.cuda.current_stream().wait_stream(_stream)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=_stream):
                out = self.forward(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — the job carries on without graphs
            print(f"graphs: {self.label} can't be recorded ({type(e).__name__}: {e}); running without", flush=True)
            self.broken = True
            torch.cuda.synchronize()
            return None
        outputs, out_spec = tree_flatten(out)
        self.capture_seconds += time.monotonic() - start
        return graph, static, outputs, out_spec


def install(module, label: str = "step"):
    """``StepGraphs`` on ``module`` when CUDA graphs are on and there's a CUDA GPU, else None."""
    import torch

    if not enabled() or not torch.cuda.is_available():
        return None
    return StepGraphs(module, label)
