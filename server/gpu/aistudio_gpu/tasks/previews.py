"""What the diffusers image workers (SDXL, SD 3.5 Large) preview mid-run: the sampler's estimate of the finished
latents. The latents themselves stay mostly noise for much of a run (a Karras schedule; SD 3.5's shifted flow schedule),
so decoding them shows static."""

from __future__ import annotations


def remember_x0(scheduler, slot: dict) -> None:
    """Wraps ``scheduler.step`` (once, at load) to keep the step's estimate of the finished latents in ``slot["x0"]``.

    Euler samplers return it; DPM++ (``dpmsolver++``) keeps it as its last model output; a flow-matching sampler's
    model predicts the velocity ``noise - x0``, so it's ``sample - sigma * velocity``."""
    step = scheduler.step
    flow = type(scheduler).__name__.startswith("FlowMatch")

    def wrapped(*args, **kwargs):
        out = step(*args, **kwargs)
        x0 = out[1] if isinstance(out, tuple) and len(out) > 1 else getattr(out, "pred_original_sample", None)
        if x0 is None and flow:
            velocity = args[0] if args else kwargs["model_output"]
            sample = args[2] if len(args) > 2 else kwargs["sample"]
            sigma = float(scheduler.sigmas[scheduler.step_index - 1])  # the step just taken
            x0 = (sample.float() - sigma * velocity.float()).to(sample.dtype)
        elif x0 is None and getattr(scheduler, "model_outputs", None):
            x0 = scheduler.model_outputs[-1]
        slot["x0"] = x0
        return out

    scheduler.step = wrapped
