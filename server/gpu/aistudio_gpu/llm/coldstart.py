"""Cold starts, timed: how long a model took from the request to answering, split into phases (waiting for a GPU,
starting the engine, loading weights, compiling, capturing CUDA graphs…). Each boot's record goes into a short
history per model, so the app can show the last one and say how long a start usually takes."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

HISTORY = 10  # records kept per model


class ColdStartClock:
    """Watches one boot's phases (the same ``on_status`` calls the engine makes) and makes its record when it's
    ready. ``requested`` is when the front desk asked for the container (the GPU wait is the time before ``started``)."""

    def __init__(self, requested: float | None = None, started: float | None = None,
                 now: Callable[[], float] = time.time, how: str = "cold"):
        self.now = now
        self.started = started if started is not None else now()
        self.requested = requested if requested is not None and requested <= self.started else None
        self.how = how  # "cold", or "snapshot" when restored from a memory snapshot
        self.phases: list[list] = []  # [name, start]
        if self.requested is not None:
            # A container was asked for: the wait for a GPU, then the container starting up (until the engine's
            # first phase).
            # From a snapshot, the clock starts after the restore: the wait includes it.
            wait = "Waiting for a GPU" if how == "cold" else "Getting a GPU and restoring the snapshot"
            if self.started - self.requested >= 0.5:
                self.phases.append([wait, self.requested])
            if how == "cold":
                self.phases.append(["Starting container", self.started])

    def phase(self, name: str) -> None:
        if name == "Ready" or (self.phases and self.phases[-1][0] == name):
            return
        self.phases.append([name, self.now()])

    def finish(self, **extra) -> dict:
        end = self.now()
        out: list[dict] = []
        seen: dict[str, float] = {}
        for i, (name, start) in enumerate(self.phases):
            stop = self.phases[i + 1][1] if i + 1 < len(self.phases) else end
            seen[name] = seen.get(name, 0.0) + max(0.0, stop - start)
        for name, _ in self.phases:  # in first-seen order, repeats merged, blinks left out
            if name in seen:
                secs = round(seen.pop(name), 1)
                if secs > 0:
                    out.append({"name": name, "seconds": secs})
        begin = self.requested if self.requested is not None else self.started
        return {"at": round(begin, 1), "seconds": round(end - begin, 1),
                "waitSeconds": round(self.started - self.requested, 1) if self.requested is not None else 0.0,
                "how": self.how, "phases": out, **extra}


def remember(history: list | None, record: dict) -> list:
    """The history with ``record`` added (newest last, at most ``HISTORY``)."""
    return (list(history or []) + [record])[-HISTORY:]


def summary(history: list | None, updated: float | None = None) -> dict | None:
    """What the app shows: the last start, and the typical one (median of the last five made with the model's
    current settings, else of any)."""
    history = [h for h in (history or []) if isinstance(h, dict) and h.get("seconds") is not None]
    if not history:
        return None
    current = [h for h in history if updated is None or (h.get("updated") or 0) >= updated] or history
    recent = [h["seconds"] for h in current[-5:]]
    return {"last": history[-1], "typicalSeconds": round(statistics.median(recent), 1), "count": len(history)}
