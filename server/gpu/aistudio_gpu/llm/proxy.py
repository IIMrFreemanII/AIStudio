"""Relaying a chat stream from the engine to the app (Server-Sent Events, as OpenAI sends them), plus:

- while the model is still starting (a cold start can take minutes), ``: ping`` comments every few seconds and
  ``event: aistudio.status`` lines (``{"state", "phase", "progress"}``) whenever the phase changes, so the stream
  stays open and the app can show what's happening;
- before ``data: [DONE]``, ``event: aistudio.usage``: time to first token, tokens per second, tokens, and what the
  request cost (its seconds × the GPUs' price);
- an engine failure becomes ``event: aistudio.error`` (``{"error"}``), then ``[DONE]``.

Clients that ignore named events (any OpenAI client) see a normal stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable

PING_SECONDS = 5.0
DONE = b"data: [DONE]"


def event(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


async def relay(upstream: AsyncIterator[bytes], status: Callable[[], Awaitable[dict]], usd_per_second: float,
                ping_seconds: float | None = None) -> AsyncIterator[bytes]:
    ping_seconds = ping_seconds or PING_SECONDS
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    end = object()

    async def pump():
        try:
            async for chunk in upstream:
                await queue.put(chunk)
            await queue.put(end)
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # noqa: BLE001 — every failure goes to the client
            await queue.put(e)

    task = asyncio.create_task(pump())
    start = time.monotonic()
    first_token: float | None = None
    usage: dict | None = None
    buf = b""
    last_status = None
    started_stream = False
    start_generation = start
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=ping_seconds)
            except asyncio.TimeoutError:
                yield b": ping\n\n"
                if started_stream:
                    continue
                try:
                    s = await status()
                except Exception:  # noqa: BLE001
                    s = None
                if s and s != last_status:
                    last_status = s
                    yield event("aistudio.status", s)
                continue
            if item is end:
                break
            if isinstance(item, BaseException):
                yield event("aistudio.error", {"error": str(item) or type(item).__name__})
                break
            if not started_stream:
                started_stream = True
                start_generation = time.monotonic()
            buf += item
            while b"\n\n" in buf:
                raw, buf = buf.split(b"\n\n", 1)
                stripped = raw.strip()
                if stripped == DONE:
                    continue  # sent after the usage event
                if stripped.startswith(b"data:"):
                    try:
                        d = json.loads(stripped[5:])
                    except ValueError:
                        d = None
                    if isinstance(d, dict):
                        if d.get("usage"):
                            usage = d["usage"]
                        if first_token is None and _has_token(d):
                            first_token = time.monotonic()
                yield raw + b"\n\n"
        if buf.strip() and buf.strip() != DONE:
            yield buf.strip() + b"\n\n"
    finally:
        task.cancel()
    now = time.monotonic()
    completion = int((usage or {}).get("completion_tokens") or 0)
    gen_start = first_token or now
    seconds = now - start
    yield event("aistudio.usage", {
        "seconds": round(seconds, 3),
        "waitSeconds": round((start_generation if started_stream else now) - start, 3),
        "ttftMs": round((gen_start - start) * 1000) if first_token else None,
        "tokPerSec": round(completion / (now - gen_start), 1) if completion and now > gen_start else None,
        "promptTokens": (usage or {}).get("prompt_tokens"), "completionTokens": completion or None,
        "usdPerSecond": usd_per_second, "costUSD": round(seconds * usd_per_second, 6)})
    yield DONE + b"\n\n"


def _has_token(d: dict) -> bool:
    for c in d.get("choices") or []:
        delta = c.get("delta") or {}
        if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False
