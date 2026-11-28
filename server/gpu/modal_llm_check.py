"""Starts installed LLMs on the deployed app the way the app does, sends a few fixed prompts at temperature 0 and
reports the cold start, time to first token and decode speed, then stops each container. For comparing a model with
its NVFP4 version (``modal_quantize.py``): run both, compare the answers and the speeds.

    python server/gpu/modal_llm_check.py qwen3-4b-test qwen3-4b-test-nvfp4 [--out answers.json] [--tokens 400]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import modal

APP = "aistudio-gpu"
VOLUME = "aistudio-gpu-models"
PROMPTS = [
    "Explain in three short paragraphs why the sky is blue, for a curious ten-year-old.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.",
    "A train leaves at 14:35 and arrives at 18:10. How long is the trip? Answer with the reasoning in two lines.",
]


def manifest(mid: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "llm.json"
        vol = modal.Volume.from_name(VOLUME)
        with open(path, "wb") as f:
            for chunk in vol.read_file(f"llm/{mid}/llm.json"):
                f.write(chunk)
        return json.loads(path.read_text())


def worker(m: dict):
    n = int(m.get("count") or 1)
    gpu = m["gpu"] if n == 1 else f"{m['gpu']}:{n}"
    if m.get("snapshot") and n == 1:
        return modal.Cls.from_name(APP, "LLMSnap").with_options(gpu=gpu)(model=m["id"], rev=str(m.get("updated") or ""))
    return modal.Cls.from_name(APP, "LLM").with_options(gpu=gpu)(model=m["id"])


def ask(obj, mid: str, prompt: str, tokens: int) -> dict:
    body = {"model": mid, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    start = time.monotonic()
    first = None
    text, usage = [], {}
    buf = b""
    for chunk in obj.chat.remote_gen(body):
        buf += chunk if isinstance(chunk, bytes) else str(chunk).encode()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            for c in ev.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    first = first or time.monotonic()
                    text.append(piece)
            usage = ev.get("usage") or usage
    end = time.monotonic()
    out_tokens = int(usage.get("completion_tokens") or 0)
    decode = (out_tokens - 1) / (end - first) if first and out_tokens > 1 and end > first else None
    return {"prompt": prompt, "text": "".join(text), "seconds": round(end - start, 2),
            "firstToken": round(first - start, 2) if first else None, "tokens": out_tokens,
            "tokensPerSecond": round(decode, 1) if decode else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--out")
    a = p.parse_args()
    report = {}
    for mid in a.models:
        m = manifest(mid)
        obj = worker(m)
        print(f"== {mid} on {m['gpu']} ({(m.get('weightBytes') or 0) / 1e9:.1f} GB)", flush=True)
        t = time.monotonic()
        try:
            answers = [ask(obj, mid, q, a.tokens) for q in PROMPTS]
            cold = answers[0]["firstToken"]
            for r in answers:
                print(f"  {r['tokens']} tokens, first {r['firstToken']} s, {r['tokensPerSecond']} tok/s: "
                      f"{r['text'][:120]!r}", flush=True)
            report[mid] = {"gpu": m["gpu"], "weightBytes": m.get("weightBytes"), "coldFirstToken": cold,
                           "answers": answers, "wall": round(time.monotonic() - t, 1)}
        finally:
            try:
                obj.stop.remote()
            except Exception as e:  # noqa: BLE001
                print(f"  stop failed: {e}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
