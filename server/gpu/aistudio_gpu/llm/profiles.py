"""How vLLM should serve a model: its tool-call parser, its reasoning parser, and extra engine flags.

Parsers are guessed from the chat template first (it shows the tool-call format the model was trained on), then
from the model type. Everything here is a default: the install sheet and ``PATCH /v1/llm/models/{id}`` can change it.
"""

from __future__ import annotations

from .facts import Facts

# model_type prefix → (tool parser, reasoning parser, extra flags). The first match wins.
BY_TYPE: list[tuple[str, str | None, str | None, list[str]]] = [
    # Qwen3.8-Flash-Next (Qwen4 preview): vLLM's recipe (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
    ("qwen4", "qwen3_coder", "qwen3", ["--kv-cache-dtype", "fp8"]),
    ("qwen3_5", "qwen3_coder", "qwen3", []),
    ("qwen3_next", "hermes", "qwen3", []),
    ("qwen3", "hermes", "qwen3", []),
    ("qwen2", "hermes", None, []),
    ("deepseek_v3", "deepseek_v3", "deepseek_r1", []),
    ("deepseek", "deepseek_v3", "deepseek_r1", []),
    ("gpt_oss", "openai", "openai_gptoss", []),
    ("glm4_moe", "glm45", "glm45", []),
    ("llama4", "llama4_pythonic", None, []),
    ("llama", "llama3_json", None, []),
    ("mistral", "mistral", None, []),
    ("gemma3", "pythonic", None, []),
]


def guess(f: Facts) -> dict:
    """{"toolParser", "reasoningParser", "engineArgs"} for a model."""
    mt = f.model_type.lower()
    tool, reasoning, args = None, None, []
    for prefix, t, r, a in BY_TYPE:
        if mt.startswith(prefix):
            tool, reasoning, args = t, r, list(a)
            break
    tpl = f.chat_template
    if tpl:
        if "<function=" in tpl and "<tool_call>" in tpl:
            tool = "qwen3_coder"
        elif "<tool_call>" in tpl and tool is None:
            tool = "hermes"
        if "<think>" in tpl and reasoning is None:
            reasoning = "deepseek_r1" if "deepseek" in mt else "qwen3"
    if tool is None:
        tool = "hermes"
    return {"toolParser": tool, "reasoningParser": reasoning, "engineArgs": args}


def kv_dtype_bytes(engine_args: list[str]) -> int:
    for i, a in enumerate(engine_args):
        if a.startswith("--kv-cache-dtype"):
            value = a.split("=", 1)[1] if "=" in a else (engine_args[i + 1] if i + 1 < len(engine_args) else "")
            return 1 if value.startswith("fp8") else 2
    return 2
