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
    if f.engine == "llamacpp":
        # llama.cpp reads tool calls through the model's own chat template (--jinja); "llama.cpp" marks that its
        # reasoning format splits out <think> blocks.
        return {"toolParser": "llama.cpp", "reasoningParser": "llama.cpp" if "<think>" in f.chat_template else None,
                "engineArgs": []}
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


# MARK: thinking effort

_LEVELS = ["low", "medium", "high"]
_SWITCH = ["off", "on"]
_HYBRID = ("qwen3", "glm45", "llama.cpp")


def _switch_key(m: dict) -> str | None:
    """The chat-template argument that turns thinking on or off for a hybrid model, else None."""
    parser = m.get("reasoningParser")
    if parser in _HYBRID:
        return "enable_thinking"
    if parser == "deepseek_r1":
        facts = m.get("facts") or {}
        if str(facts.get("model_type") or "").lower().startswith("deepseek_v3"):
            # The manifest keeps no chat template, so also go by the V3.1+ names (R1 shares the model type).
            name = f"{m.get('repo') or ''} {m.get('name') or ''}".lower()
            if "thinking" in str(facts.get("chat_template") or "") or any(v in name for v in ("v3.1", "v3.2", "v3.5")):
                return "thinking"
    return None


def efforts(m: dict) -> list[str]:
    """The thinking levels a model supports, for the app's menu: [] when it can't be controlled."""
    if m.get("reasoningParser") == "openai_gptoss":
        return list(_LEVELS)
    return list(_SWITCH) if _switch_key(m) else []


def apply_effort(body: dict, m: dict) -> dict:
    """Maps body["reasoning_effort"] to the model's own control, in place, and returns body."""
    effort = body.get("reasoning_effort")
    if effort is None:
        return body
    if m.get("reasoningParser") == "openai_gptoss":
        body["reasoning_effort"] = {"off": "low", "on": "medium"}.get(effort, effort)
        return body
    del body["reasoning_effort"]
    key = _switch_key(m)
    if key:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs[key] = effort not in ("off", "low")
        body["chat_template_kwargs"] = kwargs
    return body
