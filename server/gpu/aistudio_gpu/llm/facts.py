"""What a model link points at, read from Hugging Face without downloading the weights: the repo's file list (sizes),
``config.json`` and the chat template.

Links: ``owner/name``, ``owner/name@revision``, ``https://huggingface.co/owner/name`` (also ``/tree/<rev>``,
``/blob/<rev>/…``, ``hf.co``). GGUF files and Civitai links are recognised and refused for now with a reason.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from urllib.parse import urlsplit

import httpx

HF_HOSTS = ("huggingface.co", "hf.co")
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
# Files an engine needs besides the weights (tokenizer, chat template, processors for vision models).
SIDE_FILES = ("*.json", "*.jinja", "tokenizer*", "*.model", "*.tiktoken", "merges.txt", "vocab.txt")


class LinkError(ValueError):
    """The link can't be used (with a reason the app shows as is)."""


@dataclass
class ModelRef:
    repo: str
    revision: str = "main"

    @property
    def label(self) -> str:
        return self.repo if self.revision == "main" else f"{self.repo}@{self.revision}"


@dataclass
class Facts:
    repo: str
    revision: str
    sha: str | None
    architecture: str
    model_type: str
    weight_bytes: int
    files: int
    quant: dict  # {"method": "fp8" | "awq" | "gptq" | "compressed-tensors" | …, "bits": 8} or {}
    dtype: str
    native_context: int
    vision: bool
    layers: int
    attention_layers: int  # layers with a growing KV cache (full attention)
    attention_heads: int
    kv_heads: int
    head_dim: int
    kv_lora_rank: int  # MLA (DeepSeek): the latent's width, else 0
    rope_head_dim: int
    sliding_window: int  # 0 = none
    experts: int
    gated: bool
    chat_template: str = ""
    params: int | None = None
    notes: list[str] = field(default_factory=list)
    download: list[tuple[str, int]] = field(default_factory=list)  # the files an install fetches, with sizes

    def public(self) -> dict:
        d = asdict(self)
        d.pop("chat_template")
        d.pop("download")
        d["hasChatTemplate"] = bool(self.chat_template)
        return {_camel(k): v for k, v in d.items()}


def _camel(k: str) -> str:
    head, *rest = k.split("_")
    return head + "".join(w.title() for w in rest)


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def parse_link(link: str) -> ModelRef:
    """``owner/name[@rev]`` or a Hugging Face URL → the repo and revision. Raises ``LinkError`` with a reason."""
    link = (link or "").strip()
    if not link:
        raise LinkError("paste a Hugging Face model link or owner/name")
    if "://" not in link and not link.startswith(("huggingface.co/", "hf.co/", "civitai.com/")):
        repo, _, rev = link.partition("@")
        if REPO.match(repo) and (not rev or REVISION.match(rev)):
            return ModelRef(repo, rev or "main")
        raise LinkError(f"{link!r} isn't a Hugging Face repo (owner/name)")
    parts = urlsplit(link if "://" in link else "https://" + link)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if host == "civitai.com" or host.endswith(".civitai.com"):
        raise LinkError("Civitai LLM links aren't supported yet (they come with GGUF support in the next "
                        "milestone); use the model's Hugging Face repo")
    if host not in HF_HOSTS and host != urlsplit(endpoint()).hostname:
        raise LinkError(f"only Hugging Face links for now, not {host or link!r}")
    segs = [s for s in parts.path.split("/") if s]
    if segs and segs[0] in ("models",):
        segs = segs[1:]
    if len(segs) < 2:
        raise LinkError("the link should point at a model: huggingface.co/owner/name")
    repo = f"{segs[0]}/{segs[1]}"
    rev = "main"
    if len(segs) >= 4 and segs[2] in ("tree", "blob", "resolve"):
        rev = segs[3]
        if segs[-1].lower().endswith(".gguf"):
            raise LinkError("GGUF files aren't supported yet (next milestone); use a safetensors repo, such as "
                            "the model's FP8 or AWQ version")
    if not REPO.match(repo) or not REVISION.match(rev):
        raise LinkError(f"{link!r} isn't a Hugging Face model link")
    return ModelRef(repo, rev)


async def inspect(ref: ModelRef, token: str | None = None, client: httpx.AsyncClient | None = None) -> Facts:
    """Reads the repo's metadata and config (no weights). Raises ``LinkError`` for links that can't be served."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
    headers = {"User-Agent": "aistudio-gpu"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        base = endpoint()
        r = await client.get(f"{base}/api/models/{ref.repo}/revision/{ref.revision}",
                             params={"blobs": "true"}, headers=headers)
        if r.status_code in (401, 403):
            raise LinkError(f"{ref.repo} is private or gated: accept its license on Hugging Face and add your "
                            "token in Models › Hugging Face token")
        if r.status_code == 404:
            raise LinkError(f"no model {ref.label} on Hugging Face")
        r.raise_for_status()
        meta = r.json()
        siblings = meta.get("siblings") or []
        names = {s.get("rfilename", ""): int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
                 for s in siblings}
        weights = {n: b for n, b in names.items() if n.endswith(".safetensors") and "/" not in n}
        if not weights:
            if any(n.lower().endswith(".gguf") for n in names):
                raise LinkError("this repo only has GGUF files, which aren't supported yet (next milestone)")
            if any(n.endswith((".bin", ".pt", ".pth")) for n in names):
                raise LinkError("this repo has no .safetensors weights (only pickled .bin files, which the server "
                                "doesn't load)")
            raise LinkError("this repo has no model weights at its top level")
        if "config.json" not in names:
            raise LinkError("this repo has no config.json, so it isn't a complete model")

        async def text(name: str) -> str:
            resp = await client.get(f"{base}/{ref.repo}/resolve/{meta.get('sha') or ref.revision}/{name}",
                                    headers=headers)
            if resp.status_code in (401, 403):
                raise LinkError(f"{ref.repo} is gated: accept its license on Hugging Face and add your token in "
                                "Models › Hugging Face token")
            resp.raise_for_status()
            return resp.text

        config = json.loads(await text("config.json"))
        template = ""
        if "chat_template.jinja" in names:
            template = await text("chat_template.jinja")
        elif "tokenizer_config.json" in names:
            tc = json.loads(await text("tokenizer_config.json"))
            t = tc.get("chat_template")
            template = t if isinstance(t, str) else json.dumps(t or "")
        f = facts_from_config(ref, meta, config, weights, template)
        f.download = sorted((n, b) for n, b in names.items()
                            if "/" not in n and (n in weights or any(fnmatch(n, p) for p in SIDE_FILES)))
        return f
    except httpx.HTTPError as e:
        raise LinkError(f"couldn't read {ref.label} from Hugging Face ({type(e).__name__})") from None
    finally:
        if own:
            await client.aclose()


def facts_from_config(ref: ModelRef, meta: dict, config: dict, weights: dict[str, int], template: str) -> Facts:
    """The parts of ``config.json`` sizing and the engine care about (text config of multimodal models included)."""
    text = config.get("text_config") or config.get("llm_config") or config
    merged = {**config, **text}
    arch = (config.get("architectures") or ["?"])[0]
    layers = int(merged.get("num_hidden_layers") or merged.get("n_layer") or 0)
    heads = int(merged.get("num_attention_heads") or merged.get("n_head") or 0)
    kv = int(merged.get("num_key_value_heads") or heads)
    hidden = int(merged.get("hidden_size") or 0)
    head_dim = int(merged.get("head_dim") or (hidden // heads if heads else 0))
    types = merged.get("layer_types")
    if isinstance(types, list) and types:
        attn = sum(1 for t in types if t in ("full_attention", "attention", "sliding_attention"))
    elif merged.get("full_attention_interval"):
        attn = layers // int(merged["full_attention_interval"])
    else:
        attn = layers
    window = int(merged.get("sliding_window") or 0) if merged.get("use_sliding_window", True) else 0
    qc = config.get("quantization_config") or text.get("quantization_config") or {}
    quant = {}
    if qc:
        method = str(qc.get("quant_method") or qc.get("format") or "?").lower()
        bits = qc.get("bits") or qc.get("weight_bits")
        if bits is None and method == "fp8":
            bits = 8
        if bits is None and method == "compressed-tensors":
            groups = (qc.get("config_groups") or {}).values()
            for g in groups:
                bits = (g.get("weights") or {}).get("num_bits") or bits
        quant = {"method": method, "bits": int(bits) if bits else None}
    dtype = str(merged.get("torch_dtype") or merged.get("dtype") or "bfloat16")
    native = int(merged.get("max_position_embeddings") or merged.get("max_sequence_length")
                 or merged.get("seq_length") or 4096)
    experts = int(merged.get("num_experts") or merged.get("n_routed_experts") or merged.get("num_local_experts") or 0)
    vision = bool(config.get("vision_config") or config.get("visual") or "image_token_id" in config
                  or "VL" in arch or "ConditionalGeneration" in arch and not config.get("language_model_only"))
    params = (meta.get("safetensors") or {}).get("total")
    notes = []
    if quant.get("method") == "fp8":
        notes.append("FP8 weights: fastest on L40S, H100, H200 and B200 (A100 runs them slower)")
    return Facts(
        repo=ref.repo, revision=ref.revision, sha=meta.get("sha"), architecture=arch,
        model_type=str(config.get("model_type") or text.get("model_type") or "?"),
        weight_bytes=sum(weights.values()), files=len(weights), quant=quant, dtype=dtype, native_context=native,
        vision=vision, layers=layers, attention_layers=attn, attention_heads=heads, kv_heads=kv, head_dim=head_dim,
        kv_lora_rank=int(merged.get("kv_lora_rank") or 0), rope_head_dim=int(merged.get("qk_rope_head_dim") or 0),
        sliding_window=window, experts=experts, gated=bool(meta.get("gated")), chat_template=template,
        params=int(params) if params else None, notes=notes)
