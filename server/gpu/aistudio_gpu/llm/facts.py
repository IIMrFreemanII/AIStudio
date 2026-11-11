"""What a model link points at, read without downloading the weights.

- Hugging Face, safetensors (vLLM): the repo's file list (sizes), ``config.json`` and the chat template.
- Hugging Face, GGUF (llama.cpp): the repo's GGUF files grouped by quantization (split files summed), and the chosen
  one's header, read with HTTP Range requests. A ``mmproj`` file next to them gives vision.
- Civitai: the model version's files. A GGUF file runs on llama.cpp; a zip of a Hugging Face model folder on vLLM
  (its config.json is read from inside the zip, remotely); a bare safetensors file is refused (it needs its base).

Links: ``owner/name``, ``owner/name@revision``, ``https://huggingface.co/owner/name`` (also ``/tree/<rev>``,
``/blob/<rev>/file.gguf``, ``hf.co``), ``https://civitai.com/models/<id>`` (``?modelVersionId=``),
``https://civitai.com/api/download/models/<versionId>``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from urllib.parse import parse_qs, urlsplit

import httpx

from . import gguf
from .remote import RangeError, RangeFile, auth_headers

log = logging.getLogger(__name__)

HF_HOSTS = ("huggingface.co", "hf.co")
CIVITAI_API = "https://civitai.com/api/v1"
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
# Files an engine needs besides the weights (tokenizer, chat template, processors for vision models).
SIDE_FILES = ("*.json", "*.jinja", "tokenizer*", "*.model", "*.tiktoken", "merges.txt", "vocab.txt")
GATED = "is private or gated: accept its license on Hugging Face and add your token in Models › Hugging Face token"


class LinkError(ValueError):
    """The link can't be used (with a reason the app shows as is)."""


@dataclass
class ModelRef:
    repo: str
    revision: str = "main"
    file: str | None = None  # a GGUF file the link points at

    @property
    def label(self) -> str:
        return self.repo if self.revision == "main" else f"{self.repo}@{self.revision}"


@dataclass
class CivitaiRef:
    model_id: int | None
    version_id: int | None


@dataclass
class Facts:
    repo: str
    revision: str
    sha: str | None
    architecture: str
    model_type: str
    weight_bytes: int
    files: int
    quant: dict  # {"method": "fp8" | "awq" | "gptq" | "gguf" | …, "bits": 8, "type": "Q4_K_M"} or {}
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
    engine: str = "vllm"  # or "llamacpp" (GGUF)
    source: str = "huggingface"  # or "civitai"
    gguf_file: str | None = None  # the first (or only) file of the chosen GGUF, as named in `download`
    mmproj: str | None = None
    gguf_options: list[dict] = field(default_factory=list)  # [{"quant", "bytes"}] when the repo has several
    # What an install fetches: [{"name", "size", "url"?, "sha256"?, "extract"?}] ("extract": a zip's folder to unpack)
    download: list[dict] = field(default_factory=list)
    # Weights vLLM keeps in CPU memory instead of on the GPUs: the n-gram ("Engram") embedding tables of models like
    # Qwen3.8-Flash-Next (51 GB of its 185 GB). Sizing leaves them out of each GPU's share.
    cpu_bytes: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d.pop("chat_template")
        d.pop("download")
        d["hasChatTemplate"] = bool(self.chat_template)
        return {_camel(k): v for k, v in d.items()}

    def stored(self) -> dict:
        """What the manifest keeps, to plan GPUs again later (no template, no file list)."""
        d = asdict(self)
        d.pop("chat_template")
        d.pop("download")
        d.pop("gguf_options")
        return d


def _camel(k: str) -> str:
    head, *rest = k.split("_")
    return head + "".join(w.title() for w in rest)


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def parse_link(link: str) -> ModelRef | CivitaiRef:
    """A Hugging Face repo (and revision, and GGUF file) or a Civitai model / version. Raises ``LinkError``."""
    link = (link or "").strip()
    if not link:
        raise LinkError("paste a Hugging Face or Civitai model link, or owner/name")
    if "://" not in link and not link.startswith(("huggingface.co/", "hf.co/", "civitai.com/", "www.")):
        repo, _, rev = link.partition("@")
        if REPO.match(repo) and (not rev or REVISION.match(rev)):
            return ModelRef(repo, rev or "main")
        raise LinkError(f"{link!r} isn't a Hugging Face repo (owner/name)")
    parts = urlsplit(link if "://" in link else "https://" + link)
    host = (parts.hostname or "").lower().removeprefix("www.")
    segs = [s for s in parts.path.split("/") if s]
    if host == "civitai.com" or host.endswith(".civitai.com"):
        query = parse_qs(parts.query)
        if len(segs) >= 4 and segs[:3] == ["api", "download", "models"] and segs[3].isdigit():
            return CivitaiRef(None, int(segs[3]))
        if len(segs) >= 2 and segs[0] == "models" and segs[1].isdigit():
            v = (query.get("modelVersionId") or [None])[0]
            return CivitaiRef(int(segs[1]), int(v) if v and v.isdigit() else None)
        raise LinkError("a Civitai link should be a model page (civitai.com/models/…)")
    if host not in HF_HOSTS and host != urlsplit(endpoint()).hostname:
        raise LinkError(f"only Hugging Face and Civitai links, not {host or link!r}")
    if segs and segs[0] == "models":
        segs = segs[1:]
    if len(segs) < 2:
        raise LinkError("the link should point at a model: huggingface.co/owner/name")
    repo = f"{segs[0]}/{segs[1]}"
    rev, file = "main", None
    if len(segs) >= 4 and segs[2] in ("tree", "blob", "resolve"):
        rev = segs[3]
        if len(segs) >= 5 and segs[-1].lower().endswith(".gguf"):
            file = "/".join(segs[4:])
    if not REPO.match(repo) or not REVISION.match(rev):
        raise LinkError(f"{link!r} isn't a Hugging Face model link")
    return ModelRef(repo, rev, file)


SyncClient = Callable[[], httpx.Client]


async def inspect(ref: ModelRef | CivitaiRef, tokens: dict | None = None, client: httpx.AsyncClient | None = None,
                  sync_client: SyncClient | None = None, quant: str | None = None) -> Facts:
    """Reads what the link points at (no weights). Raises ``LinkError`` for links that can't be served."""
    tokens = tokens or {}
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
    try:
        if isinstance(ref, CivitaiRef):
            return await _inspect_civitai(ref, tokens, client, sync_client)
        return await _inspect_hf(ref, tokens, client, sync_client, quant)
    except httpx.HTTPError as e:
        raise LinkError(f"couldn't read the model's details ({type(e).__name__})") from None
    finally:
        if own:
            await client.aclose()


def _range_file(url: str, tokens: dict, sync_client: SyncClient | None) -> RangeFile:
    return RangeFile(url, tokens, client=sync_client() if sync_client else None)


async def _inspect_hf(ref: ModelRef, tokens: dict, client: httpx.AsyncClient, sync_client, quant) -> Facts:
    headers = {"User-Agent": "aistudio-gpu", **({"Authorization": f"Bearer {tokens['huggingface']}"}
                                                if tokens.get("huggingface") else {})}
    base = endpoint()
    r = await client.get(f"{base}/api/models/{ref.repo}/revision/{ref.revision}", params={"blobs": "true"},
                         headers=headers)
    if r.status_code in (401, 403):
        raise LinkError(f"{ref.repo} {GATED}")
    if r.status_code == 404:
        raise LinkError(f"no model {ref.label} on Hugging Face")
    r.raise_for_status()
    meta = r.json()
    sha = meta.get("sha") or ref.revision
    names = {s.get("rfilename", ""): int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
             for s in meta.get("siblings") or []}
    weights = {n: b for n, b in names.items() if n.endswith(".safetensors") and "/" not in n}
    has_gguf = any(n.lower().endswith(".gguf") for n in names)
    if ref.file or (not weights and has_gguf):
        return await _hf_gguf(ref, meta, sha, names, tokens, sync_client, quant)
    if not weights:
        if any(n.endswith((".bin", ".pt", ".pth")) for n in names):
            raise LinkError("this repo has no .safetensors weights (only pickled .bin files, which the server "
                            "doesn't load)")
        raise LinkError("this repo has no model weights at its top level")
    if "config.json" not in names:
        raise LinkError("this repo has no config.json, so it isn't a complete model")

    async def text(name: str) -> str:
        resp = await client.get(f"{base}/{ref.repo}/resolve/{sha}/{name}", headers=headers)
        if resp.status_code in (401, 403):
            raise LinkError(f"{ref.repo} {GATED}")
        resp.raise_for_status()
        return resp.text

    config = json.loads(await text("config.json"))
    template = ""
    if "chat_template.jinja" in names:
        template = await text("chat_template.jinja")
    elif "tokenizer_config.json" in names:
        t = json.loads(await text("tokenizer_config.json")).get("chat_template")
        template = t if isinstance(t, str) else json.dumps(t or "")
    f = facts_from_config(ref, meta, config, weights, template)
    f.download = [{"name": n, "size": b} for n, b in sorted(names.items())
                  if "/" not in n and (n in weights or any(fnmatch(n, p) for p in SIDE_FILES))]
    text_config = config.get("text_config") or config
    if text_config.get("ngram_vocab_size_base") and INDEX in names:
        index = json.loads(await text(INDEX))
        f.cpu_bytes = await asyncio.to_thread(
            ngram_bytes, index, lambda name: f"{base}/{ref.repo}/resolve/{sha}/{name}", tokens, sync_client)
        if not f.cpu_bytes:  # the headers couldn't be read: the table's size from the config
            per = 0.5 if f.quant.get("bits") == 4 else 1 if f.quant.get("bits") == 8 else 2
            f.cpu_bytes = int(text_config["ngram_vocab_size_base"] * int(text_config.get("ple_embed_dim") or
                                                                         text_config.get("hidden_size") or 0) * per)
        if f.cpu_bytes:
            f.notes.append(f"vLLM keeps the n-gram embeddings ({f.cpu_bytes / 1e9:.0f} GB) in CPU memory, so the "
                           "GPUs hold the rest")
    return f


INDEX = "model.safetensors.index.json"
NGRAM_TENSOR = "ngram_embedding"  # vLLM's name for the tables it offloads (EngramConfig(cpu_offload=True))


def ngram_bytes(index: dict, url_of: Callable[[str], str], tokens: dict, sync_client) -> int:
    """The n-gram embedding tables' size in bytes, from the safetensors headers of the files that hold them (each
    header is read with Range requests; the tensors' data isn't). 0 when a header can't be read."""
    from concurrent.futures import ThreadPoolExecutor

    files = sorted({file for name, file in (index.get("weight_map") or {}).items() if NGRAM_TENSOR in name})

    def size_in(file: str) -> int:
        rf = _range_file(url_of(file), tokens, sync_client)
        try:
            head = safetensors_header(rf)
        finally:
            rf.close()
        return sum(int(t["data_offsets"][1]) - int(t["data_offsets"][0]) for name, t in head.items()
                   if NGRAM_TENSOR in name and isinstance(t, dict) and "data_offsets" in t)

    try:  # 8 at a time: Qwen3.8-Flash-Next spreads its tables over 33 files
        with ThreadPoolExecutor(8) as pool:
            return sum(pool.map(size_in, files))
    except (RangeError, OSError, ValueError, KeyError) as e:
        log.info("couldn't read the n-gram tables' size: %s", e)
        return 0


def safetensors_header(rf) -> dict:
    """A safetensors file's JSON header: an 8-byte little-endian length, then that much JSON."""
    def exactly(n: int) -> bytes:
        out = b""
        while len(out) < n:
            chunk = rf.read(n - len(out))
            if not chunk:
                raise ValueError("the safetensors header is cut short")
            out += chunk
        return out

    size = int.from_bytes(exactly(8), "little")
    if size > 100 << 20:
        raise ValueError("the safetensors header is too big")
    return json.loads(exactly(size))


async def _hf_gguf(ref: ModelRef, meta: dict, sha: str, names: dict, tokens: dict, sync_client, quant) -> Facts:
    groups = gguf.group(names)
    if not groups:
        raise LinkError(f"{ref.file} isn't in {ref.repo}" if ref.file else "this repo has no GGUF model files")
    chosen = None
    if ref.file:
        chosen = next((g for g in groups if ref.file in g["files"]), None)
        if chosen is None:
            raise LinkError(f"{ref.file} isn't in {ref.repo}")
    elif quant:
        chosen = next((g for g in groups if g["quant"].upper() == quant.upper()), None)
        if chosen is None:
            raise LinkError(f"{ref.repo} has no {quant} file (it has {', '.join(g['quant'] for g in groups)})")
    else:
        chosen = groups[0]
    url = f"{endpoint()}/{ref.repo}/resolve/{sha}/{chosen['files'][0]}"
    head = await asyncio.to_thread(_gguf_header, url, tokens, sync_client)
    proj = gguf.mmproj(names)
    f = facts_from_gguf(head, ref.repo, ref.revision, meta.get("sha"), chosen, bool(meta.get("gated")))
    f.mmproj = proj
    f.vision = proj is not None
    f.gguf_options = [{"quant": g["quant"], "bytes": g["bytes"]} for g in groups]
    f.download = [{"name": n, "size": names[n]} for n in chosen["files"]]
    if proj:
        f.download.append({"name": proj, "size": names[proj]})
    return f


def _gguf_header(url: str, tokens: dict, sync_client) -> dict:
    try:
        with _range_file(url, tokens, sync_client) as rf:
            return gguf.metadata(rf)
    except PermissionError as e:
        raise LinkError(str(e)) from None
    except (RangeError, gguf.GGUFError, OSError) as e:
        raise LinkError(f"couldn't read the GGUF file's header: {e}") from None


def facts_from_gguf(meta: dict, repo: str, revision: str, sha: str | None, chosen: dict, gated: bool) -> Facts:
    a = gguf.arch_facts(meta)
    q = chosen["quant"]
    bits = {"Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5, "Q6": 6, "Q8": 8, "IQ1": 1, "IQ2": 2, "IQ3": 3, "IQ4": 4,
            "BF16": 16, "F16": 16, "F32": 32, "MXFP4": 4}.get(re.split(r"_", q)[0], None)
    return Facts(
        repo=repo, revision=revision, sha=sha, architecture=a["architecture"], model_type=a["architecture"],
        weight_bytes=chosen["bytes"], files=len(chosen["files"]), quant={"method": "gguf", "type": q, "bits": bits},
        dtype=q, native_context=a["native_context"], vision=False, layers=a["layers"],
        attention_layers=a["attention_layers"], attention_heads=a["attention_heads"], kv_heads=a["kv_heads"],
        head_dim=a["head_dim"], kv_lora_rank=a["kv_lora_rank"], rope_head_dim=a["rope_head_dim"],
        sliding_window=a["sliding_window"], experts=a["experts"], gated=gated, chat_template=a["chat_template"],
        engine="llamacpp", gguf_file=chosen["files"][0],
        notes=["GGUF runs on llama.cpp: slower than vLLM on the same GPUs, but smaller files"])


# MARK: Civitai

async def _inspect_civitai(ref: CivitaiRef, tokens: dict, client: httpx.AsyncClient, sync_client) -> Facts:
    headers = {"User-Agent": "aistudio-gpu", **auth_headers("https://civitai.com/", tokens)}
    if ref.version_id is not None:
        r = await client.get(f"{CIVITAI_API}/model-versions/{ref.version_id}", headers=headers)
        if r.status_code == 404:
            raise LinkError("no such Civitai model version")
        r.raise_for_status()
        version = r.json()
        name = (version.get("model") or {}).get("name") or version.get("name") or str(ref.version_id)
    else:
        r = await client.get(f"{CIVITAI_API}/models/{ref.model_id}", headers=headers)
        if r.status_code == 404:
            raise LinkError("no such Civitai model")
        r.raise_for_status()
        model = r.json()
        versions = model.get("modelVersions") or []
        if not versions:
            raise LinkError("this Civitai model has no versions")
        version = versions[0]
        name = model.get("name") or str(ref.model_id)
    files = version.get("files") or []
    pick = next((f for f in files if str(f.get("name", "")).lower().endswith(".gguf")
                 or (f.get("metadata") or {}).get("format", "").upper() == "GGUF"), None)
    kind = "gguf"
    if pick is None:
        pick = next((f for f in files if str(f.get("name", "")).lower().endswith(".zip")), None)
        kind = "zip"
    if pick is None:
        if any(str(f.get("name", "")).lower().endswith(".safetensors") for f in files):
            raise LinkError("this Civitai file is a single .safetensors file without its config and tokenizer, so "
                            "it can't run on its own; use a GGUF version or the base model's Hugging Face repo")
        raise LinkError("this Civitai model has no GGUF file or zipped model folder")
    url = pick.get("downloadUrl") or f"https://civitai.com/api/download/models/{version.get('id')}"
    size = int(float(pick.get("sizeKB") or 0) * 1024)
    sha = ((pick.get("hashes") or {}).get("SHA256") or "").lower() or None
    repo = f"civitai/{re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('-')[:60] or version.get('id')}"
    rev = str(version.get("id") or ref.version_id or "")
    if kind == "gguf":
        head = await asyncio.to_thread(_gguf_header, url, tokens, sync_client)
        fname = pick.get("name") or "model.gguf"
        chosen = {"quant": gguf.quant_of(fname) or "GGUF", "bytes": size, "files": [fname]}
        f = facts_from_gguf(head, repo, rev, None, chosen, False)
        f.source = "civitai"
        f.download = [{"name": fname, "size": size, "url": url, "sha256": sha}]
        return f
    config, weights, template, prefix = await asyncio.to_thread(_zip_contents, url, tokens, sync_client)
    f = facts_from_config(ModelRef(repo, rev), {"sha": None}, config, weights, template)
    f.source = "civitai"
    f.download = [{"name": "model.zip", "size": size, "url": url, "sha256": sha, "extract": prefix}]
    return f


def _zip_contents(url: str, tokens: dict, sync_client) -> tuple[dict, dict, str, str]:
    """(config, {weight file: bytes}, chat template, the folder inside the zip) of a zipped model folder."""
    try:
        with _range_file(url, tokens, sync_client) as rf, zipfile.ZipFile(rf) as z:
            names = z.namelist()
            configs = sorted((n for n in names if n.rsplit("/", 1)[-1] == "config.json"), key=len)
            if not configs:
                raise LinkError("the zip has no config.json, so it isn't a model folder")
            prefix = configs[0][:-len("config.json")]
            weights = {n[len(prefix):]: z.getinfo(n).file_size for n in names
                       if n.startswith(prefix) and n.endswith(".safetensors") and "/" not in n[len(prefix):]}
            if not weights:
                raise LinkError("the zip has no .safetensors weights next to its config.json")
            config = json.loads(z.read(configs[0]))
            template = ""
            if prefix + "chat_template.jinja" in names:
                template = z.read(prefix + "chat_template.jinja").decode()
            elif prefix + "tokenizer_config.json" in names:
                t = json.loads(z.read(prefix + "tokenizer_config.json")).get("chat_template")
                template = t if isinstance(t, str) else ""
            return config, weights, template, prefix
    except PermissionError as e:
        raise LinkError(str(e)) from None
    except (RangeError, zipfile.BadZipFile, OSError, ValueError) as e:
        if isinstance(e, LinkError):
            raise
        raise LinkError(f"couldn't read the zip: {e}") from None


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
            for g in (qc.get("config_groups") or {}).values():
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
