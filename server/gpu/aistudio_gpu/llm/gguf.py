"""GGUF files (llama.cpp's format): the metadata at the start of the file, and the naming of quantizations and split
files in model repos.

Header: ``GGUF``, version (u32), tensor count (u64), metadata count (u64), then metadata key/value pairs. Only the
metadata is read (the tokenizer's arrays are skipped value by value); the tensor data never is.
"""

from __future__ import annotations

import os
import re
import struct
from typing import BinaryIO

MAGIC = b"GGUF"
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
# Quantization in a file name: Q4_K_M, IQ2_XXS, Q8_0, BF16, F16, MXFP4…
QUANT = re.compile(r"(?<![A-Za-z0-9])(I?Q\d(?:_[0-9A-Z]+)*|BF16|F16|F32|MXFP4(?:_MOE)?)(?![A-Za-z0-9])", re.I)
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
# Which quantization to offer first: good quality per byte, then the others.
PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "MXFP4", "MXFP4_MOE", "Q6_K", "Q8_0", "IQ4_XS", "Q3_K_M", "BF16", "F16"]


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("the file ended inside its header")
    return b


def _string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", _read(f, 8))
    if n > 1 << 26:
        raise GGUFError("implausible string length in the header")
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, t: int, keep_arrays: bool):
    if t in SCALARS:
        fmt = SCALARS[t]
        return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
    if t == STRING:
        return _string(f)
    if t == ARRAY:
        (it,) = struct.unpack("<I", _read(f, 4))
        (n,) = struct.unpack("<Q", _read(f, 8))
        if it in SCALARS and not keep_arrays:  # skip numbers in one go
            f.seek(n * struct.calcsize(SCALARS[it]), 1)
            return None
        if it == STRING and not keep_arrays:  # skip strings by their lengths (the vocabulary, the merges)
            for _ in range(n):
                (ln,) = struct.unpack("<Q", _read(f, 8))
                f.seek(ln, 1)
            return None
        items = [_value(f, it, keep_arrays) for _ in range(n)]
        return items if keep_arrays else None
    raise GGUFError(f"unknown value type {t} in the header")


def metadata(f: BinaryIO, small_arrays: int = 64) -> dict:
    """The header's key/value pairs. Arrays are kept only when short (per-layer values); long ones (the vocabulary)
    are skipped."""
    if _read(f, 4) != MAGIC:
        raise GGUFError("not a GGUF file")
    (version,) = struct.unpack("<I", _read(f, 4))
    if version < 2:
        raise GGUFError(f"GGUF version {version} is too old")
    _tensors, count = struct.unpack("<QQ", _read(f, 16))
    out: dict = {"_version": version}
    for _ in range(count):
        key = _string(f)
        (t,) = struct.unpack("<I", _read(f, 4))
        if t == ARRAY:
            pos = f.tell()
            (it,) = struct.unpack("<I", _read(f, 4))
            (n,) = struct.unpack("<Q", _read(f, 8))
            f.seek(pos)
            out[key] = _value(f, t, keep_arrays=n <= small_arrays)
        else:
            out[key] = _value(f, t, keep_arrays=False)
    return out


def quant_of(name: str) -> str | None:
    found = QUANT.findall(name.rsplit("/", 1)[-1])
    return found[-1].upper() if found else None


def _before_quant(name: str) -> str:
    """A file's name (no folder, shard suffix or extension) up to its quantization: what tells variants apart."""
    base = SPLIT.sub("", name.rsplit("/", 1)[-1]).removesuffix(".gguf").removesuffix(".GGUF")
    found = list(QUANT.finditer(base))
    return base[:found[-1].start()] if found else base


def _variant_rank(variant: str | None) -> int:
    """Which of a quantization's variants to offer first: one with the model's MTP layers (faster replies), then the
    plain one, then others (``LOW-MTP``, ``UD``…)."""
    return 0 if variant and variant.upper() == "MTP" else 1 if not variant else 2


def group(files: dict[str, int]) -> list[dict]:
    """GGUF files of a repo → one entry per quantization: {"quant", "variant", "choice", "bytes", "files" (shards in
    order)}. ``variant`` is what the file names add before the quantization when a repo has several kinds of files
    ("MTP", "LOW-MTP": ``x-Q4_K_M.gguf`` and ``x-MTP-Q4_K_M.gguf``), else None; ``choice`` names the entry uniquely
    ("Q4_K_M", "Q4_K_M MTP"). Projector files (``mmproj``) are left out (see ``mmproj``)."""
    groups: dict[str, dict] = {}
    for name, size in sorted(files.items()):
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
            continue
        q = quant_of(name) or name.rsplit("/", 1)[-1].removesuffix(".gguf")
        m = SPLIT.search(name)
        key = f"{q}|{SPLIT.sub('', name) if m else name}"
        g = groups.setdefault(key, {"quant": q, "bytes": 0, "files": [], "before": _before_quant(name),
                                    "named": quant_of(name) is not None})
        g["bytes"] += size
        g["files"].append(name)
    out = list(groups.values())
    named = [g["before"] for g in out if g["named"]]
    common = os.path.commonprefix(named) if len(named) > 1 else (named[0] if named else "")
    common = common[:max(common.rfind(c) for c in "-_. ") + 1] if any(c in common for c in "-_. ") else ""
    for g in out:
        g["files"].sort()
        rest = g.pop("before")[len(common):].strip("-_. ") if g.pop("named") else ""
        g["variant"] = rest if rest and len(rest) <= 24 else None
        g["choice"] = f"{g['quant']} {g['variant']}" if g["variant"] else g["quant"]
    seen: dict[str, int] = {}
    for g in out:
        seen[g["choice"]] = seen.get(g["choice"], 0) + 1
    for g in out:  # still the same name (long or odd file names): the file's own name tells them apart
        if seen[g["choice"]] > 1:
            g["choice"] = SPLIT.sub("", g["files"][0].rsplit("/", 1)[-1]).removesuffix(".gguf")
    smallest: dict[str, int] = {}  # a quantization's variants stay together, by its smallest file
    for g in out:
        smallest[g["quant"]] = min(smallest.get(g["quant"], g["bytes"]), g["bytes"])
    return sorted(out, key=lambda g: (PREFERENCE.index(g["quant"]) if g["quant"] in PREFERENCE else 99,
                                      smallest[g["quant"]], g["quant"], _variant_rank(g["variant"]), g["bytes"]))


def mmproj(files: dict[str, int]) -> str | None:
    """The vision projector of a repo, if it has one (F16 / BF16 preferred)."""
    found = sorted(n for n in files if n.lower().endswith(".gguf") and "mmproj" in n.lower())
    for pref in (r"(?<!b)f16", r"bf16"):
        for n in found:
            if re.search(pref, n.lower()):
                return n
    return found[0] if found else None


def arch_facts(meta: dict) -> dict:
    """The fields sizing needs, from the header (``{arch}.block_count`` and friends)."""
    arch = str(meta.get("general.architecture") or "?")

    def get(key, default=0):
        v = meta.get(f"{arch}.{key}", default)
        if isinstance(v, list):  # per-layer values (some hybrid models)
            nums = [x for x in v if isinstance(x, (int, float))]
            return max(nums) if nums else default
        return v if v is not None else default

    # Multi-token-prediction (NextN) layers come after the model's own in ``block_count``; each has full attention.
    mtp = int(get("nextn_predict_layers"))
    layers = int(get("block_count")) - mtp
    heads = int(get("attention.head_count"))
    kv_list = meta.get(f"{arch}.attention.head_count_kv")
    kv = int(get("attention.head_count_kv", heads))
    attn = layers
    if isinstance(kv_list, list):
        attn = sum(1 for x in kv_list[:layers] if x)  # layers without KV heads are linear-attention ones
    elif get("full_attention_interval"):
        attn = layers // int(get("full_attention_interval"))
    attn += mtp
    embed = int(get("embedding_length"))
    head_dim = int(get("attention.key_length") or (embed // heads if heads else 0))
    return {
        "architecture": arch, "layers": layers, "attention_layers": attn or layers, "attention_heads": heads,
        "kv_heads": kv, "head_dim": head_dim, "native_context": int(get("context_length") or 4096),
        "experts": int(get("expert_count")), "kv_lora_rank": int(get("attention.kv_lora_rank")),
        "rope_head_dim": int(get("rope.dimension_count") if get("attention.kv_lora_rank") else 0),
        "sliding_window": int(get("attention.sliding_window")),
        "chat_template": str(meta.get("tokenizer.chat_template") or ""),
        "name": str(meta.get("general.name") or ""), "mtp_layers": mtp,
    }
