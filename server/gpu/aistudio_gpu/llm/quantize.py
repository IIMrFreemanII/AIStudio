"""NVFP4 versions of installed LLMs, made once and kept on the Volume as models of their own (``<id>-nvfp4``).

A one-time GPU job (``modal_app.quantize_llm``) loads an installed vLLM model in bf16 (fp8 and compressed-tensors
checkpoints are dequantized on load), quantizes it with llm-compressor using calibration data, and saves a
compressed-tensors NVFP4 checkpoint: 4-bit weights in blocks of 16 with an fp8 scale each and an fp32 scale per
tensor, plus a calibrated per-tensor scale for the activations. vLLM loads it like any other checkpoint: on a B200
the activations are quantized too (fp4 tensor cores); older GPUs run it weight-only (fp4 weights, bf16 activations),
which still reads a quarter of the bytes per token. The original stays installed; the app lists both.

Recipes (``recipe``), by family:
- Qwen3.5/3.8 dense (``qwen3_5``): llm-compressor's Qwen3.8-27B recipe, AWQ then GPTQ, MLPs in NVFP4, attention,
  linear attention, lm_head and the last 8 layers' MLPs in fp8 (dynamic), the KV cache in fp8.
- other dense models: AWQ then GPTQ, every Linear in NVFP4 but lm_head, embeddings and the vision tower.
- mixtures of experts: calibrated NVFP4 without GPTQ (too slow for hundreds of experts), every expert calibrated.
MTP layers are left out of llm-compressor's model classes, so they're copied from the original checkpoint as they
are (dequantized to bf16 when stored in fp8) and kept out of the quantization, so drafting still works.

Not working yet (llm-compressor 0.15, 2026-11-28): Qwen4-Exp (Flash-Next) keeps its experts fused, so no expert
layer is found to quantize; DeepSeek V4.1 has no transformers class; Mistral3 (Dolphin) fails in calibration (an
activation observer's statistics change shape with the sample's length).

This module's top level imports nothing heavy: ``run`` imports torch, transformers and llm-compressor itself."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from . import store

SUFFIXES = ("-fp8", "-bf16", "-fp16", "-nvfp4")
TAG = "-nvfp4"
# What llm-compressor's examples keep in 16-bit: the output head, the embeddings, a vision tower and its projector,
# MoE routers and shared-expert gates (tiny, precision-sensitive), linear attention's a/b projections (per-head
# scalars) and MTP layers (not in the model it calibrates; copied over as they are).
IGNORE = [
    r"re:.*lm_head",
    r"re:.*embed_tokens$",
    r"re:(model\.)?visual\..*",
    r"re:.*\.visual\..*",
    r"re:.*(vision_tower|vision_model|multi_modal_projector|mm_projector)\..*",
    r"re:.*mlp\.gate$",
    r"re:.*shared_expert_gate$",
    r"re:.*linear_attn\.in_proj_(a|b)$",
    r"re:(.*\.)?mtp\..*",
]
MTP_IGNORE = r"re:(.*\.)?mtp\..*"
# Files of the original copied next to the new weights (the chat template and processor exactly as installed).
SKIP_FILES = re.compile(r"(\.safetensors|\.safetensors\.index\.json|\.bin|\.pt|\.pth|\.gguf)$|^config\.json$|^\.")
FP8_KV = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "tensor", "dynamic": False,
          "observer": "static_minmax"}


def derived_id(mid: str) -> str:
    """``qwen3.8-flash-next-fp8`` → ``qwen3.8-flash-next-nvfp4``."""
    base = mid
    for s in SUFFIXES:
        if base.endswith(s):
            base = base[: -len(s)]
            break
    return base[: 64 - len(TAG)].rstrip("-._") + TAG


def _text(config: dict) -> dict:
    return {**config, **(config.get("text_config") or config.get("llm_config") or {})}


def is_moe(config: dict) -> bool:
    t = _text(config)
    return int(t.get("num_experts") or t.get("n_routed_experts") or t.get("num_local_experts") or 0) > 1


def source_format(config: dict) -> str | None:
    """How the original stores its weights: "fp8" (block-scaled, transformers' FineGrainedFP8), "compressed-tensors",
    another method's name, or None for plain bf16/fp16."""
    qc = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config")
    if not qc:
        return None
    return str(qc.get("quant_method") or qc.get("format") or "?").lower()


def recipe(config: dict) -> dict:
    """What ``run`` does for a model (JSON, so the plan can be printed and checked before the paid run)."""
    t = _text(config)
    mtype = str(config.get("model_type") or t.get("model_type") or "").lower()
    layers = int(t.get("num_hidden_layers") or 0)
    ignore = list(IGNORE)
    qc = config.get("quantization_config") or {}
    # Whatever the original left in 16-bit (Qwen4-Exp's hyper-connection mixers, its linear-attention layers) stays so.
    ignore += [n for n in (qc.get("modules_to_not_convert") or qc.get("ignore") or []) if isinstance(n, str)
               and not n.startswith("re:")]
    if is_moe(config):
        return {"kind": "ptq", "scheme": "NVFP4", "ignore": ignore, "samples": 64, "seq_len": 2048, "moe": True,
                "kv_fp8": False}
    if mtype.startswith("qwen3_5") and layers > 8:
        last = "|".join(str(i) for i in range(layers - 8, layers))
        fp8 = [r"re:.*self_attn\.(q|k|v|o)_proj$", r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
               r"re:.*lm_head", rf"re:.*layers\.({last})\.mlp\..*(gate|up|down)_proj$"]
        return {"kind": "awq_gptq_mixed", "fp8": fp8, "nvfp4": [r"re:.*mlp\..*(gate|up|down)_proj$"],
                "ignore": [r"re:(model\.)?visual\..*", r"re:.*\.visual\..*", MTP_IGNORE, r"re:.*embed_tokens$",
                           r"re:.*linear_attn\.in_proj_(a|b)$"],
                "samples": 512, "seq_len": 4096, "moe": False, "kv_fp8": True}
    # A vision tower with its own q/k/v projections (Pixtral's) breaks AWQ's per-layer grouping: AWQ has no ignore
    # list, so its mappings are kept to the language model.
    scope = "language_model" if "vision_config" in config else None
    return {"kind": "awq_gptq", "scheme": "NVFP4", "ignore": ignore, "samples": 256, "seq_len": 2048, "moe": False,
            "kv_fp8": False, "awq_scope": scope}


def mtp_names(names) -> list[str]:
    return [n for n in names if n.startswith("mtp.") or ".mtp." in n]


def checkpoint_names(weights: Path) -> dict[str, str]:
    """{tensor name: file} of a safetensors checkpoint."""
    from .facts import safetensors_header

    index = weights / "model.safetensors.index.json"
    if index.is_file():
        return dict(json.loads(index.read_text()).get("weight_map") or {})
    out = {}
    for f in sorted(weights.glob("*.safetensors")):
        with open(f, "rb") as fh:
            out.update({k: f.name for k in safetensors_header(fh) if k != "__metadata__"})
    return out


def dequant_block_fp8(w, scale_inv, block=(128, 128)):
    """A block-scaled fp8 weight (DeepSeek / Qwen "fp8" checkpoints: ``weight`` × ``weight_scale_inv`` per block) in
    bf16."""
    import torch

    rows, cols = w.shape
    br, bc = block
    s = scale_inv.float().repeat_interleave(br, 0)[:rows].repeat_interleave(bc, 1)[:, :cols]
    return (w.float() * s).to(torch.bfloat16)


def carry_mtp(src: Path, dst: Path, config: dict) -> int:
    """Copies the original's MTP tensors into ``dst`` (``mtp.safetensors``, added to the index; fp8 ones in bf16)
    when the new checkpoint lacks them, and keeps them out of the quantization config. Returns the tensors copied."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    names = checkpoint_names(src)
    wanted = mtp_names(names)
    have = set(checkpoint_names(dst))
    if not wanted or set(mtp_names(have)):
        return 0
    qc = config.get("quantization_config") or {}
    block = tuple(qc.get("weight_block_size") or (128, 128))
    tensors = {}
    by_file: dict[str, list[str]] = {}
    for n in wanted:
        by_file.setdefault(names[n], []).append(n)
    for fname, keys in by_file.items():
        with safe_open(str(src / fname), framework="pt") as f:
            for k in keys:
                if k.endswith("_scale_inv"):
                    continue
                t = f.get_tensor(k)
                scale = k[: -len("weight")] + "weight_scale_inv" if k.endswith("weight") else None
                if t.dtype.is_floating_point and t.element_size() == 1 and scale in names:
                    with safe_open(str(src / names[scale]), framework="pt") as g:
                        t = dequant_block_fp8(t, g.get_tensor(scale), block)
                tensors[k] = t.contiguous()
    save_file(tensors, str(dst / "mtp.safetensors"), metadata={"format": "pt"})
    index = dst / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
    else:
        data = {"metadata": {}, "weight_map": {n: f for n, f in checkpoint_names(dst).items()}}
    data["weight_map"].update({k: "mtp.safetensors" for k in tensors})
    index.write_text(json.dumps(data, indent=2))
    keep_out_of_quant(dst, MTP_IGNORE)
    return len(tensors)


def keep_out_of_quant(weights: Path, pattern: str) -> None:
    """Adds ``pattern`` to the saved checkpoint's quantization ``ignore`` list."""
    path = weights / "config.json"
    config = json.loads(path.read_text())
    qc = config.get("quantization_config") or {}
    ignore = list(qc.get("ignore") or [])
    if pattern not in ignore:
        qc["ignore"] = ignore + [pattern]
        config["quantization_config"] = qc
        path.write_text(json.dumps(config, indent=2))


def copy_extras(src: Path, dst: Path) -> list[str]:
    """The original's tokenizer, chat template, processor and generation files, as installed."""
    copied = []
    for p in sorted(src.iterdir()):
        if p.is_file() and not SKIP_FILES.search(p.name):
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    return copied


def manifest_for(source: dict, new_id: str, weights: Path) -> dict:
    """The new model's manifest: the original's settings (GPU, context, parsers, engine arguments), its own facts."""
    from .facts import ModelRef, facts_from_config

    config = json.loads((weights / "config.json").read_text())
    sizes = {p.name: p.stat().st_size for p in weights.glob("*.safetensors")}
    template = ""
    if (weights / "chat_template.jinja").is_file():
        template = (weights / "chat_template.jinja").read_text()
    f = facts_from_config(ModelRef(source.get("repo") or new_id, source.get("revision") or "main"),
                          {"sha": None}, config, sizes, template)
    old = source.get("facts") or {}
    f.cpu_bytes = int(old.get("cpu_bytes") or 0)
    m = dict(source)
    for k in ("updated", "mtpLayers"):
        m.pop(k, None)
    m.update({"id": new_id, "name": f"{source.get('name') or source['id']} (NVFP4)", "facts": f.stored(),
              "quant": f.quant, "weightBytes": f.weight_bytes, "derivedFrom": source["id"], "created": time.time()})
    m["mtpLayers"] = store.mtp_layers(weights)
    if m.get("loadFormat") == "processed":
        # vLLM 0.33's compressed-tensors NVFP4 layers don't take pre-processed weights, so the plugin's fast start
        # would refuse them; the checkpoint itself is already the small read.
        m["loadFormat"] = "auto"
    return m


def plan(models_root: Path, mid: str) -> dict:
    """What a quantization of ``mid`` would do, without doing it (for the cost sheet and the dry run)."""
    m = store.read(models_root, mid)
    if m is None:
        raise ValueError(f"no model {mid}")
    if m.get("engine") in store.NOT_VLLM:
        raise ValueError(f"{mid} isn't a vLLM model ({m.get('engine')} models are already quantized files)")
    weights = store.weights_dir(models_root, mid)
    config = json.loads((weights / "config.json").read_text())
    return {"source": mid, "id": derived_id(mid), "architecture": (config.get("architectures") or ["?"])[0],
            "from": source_format(config) or "bf16", "recipe": recipe(config),
            "mtp": len(mtp_names(checkpoint_names(weights))), "weightBytes": m.get("weightBytes"), "gpu": m.get("gpu")}


def matches(name: str, patterns) -> bool:
    """llm-compressor's target rule: ``re:<regex>`` matches from the start of the module name, else the exact name."""
    return any(re.match(p[3:], name) if p.startswith("re:") else p == name for p in patterns)


def classify(linears: list[str], r: dict) -> dict[str, list[str]]:
    """{"nvfp4", "fp8", "bf16"}: which Linear layers a recipe quantizes how (ignore wins; then the first group)."""
    out: dict[str, list[str]] = {"nvfp4": [], "fp8": [], "bf16": []}
    for n in linears:
        if matches(n, r["ignore"]):
            out["bf16"].append(n)
        elif r["kind"] == "awq_gptq_mixed":
            if matches(n, r["fp8"]):
                out["fp8"].append(n)
            elif matches(n, r["nvfp4"]):
                out["nvfp4"].append(n)
            else:
                out["bf16"].append(n)
        else:
            out["nvfp4"].append(n)
    return out


def dry_run(models_root: Path, mid: str) -> dict:
    """``plan`` plus a check that this transformers builds the model, and the layers per precision (counts, a few
    names each), with the model built on the meta device (no weights read)."""
    import torch
    import transformers

    info = plan(models_root, mid)
    weights = store.weights_dir(models_root, mid)
    arch = info["architecture"]
    cls = getattr(transformers, arch, None)
    info["transformers"] = transformers.__version__
    if cls is None:
        info["error"] = f"transformers {transformers.__version__} has no {arch}"
        return info
    config = transformers.AutoConfig.from_pretrained(str(weights))
    with torch.device("meta"):
        model = cls._from_config(config, dtype=torch.bfloat16)
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    groups = classify(linears, info["recipe"])
    params = {k: sum(model.get_submodule(n).weight.numel() for n in v) for k, v in groups.items()}
    total = sum(p.numel() for p in model.parameters())
    info["layers"] = {k: {"count": len(v), "params": params[k], "examples": v[:3]} for k, v in groups.items()}
    info["params"] = total
    # Weights after: 4.5 bits for NVFP4 (one fp8 scale per 16), 8 for fp8, 16 for the rest.
    other = total - sum(params.values())
    info["estimatedBytes"] = int(params["nvfp4"] * 0.5625 + params["fp8"] + (params["bf16"] + other) * 2)
    return info


SCALES = (".weight_scale_inv", ".weight_scale", ".input_scale", ".activation_scale", ".weight_zero_point",
          ".input_zero_point", ".weight_global_scale", ".input_global_scale")


def dequant_scaled(w, scale):
    """An fp8 weight × its compressed-tensors scale (per tensor, per output channel, or per group of columns)."""
    import torch

    s = scale.float()
    if s.dim() == 2 and s.shape[0] == w.shape[0] and s.shape[1] not in (1, w.shape[1]):
        s = s.repeat_interleave(w.shape[1] // s.shape[1], 1)
    return (w.float() * s).to(torch.bfloat16)


def bf16_copy(src: Path, dst: Path, config: dict) -> int:
    """A plain bf16 checkpoint of an fp8 one (block-scaled "fp8" or compressed-tensors float8) in ``dst``: weights
    times their scales, the scale tensors and the quantization config dropped, everything else as it is. Loading the
    fp8 checkpoint through transformers instead leaves its quantization schemes on the layers, whose observers then
    break calibration. Returns the weights dequantized."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    qc = config.get("quantization_config") or {}
    block = tuple(qc.get("weight_block_size") or ()) or None
    names = checkpoint_names(src)
    files = sorted(set(names.values()))
    dst.mkdir(parents=True, exist_ok=True)
    weight_map, done = {}, 0
    for i, fname in enumerate(files):
        out = {}
        with safe_open(str(src / fname), framework="pt") as f:
            for k in f.keys():
                if k.endswith(SCALES):
                    continue
                t = f.get_tensor(k)
                if t.dtype.is_floating_point and t.element_size() == 1 and k.endswith(".weight"):
                    stem = k[: -len("weight")]
                    if stem + "weight_scale_inv" in names:
                        with safe_open(str(src / names[stem + "weight_scale_inv"]), framework="pt") as g:
                            t = dequant_block_fp8(t, g.get_tensor(stem + "weight_scale_inv"), block or (128, 128))
                    elif stem + "weight_scale" in names:
                        with safe_open(str(src / names[stem + "weight_scale"]), framework="pt") as g:
                            t = dequant_scaled(t, g.get_tensor(stem + "weight_scale"))
                    else:
                        raise ValueError(f"{k} is fp8 without a scale")
                    done += 1
                out[k] = t.contiguous()
        part = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
        save_file(out, str(dst / part), metadata={"format": "pt"})
        weight_map.update({k: part for k in out})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    plain = dict(config)
    plain.pop("quantization_config", None)
    if isinstance(plain.get("text_config"), dict):
        plain["text_config"] = {k: v for k, v in plain["text_config"].items() if k != "quantization_config"}
    (dst / "config.json").write_text(json.dumps(plain, indent=2))
    copy_extras(src, dst)
    return done


# MARK: the GPU job (llm-compressor's image)


def _load(weights: Path, config: dict):
    import torch
    import transformers
    from llmcompressor.utils import load_context

    arch = (config.get("architectures") or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise RuntimeError(f"transformers {transformers.__version__} has no {arch}, so llm-compressor can't load it")
    kwargs: dict = {"dtype": torch.bfloat16}
    fmt = source_format(config)
    if fmt in ("fp8", "compressed-tensors"):
        plain = Path(os.environ.get("AISTUDIO_QUANT_TMP", "/tmp/aistudio-bf16"))
        shutil.rmtree(plain, ignore_errors=True)
        t = time.monotonic()
        n = bf16_copy(weights, plain, config)
        print(f"quantize: {n} fp8 weights in bf16 in {time.monotonic() - t:.0f} s", flush=True)
        weights = plain
    elif fmt:
        raise RuntimeError(f"can't requantize a {fmt} checkpoint")
    with load_context(cls):
        model = cls.from_pretrained(str(weights), **kwargs)
    try:
        processor = transformers.AutoProcessor.from_pretrained(str(weights))
    except Exception:  # noqa: BLE001 — text-only models have just a tokenizer
        processor = transformers.AutoTokenizer.from_pretrained(str(weights))
    return model, processor


def _modifiers(r: dict):
    from llmcompressor.modifiers.quantization import QuantizationModifier

    kv = FP8_KV if r.get("kv_fp8") else None
    if r["kind"] == "ptq":
        return QuantizationModifier(targets="Linear", scheme=r["scheme"], ignore=r["ignore"])
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    if r["kind"] == "awq_gptq_mixed":
        from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4, QuantizationScheme

        groups = {"attention": QuantizationScheme(targets=r["fp8"], **FP8_DYNAMIC),
                  "mlp": QuantizationScheme(targets=r["nvfp4"], **NVFP4)}
        return [AWQModifier(duo_scaling="both"),
                GPTQModifier(config_groups=groups, ignore=r["ignore"], kv_cache_scheme=kv)]
    return [AWQModifier(duo_scaling="both", **({"mappings": awq_mappings(r["awq_scope"])} if r.get("awq_scope") else {})),
            GPTQModifier(targets="Linear", scheme=r["scheme"], ignore=r["ignore"], kv_cache_scheme=kv)]


def scoped(pattern: str, scope: str) -> str:
    """An AWQ layer pattern kept to ``scope`` (``re:.*q_proj$`` → ``re:.*language_model\\..*q_proj$``)."""
    body = pattern[3:] if pattern.startswith("re:") else re.escape(pattern) + "$"
    body = body[2:] if body.startswith(".*") else body
    return rf"re:.*{scope}\..*{body}"


def awq_mappings(scope: str):
    """llm-compressor's default AWQ mappings, kept to the ``scope`` submodule."""
    from llmcompressor.modifiers.transform.awq.mappings import AWQMapping, default_mappings

    return [AWQMapping(scoped(m.smooth_layer, scope), [scoped(b, scope) for b in m.balance_layers])
            for m in default_mappings]


def run(models_root: Path, mid: str, log=print, samples: int | None = None, seq_len: int | None = None) -> dict:
    """Quantizes ``mid`` into ``<id>-nvfp4`` on the Volume (weights, then the manifest; ``.complete`` last)."""
    from llmcompressor import oneshot

    info = plan(models_root, mid)
    new_id = info["id"]
    if store.read(models_root, new_id) is not None:
        raise ValueError(f"{new_id} is already installed")
    source = store.read(models_root, mid)
    src = store.weights_dir(models_root, mid)
    config = json.loads((src / "config.json").read_text())
    r = info["recipe"]
    n = int(samples or r["samples"])
    seq = int(seq_len or r["seq_len"])
    log(f"quantize: {mid} → {new_id}: {r['kind']}, {n} samples × {seq} tokens")
    start = time.monotonic()
    model, processor = _load(src, config)
    log(f"quantize: loaded in {time.monotonic() - start:.0f} s")
    t = time.monotonic()
    oneshot(model=model, processor=processor, recipe=_modifiers(r), dataset="perfectblend",
            splits=f"train[:{max(n, 64)}]", max_seq_length=seq, num_calibration_samples=n,
            moe_calibrate_all_experts=bool(r.get("moe")))
    log(f"quantize: calibrated and quantized in {time.monotonic() - t:.0f} s")
    out_dir = store.model_dir(models_root, new_id)
    try:
        tmp = out_dir / "weights.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        t = time.monotonic()
        model.save_pretrained(str(tmp), save_compressed=True)
        processor.save_pretrained(str(tmp))
        del model
        extras = copy_extras(src, tmp)
        mtp = carry_mtp(src, tmp, config)
        log(f"quantize: saved in {time.monotonic() - t:.0f} s ({len(extras)} files copied, {mtp} MTP tensors kept)")
        weights = store.weights_dir(models_root, new_id)
        shutil.rmtree(weights, ignore_errors=True)
        os.replace(tmp, weights)
        manifest = manifest_for(source, new_id, weights)
        store.write(models_root, manifest)
        (weights / ".complete").write_text(str(time.time()))
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)  # no half-made model in the app's list
        raise
    gb = manifest["weightBytes"] / 1e9
    log(f"quantize: {new_id} installed, {gb:.1f} GB (was {(source.get('weightBytes') or 0) / 1e9:.1f} GB), "
        f"{time.monotonic() - start:.0f} s in all")
    return {"id": new_id, "weightBytes": manifest["weightBytes"], "seconds": time.monotonic() - start}
