"""llm/quantize.py: NVFP4 versions of installed LLMs (the plan side; the llm-compressor run needs a GPU)."""

import json

import pytest

from aistudio_gpu.llm import facts, quantize, store


def test_derived_id_swaps_the_precision_suffix():
    assert quantize.derived_id("qwen3.8-flash-next-fp8") == "qwen3.8-flash-next-nvfp4"
    assert quantize.derived_id("qwen3.8-27b-abliterated") == "qwen3.8-27b-abliterated-nvfp4"
    assert quantize.derived_id("x-nvfp4") == "x-nvfp4"
    long = "a" * 64
    assert len(quantize.derived_id(long)) == 64 and store.ID.match(quantize.derived_id(long))


QWEN35 = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
          "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64, "mtp_num_hidden_layers": 1},
          "vision_config": {}}


def test_qwen35_dense_gets_the_mixed_awq_gptq_recipe():
    r = quantize.recipe(QWEN35)
    assert r["kind"] == "awq_gptq_mixed" and r["samples"] == 512 and r["kv_fp8"]
    names = {
        "model.language_model.layers.3.mlp.gate_proj": "nvfp4",
        "model.language_model.layers.55.mlp.down_proj": "nvfp4",
        "model.language_model.layers.56.mlp.up_proj": "fp8",  # the last 8 layers' MLPs stay fp8
        "model.language_model.layers.63.mlp.down_proj": "fp8",
        "model.language_model.layers.3.self_attn.q_proj": "fp8",
        "model.language_model.layers.0.linear_attn.in_proj_qkv": "fp8",
        "model.language_model.layers.0.linear_attn.in_proj_a": "bf16",
        "lm_head": "fp8",
        "model.visual.blocks.0.attn.qkv": "bf16",
        "mtp.layers.0.mlp.gate_proj": "bf16",
    }
    got = quantize.classify(list(names), r)
    assert {n: k for k, v in got.items() for n in v} == names


def test_moe_gets_calibrated_nvfp4_with_the_originals_16_bit_layers_kept():
    config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
              "text_config": {"num_hidden_layers": 48, "num_experts": 512},
              "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128],
                                      "modules_to_not_convert": ["model.language_model.layers.0.attn_mix.w"]}}
    r = quantize.recipe(config)
    assert r["kind"] == "ptq" and r["moe"] and quantize.source_format(config) == "fp8"
    got = quantize.classify(["model.language_model.layers.0.mlp.experts.3.up_proj",
                             "model.language_model.layers.0.mlp.gate",
                             "model.language_model.layers.0.mlp.shared_expert_gate",
                             "model.language_model.layers.0.attn_mix.w", "lm_head"], r)
    assert got["nvfp4"] == ["model.language_model.layers.0.mlp.experts.3.up_proj"]
    assert len(got["bf16"]) == 4


def test_a_dense_vision_model_keeps_its_vision_tower_and_projector():
    config = {"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
              "text_config": {"model_type": "mistral", "num_hidden_layers": 40},
              "quantization_config": {"quant_method": "compressed-tensors", "ignore": ["re:.*vision.*"]}}
    r = quantize.recipe(config)
    assert r["kind"] == "awq_gptq" and quantize.source_format(config) == "compressed-tensors"
    got = quantize.classify(["model.vision_tower.transformer.layers.0.feed_forward.up_proj",
                             "model.multi_modal_projector.linear_1",
                             "model.language_model.layers.0.self_attn.q_proj", "model.language_model.embed_tokens"],
                            r)
    assert got["nvfp4"] == ["model.language_model.layers.0.self_attn.q_proj"]


def _nvfp4_config(groups):
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "num_hidden_layers": 2,
            "quantization_config": {"quant_method": "compressed-tensors", "format": "mixed-precision",
                                    "config_groups": groups}}


def test_facts_name_an_nvfp4_checkpoint():
    nv = {"weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"}}
    fp8 = {"weights": {"num_bits": 8, "type": "float", "strategy": "channel"}}
    f = facts.facts_from_config(facts.ModelRef("x/y", "main"), {}, _nvfp4_config({"a": fp8, "b": nv}), {"m": 1}, "")
    assert f.quant == {"method": "compressed-tensors", "bits": 4, "format": "nvfp4"}
    assert any("NVFP4" in n for n in f.notes)
    f = facts.facts_from_config(facts.ModelRef("x/y", "main"), {}, _nvfp4_config({"a": fp8}), {"m": 1}, "")
    assert f.quant == {"method": "compressed-tensors", "bits": 8}


def test_an_nvfp4_checkpoint_fits_gpus_below_b200():
    from aistudio_gpu.llm import sizing

    nv = {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}
    f = facts.facts_from_config(facts.ModelRef("x/y", "main"), {}, _nvfp4_config({"b": nv}), {"m": 1}, "")
    assert sizing.unsupported(f, "H100") is None and sizing.unsupported(f, "L40S") is None


def _installed(root, mid, config, manifest=None, files=None):
    w = store.weights_dir(root, mid)
    w.mkdir(parents=True)
    (w / "config.json").write_text(json.dumps(config))
    for name, text in (files or {}).items():
        (w / name).write_text(text)
    store.write(root, {"id": mid, "engine": "vllm", "gpu": "B200", "count": 1, "context": 32768,
                       "name": "Test", "repo": "x/test", **(manifest or {})})
    return w


def test_copy_extras_takes_the_template_and_tokenizer_not_the_weights(tmp_path):
    src = _installed(tmp_path, "m", QWEN35, files={"chat_template.jinja": "{{x}}", "tokenizer.json": "{}",
                                                   "model-00001.safetensors": "", "model.safetensors.index.json": "{}",
                                                   ".complete": "1"})
    dst = tmp_path / "out"
    dst.mkdir()
    assert quantize.copy_extras(src, dst) == ["chat_template.jinja", "tokenizer.json"]


def test_plan_refuses_files_vllm_doesnt_serve(tmp_path):
    _installed(tmp_path, "g", QWEN35, {"engine": "llamacpp"})
    with pytest.raises(ValueError, match="isn't a vLLM model"):
        quantize.plan(tmp_path, "g")


def test_mtp_layers_are_carried_over_in_bf16(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    config = {**QWEN35, "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]}}
    src = _installed(tmp_path, "m", config)
    w = torch.randn(4, 4)
    scale = torch.tensor([[2.0, 0.5], [1.0, 4.0]])
    stored = (w / scale.repeat_interleave(2, 0).repeat_interleave(2, 1)).to(torch.float8_e4m3fn)
    save_file({"mtp.layers.0.mlp.up_proj.weight": stored, "mtp.layers.0.mlp.up_proj.weight_scale_inv": scale,
               "mtp.norm.weight": torch.ones(4, dtype=torch.bfloat16),
               "model.language_model.layers.0.mlp.up_proj.weight": torch.zeros(4, 4)}, str(src / "model.safetensors"))
    dst = tmp_path / "out"
    dst.mkdir()
    save_file({"model.language_model.layers.0.mlp.up_proj.weight_packed": torch.zeros(4, 2, dtype=torch.uint8)},
              str(dst / "model.safetensors"))
    (dst / "config.json").write_text(json.dumps({"quantization_config": {"ignore": ["lm_head"]}}))
    assert quantize.carry_mtp(src, dst, config) == 2
    got = load_file(str(dst / "mtp.safetensors"))
    want = stored.float() * scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert got["mtp.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert torch.equal(got["mtp.layers.0.mlp.up_proj.weight"], want.to(torch.bfloat16))
    assert torch.equal(got["mtp.norm.weight"], torch.ones(4, dtype=torch.bfloat16))
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert index["mtp.norm.weight"] == "mtp.safetensors" and index[
        "model.language_model.layers.0.mlp.up_proj.weight_packed"] == "model.safetensors"
    assert quantize.MTP_IGNORE in json.loads((dst / "config.json").read_text())["quantization_config"]["ignore"]
    assert quantize.carry_mtp(src, dst, config) == 0  # already there


def test_the_new_manifest_keeps_the_settings_and_takes_its_own_facts(tmp_path):
    nv = {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}
    _installed(tmp_path, "m-fp8", QWEN35, {"loadFormat": "processed", "toolParser": "qwen3_coder",
                                           "engineArgs": ["--kv-cache-dtype", "fp8"], "updated": 5, "weightBytes": 9})
    w = tmp_path / "new"
    w.mkdir()
    (w / "config.json").write_text(json.dumps(_nvfp4_config({"b": nv})))
    (w / "model.safetensors").write_bytes(b"0" * 100)
    m = quantize.manifest_for(store.read(tmp_path, "m-fp8"), "m-nvfp4", w)
    assert m["id"] == "m-nvfp4" and m["derivedFrom"] == "m-fp8" and m["name"] == "Test (NVFP4)"
    assert m["quant"]["format"] == "nvfp4" and m["weightBytes"] == 100 and m["loadFormat"] == "auto"
    assert m["toolParser"] == "qwen3_coder" and m["engineArgs"] == ["--kv-cache-dtype", "fp8"] and "updated" not in m


def test_awq_is_kept_to_the_language_model_of_a_vision_model():
    config = {"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
              "text_config": {"model_type": "mistral", "num_hidden_layers": 40}, "vision_config": {}}
    assert quantize.recipe(config)["awq_scope"] == "language_model"
    assert quantize.recipe({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})["awq_scope"] is None
    p = quantize.scoped("re:.*q_proj$", "language_model")
    assert quantize.matches("model.language_model.layers.0.self_attn.q_proj", [p])
    assert not quantize.matches("model.vision_tower.transformer.layers.0.attention.q_proj", [p])
    assert quantize.matches("model.language_model.layers.3.input_layernorm",
                            [quantize.scoped("re:.*input_layernorm$", "language_model")])


def test_an_fp8_checkpoint_gets_a_plain_bf16_copy_to_calibrate(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    src.mkdir()
    w = torch.randn(4, 8)
    channel = torch.tensor([[0.5], [2.0], [1.0], [4.0]])
    stored = (w / channel).to(torch.float8_e4m3fn)
    norm = torch.ones(8, dtype=torch.bfloat16)
    save_file({"a.weight": stored, "a.weight_scale": channel, "a.input_scale": torch.ones(1), "norm.weight": norm},
              str(src / "model.safetensors"))
    config = {"architectures": ["MistralForCausalLM"], "quantization_config": {"quant_method": "compressed-tensors"},
              "text_config": {"quantization_config": {}, "hidden_size": 8}}
    (src / "config.json").write_text(json.dumps(config))
    (src / "tokenizer.json").write_text("{}")
    dst = tmp_path / "dst"
    assert quantize.bf16_copy(src, dst, config) == 1
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    got = {}
    for part in set(index.values()):
        got.update(load_file(str(dst / part)))
    assert set(got) == {"a.weight", "norm.weight"}
    assert torch.equal(got["a.weight"], (stored.float() * channel).to(torch.bfloat16))
    assert torch.equal(got["norm.weight"], norm)
    plain = json.loads((dst / "config.json").read_text())
    assert "quantization_config" not in plain and "quantization_config" not in plain["text_config"]
    assert (dst / "tokenizer.json").is_file()


def test_block_fp8_and_grouped_scales_are_expanded():
    torch = pytest.importorskip("torch")
    w = torch.randn(4, 8).to(torch.float8_e4m3fn)
    groups = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])  # one per 4 columns
    want = w.float() * groups.repeat_interleave(4, 1)
    assert torch.equal(quantize.dequant_scaled(w, groups), want.to(torch.bfloat16))
    assert torch.equal(quantize.dequant_scaled(w, torch.tensor(2.0)), (w.float() * 2).to(torch.bfloat16))
    blocks = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2×4 blocks
    want = w.float() * blocks.repeat_interleave(2, 0).repeat_interleave(4, 1)
    assert torch.equal(quantize.dequant_block_fp8(w, blocks, (2, 4)), want.to(torch.bfloat16))
