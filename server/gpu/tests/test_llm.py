"""LLMs from links: link parsing, GPU sizing, parser guesses, and the /v1/llm API end to end with a fake Hugging
Face (httpx.MockTransport) and the fake engine (no GPU, no vLLM)."""

import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aistudio_gpu.api import create_app
from aistudio_gpu.files import FileStore
from aistudio_gpu.jobs import JobManager
from aistudio_gpu.llm import proxy, sizing
from aistudio_gpu.llm.engine import FAKE_COMMIT, FAKE_PR, FakeEngine, phase_of, vllm_command
from aistudio_gpu.llm.facts import LinkError, ModelRef, facts_from_config, parse_link
from aistudio_gpu.llm.profiles import apply_effort, efforts, guess
from aistudio_gpu.llm.service import LLMService, LocalLLMService, local_media, stop_pressed
from aistudio_gpu.local import LocalRunner

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GIB = 1 << 30

# The parts of Qwen/Qwen3.8-Flash-Next-FP8's config.json that sizing reads (hybrid: 1 in 4 layers full attention).
QWEN38 = {
    "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "image_token_id": 248056,
    "vision_config": {"depth": 24},
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    "text_config": {"model_type": "qwen4_exp_text", "num_hidden_layers": 48, "num_attention_heads": 24,
                    "num_key_value_heads": 2, "head_dim": 256, "hidden_size": 2560, "num_experts": 512,
                    "max_position_embeddings": 262144, "full_attention_interval": 4,
                    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12},
}
QWEN38_TEMPLATE = "{% for m in messages %}<think>…</think><tool_call>\n<function=name>{% endfor %}"
LLAMA8B = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 32,
           "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
           "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}
DEEPSEEK = {"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "num_hidden_layers": 61,
            "num_attention_heads": 128, "num_key_value_heads": 128, "hidden_size": 7168, "kv_lora_rank": 512,
            "qk_rope_head_dim": 64, "n_routed_experts": 256, "max_position_embeddings": 163840,
            "quantization_config": {"quant_method": "fp8"}}


def facts(config, weight_bytes, template="", repo="o/m"):
    return facts_from_config(ModelRef(repo), {"sha": "a" * 40}, config, {"model.safetensors": weight_bytes},
                             template)


# MARK: links

@pytest.mark.parametrize("link,repo,rev", [
    ("Qwen/Qwen3.8-Flash-Next-FP8", "Qwen/Qwen3.8-Flash-Next-FP8", "main"),
    ("Qwen/Qwen3.8-Flash-Next-FP8@abc123", "Qwen/Qwen3.8-Flash-Next-FP8", "abc123"),
    ("https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8", "Qwen/Qwen3.8-Flash-Next-FP8", "main"),
    ("https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/v2", "Qwen/Qwen3.8-Flash-Next-FP8", "v2"),
    ("hf.co/meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "main"),
])
def test_parse_link(link, repo, rev):
    ref = parse_link(link)
    assert (ref.repo, ref.revision) == (repo, rev)


def test_parse_gguf_and_civitai_links():
    from aistudio_gpu.llm.facts import CivitaiRef
    ref = parse_link("https://huggingface.co/bartowski/x-GGUF/blob/main/Q8/x-Q8_0-00001-of-00002.gguf")
    assert (ref.repo, ref.revision, ref.file) == ("bartowski/x-GGUF", "main", "Q8/x-Q8_0-00001-of-00002.gguf")
    assert parse_link("https://civitai.com/models/123/some-name?modelVersionId=456") == CivitaiRef(123, 456)
    assert parse_link("https://civitai.com/api/download/models/789") == CivitaiRef(None, 789)


@pytest.mark.parametrize("link,why", [
    ("https://civitai.com/images/123", "Civitai link"),
    ("https://example.com/o/m", "only Hugging Face"),
    ("not a repo", "isn't a Hugging Face repo"),
    ("", "paste"),
])
def test_parse_link_refuses(link, why):
    with pytest.raises(LinkError, match=why):
        parse_link(link)


# MARK: sizing and profiles

def test_qwen38_fp8_picks_two_h200_with_full_context():
    f = facts(QWEN38, int(172.78 * GIB), QWEN38_TEMPLATE)
    assert f.attention_layers == 12 and f.kv_heads == 2 and f.vision and f.quant == {"method": "fp8", "bits": 8}
    profile = guess(f)
    assert profile["toolParser"] == "qwen3_coder" and profile["reasoningParser"] == "qwen3"
    plan = sizing.plan(f, kv_dtype_bytes=1)
    rec = plan["recommended"]
    assert (rec["gpu"], rec["count"]) == ("H200", 2)
    assert rec["maxContext"] == 262144 and plan["context"] == 131072
    by = {(o["gpu"], o["count"]): o for o in plan["options"]}
    assert not by[("B200", 1)]["fits"] and not by[("H100", 2)]["fits"] and by[("H100", 4)]["fits"]
    assert ("L40S", 4) not in by  # no NVLink: at most 2


def test_ngram_tables_in_cpu_memory_let_qwen_fit_one_b200():
    # Measured on Modal: vLLM keeps Qwen3.8-Flash-Next's n-gram tables (51.2 GB) in CPU memory, and 124.3 GiB of
    # weights went on one B200. One H200 is too small for that.
    f = facts(QWEN38, int(172.78 * GIB), QWEN38_TEMPLATE)
    f.cpu_bytes = 51_200_000_000
    by = {(o["gpu"], o["count"]): o for o in sizing.plan(f, kv_dtype_bytes=1)["options"]}
    assert by[("B200", 1)]["fits"] and not by[("H200", 1)]["fits"]
    assert 128 < by[("B200", 1)]["weightsPerGPUGB"] < 140  # 124.3 GiB measured
    assert by[("A100-80GB", 2)]["fits"] and by[("A100-80GB", 2)]["note"]  # cheaper, but FP8 runs slower there
    rec = sizing.plan(f, kv_dtype_bytes=1)["recommended"]
    assert (rec["gpu"], rec["count"]) == ("B200", 1)  # $6.25/h against 2× H200's $9.08


def safetensors_file(tensors: dict[str, int]) -> bytes:
    """A safetensors file's header and data: name → byte size, laid out one after another."""
    header, at = {"__metadata__": {"format": "pt"}}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "F8_E4M3", "shape": [size], "data_offsets": [at, at + size]}
        at += size
    raw = json.dumps(header).encode()
    return len(raw).to_bytes(8, "little") + raw + b"\0" * at


def test_ngram_table_sizes_come_from_the_safetensors_headers():
    from aistudio_gpu.llm.facts import inspect
    files = {"model-1.safetensors": safetensors_file({"layers.1.ple.ple_embedding.ngram_embedding.weight": 3000,
                                                      "layers.1.ple.key_proj.weight": 100}),
             "model-2.safetensors": safetensors_file({"layers.2.mlp.weight": 500})}
    config = {**QWEN38, "text_config": {**QWEN38["text_config"], "ngram_vocab_size_base": 20_000_000,
                                        "ple_embed_dim": 2560}}
    index = {"metadata": {"total_size": 3600}, "weight_map": {
        "layers.1.ple.ple_embedding.ngram_embedding.weight": "model-1.safetensors",
        "layers.1.ple.key_proj.weight": "model-1.safetensors", "layers.2.mlp.weight": "model-2.safetensors"}}

    def site(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/models/o/ngram/revision/main":
            return httpx.Response(200, json={"id": "o/ngram", "sha": SHA, "siblings": [
                {"rfilename": "config.json", "size": 100},
                {"rfilename": "model.safetensors.index.json", "size": 100},
                *({"rfilename": n, "size": len(b)} for n, b in files.items())]})
        if path == f"/o/ngram/resolve/{SHA}/config.json":
            return httpx.Response(200, json=config)
        if path == f"/o/ngram/resolve/{SHA}/model.safetensors.index.json":
            return httpx.Response(200, json=index)
        name = path.rpartition("/")[2]
        if name in files:
            assert name == "model-1.safetensors"  # only the files that hold the tables are read
            data = files[name]
            start, _, end = request.headers["range"].removeprefix("bytes=").partition("-")
            start, end = int(start), min(int(end), len(data) - 1)
            return httpx.Response(206, content=data[start:end + 1],
                                  headers={"content-range": f"bytes {start}-{end}/{len(data)}"})
        return httpx.Response(404)

    import asyncio
    transport = httpx.MockTransport(site)
    f = asyncio.run(inspect(parse_link("o/ngram"), {}, httpx.AsyncClient(transport=transport),
                            lambda: httpx.Client(transport=transport)))
    assert f.cpu_bytes == 3000
    assert any("n-gram embeddings" in n for n in f.notes)

    def unreadable(request: httpx.Request) -> httpx.Response:  # no Range support: the size from the config
        return httpx.Response(200, content=b"x") if request.url.path.endswith(".safetensors") else site(request)

    transport = httpx.MockTransport(unreadable)
    f = asyncio.run(inspect(parse_link("o/ngram"), {}, httpx.AsyncClient(transport=transport),
                            lambda: httpx.Client(transport=transport)))
    assert f.cpu_bytes == 20_000_000 * 2560  # FP8: a byte each


def test_dense_8b_fits_one_l40s():
    f = facts(LLAMA8B, 16 * 10**9)
    assert f.head_dim == 128 and f.attention_layers == 32
    rec = sizing.plan(f)["recommended"]
    assert (rec["gpu"], rec["count"]) == ("L40S", 1) and rec["maxContext"] == 131072
    assert guess(f)["toolParser"] == "llama3_json"


def test_mla_model_needs_eight_h200():
    f = facts(DEEPSEEK, 671 * 10**9)
    assert f.kv_lora_rank == 512
    # The latent cache isn't split across GPUs: 61 layers × (512 + 64) × 2 bytes per token on every GPU.
    assert sizing.kv_bytes_per_token(f, 8) == 61 * 576 * 2
    rec = sizing.plan(f)["recommended"]
    assert (rec["gpu"], rec["count"]) == ("H200", 8)
    assert guess(f)["reasoningParser"] == "deepseek_r1"


def test_vllm_command_has_parsers_and_expert_parallel(tmp_path):
    m = {"id": "qwen38", "count": 2, "context": 131072, "toolParser": "qwen3_coder", "reasoningParser": "qwen3",
         "experts": 512, "engineArgs": ["--kv-cache-dtype", "fp8"]}
    cmd = vllm_command(m, tmp_path, 8000)
    assert cmd[:3] == ["vllm", "serve", str(tmp_path)]
    joined = " ".join(cmd)
    assert "--tensor-parallel-size 2" in joined and "--max-model-len 131072" in joined
    assert "--tool-call-parser qwen3_coder" in joined and "--reasoning-parser qwen3" in joined
    assert "--enable-expert-parallel" in cmd and cmd[-2:] == ["--kv-cache-dtype", "fp8"]
    assert "--enable-expert-parallel" not in vllm_command({**m, "count": 1}, tmp_path, 8000)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "64"  # one person's chats, not vLLM's 1024
    own = vllm_command({**m, "engineArgs": ["--max-num-seqs", "8"]}, tmp_path, 8000)
    assert own.count("--max-num-seqs") == 1 and own[own.index("--max-num-seqs") + 1] == "8"


def test_vllm_records_few_cuda_graphs_unless_the_model_sets_its_own(tmp_path):
    m = {"id": "qwen38", "count": 1, "context": 8192, "engineArgs": []}
    cmd = vllm_command(m, tmp_path, 8000)
    assert json.loads(cmd[cmd.index("--compilation-config") + 1]) == {"cudagraph_capture_sizes": [1, 2, 4, 8]}
    for own in (["--compilation-config", "{}"], ["-O3"], ["-cc.mode=0"], ["--enforce-eager"]):
        assert vllm_command({**m, "engineArgs": own}, tmp_path, 8000).count("--compilation-config") == \
            own.count("--compilation-config")


def test_vllm_reuses_the_measured_kv_cache_size_unless_the_model_sets_memory(tmp_path):
    m = {"id": "qwen38", "count": 1, "context": 8192, "engineArgs": []}
    assert "--kv-cache-memory" not in vllm_command(m, tmp_path, 8000)
    cmd = vllm_command(m, tmp_path, 8000, kv_bytes=34803106714)
    assert cmd[cmd.index("--kv-cache-memory") + 1] == "34803106714"
    for own in (["--kv-cache-memory", "1000"], ["--gpu-memory-utilization=0.8"]):
        assert vllm_command({**m, "engineArgs": own}, tmp_path, 8000, kv_bytes=5).count("--kv-cache-memory") == \
            own.count("--kv-cache-memory")


def test_engine_remembers_vllms_kv_advice_per_setup_and_forgets_it_when_a_start_fails(tmp_path):
    from aistudio_gpu.llm.engine import EngineError, EngineProcess

    m = {"id": "qwen38", "gpu": "B200", "count": 1, "context": 131072, "engineArgs": ["--kv-cache-dtype", "fp8"]}

    def engine(manifest, lines, exit_code=None):
        p = EngineProcess(manifest, tmp_path, lambda *a: None)
        p.kv_bytes = p._remembered_kv()
        p.proc = type("P", (), {"stdout": iter(lines), "poll": lambda self: exit_code})()
        return p

    advice = ("INFO [gpu_worker.py:1066] Free memory on device (177.53/178.34 GiB) on startup. Replace "
              "gpu_memory_utilization config with `--kv-cache-memory=34803106714` (32.41 GiB) to fit into requested "
              "memory, or `--kv-cache-memory=53084203520` (49.44 GiB) to fully utilize gpu memory.\n")
    engine(m, [advice])._read()
    p = engine(m, [])
    assert p.kv_bytes == 34803106714
    other = engine({**m, "gpu": "H200"}, [])
    assert other.kv_bytes is None  # another GPU kind measures its own
    assert engine({**m, "context": 8192}, []).kv_bytes is None
    failed = engine(m, [], exit_code=1)
    with pytest.raises(EngineError):
        failed.wait_ready(timeout=5)
    assert engine(m, []).kv_bytes is None  # measured again on the next start


def test_vllm_reads_chat_media_from_the_uploads_folder_for_vision_models(tmp_path):
    m = {"id": "qwen38", "count": 1, "context": 8192, "vision": True, "engineArgs": ["--kv-cache-dtype", "fp8"]}
    cmd = vllm_command(m, tmp_path, 8000, media="/uploads")
    assert cmd[cmd.index("--allowed-local-media-path") + 1] == "/uploads" and cmd[-2:] == ["--kv-cache-dtype", "fp8"]
    assert "--allowed-local-media-path" not in vllm_command({**m, "vision": False}, tmp_path, 8000, media="/uploads")
    assert "--allowed-local-media-path" not in vllm_command(m, tmp_path, 8000)


def test_phases_from_vllm_log():
    assert phase_of("Loading safetensors checkpoint shards:  41% Completed | 12/29") == ("Loading weights", 0.41)
    assert phase_of("Capturing CUDA graphs (mixed prefill-decode): 100%|") == ("Capturing CUDA graphs", 1.0)
    assert phase_of("INFO 11-11 hello") is None
    # From a real Qwen3.8-Flash-Next start on Modal: the config line about the KV cache comes before the weights.
    assert phase_of("INFO [cache.py:484] Using fp8 data type to store kv cache. It reduces the GPU memory") is None
    assert phase_of("INFO [utils.py:381] Using BLNHC KV cache layout.") == ("Allocating KV cache", None)
    assert phase_of("Loading safetensors using Runai Model Streamer:  61% Completed | 93070/152089") == \
        ("Loading weights", 0.61)
    assert phase_of("flashinfer.jit: Building JIT module fused_moe_90; this can take several minutes") == \
        ("Building GPU kernels", None)


def test_engine_phases_dont_go_back_to_starting():
    from aistudio_gpu.llm.engine import EngineProcess

    seen = []
    p = EngineProcess.__new__(EngineProcess)
    p.on_status = lambda name, progress: seen.append(name)
    p.log = []
    p.proc = type("P", (), {"stdout": iter([
        "Initializing a V1 LLM engine (v0.33.0)\n", "Loading weights took 128.63 seconds\n",
        "INFO [base_loader.py:82] Initializing model took 9.6 seconds\n", "Using BLNHC KV cache layout.\n"])})()
    p._read()
    assert seen == ["Starting engine", "Loading weights", "Allocating KV cache"]


# MARK: the API

SHA = "b" * 40


def fake_hf(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/models/o/qwen38/revision/main":
        return httpx.Response(200, json={"id": "o/qwen38", "sha": SHA, "siblings": [
            {"rfilename": "config.json", "size": 100}, {"rfilename": "chat_template.jinja", "size": 50},
            {"rfilename": "tokenizer.json", "size": 10}, {"rfilename": "README.md", "size": 5},
            {"rfilename": "model-00001-of-00002.safetensors", "size": 100 * GIB},
            {"rfilename": "model-00002-of-00002.safetensors", "size": int(72.78 * GIB)}]})
    if path == "/api/models/o/gated/revision/main":
        return httpx.Response(401, json={"error": "gated"})
    if path == f"/o/qwen38/resolve/{SHA}/config.json":
        return httpx.Response(200, json=QWEN38)
    if path == f"/o/qwen38/resolve/{SHA}/chat_template.jinja":
        return httpx.Response(200, text=QWEN38_TEMPLATE)
    return httpx.Response(404)


def fetched(repo, revision, name, dest, token):
    assert repo == "o/qwen38" and revision == SHA
    (dest / name).write_bytes(b"{}" if name.endswith(".json") else b"x")


@pytest.fixture()
def llm(tmp_path: Path):
    return LocalLLMService(tmp_path / "models", http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake_hf)),
                           fetch=fetched, engine_factory=lambda m, r, s: FakeEngine(m, r, s, boot_seconds=0.4),
                           idle=60, lease=60, billed=True)


@pytest.fixture()
def client(tmp_path: Path, llm, monkeypatch):
    monkeypatch.setattr(proxy, "PING_SECONDS", 0.05)
    runner = LocalRunner(tmp_path / "models", allow_gpu_tasks=False)
    manager = JobManager(runner, FileStore(tmp_path / "uploads"), tmp_path / "results")
    with TestClient(create_app(manager, TOKEN, llm=llm)) as c:
        yield c


def sse(text: str) -> list[tuple[str | None, str]]:
    """(event name, data) per event; comments as ("comment", text)."""
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith(":"):
                out.append(("comment", line[1:].strip()))
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            out.append((name, "\n".join(data)))
    return out


def install(client, **extra):
    r = client.post("/v1/llm/install", json={"url": "o/qwen38", **extra}, headers=AUTH)
    assert r.status_code == 202, r.text
    for _ in range(100):
        listing = client.get("/v1/llm/models", headers=AUTH).json()
        if any(m["id"] == r.json()["id"] and m["installed"] for m in listing["models"]):
            return r.json()["id"]
        time.sleep(0.05)
    raise AssertionError(f"install didn't finish: {listing}")


def chat(client, mid, text, **extra):
    body = {"model": mid, "stream": True, "messages": [{"role": "user", "content": text}], **extra}
    with client.stream("POST", "/v1/llm/oai/v1/chat/completions", json=body, headers=AUTH) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        return sse(r.read().decode())


def test_needs_token_and_info_says_llm(client):
    assert client.get("/v1/llm/models").status_code == 401
    assert client.get("/v1/info", headers=AUTH).json()["llm"] is True


def test_inspect(client):
    r = client.post("/v1/llm/inspect", json={"url": "https://huggingface.co/o/qwen38"}, headers=AUTH)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["facts"]["modelType"] == "qwen4_exp" and d["facts"]["vision"] and d["facts"]["hasChatTemplate"]
    assert d["profile"]["toolParser"] == "qwen3_coder"
    assert (d["plan"]["recommended"]["gpu"], d["plan"]["recommended"]["count"]) == ("H200", 2)
    assert d["suggestedId"] == "qwen38"
    gated = client.post("/v1/llm/inspect", json={"url": "o/gated"}, headers=AUTH)
    assert gated.status_code == 400 and "gated" in gated.json()["error"]
    bad = client.post("/v1/llm/inspect", json={"url": "https://civitai.com/models/1"}, headers=AUTH)
    assert bad.status_code == 400 and "Civitai" in bad.json()["error"]


def test_install_list_patch_remove(client, tmp_path):
    mid = install(client)
    m = next(m for m in client.get("/v1/llm/models", headers=AUTH).json()["models"] if m["id"] == mid)
    assert (m["gpu"], m["count"], m["context"]) == ("H200", 2, 131072)
    assert m["status"]["state"] == "cold" and m["status"]["usdPerSecond"] > 0
    weights = tmp_path / "models" / "llm" / mid / "weights"
    assert (weights / ".complete").exists() and (weights / "config.json").exists()
    assert not (weights / "README.md").exists()
    assert client.post("/v1/llm/install", json={"url": "o/qwen38", "id": mid}, headers=AUTH).status_code == 409
    r = client.patch(f"/v1/llm/models/{mid}", json={"gpu": "H100", "count": 4}, headers=AUTH)
    assert r.status_code == 200 and r.json()["count"] == 4
    assert client.patch(f"/v1/llm/models/{mid}", json={"repo": "x/y"}, headers=AUTH).status_code == 400
    models = client.get("/v1/llm/oai/v1/models", headers=AUTH).json()
    assert [d["id"] for d in models["data"]] == [mid]
    assert client.delete(f"/v1/llm/models/{mid}", headers=AUTH).status_code == 204
    assert client.delete(f"/v1/llm/models/{mid}", headers=AUTH).status_code == 404


def test_install_picked_gpu_must_be_an_option(client):
    r = client.post("/v1/llm/install", json={"url": "o/qwen38", "gpu": "L40S", "count": 1}, headers=AUTH)
    assert r.status_code == 400


def test_chat_cold_start_streams_status_then_reply_and_usage(client):
    mid = install(client)
    events = chat(client, mid, "hello there")
    names = [n for n, _ in events]
    assert "comment" in names  # pings while the fake engine boots
    statuses = [json.loads(d) for n, d in events if n == "aistudio.status"]
    assert statuses and statuses[0]["state"] in ("starting", "loading")
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning") or "" for c in chunks if c["choices"])
    assert text == "You said: hello there" and reasoning == "Thinking about it."
    assert any(c.get("usage") for c in chunks)  # include_usage was forced
    usage = json.loads(next(d for n, d in events if n == "aistudio.usage"))
    assert usage["completionTokens"] and usage["costUSD"] >= 0 and usage["ttftMs"] is not None
    assert events[-1] == (None, "[DONE]")
    status = client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()
    assert status["state"] == "ready" and status["uptime"] > 0


def test_tool_call_and_error_pass_through(client):
    mid = install(client)
    events = chat(client, mid, "please use a tool")
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in
             (json.loads(d) for n, d in events if n is None and d != "[DONE]") if c["choices"]
             and c["choices"][0]["delta"].get("tool_calls")]
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"path": "README.md"}
    failed = chat(client, mid, "now fail")
    assert any(n == "aistudio.error" and "on purpose" in d for n, d in failed)
    assert failed[-1] == (None, "[DONE]")


def test_non_streaming_chat(client):
    mid = install(client)
    r = client.post("/v1/llm/oai/v1/chat/completions", headers=AUTH,
                    json={"model": mid, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "You said: hi"
    assert client.post("/v1/llm/oai/v1/chat/completions", headers=AUTH,
                       json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]}).status_code == 404


def test_warm_keepalive_lease_and_idle_stop(client, llm):
    mid = install(client)
    s = client.post(f"/v1/llm/models/{mid}/warm", headers=AUTH).json()
    assert s["state"] in ("starting", "loading")
    for _ in range(50):
        if client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()["state"] == "ready":
            break
        time.sleep(0.05)
    assert client.post(f"/v1/llm/models/{mid}/keepalive", headers=AUTH).json().get("ignored") is None
    llm.lease = 0.0
    assert client.post(f"/v1/llm/models/{mid}/keepalive", headers=AUTH).json()["ignored"] is True
    llm.idle = 0.2
    for _ in range(60):
        if client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()["state"] == "cold":
            break
        time.sleep(0.05)
    assert client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()["state"] == "cold"
    assert client.post(f"/v1/llm/models/{mid}/stop", headers=AUTH).json()["state"] == "cold"


def test_fake_agent_script(client):
    mid = install(client)
    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "create hello.txt and show it"}]
    names = []
    for step in range(2):
        events = chat(client, mid, "", messages=msgs, tools=tools)
        chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
        call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                    if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
        names.append(call["function"]["name"])
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"], "content": f"result {step}"}]
    assert names == ["write_file", "run_command"]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    text = "".join(json.loads(d)["choices"][0]["delta"].get("content") or "" for n, d in events
                   if n is None and d != "[DONE]" and json.loads(d)["choices"])
    assert text == "Done: result 1"


def test_fake_session_script(client):
    mid = install(client)
    tools = [{"type": "function", "function": {"name": "start_session", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "start a session to create hello.txt"}]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
    assert call["function"]["name"] == "start_session"
    assert json.loads(call["function"]["arguments"]) == {"title": "Hello file", "prompt": "create hello.txt and show it",
                                                         "worktree": True}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": call["id"], "content": "Started"}]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    text = "".join(json.loads(d)["choices"][0]["delta"].get("content") or "" for n, d in events
                   if n is None and d != "[DONE]" and json.loads(d)["choices"])
    assert text == "Done: Started"
    # without start_session offered, the same message falls through to the hello.txt script
    write = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]
    events = chat(client, mid, "", messages=msgs[:1], tools=write)
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
    assert call["function"]["name"] == "write_file"


def test_fake_agent_todo_script(client):
    mid = install(client)
    tools = [{"type": "function", "function": {"name": "todo_write", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "create hello.txt with a todo list"}]
    names = []
    for step in range(3):
        events = chat(client, mid, "", messages=msgs, tools=tools)
        chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
        call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                    if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
        names.append(call["function"]["name"])
        if step == 0:
            todos = json.loads(call["function"]["arguments"])["todos"]
            assert [t["activeForm"] for t in todos] == ["Writing hello.txt", "Showing it"]
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"], "content": f"result {step}"}]
    assert names == ["todo_write", "write_file", "run_command"]


def test_fake_commit_and_pr_drafts(client):
    mid = install(client)
    for ask, reply in [("Write the commit message now.", FAKE_COMMIT), ("Write the pull request now.", FAKE_PR)]:
        events = chat(client, mid, "", messages=[{"role": "user", "content": f"Diff:\n...\n\n{ask}"}])
        chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        assert text == reply


def test_fake_graph_helper_script(client):
    mid = install(client)
    tools = [{"type": "function", "function": {"name": "edit_graph", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "Add an upscale after Save"}]
    calls = []
    for step in range(2):
        events = chat(client, mid, "", messages=msgs, tools=tools)
        chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
        call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                    if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
        calls.append(call)
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"], "content": f"line {step}\nmore"}]
    assert [c["function"]["name"] for c in calls] == ["get_graph", "edit_graph"]
    ops = json.loads(calls[1]["function"]["arguments"])["operations"]
    assert [o["op"] for o in ops] == ["add", "connect"]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    text = "".join(json.loads(d)["choices"][0]["delta"].get("content") or "" for n, d in events
                   if n is None and d != "[DONE]" and json.loads(d)["choices"])
    assert text == "Done: line 1"


def test_fake_code_graph_script(client):
    mid = install(client)
    tools = [{"type": "function", "function": {"name": "code_graph", "parameters": {"type": "object"}}},
             {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "Map the project"}]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
    assert call["function"]["name"] == "code_graph"
    assert json.loads(call["function"]["arguments"]) == {"action": "query", "target": "greet"}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": call["id"],
              "content": "Start: greet() (5 nodes)\ngreet()  Sources/Greeter.swift:1\n.main()  Sources/Main.swift:2"}]
    events = chat(client, mid, "", messages=msgs, tools=tools)
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "Sources/Greeter.swift", "offset": 1, "limit": 20}
    # without code_graph offered, the same message gets no code_graph call
    events = chat(client, mid, "", messages=[{"role": "user", "content": "Map the project"}],
                  tools=[{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}])
    _, calls = reply_text(events)
    assert not any(c["function"]["name"] == "code_graph" for c in calls)


# MARK: cold starts (M28)

def test_cold_start_clock_splits_phases():
    from aistudio_gpu.llm.coldstart import ColdStartClock, remember, summary

    t = [100.0]
    clock = ColdStartClock(requested=90.0, now=lambda: t[0])
    for name, at in [("Starting engine", 101), ("Loading weights", 105), ("Loading weights", 140), ("Compiling", 150),
                     ("Loading weights", 160), ("Ready", 170)]:
        t[0] = at
        clock.phase(name)
    t[0] = 171
    r = clock.finish(gpu="H200", count=2)
    assert r["seconds"] == 81 and r["waitSeconds"] == 10 and r["how"] == "cold" and r["gpu"] == "H200"
    assert r["phases"] == [{"name": "Waiting for a GPU", "seconds": 10.0}, {"name": "Starting container", "seconds": 1.0},
                           {"name": "Starting engine", "seconds": 4.0}, {"name": "Loading weights", "seconds": 56.0},
                           {"name": "Compiling", "seconds": 10.0}]
    # A container that started before the recorded request (a stale one) has no GPU wait.
    assert ColdStartClock(requested=200.0, started=100.0).finish()["waitSeconds"] == 0.0
    # Restored from a snapshot: the wait includes the restore, and there's no container start-up phase.
    snap = ColdStartClock(requested=90.0, started=100.0, how="snapshot", now=lambda: 100.0)
    assert [p["name"] for p in snap.finish()["phases"]] == ["Getting a GPU and restoring the snapshot"]

    history = []
    for i, secs in enumerate([300, 200, 100, 120, 110, 90]):
        history = remember(history, {"seconds": secs, "updated": 1.0 if i < 2 else 2.0})
    s = summary(history, updated=2.0)
    assert s["last"]["seconds"] == 90 and s["typicalSeconds"] == 105 and s["count"] == 6
    assert summary([], None) is None
    assert len(remember([{"seconds": 1}] * 10, {"seconds": 2})) == 10


def test_load_format_and_snapshot_settings(tmp_path):
    from aistudio_gpu.llm import store

    m = {"id": "x", "engine": "vllm", "count": 1, "context": 8192, "engineArgs": []}
    assert store.patch(m, {"loadFormat": "runai", "snapshot": True})["snapshot"] is True
    for bad in [{"loadFormat": "turbo"}, {"snapshot": "yes"}]:
        with pytest.raises(ValueError):
            store.patch(m, bad)
    with pytest.raises(ValueError, match="one GPU"):
        store.patch({**m, "count": 2}, {"snapshot": True})
    with pytest.raises(ValueError, match="vLLM"):
        store.patch({**m, "engine": "llamacpp"}, {"loadFormat": "runai"})

    cmd = vllm_command({**m, "loadFormat": "runai", "snapshot": True}, tmp_path, 8000)
    i = cmd.index("--load-format")
    assert cmd[i + 1] == "runai_streamer" and json.loads(cmd[i + 3]) == {"concurrency": 16}
    assert "--enable-sleep-mode" in cmd
    plain = vllm_command(m, tmp_path, 8000)
    assert "--load-format" not in plain and "--enable-sleep-mode" not in plain


def test_fast_start_loads_the_saved_processed_copy_per_gpu(tmp_path):
    from aistudio_gpu.llm import store

    m = {"id": "x", "engine": "vllm", "count": 1, "context": 8192, "engineArgs": [], "gpu": "B200"}
    assert store.patch(m, {"loadFormat": "processed"})["loadFormat"] == "processed"
    with pytest.raises(ValueError, match="one GPU"):
        store.patch({**m, "count": 2}, {"loadFormat": "processed"})
    with pytest.raises(ValueError, match="vLLM"):
        store.patch({**m, "engine": "llamacpp"}, {"loadFormat": "processed"})
    weights = tmp_path / "llm" / "x" / "weights"
    cmd = vllm_command({**m, "loadFormat": "processed"}, weights, 8000)
    i = cmd.index("--load-format")
    assert cmd[i + 1] == "aistudio_processed"
    assert json.loads(cmd[i + 3]) == {"dir": str(tmp_path / "llm" / "x" / "processed" / "B200")}
    assert store.processed_dir(tmp_path, "x") == tmp_path / "llm" / "x" / "processed"


def test_fast_start_phases():
    assert phase_of("INFO [__init__.py:9] Saving processed weights to /models/llm/x/processed/B200 (later starts")[0] \
        == "Saving fast-start weights"
    assert phase_of("INFO Loading processed weights from /models/llm/x/processed/B200")[0] == "Loading weights"
    assert phase_of("INFO No usable processed weights in /m (there is none yet): loading the checkpoint")[0] \
        == "Loading weights"


def test_turning_fast_start_off_deletes_its_copy(client, llm, tmp_path):
    from aistudio_gpu.llm import store

    mid = install(client)
    m = store.read(tmp_path / "models", mid)
    store.write(tmp_path / "models", {**m, "gpu": "B200", "count": 1})  # fast start is for one GPU
    r = client.patch(f"/v1/llm/models/{mid}", json={"loadFormat": "processed"}, headers=AUTH)
    assert r.status_code == 200, r.text
    copy = store.processed_dir(tmp_path / "models", mid) / "B200"
    copy.mkdir(parents=True)
    (copy / "processed-00000.safetensors").write_bytes(b"x")
    assert client.patch(f"/v1/llm/models/{mid}", json={"name": "renamed"}, headers=AUTH).status_code == 200
    assert copy.exists()  # other changes keep it
    assert client.patch(f"/v1/llm/models/{mid}", json={"loadFormat": "auto"}, headers=AUTH).status_code == 200
    assert not store.processed_dir(tmp_path / "models", mid).exists()


def test_plugin_marks_only_validation_only_processing():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "aistudio_vllm_plugin", Path(__file__).parent.parent / "vllm_plugin" / "aistudio_vllm_plugin" / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)  # stdlib only at import: vLLM isn't needed here
    sys.modules.pop("aistudio_vllm_plugin", None)
    ple = '''    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")
'''
    marker = plugin.VALIDATION_ONLY["Qwen4ExpPLEFp8EmbeddingMethod"]
    assert plugin.validation_only(ple, marker)
    for write in ["layer.weight_scale = s", "layer.weight.data.mul_(2)", "replace_parameter(layer, 'weight', w)",
                  "layer.weight.copy_(w)", "layer.w13 += 1", "setattr(layer, 'weight', w)",
                  "layer.register_buffer('b', t)"]:
        assert not plugin.validation_only(ple + "        " + write + "\n", marker), write
    assert not plugin.validation_only(ple.replace("missing its global scale", "other"), marker)


def test_status_reports_the_last_and_typical_cold_start(client):
    mid = install(client)
    status = client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()
    assert status["coldStart"] is None and status["bootSeconds"] is None
    client.post(f"/v1/llm/models/{mid}/warm", headers=AUTH)
    seen_boot = None
    for _ in range(100):
        status = client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()
        if status["state"] in ("starting", "loading") and status["bootSeconds"] is not None:
            seen_boot = status["bootSeconds"]
        if status["state"] == "ready" and status["coldStart"]:
            break
        time.sleep(0.05)
    assert seen_boot is not None and seen_boot >= 0
    last = status["coldStart"]["last"]
    assert status["coldStart"]["count"] == 1 and last["how"] == "cold" and last["loadFormat"] == "auto"
    assert [p["name"] for p in last["phases"]][:2] == ["Starting engine", "Loading weights"]
    assert 0.3 <= last["seconds"] < 5 and status["coldStart"]["typicalSeconds"] == last["seconds"]


def test_fake_engine_sleeps_and_wakes():
    seen = []
    e = FakeEngine({"id": "m"}, Path("."), lambda p, x: seen.append(p), boot_seconds=0)
    e.start()
    e.wait_ready()
    e.warmup()
    e.sleep()
    e.wake_up()
    assert seen[-3:] == ["Asleep", "Waking up", "Ready"]


def video_message(name: str, text: str = "what happens here?") -> list:
    return [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": f"upload:{name}"}},
                                         {"type": "text", "text": text}]}]


def test_chat_videos_are_uploads_the_engine_reads_from_disk(client, tmp_path, monkeypatch):
    seen = []
    real = FakeEngine.chat

    def recording(self, body):
        seen.append(body)
        return real(self, body)

    monkeypatch.setattr(FakeEngine, "chat", recording)
    mid = install(client)
    data = b"\x00\x00\x00\x18ftypmp42 a tiny video"
    name = hashlib.sha256(data).hexdigest() + ".mp4"
    url = "/v1/llm/oai/v1/chat/completions"
    missing = client.post(url, json={"model": mid, "messages": video_message(name)}, headers=AUTH)
    assert missing.status_code == 404 and "upload it first" in missing.json()["error"]

    assert client.put(f"/v1/files/{name}", content=data, headers=AUTH).status_code == 201
    events = chat(client, mid, "", messages=video_message(name))
    assert events[-1] == (None, "[DONE]")
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]).endswith("[1 video]")
    part = seen[-1]["messages"][0]["content"][0]
    path = (tmp_path / "uploads" / name).absolute()
    assert part == {"type": "video_url", "video_url": {"url": path.as_uri()}}
    assert local_media(seen[-1]["messages"], (tmp_path / "uploads").absolute()) == [path]
    assert local_media(seen[-1]["messages"], tmp_path / "elsewhere") == []

    bad = client.post(url, json={"model": mid, "messages": video_message("../../etc/passwd")}, headers=AUTH)
    assert bad.status_code == 400


def test_models_without_vision_or_on_llamacpp_refuse_videos(llm, tmp_path):
    llm.files = FileStore(tmp_path / "uploads")
    for m in ({"id": "text", "vision": False}, {"id": "gguf", "vision": True, "engine": "llamacpp"}):
        with pytest.raises(ValueError, match="can't watch videos"):
            llm.resolve_uploads(m, video_message("a" * 64 + ".mp4"))
    plain = [{"role": "user", "content": "hi"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
    assert llm.resolve_uploads({"id": "text", "vision": False}, plain) == plain


def chat_tools():
    return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
            for n in ("ask_user", "show_widget")]


def reply_text(events):
    chunks = [json.loads(d) for n, d in events if n is None and d != "[DONE]"]
    calls = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls") or []]
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]), calls


def test_fake_engine_scripts_the_chat_tools(client):
    mid = install(client)
    text, calls = reply_text(chat(client, mid, "ask me something", tools=chat_tools()))
    assert calls and calls[0]["function"]["name"] == "ask_user"
    assert json.loads(calls[0]["function"]["arguments"])["questions"][0]["header"] == "Database"
    msgs = [{"role": "user", "content": "ask me something"},
            {"role": "assistant", "content": "Working on it.", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "ask_user", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "The user answered:\nDatabase: SQLite"}]
    text, calls = reply_text(chat(client, mid, "", messages=msgs, tools=chat_tools()))
    assert text == "Done: The user answered: Database: SQLite" and not calls

    _, calls = reply_text(chat(client, mid, "show a widget", tools=chat_tools()))
    assert calls[0]["function"]["name"] == "show_widget"
    assert "aistudio.sendPrompt" in json.loads(calls[0]["function"]["arguments"])["html"]
    # Without the tool in the request, no script: an echo.
    text, calls = reply_text(chat(client, mid, "show a widget"))
    assert text == "You said: show a widget" and not calls

    text, _ = reply_text(chat(client, mid, "suggest something"))
    assert text.endswith("<followups>\n- Tell me more\n- Show an example\n</followups>")


def test_upload_urls_keep_the_uploads_path_through_links(llm, tmp_path):
    # On Modal /uploads is a symlink to the container's own Volume mount: the URL must say /uploads.
    real = tmp_path / "volume-mount"
    real.mkdir()
    link = tmp_path / "uploads"
    link.symlink_to(real)
    llm.files = FileStore(link)
    name = "b" * 64 + ".mp4"
    (real / name).write_bytes(b"x")
    msgs = llm.resolve_uploads({"id": "m", "vision": True}, video_message(name))
    url = msgs[0]["content"][0]["video_url"]["url"]
    assert url == (link / name).absolute().as_uri() and "volume-mount" not in url


def test_plugin_keeps_a_speculative_drafts_copy_apart_from_the_models():
    import importlib.util
    import sys
    from types import SimpleNamespace as NS

    spec = importlib.util.spec_from_file_location(
        "aistudio_vllm_plugin", Path(__file__).parent.parent / "vllm_plugin" / "aistudio_vllm_plugin" / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    sys.modules.pop("aistudio_vllm_plugin", None)
    target = NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"]))
    mtp = NS(hf_config=NS(architectures=["Qwen4ExpMTP"]))
    vc = NS(model_config=target)
    assert plugin.copy_dir("/m/processed/B200", vc, target) == "/m/processed/B200"
    assert plugin.copy_dir("/m/processed/B200", vc, None) == "/m/processed/B200"
    assert plugin.draft_of(vc, mtp) == "Qwen4ExpMTP"
    assert plugin.copy_dir("/m/processed/B200", vc, mtp) == "/m/processed/B200/draft-Qwen4ExpMTP"


def _checkpoint(d, config, names, index=True):
    import struct

    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    if index:
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "a.safetensors" for n in names}}))
    else:
        head = json.dumps({"__metadata__": {}, **{n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
                                                  for n in names}}).encode()
        (d / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")
    return d


def test_mtp_layers_need_the_config_and_the_tensors(tmp_path):
    from aistudio_gpu.llm import store

    qwen = {"text_config": {"mtp_num_hidden_layers": 1, "num_hidden_layers": 48}}
    assert store.mtp_layers(_checkpoint(tmp_path / "q", qwen, ["model.layers.0.w", "mtp.layers.0.w"])) == 1
    assert store.mtp_layers(_checkpoint(tmp_path / "s", qwen, ["mtp.fc.weight"], index=False)) == 1
    # a quantized re-upload that kept the config but dropped the MTP tensors
    assert store.mtp_layers(_checkpoint(tmp_path / "d", qwen, ["model.layers.0.w"])) == 0
    deepseek = {"num_nextn_predict_layers": 1, "num_hidden_layers": 61}
    assert store.mtp_layers(_checkpoint(tmp_path / "ds", deepseek, ["model.layers.61.eh_proj.weight"])) == 1
    assert store.mtp_layers(_checkpoint(tmp_path / "n", {"num_hidden_layers": 4}, ["mtp.x"])) == 0
    assert store.mtp_layers(tmp_path / "missing") == 0


def test_vllm_drafts_with_mtp_layers_unless_turned_off(tmp_path):
    from aistudio_gpu.llm import store

    m = {"id": "qwen38", "count": 1, "context": 8192, "engineArgs": [], "mtpLayers": 1}
    cmd = vllm_command(m, tmp_path, 8000)
    assert json.loads(cmd[cmd.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 2}
    assert "--speculative-config" not in vllm_command({**m, "mtp": False}, tmp_path, 8000)
    assert "--speculative-config" not in vllm_command({**m, "mtpLayers": 0}, tmp_path, 8000)
    own = vllm_command({**m, "engineArgs": ["--speculative-config", "{}"]}, tmp_path, 8000)
    assert own.count("--speculative-config") == 1 and own[-1] == "{}"
    # models installed before the field: read from the checkpoint
    weights = _checkpoint(tmp_path / "w", {"mtp_num_hidden_layers": 1}, ["mtp.x"])
    assert "--speculative-config" in vllm_command({k: v for k, v in m.items() if k != "mtpLayers"}, weights, 8000)
    assert store.patch(m, {"mtp": False})["mtp"] is False
    with pytest.raises(ValueError):
        store.patch(m, {"mtp": "no"})


def test_listing_fills_in_mtp_layers_once(client, llm, tmp_path):
    from aistudio_gpu.llm import store

    mid = install(client)
    root = tmp_path / "models"
    m = store.read(root, mid)
    m.pop("mtpLayers", None)
    store.write(root, m)
    _checkpoint(store.weights_dir(root, mid), {"mtp_num_hidden_layers": 1}, ["mtp.x"])
    listed = {x["id"]: x for x in store.listing(root)}[mid]
    assert listed["mtpLayers"] == 1 and store.read(root, mid)["mtpLayers"] == 1


# MARK: flows: KV capacity, context cap, fake flow scripts

KV_LINES = [
    "(EngineCore pid=1) INFO [kv_cache_utils.py:1087] GPU KV cache size: 1,234,560 tokens",
    "(EngineCore pid=1) INFO [kv_cache_utils.py:1091] Maximum concurrency for 131,072 tokens per request: 9.42x",
]


def test_kv_capacity_parsed_from_vllm_lines(tmp_path):
    from aistudio_gpu.llm.engine import EngineProcess, parse_kv_capacity
    assert parse_kv_capacity(KV_LINES[0]) == {"kv_tokens": 1234560}
    assert parse_kv_capacity(KV_LINES[1]) == {"max_concurrency": 9.42}
    assert parse_kv_capacity("GPU KV cache size: 1234560 tokens") == {"kv_tokens": 1234560}
    assert parse_kv_capacity("Maximum concurrency for 131072 tokens per request: 9.42") == {"max_concurrency": 9.42}
    assert parse_kv_capacity("Loading weights 41%") == {}
    m = {"id": "q", "gpu": "B200", "count": 1, "context": 131072, "engineArgs": []}
    p = EngineProcess(m, tmp_path, lambda *a: None)
    p.proc = type("P", (), {"stdout": iter(l + "\n" for l in KV_LINES), "poll": lambda self: None})()
    p._read()
    assert (p.kv_tokens, p.max_concurrency) == (1234560, 9.42)
    gguf = EngineProcess({**m, "engine": "llamacpp"}, tmp_path, lambda *a: None)
    gguf.proc = type("P", (), {"stdout": iter(l + "\n" for l in KV_LINES), "poll": lambda self: None})()
    gguf._read()
    assert (gguf.kv_tokens, gguf.max_concurrency) == (0, 0.0)


def test_fake_engine_status_reports_kv_capacity(client):
    mid = install(client)
    assert "kvTokens" not in client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()
    chat(client, mid, "hi")
    s = client.get(f"/v1/llm/models/{mid}/status", headers=AUTH).json()
    assert s["kvTokens"] == 262144 and s["maxConcurrency"] == 2.0


def test_context_is_capped_at_1m_and_the_models_window():
    big = facts({**LLAMA8B, "max_position_embeddings": 2_000_000}, 16 * GIB)
    assert big.native_context == 2_000_000
    p = sizing.plan(big, 2_000_000)
    assert p["context"] == sizing.MAX_CONTEXT == 1_048_576 and p["nativeContext"] == 1_048_576
    assert all(o["maxContext"] <= sizing.MAX_CONTEXT for o in p["options"])
    assert all("kvTokens" in o for o in p["options"])
    small = facts({**LLAMA8B, "max_position_embeddings": 32768}, 16 * GIB)
    p = sizing.plan(small, 65536)
    assert p["context"] == 32768 and p["nativeContext"] == 32768
    o = next(o for o in p["options"] if o["fits"])
    assert o["kvTokens"] > 0 and o["maxContext"] == 32768
    assert not any(o["kvTokens"] for o in p["options"] if not o["fits"] and o["maxContext"] == 0)


def test_bigger_context_recommends_more_memory():
    f = facts({**LLAMA8B, "max_position_embeddings": 1_048_576}, 16 * GIB)

    def memory(ctx):
        rec = sizing.plan(f, ctx)["recommended"]
        return sizing.GPUS[rec["gpu"]] * rec["count"]
    assert memory(8192) < memory(1_048_576)


def test_too_large_a_context_is_refused(client):
    r = client.post("/v1/llm/install", json={"url": "o/qwen38", "context": 300000}, headers=AUTH)
    assert r.status_code == 400 and "more than this model's window" in r.text
    mid = install(client)
    r = client.patch(f"/v1/llm/models/{mid}", json={"context": 300000}, headers=AUTH)
    assert r.status_code == 400
    assert "more than this model's window" in r.text
    plan = client.get(f"/v1/llm/models/{mid}/plan?context=65536", headers=AUTH).json()
    assert plan["nativeContext"] == 262144 and "kvTokens" in plan["options"][0]


def tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in names]


def first_call(events):
    _, calls = reply_text(events)
    return calls[0]["function"] if calls else None


def test_fake_flow_scripts(client):
    from aistudio_gpu.llm.engine import FAKE_FLOW, FAKE_SLOW_FLOW
    mid = install(client)
    call = first_call(chat(client, mid, "", messages=[{"role": "user", "content": "Run a flow"}],
                           tools=tool_defs("flow")))
    assert call["name"] == "flow" and json.loads(call["arguments"]) == {"script": FAKE_FLOW}
    call = first_call(chat(client, mid, "", messages=[{"role": "user", "content": "Run a slow flow"}],
                           tools=tool_defs("flow")))
    assert json.loads(call["arguments"]) == {"script": FAKE_SLOW_FLOW}
    assert first_call(chat(client, mid, "Run a flow")) is None  # not offered: plain reply
    call = first_call(chat(client, mid, "", messages=[{"role": "user", "content": "write flow.txt"}],
                           tools=tool_defs("write_file", "run_command")))
    assert call["name"] == "write_file" and json.loads(call["arguments"])["path"] == "flow.txt"


def test_fake_schema_reply_and_sleep(client):
    mid = install(client)
    schema = {"type": "object", "properties": {"findings": {"type": "array", "items": {
        "type": "object", "properties": {"title": {"type": "string"}, "n": {"type": "integer"},
                                         "ok": {"type": "boolean"}, "k": {"enum": ["a", "b"]}},
        "required": ["title"]}}}, "required": ["findings"]}
    prompt = f"Review the project for bugs. sleep 0.2\n\nReply with only JSON matching this schema:\n{json.dumps(schema)}"
    t = time.monotonic()
    text, calls = reply_text(chat(client, mid, "", messages=[{"role": "user", "content": prompt}],
                                  tools=tool_defs("read_file")))
    assert time.monotonic() - t >= 0.2 and not calls
    got = json.loads(text)
    assert got["findings"][0] == {"title": "x", "n": 1, "ok": True, "k": "a"}
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "oops"},
            {"role": "user", "content": "That reply wasn't valid JSON for the schema. Reply again."}]
    text, _ = reply_text(chat(client, mid, "", messages=msgs))
    assert json.loads(text)["findings"][0]["title"] == "x"
    t = time.monotonic()
    text, _ = reply_text(chat(client, mid, "sleep 0.2 then hi"))
    assert time.monotonic() - t >= 0.2 and text.startswith("You said")


# MARK: thinking effort

GPTOSS = {"reasoningParser": "openai_gptoss"}
QWEN = {"reasoningParser": "qwen3"}
V31 = {"reasoningParser": "deepseek_r1", "facts": {"model_type": "deepseek_v3", "chat_template": "{% if thinking %}"}}
R1 = {"reasoningParser": "deepseek_r1", "facts": {"model_type": "deepseek_v3", "chat_template": "<think>"}}


def test_efforts_per_family():
    assert efforts(GPTOSS) == ["low", "medium", "high"]
    for m in (QWEN, {"reasoningParser": "glm45"}, {"reasoningParser": "llama.cpp"}, V31):
        assert efforts(m) == ["off", "on"]
    assert efforts(R1) == [] and efforts({"reasoningParser": None}) == [] and efforts({"reasoningParser": "x"}) == []


def test_apply_effort_mapping():
    assert apply_effort({"reasoning_effort": "high"}, GPTOSS) == {"reasoning_effort": "high"}
    assert apply_effort({"reasoning_effort": "off"}, GPTOSS) == {"reasoning_effort": "low"}
    assert apply_effort({"reasoning_effort": "on"}, GPTOSS) == {"reasoning_effort": "medium"}
    assert apply_effort({"reasoning_effort": "off"}, QWEN) == {"chat_template_kwargs": {"enable_thinking": False}}
    body = apply_effort({"reasoning_effort": "high", "chat_template_kwargs": {"x": 1}}, QWEN)
    assert body == {"chat_template_kwargs": {"x": 1, "enable_thinking": True}}
    assert apply_effort({"reasoning_effort": "low"}, V31) == {"chat_template_kwargs": {"thinking": False}}
    assert apply_effort({"reasoning_effort": "on"}, V31) == {"chat_template_kwargs": {"thinking": True}}
    assert apply_effort({"reasoning_effort": "high", "a": 1}, R1) == {"a": 1}
    assert apply_effort({"reasoning_effort": "high"}, {}) == {}
    for m in (GPTOSS, QWEN, R1, {}):
        assert apply_effort({"a": 1}, m) == {"a": 1}
        assert apply_effort({"reasoning_effort": None}, m) == {"reasoning_effort": None}


def test_listing_and_relay_carry_efforts(client, monkeypatch):
    seen = []
    real = FakeEngine.chat

    def recording(self, body):
        seen.append(body)
        return real(self, body)

    monkeypatch.setattr(FakeEngine, "chat", recording)
    mid = install(client)
    m = next(m for m in client.get("/v1/llm/models", headers=AUTH).json()["models"] if m["id"] == mid)
    assert m["efforts"] == efforts(m) and "efforts" in m
    assert client.patch(f"/v1/llm/models/{mid}", json={"reasoningParser": "qwen3"}, headers=AUTH).json()["efforts"] \
        == ["off", "on"]
    chat(client, mid, "hi", reasoning_effort="low")
    assert seen[-1]["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning_effort" not in seen[-1]
    assert client.patch(f"/v1/llm/models/{mid}", json={"reasoningParser": "openai_gptoss"}, headers=AUTH).status_code == 200
    chat(client, mid, "hi", reasoning_effort="off")
    assert seen[-1]["reasoning_effort"] == "low"


def test_stop_pressed_only_for_this_container():
    # The Modal containers' heartbeat stops a ready (or starting) engine when the front desk's Stop is newer than
    # the container's start; a Stop from before it started was for an earlier container.
    assert stop_pressed({"started": 100.0}, 150.0)
    assert not stop_pressed({"started": 100.0}, 90.0)
    assert not stop_pressed({"started": 100.0}, None) and not stop_pressed({"started": 100.0}, 0)
    assert not stop_pressed({}, 150.0)


class _StartingEngine(LLMService):
    """A deployment whose engine goes through the given states, one per poll, after warm()."""

    def __init__(self, tmp_path, states):
        super().__init__(tmp_path)
        self.states, self.warmed_calls = list(states), 0

    async def _engine_warm(self, mid, m):
        self.warmed_calls += 1

    async def _engine_state(self, mid, m):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def test_until_ready_waits_then_returns(tmp_path):
    import asyncio

    svc = _StartingEngine(tmp_path, [{"state": "starting"}, {"state": "loading"}, {"state": "ready"}])
    asyncio.run(svc._until_ready("m", {}, poll=0))
    assert svc.warmed_calls == 1 and svc.states == [{"state": "ready"}]


def test_until_ready_raises_when_stopped_or_failed(tmp_path):
    import asyncio

    from aistudio_gpu.llm.engine import EngineError

    stopped = _StartingEngine(tmp_path, [{"state": "starting"}, {"state": "cold"}])
    with pytest.raises(EngineError, match="stopped before it was ready"):
        asyncio.run(stopped._until_ready("m", {}, poll=0))
    failed = _StartingEngine(tmp_path, [{"state": "loading"}, {"state": "failed", "error": "out of memory"}])
    with pytest.raises(EngineError, match="out of memory"):
        asyncio.run(failed._until_ready("m", {}, poll=0))


def test_until_ready_ends_quietly_when_the_client_leaves(tmp_path):
    # The chat isn't sent until the engine is ready: cancelling the wait (the app closed the stream) sends nothing.
    import asyncio

    svc = _StartingEngine(tmp_path, [{"state": "starting"}])

    async def leave():
        task = asyncio.create_task(svc._until_ready("m", {}, poll=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(leave())
    assert svc.warmed_calls == 1
