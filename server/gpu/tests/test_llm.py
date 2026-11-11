"""LLMs from links: link parsing, GPU sizing, parser guesses, and the /v1/llm API end to end with a fake Hugging
Face (httpx.MockTransport) and the fake engine (no GPU, no vLLM)."""

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
from aistudio_gpu.llm.engine import FakeEngine, phase_of, vllm_command
from aistudio_gpu.llm.facts import LinkError, ModelRef, facts_from_config, parse_link
from aistudio_gpu.llm.profiles import guess
from aistudio_gpu.llm.service import LocalLLMService
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


def test_phases_from_vllm_log():
    assert phase_of("Loading safetensors checkpoint shards:  41% Completed | 12/29") == ("Loading weights", 0.41)
    assert phase_of("Capturing CUDA graphs (mixed prefill-decode): 100%|") == ("Capturing CUDA graphs", 1.0)
    assert phase_of("INFO 11-11 hello") is None


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
