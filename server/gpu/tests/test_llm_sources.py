"""GGUF and Civitai models: the GGUF header parser on bytes built here, GGUF repos on a fake Hugging Face (Range
requests), Civitai GGUF files and zipped model folders on a fake Civitai, installs, llama.cpp's command line, and
changing an installed model's GPUs."""

import io
import json
import struct
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aistudio_gpu.api import create_app
from aistudio_gpu.files import FileStore
from aistudio_gpu.jobs import JobManager
from aistudio_gpu.llm import gguf, proxy, sizing
from aistudio_gpu.llm.engine import FakeEngine, llama_command
from aistudio_gpu.llm.install import unzip_folder
from aistudio_gpu.llm.service import LocalLLMService
from aistudio_gpu.local import LocalRunner

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GIB = 1 << 30
SHA = "c" * 40


def gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gvalue(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<II", 4, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + gstr(v)
    if isinstance(v, list) and v and isinstance(v[0], str):
        return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(gstr(x) for x in v)
    if isinstance(v, list):
        return struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
    raise TypeError(v)


def gguf_bytes(kv: dict, pad: int = 0) -> bytes:
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, v in kv.items():
        out += gstr(k) + gvalue(v)
    return out + b"\0" * pad


# A Qwen3-MoE-like GGUF header with a big vocabulary (so reading it needs several Range blocks).
QWEN_GGUF = gguf_bytes({
    "general.architecture": "qwen3moe", "general.name": "Qwen3 30B A3B",
    "qwen3moe.block_count": 48, "qwen3moe.attention.head_count": 32, "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.attention.key_length": 128, "qwen3moe.embedding_length": 2048, "qwen3moe.context_length": 262144,
    "qwen3moe.expert_count": 128,
    "tokenizer.ggml.tokens": [f"tok{i}" for i in range(200_000)],
    "tokenizer.chat_template": "{% for m in messages %}<think></think><tool_call>{% endfor %}",
}, pad=4096)


def test_gguf_header_and_groups():
    meta = gguf.metadata(io.BytesIO(QWEN_GGUF))
    assert meta["general.architecture"] == "qwen3moe" and meta["tokenizer.ggml.tokens"] is None
    a = gguf.arch_facts(meta)
    assert (a["layers"], a["attention_heads"], a["kv_heads"], a["head_dim"], a["experts"]) == (48, 32, 4, 128, 128)
    assert a["native_context"] == 262144 and "<think>" in a["chat_template"]
    files = {"x-Q4_K_M.gguf": 18 * GIB, "Q8_0/x-Q8_0-00001-of-00002.gguf": 20 * GIB,
             "Q8_0/x-Q8_0-00002-of-00002.gguf": 12 * GIB, "x-IQ2_XXS.gguf": 9 * GIB, "mmproj-x-f16.gguf": GIB,
             "mmproj-x-bf16.gguf": GIB, "README.md": 1}
    groups = gguf.group(files)
    assert [g["quant"] for g in groups] == ["Q4_K_M", "Q8_0", "IQ2_XXS"]
    assert groups[1]["bytes"] == 32 * GIB and groups[1]["files"][0].endswith("00001-of-00002.gguf")
    assert gguf.mmproj(files) == "mmproj-x-f16.gguf"
    with pytest.raises(gguf.GGUFError):
        gguf.metadata(io.BytesIO(b"GGML" + b"\0" * 20))
    with pytest.raises(gguf.GGUFError):
        gguf.metadata(io.BytesIO(QWEN_GGUF[:1000]))


def ranged(data: bytes, request: httpx.Request) -> httpx.Response:
    rng = request.headers.get("range", "")
    start, _, end = rng.removeprefix("bytes=").partition("-")
    start, end = int(start), min(int(end), len(data) - 1)
    return httpx.Response(206, content=data[start:end + 1], headers={"content-range": f"bytes {start}-{end}/{len(data)}"})


def model_zip() -> bytes:
    buf = io.BytesIO()
    llama = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 32,
             "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
             "max_position_embeddings": 131072}
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("my-finetune/config.json", json.dumps(llama))
        z.writestr("my-finetune/tokenizer_config.json", json.dumps({"chat_template": "<|start|>"}))
        z.writestr("my-finetune/model.safetensors", b"\0" * 5000)
        z.writestr("README.txt", "hello")
    return buf.getvalue()


ZIP = model_zip()


def fake_sites(request: httpx.Request) -> httpx.Response:
    path, host = request.url.path, request.url.host
    if host == "huggingface.co":
        if path == "/api/models/o/qwen-gguf/revision/main":
            return httpx.Response(200, json={"id": "o/qwen-gguf", "sha": SHA, "siblings": [
                {"rfilename": "Qwen3-30B-A3B-Q4_K_M.gguf", "size": 18 * GIB},
                {"rfilename": "Qwen3-30B-A3B-Q8_0.gguf", "size": 32 * GIB},
                {"rfilename": "mmproj-Qwen3-f16.gguf", "size": GIB}]})
        if path.startswith(f"/o/qwen-gguf/resolve/{SHA}/") and path.endswith(".gguf"):
            return ranged(QWEN_GGUF, request)
    if host == "civitai.com":
        if path == "/api/v1/models/42":
            return httpx.Response(200, json={"name": "Story Writer 8B", "modelVersions": [{"id": 7, "files": [
                {"name": "story-Q5_K_M.gguf", "sizeKB": 5 * 1024 * 1024, "hashes": {"SHA256": "AB" * 32},
                 "downloadUrl": "https://civitai.com/api/download/models/7"}]}]})
        if path == "/api/v1/models/43":
            return httpx.Response(200, json={"name": "Zipped Llama", "modelVersions": [{"id": 8, "files": [
                {"name": "model.zip", "sizeKB": len(ZIP) / 1024, "downloadUrl": "https://civitai.com/api/download/models/8"}]}]})
        if path == "/api/v1/models/44":
            return httpx.Response(200, json={"name": "Lone file", "modelVersions": [{"id": 9, "files": [
                {"name": "model.safetensors", "sizeKB": 100}]}]})
        if path == "/api/download/models/7":  # a redirect to the CDN, as Civitai does
            return httpx.Response(307, headers={"location": "https://cdn.example.com/story.gguf"})
        if path == "/api/download/models/8":
            return ranged(ZIP, request)
    if host == "cdn.example.com" and path == "/story.gguf":
        assert "authorization" not in request.headers  # the login stays with the site
        return ranged(QWEN_GGUF, request)
    return httpx.Response(404)


def fetched(repo, revision, name, dest, token):
    (dest / name).parent.mkdir(parents=True, exist_ok=True)
    (dest / name).write_bytes(b"x")


def fetched_url(url, dest, tokens, sha, cancelled):
    dest.write_bytes(ZIP if url.endswith("/8") else b"GGUF")


@pytest.fixture()
def llm(tmp_path: Path):
    transport = httpx.MockTransport(fake_sites)
    return LocalLLMService(tmp_path / "models", http=lambda: httpx.AsyncClient(transport=transport, follow_redirects=True),
                           sync_http=lambda: httpx.Client(transport=transport), fetch=fetched, fetch_url=fetched_url,
                           engine_factory=lambda m, r, s: FakeEngine(m, r, s, boot_seconds=0.1), billed=True)


@pytest.fixture()
def client(tmp_path: Path, llm, monkeypatch):
    monkeypatch.setattr(proxy, "PING_SECONDS", 0.05)
    manager = JobManager(LocalRunner(tmp_path / "models", allow_gpu_tasks=False), FileStore(tmp_path / "uploads"),
                         tmp_path / "results")
    with TestClient(create_app(manager, TOKEN, llm=llm)) as c:
        yield c


def wait_installed(client, mid):
    for _ in range(100):
        listing = client.get("/v1/llm/models", headers=AUTH).json()
        if any(m["id"] == mid and m["installed"] for m in listing["models"]):
            return next(m for m in listing["models"] if m["id"] == mid)
        failed = listing["installs"].get(mid, {})
        assert failed.get("state") != "failed", failed
        time.sleep(0.05)
    raise AssertionError(listing)


def test_gguf_repo_inspect_pick_quant_and_install(client, tmp_path):
    d = client.post("/v1/llm/inspect", json={"url": "o/qwen-gguf"}, headers=AUTH).json()
    f = d["facts"]
    assert f["engine"] == "llamacpp" and f["quant"]["type"] == "Q4_K_M" and f["weightBytes"] == 18 * GIB
    assert f["vision"] and f["mmproj"] == "mmproj-Qwen3-f16.gguf" and f["layers"] == 48
    assert [o["quant"] for o in f["ggufOptions"]] == ["Q4_K_M", "Q8_0"]
    assert d["profile"]["reasoningParser"] == "llama.cpp"
    rec = d["plan"]["recommended"]
    assert rec["fits"] and rec["gpu"] == "L40S" and rec["count"] == 1
    q8 = client.post("/v1/llm/inspect", json={"url": "o/qwen-gguf", "quant": "Q8_0"}, headers=AUTH).json()
    assert q8["facts"]["weightBytes"] == 32 * GIB
    r = client.post("/v1/llm/install", json={"url": "o/qwen-gguf", "quant": "Q8_0"}, headers=AUTH)
    assert r.status_code == 202, r.text
    m = wait_installed(client, r.json()["id"])
    assert m["engine"] == "llamacpp" and m["ggufFile"] == "Qwen3-30B-A3B-Q8_0.gguf"
    weights = tmp_path / "models" / "llm" / m["id"] / "weights"
    assert (weights / "Qwen3-30B-A3B-Q8_0.gguf").exists() and (weights / "mmproj-Qwen3-f16.gguf").exists()
    assert not (weights / "Qwen3-30B-A3B-Q4_K_M.gguf").exists()
    cmd = llama_command({**m, "count": 2}, weights, 8000)
    assert cmd[:3] == ["llama-server", "-m", str(weights / "Qwen3-30B-A3B-Q8_0.gguf")]
    joined = " ".join(cmd)
    assert "--jinja" in cmd and "--reasoning-format deepseek" in joined and "--split-mode layer" in joined
    assert f"--mmproj {weights / 'mmproj-Qwen3-f16.gguf'}" in joined and f"-c {m['context']}" in joined
    # The fake engine chats with it like any other.
    body = {"model": m["id"], "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/llm/oai/v1/chat/completions", json=body, headers=AUTH).json()["choices"][0]["message"][
        "content"] == "You said: hi"


def test_civitai_gguf_and_zip(client, tmp_path):
    d = client.post("/v1/llm/inspect", json={"url": "https://civitai.com/models/42"}, headers=AUTH).json()
    assert d["facts"]["source"] == "civitai" and d["facts"]["engine"] == "llamacpp"
    assert d["facts"]["quant"]["type"] == "Q5_K_M" and d["facts"]["repo"] == "civitai/Story-Writer-8B"
    r = client.post("/v1/llm/install", json={"url": "https://civitai.com/models/42"}, headers=AUTH)
    m = wait_installed(client, r.json()["id"])
    assert (tmp_path / "models" / "llm" / m["id"] / "weights" / "story-Q5_K_M.gguf").exists()

    z = client.post("/v1/llm/inspect", json={"url": "https://civitai.com/models/43"}, headers=AUTH).json()
    assert z["facts"]["engine"] == "vllm" and z["facts"]["modelType"] == "llama" and z["facts"]["weightBytes"] == 5000
    r = client.post("/v1/llm/install", json={"url": "https://civitai.com/models/43"}, headers=AUTH)
    m = wait_installed(client, r.json()["id"])
    weights = tmp_path / "models" / "llm" / m["id"] / "weights"
    assert (weights / "config.json").exists() and (weights / "model.safetensors").stat().st_size == 5000
    assert not (weights / "model.zip").exists() and not (weights / "README.txt").exists()

    lone = client.post("/v1/llm/inspect", json={"url": "https://civitai.com/models/44"}, headers=AUTH)
    assert lone.status_code == 400 and "single .safetensors" in lone.json()["error"]


def test_unzip_refuses_paths_outside(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m/config.json", "{}")
        z.writestr("m/../../evil.txt", "x")
    zp = tmp_path / "m.zip"
    zp.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="unsafe"):
        unzip_folder(zp, "m/", tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_plan_and_changing_gpus(client):
    r = client.post("/v1/llm/install", json={"url": "o/qwen-gguf"}, headers=AUTH)
    mid = r.json()["id"]
    wait_installed(client, mid)
    p = client.get(f"/v1/llm/models/{mid}/plan", headers=AUTH).json()
    assert p["current"]["gpu"] == "L40S" and any(o["gpu"] == "H100" and o["fits"] for o in p["options"])
    ok = client.patch(f"/v1/llm/models/{mid}", json={"gpu": "H100", "count": 2, "context": 65536}, headers=AUTH)
    assert ok.status_code == 200 and ok.json()["count"] == 2
    too_long = client.patch(f"/v1/llm/models/{mid}", json={"context": 10_000_000}, headers=AUTH)
    assert too_long.status_code == 400


def test_llamacpp_sizing_ignores_head_split():
    from aistudio_gpu.llm.facts import facts_from_gguf
    f = facts_from_gguf(gguf.metadata(io.BytesIO(QWEN_GGUF)), "o/x", "main", None,
                        {"quant": "Q4_K_M", "bytes": 18 * GIB, "files": ["x.gguf"]}, False)
    # 2 × 48 layers × 4 KV heads × 128 × 2 bytes, split across 4 GPUs by layer.
    assert sizing.kv_bytes_per_token(f, 4) == 2 * 48 * 4 * 128 * 2 / 4
    counts = {o.count for o in sizing.options(f) if o.gpu == "H100"}
    assert counts == {1, 2, 4, 8}
