"""Edge0 models (strata packs, on Edge0's own engine): detecting one on a fake Hugging Face, its facts and GPU plan,
installing it, the engine's config and command line, and the request guard."""

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aistudio_gpu.api import create_app
from aistudio_gpu.files import FileStore
from aistudio_gpu.jobs import JobManager
from aistudio_gpu.llm import proxy
from aistudio_gpu.llm.engine import FakeEngine, _loaded, edge0_command, edge0_config, phase_of
from aistudio_gpu.llm.profiles import apply_effort, efforts, guess
from aistudio_gpu.llm.service import LocalLLMService, edge0_body
from aistudio_gpu.local import LocalRunner

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GIB = 1 << 30
SHA = "e" * 40
REPO = "Edge0/Edge0-145B-NVFP4"

# Edge0-145B-NVFP4's model.json (trimmed to what matters here).
PACK = {
    "model_name": "edge0-145b-fp4", "display_name": "Edge0-145B", "params": "145B", "layers": 48, "experts": 512,
    "context": 262144, "build": "NVFP4", "pack_format": "strata pack index v3", "exe": "strata",
    "args": ["--pack", ".", "--native", "./model-skeleton.gguf", "--ple-gguf", "./ple-fp8.gguf", "--ple-io", "direct",
             "--expert-cache", "auto", "--pcie-frac", "0.1", "--spec", "4", "--mtp", "./mtp", "--max-context",
             "131072", "--kv", "int8", "--kv-resident", "32768", "--pool-workers", "24", "--pcie-mode", "dma"],
    "cwd": ".", "tokenizer": "./tokenizer", "log": "./serve.log", "port": 8124, "gpu": 0, "gpus_asked": True,
    "env": {"STRATA_LATENT_A": "./experts.bin", "STRATA_EMBED_VRAM": "1"},
}
FILES = {
    "model.json": 2000, "experts.bin": int(44.3 * GIB), "dense.bin": int(1.43 * GIB),
    "latent_dense_q8.bin": int(3.49 * GIB), "model-skeleton.gguf": int(5.58 * GIB), "ple-fp8.gguf": int(47.68 * GIB),
    "mtp/experts.bin": int(0.66 * GIB), "tokenizer/tokenizer.json": 7_000_000, "tokenizer/chat_template.jinja": 9000,
    "LICENSE": 9000, "README.md": 40_000, "start.sh": 1200, "assets/demos/showcase.mp4": 90_000_000,
}


def listing(files: dict) -> dict:
    return {"id": REPO, "sha": SHA, "cardData": {"license": "other", "license_name": "edge0-community-model-1.0"},
            "siblings": [{"rfilename": n, "size": b} for n, b in files.items()]}


def fake_hf(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == f"/api/models/{REPO}/revision/main":
        return httpx.Response(200, json=listing(FILES))
    if path == "/api/models/Edge0/broken/revision/main":
        return httpx.Response(200, json=listing({n: b for n, b in FILES.items() if n != "experts.bin"}))
    if path.endswith("/model.json"):
        return httpx.Response(200, text=json.dumps(PACK))
    return httpx.Response(404)


def fetched(repo, revision, name, dest, token):
    (dest / name).parent.mkdir(parents=True, exist_ok=True)
    (dest / name).write_text(json.dumps(PACK) if name == "model.json" else "x")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(proxy, "PING_SECONDS", 0.05)
    transport = httpx.MockTransport(fake_hf)
    llm = LocalLLMService(tmp_path / "models", http=lambda: httpx.AsyncClient(transport=transport),
                          sync_http=lambda: httpx.Client(transport=transport), fetch=fetched,
                          engine_factory=lambda m, r, s: FakeEngine(m, r, s, boot_seconds=0.1), billed=True)
    manager = JobManager(LocalRunner(tmp_path / "models", allow_gpu_tasks=False), FileStore(tmp_path / "uploads"),
                         tmp_path / "results")
    with TestClient(create_app(manager, TOKEN, llm=llm)) as c:
        yield c


def wait_installed(client, mid):
    for _ in range(100):
        out = client.get("/v1/llm/models", headers=AUTH).json()
        if any(m["id"] == mid and m["installed"] for m in out["models"]):
            return next(m for m in out["models"] if m["id"] == mid)
        assert out["installs"].get(mid, {}).get("state") != "failed", out["installs"][mid]
        time.sleep(0.05)
    raise AssertionError(out)


def test_an_edge0_folder_is_found_and_planned_on_an_h100(client):
    d = client.post("/v1/llm/inspect", json={"url": f"https://huggingface.co/{REPO}"}, headers=AUTH).json()
    f = d["facts"]
    assert f["engine"] == "edge0" and f["quant"] == {"method": "edge0", "type": "NVFP4", "bits": 4}
    assert not f["vision"] and f["nativeContext"] == 131072 and f["layers"] == 48 and f["experts"] == 512
    assert f["cpuBytes"] == FILES["ple-fp8.gguf"] and f["params"] == 145_000_000_000
    skipped = FILES["README.md"] + FILES["start.sh"] + FILES["assets/demos/showcase.mp4"]
    assert f["weightBytes"] == sum(FILES.values()) - skipped
    assert any("edge0-community-model-1.0" in n for n in f["notes"])
    assert d["profile"] == {"toolParser": "edge0", "reasoningParser": "edge0", "engineArgs": []}
    plan = d["plan"]
    assert plan["recommended"]["gpu"] == "H100" and plan["recommended"]["count"] == 1
    by_gpu = {o["gpu"]: o for o in plan["options"]}
    assert {o["count"] for o in plan["options"]} == {1}
    assert by_gpu["L40S"]["fits"] and "slower" in by_gpu["L40S"]["note"]
    assert not by_gpu["H100"]["note"] and not by_gpu["A100-80GB"]["note"]


def test_an_incomplete_edge0_folder_is_refused(client):
    r = client.post("/v1/llm/inspect", json={"url": "Edge0/broken"}, headers=AUTH)
    assert r.status_code == 400 and "experts.bin" in r.text


def test_installing_skips_the_card_and_vllm_settings_are_refused(client, tmp_path):
    r = client.post("/v1/llm/install", json={"url": REPO}, headers=AUTH)
    assert r.status_code == 202, r.text
    m = wait_installed(client, r.json()["id"])
    assert m["engine"] == "edge0" and m["gpu"] == "H100" and m["count"] == 1 and m["efforts"] == ["off", "low", "medium", "high"]
    weights = tmp_path / "models" / "llm" / m["id"] / "weights"
    assert (weights / "tokenizer" / "tokenizer.json").exists() and (weights / "mtp" / "experts.bin").exists()
    assert not (weights / "assets").exists() and not (weights / "README.md").exists()
    for change in ({"snapshot": True}, {"loadFormat": "processed"}, {"mtp": False}, {"count": 2}):
        assert client.patch(f"/v1/llm/models/{m['id']}", json=change, headers=AUTH).status_code == 400, change
    ok = client.patch(f"/v1/llm/models/{m['id']}", json={"gpu": "L40S"}, headers=AUTH)
    assert ok.status_code == 200 and ok.json()["gpu"] == "L40S"


def test_the_engine_config_makes_paths_absolute_and_drops_the_pc_tuning(tmp_path):
    weights = tmp_path / "weights"
    m = {"id": "edge0-145b", "context": 65536, "engineArgs": ["--batch", "2"]}
    cfg = edge0_config(m, PACK, weights, engine_dir=Path("/opt/e"))
    assert cfg["exe"] == "/opt/e/build/strata" and cfg["cwd"] == str(weights) and cfg["aliases"] == ["edge0-145b"]
    assert cfg["tokenizer"] == str(weights / "tokenizer") and "log" not in cfg and "port" not in cfg
    assert cfg["env"] == {"STRATA_VERIFY_ALL_RESIDENT": "0", "STRATA_LATENT_A": str(weights / "experts.bin"),
                          "STRATA_EMBED_VRAM": "1"}
    args = cfg["args"]
    assert args[:4] == ["--pack", str(weights), "--native", str(weights / "model-skeleton.gguf")]
    for gone in ("--pool-workers", "--pcie-mode", "--pcie-frac", "--kv-resident", "direct", "131072"):
        assert gone not in args
    assert args[-6:] == ["--max-context", "65536", "--ple-io", "ram", "--batch", "2"]  # the model's own args last
    assert args.count("--max-context") == 1 and args.count("--ple-io") == 1
    assert "--spec" in args and str(weights / "mtp") in args
    cmd = edge0_command(m, tmp_path / "c.json", 8000, engine_dir=Path("/opt/e"))
    assert cmd == ["/opt/e/.venv/bin/python", "/opt/e/serve/server.py", "--engine", "strata", "--config",
                   str(tmp_path / "c.json"), "--host", "127.0.0.1", "--port", "8000"]


def test_serving_lines_are_not_load_phases():
    assert phase_of("loading the model (the first start takes a minute or two) ...")[0] == "Loading weights"
    assert phase_of("[strata] reading the prompt: 6,925 of 6,960 tokens, 2 s so far") is None


def test_ready_only_once_the_model_is_loaded():
    def health(body):
        return httpx.Response(200, json=body, request=httpx.Request("GET", "http://x/health"))
    assert not _loaded(health({"status": "ok", "loaded": False}))
    assert _loaded(health({"status": "ok", "loaded": True})) and _loaded(health({"status": "ok"}))


def test_thinking_has_levels():
    # The engine maps the level onto its template (high is the template's xhigh, its default); "on" is what chats
    # saved before Edge0 had levels.
    m = {"id": "e", "engine": "edge0", "reasoningParser": "edge0"}
    assert efforts(m) == ["off", "low", "medium", "high"]
    for level in ("off", "low", "medium", "high"):
        assert apply_effort({"reasoning_effort": level}, m) == {"reasoning_effort": level}
        assert edge0_body(m, apply_effort({"reasoning_effort": level, "messages": []}, m))["reasoning_effort"] == level
    assert apply_effort({"reasoning_effort": "on"}, m) == {"reasoning_effort": "high"}
    assert apply_effort({"reasoning_effort": "extreme", "a": 1}, m) == {"a": 1}
    assert apply_effort({"a": 1}, m) == {"a": 1}


def test_the_request_guard_keeps_what_the_engine_takes():
    m = {"id": "e", "name": "Edge0", "engine": "edge0"}
    body = {"model": "e", "stream": True, "stream_options": {"include_usage": True, "other": 1}, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": True, "thinking": True}, "reasoning_effort": "on",
            "skip_special_tokens": False,
            "messages": [{"role": "user", "content": "hi", "name": "me"},
                         {"role": "assistant", "content": None, "reasoning_content": "hm",
                          "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "ok"}]}]}
    out = edge0_body(m, body)
    assert set(out) == {"model", "stream", "stream_options", "temperature", "chat_template_kwargs", "messages"}
    assert out["chat_template_kwargs"] == {"enable_thinking": True} and out["stream_options"] == {"include_usage": True}
    assert out["messages"][0] == {"role": "user", "content": "hi"}
    assert out["messages"][1]["reasoning_content"] == "hm" and out["messages"][2]["tool_call_id"] == "c"
    with pytest.raises(ValueError, match="can't read images"):
        edge0_body(m, {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]})


def test_guess_for_an_edge0_model():
    from aistudio_gpu.llm.facts import ModelRef, facts_from_edge0
    names = {n: b for n, b in FILES.items()}
    f = facts_from_edge0(ModelRef(REPO), {"sha": SHA}, PACK, names)
    assert guess(f)["toolParser"] == "edge0" and f.download[0]["name"] == "LICENSE"
