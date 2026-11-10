"""Server-side downloads (POST /v1/fetch) against a local HTTP server: a redirect like Civitai's and Hugging Face's
CDN links, logins, the safetensors and SHA-256 checks, the host allow-list, cancel, and a file already stored."""

import hashlib
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from aistudio_gpu.api import create_app
from aistudio_gpu.fetch import Fetcher
from aistudio_gpu.files import FileStore
from aistudio_gpu.jobs import JobManager
from aistudio_gpu.local import LocalRunner


def safetensors_bytes(size: int = 300_000) -> bytes:
    header = json.dumps({"w": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    return struct.pack("<Q", len(header)) + header + bytes(i % 251 for i in range(size))


LORA = safetensors_bytes()
SLOW = safetensors_bytes(2_000_000)


class Handler(BaseHTTPRequestHandler):
    seen_auth: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        Handler.seen_auth.append((self.path, self.headers.get("Authorization")))
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/cdn/lora.safetensors")
            self.end_headers()
        elif self.path == "/cdn/lora.safetensors":
            self._send(LORA)
        elif self.path == "/private":
            self.send_response(401)
            self.end_headers()
        elif self.path == "/not-a-lora":
            self._send(b"<html>sign in</html>" * 10)
        elif self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", str(len(SLOW)))
            self.end_headers()
            for i in range(0, len(SLOW), 100_000):
                self.wfile.write(SLOW[i:i + 100_000])
                time.sleep(0.05)
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def origin():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def client(tmp_path):
    files = FileStore(tmp_path / "uploads")
    manager = JobManager(LocalRunner(tmp_path / "models"), files, tmp_path / "results")
    fetcher = Fetcher(files, extra_hosts=("127.0.0.1",), tokens={"civitai": None, "huggingface": None})
    with TestClient(create_app(manager, fetcher=fetcher)) as c:
        c.files, c.fetcher = files, fetcher
        yield c


def wait(client, fetch_id: str, timeout: float = 10.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = client.get(f"/v1/fetch/{fetch_id}").json()
        if s["state"] != "downloading":
            return s
        time.sleep(0.02)
    raise AssertionError("fetch didn't finish")


def test_fetch_follows_the_redirect_and_stores_the_file_by_its_hash(client, origin):
    sha = hashlib.sha256(LORA).hexdigest()
    r = client.post("/v1/fetch", json={"url": f"{origin}/redirect", "sha256": sha.upper(), "bytes": len(LORA)})
    assert r.status_code == 202
    s = wait(client, r.json()["id"])
    assert s["state"] == "done" and s["name"] == f"{sha}.lora" and s["bytes"] == len(LORA)
    assert client.files.path(f"{sha}.lora").read_bytes() == LORA
    # Already stored: answered at once, no download.
    r = client.post("/v1/fetch", json={"url": f"{origin}/redirect", "sha256": sha})
    assert r.status_code == 200 and r.json()["state"] == "done" and r.json()["name"] == f"{sha}.lora"
    assert not list(client.files.directory.glob("*.partial"))


def test_fetch_without_a_known_hash_names_the_file_by_its_content(client, origin):
    s = wait(client, client.post("/v1/fetch", json={"url": f"{origin}/cdn/lora.safetensors"}).json()["id"])
    assert s["name"] == hashlib.sha256(LORA).hexdigest() + ".lora"


@pytest.mark.parametrize("path, sha, message", [
    ("/cdn/lora.safetensors", "0" * 64, "doesn't match"),
    ("/not-a-lora", None, "isn't a .safetensors file"),
    ("/private", None, "refused the download"),
    ("/missing", None, "HTTP 404"),
])
def test_fetch_failures_are_clear(client, origin, path, sha, message):
    body = {"url": origin + path} | ({"sha256": sha} if sha else {})
    s = wait(client, client.post("/v1/fetch", json=body).json()["id"])
    assert s["state"] == "failed" and message in s["error"]
    assert not list(client.files.directory.iterdir())  # nothing half-written is left


def test_only_civitai_and_hugging_face_links_are_fetched(client):
    for url in ["https://example.com/x.safetensors", "file:///etc/passwd", "http://huggingface.co/a/b/resolve/main/x"]:
        r = client.post("/v1/fetch", json={"url": url})
        assert r.status_code == 400, url
    assert client.post("/v1/fetch", json={"url": "https://civitai.com/x", "ext": "../x"}).status_code == 400
    assert client.post("/v1/fetch", json={"url": "https://civitai.com/x", "sha256": "abc"}).status_code == 400


def test_login_hints_name_the_secret():
    f = Fetcher(FileStore.__new__(FileStore), extra_hosts=(), tokens={})
    assert "modal secret create civitai" in f._login_hint("civitai.com")
    assert "huggingface secret" in f._login_hint("huggingface.co")


def test_cancelling_a_fetch_stops_it_and_leaves_nothing(client, origin):
    fid = client.post("/v1/fetch", json={"url": f"{origin}/slow"}).json()["id"]
    time.sleep(0.2)
    assert client.delete(f"/v1/fetch/{fid}").json()["state"] == "cancelled"
    time.sleep(0.3)
    assert client.get(f"/v1/fetch/{fid}").json()["state"] == "cancelled"
    assert not list(client.files.directory.iterdir())


def test_a_fetch_reports_progress_and_the_same_link_twice_is_one_download(client, origin):
    a = client.post("/v1/fetch", json={"url": f"{origin}/slow"}).json()
    b = client.post("/v1/fetch", json={"url": f"{origin}/slow"}).json()
    assert a["id"] == b["id"]
    time.sleep(0.5)
    mid = client.get(f"/v1/fetch/{a['id']}").json()
    assert mid["state"] == "downloading" and 0 < mid["bytes"] < len(SLOW) and mid["total"] == len(SLOW)
    assert wait(client, a["id"])["state"] == "done"


def test_check_says_whether_a_link_could_be_downloaded(client, origin):
    def check(path, **extra):
        r = client.post("/v1/fetch/check", json={"url": origin + path, **extra})
        assert r.status_code == 200
        return r.json()

    assert check("/redirect") == {"stored": False, "ok": True, "error": None}
    assert "refused the download" in check("/private")["error"]
    assert "isn't a .safetensors file" in check("/not-a-lora")["error"]
    assert "HTTP 404" in check("/missing")["error"]
    assert not list(client.files.directory.iterdir())  # nothing downloaded
    sha = hashlib.sha256(LORA).hexdigest()
    wait(client, client.post("/v1/fetch", json={"url": f"{origin}/redirect", "sha256": sha}).json()["id"])
    assert check("/redirect", sha256=sha)["stored"] is True
    assert client.post("/v1/fetch/check", json={"url": "https://example.com/x"}).status_code == 400


def test_deleting_an_upload(client, origin):
    sha = hashlib.sha256(LORA).hexdigest()
    wait(client, client.post("/v1/fetch", json={"url": f"{origin}/redirect", "sha256": sha}).json()["id"])
    assert client.head(f"/v1/files/{sha}.lora").status_code == 200
    assert client.delete(f"/v1/files/{sha}.lora").status_code == 204
    assert client.head(f"/v1/files/{sha}.lora").status_code == 404
    assert client.delete(f"/v1/files/{sha}.lora").status_code == 404
    assert client.delete("/v1/files/..%2Fx").status_code in (400, 404)


def test_model_import_lists_and_removes_a_fine_tune(client, origin, monkeypatch):
    """A fine-tune from a link, in fake-models mode (the file is kept, no conversion): imported in a thread,
    listed by family, offered in /v1/info, then removed."""
    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    sha = hashlib.sha256(LORA).hexdigest()
    r = client.post("/v1/models/import", json={"family": "flux.1", "id": "realism", "name": "Realism",
                                                "files": [{"url": f"{origin}/redirect", "sha256": sha}]})
    assert r.status_code == 200 and r.json()["state"] in ("starting", "downloading", "converting", "done")
    end = time.monotonic() + 10
    while time.monotonic() < end:
        listing = client.get("/v1/models").json()
        if listing["families"]["flux.1"]:
            break
        time.sleep(0.05)
    assert listing["families"]["flux.1"][0]["id"] == "realism" and listing["families"]["flux.1"][0]["name"] == "Realism"
    assert client.get("/v1/info").json()["variants"]["flux.1"][0]["id"] == "realism"
    assert client.delete("/v1/models/flux.1/realism").status_code == 204
    assert client.get("/v1/models").json()["families"]["flux.1"] == []
    assert client.delete("/v1/models/flux.1/realism").status_code == 404
    # Refusals: an unknown family, a bad id, a link elsewhere (the failure shows in the import's state).
    assert client.post("/v1/models/import", json={"family": "sdxl", "id": "x", "files": []}).status_code == 400
    assert client.post("/v1/models/import", json={"family": "flux.1", "id": "../x", "files": []}).status_code == 400
    client.post("/v1/models/import", json={"family": "flux.1", "id": "bad", "files": [{"url": f"{origin}/not-a-lora"}]})
    time.sleep(0.5)
    state = client.get("/v1/models").json()["imports"]["flux.1/bad"]
    assert state["state"] == "failed" and "safetensors" in state["error"]


def test_a_wan_fine_tune_has_a_file_per_expert(client, origin, monkeypatch, tmp_path):
    """Wan 2.2: each file says which expert it replaces (high → transformer, low → transformer_2); one expert alone
    is fine (the other stays the base's), a file without a role is refused."""
    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    sha = hashlib.sha256(LORA).hexdigest()

    def wait(key):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            listing = client.get("/v1/models").json()
            if listing["imports"].get(key, {}).get("state") == "failed" or any(
                    v["id"] == key.split("/")[1] for v in listing["families"].get(key.split("/")[0], [])):
                return listing
            time.sleep(0.05)
        return listing

    client.post("/v1/models/import", json={"family": "wan-2.2-t2v", "id": "pair", "files": [
        {"url": f"{origin}/redirect", "sha256": sha, "role": "high"},
        {"url": f"{origin}/redirect", "sha256": sha, "role": "low"}]})
    client.post("/v1/models/import", json={"family": "wan-2.2-t2v", "id": "low-only", "files": [
        {"url": f"{origin}/redirect", "sha256": sha, "role": "low"}]})
    client.post("/v1/models/import", json={"family": "wan-2.2-t2v", "id": "no-role", "files": [
        {"url": f"{origin}/redirect", "sha256": sha}]})
    wait("wan-2.2-t2v/pair")
    wait("wan-2.2-t2v/low-only")
    listing = wait("wan-2.2-t2v/no-role")
    assert sorted(v["id"] for v in listing["families"]["wan-2.2-t2v"]) == ["low-only", "pair"]
    assert "which expert" in listing["imports"]["wan-2.2-t2v/no-role"]["error"]
