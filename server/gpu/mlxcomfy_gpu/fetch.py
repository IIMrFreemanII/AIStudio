"""Downloads the server runs for its clients (``POST /v1/fetch``): a LoRA from a Civitai or Hugging Face link, straight
into the uploads store, so a big file never goes through the client's (often slow) upload.

- Only links on ``civitai.com``, ``huggingface.co`` and ``hf.co`` (plus ``MLXCOMFY_FETCH_HOSTS``, for tests) are
  fetched; their redirects to signed CDN links are followed, without the Authorization header.
- Logins: ``CIVITAI_TOKEN`` (Modal secret ``civitai``) and ``HF_TOKEN`` (the ``huggingface`` secret), when present.
- The file must be a safetensors file (8-byte header length, then JSON), at most 4 GB, and match the SHA-256 the
  client expects (when the site publishes one). It's stored as ``<sha256>.<ext>``, like an upload; a known SHA-256
  that's already stored answers at once.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .files import FileStore

DEFAULT_HOSTS = ("civitai.com", "huggingface.co", "hf.co")
MAX_BYTES = 4_000_000_000
KEEP = 100  # finished fetches remembered
EXT = re.compile(r"^[a-z0-9]{1,8}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
REDIRECTS = (301, 302, 303, 307, 308)


class FetchError(Exception):
    pass


@dataclass
class Fetch:
    id: str
    url: str
    ext: str
    sha256: str | None = None
    state: str = "downloading"  # downloading, done, failed, cancelled
    bytes: int = 0
    total: int | None = None
    name: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    task: asyncio.Task | None = None

    def summary(self) -> dict:
        return {"id": self.id, "state": self.state, "bytes": self.bytes, "total": self.total, "name": self.name,
                "error": self.error}


def _family(host: str) -> str | None:
    if host == "civitai.com" or host.endswith(".civitai.com"):
        return "civitai"
    if host in ("huggingface.co", "hf.co"):
        return "huggingface"
    return None


class Fetcher:
    def __init__(self, files: FileStore, extra_hosts: tuple[str, ...] | None = None,
                 tokens: dict[str, str | None] | None = None, max_bytes: int = MAX_BYTES):
        self.files, self.max_bytes = files, max_bytes
        if extra_hosts is None:
            extra_hosts = tuple(h.strip().lower() for h in os.environ.get("MLXCOMFY_FETCH_HOSTS", "").split(",")
                                if h.strip())
        self.extra_hosts = set(extra_hosts)  # also allowed over plain http (local test servers)
        self.hosts = set(DEFAULT_HOSTS) | self.extra_hosts
        self.tokens = tokens if tokens is not None else {
            "civitai": os.environ.get("CIVITAI_TOKEN") or None, "huggingface": os.environ.get("HF_TOKEN") or None}
        self.fetches: dict[str, Fetch] = {}

    def _check_url(self, url: str) -> str:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().removeprefix("www.")
        if host not in self.hosts:
            raise ValueError(f"the server only downloads from {', '.join(DEFAULT_HOSTS)}, not {host or url!r}")
        if parts.scheme != "https" and not (parts.scheme == "http" and host in self.extra_hosts):
            raise ValueError("only https links")
        return host

    async def start(self, url: str, sha256: str | None = None, size: int | None = None, ext: str = "lora") -> Fetch:
        host = self._check_url(url)
        if not EXT.match(ext):
            raise ValueError(f"bad extension {ext!r}")
        if sha256 is not None:
            sha256 = sha256.lower()
            if not SHA.match(sha256):
                raise ValueError("sha256 must be 64 hex digits")
        if size is not None and size > self.max_bytes:
            raise ValueError(f"the file is {size / 1e9:.1f} GB; the limit is {self.max_bytes / 1e9:.0f} GB")
        f = Fetch(id=secrets.token_hex(12), url=url, ext=ext, sha256=sha256, total=size)
        if sha256 and self.files.exists(f"{sha256}.{ext}"):
            f.state, f.name = "done", f"{sha256}.{ext}"
            self.files.touch(f.name)
        else:
            for other in self.fetches.values():  # the same link already on its way
                if other.url == url and other.ext == ext and other.state == "downloading":
                    return other
            f.task = asyncio.create_task(self._run(f, host))
        self.fetches[f.id] = f
        self._forget_old()
        return f

    async def probe(self, url: str, sha256: str | None = None, ext: str = "lora") -> dict:
        """Whether the server could download the link (with its own keys), for a client's link preview: already
        stored, or the first bytes of a safetensors file come back. Downloads nothing more."""
        import httpx

        host = self._check_url(url)
        if sha256 and SHA.match(sha256.lower()) and EXT.match(ext) and self.files.exists(f"{sha256.lower()}.{ext}"):
            return {"stored": True, "ok": True, "error": None}
        token = self.tokens.get(_family(host) or "")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=False) as client:
                for hop in range(10):
                    headers = {"User-Agent": "mlxcomfy-gpu", "Range": "bytes=0-15"}
                    if hop == 0 and token:
                        headers["Authorization"] = f"Bearer {token}"
                    async with client.stream("GET", url, headers=headers) as r:
                        if r.status_code in REDIRECTS and "location" in r.headers:
                            url = urljoin(url, r.headers["location"])
                            continue
                        if r.status_code in (401, 403):
                            return {"stored": False, "ok": False, "error": self._login_hint(host)}
                        if r.status_code not in (200, 206):
                            return {"stored": False, "ok": False,
                                    "error": f"HTTP {r.status_code} from {urlsplit(url).hostname}"}
                        head = b""
                        async for chunk in r.aiter_bytes():
                            head += chunk
                            if len(head) >= 16:
                                break
                        if not looks_like_safetensors(head):
                            return {"stored": False, "ok": False, "error": "the link isn't a .safetensors file"}
                        return {"stored": False, "ok": True, "error": None}
            return {"stored": False, "ok": False, "error": "too many redirects"}
        except httpx.HTTPError as e:
            return {"stored": False, "ok": False, "error": f"{type(e).__name__}: {e}"}

    def get(self, fetch_id: str) -> Fetch | None:
        return self.fetches.get(fetch_id)

    def cancel(self, fetch_id: str) -> Fetch | None:
        f = self.fetches.get(fetch_id)
        if f and f.state == "downloading":
            f.state = "cancelled"
            if f.task:
                f.task.cancel()
        return f

    def _forget_old(self) -> None:
        done = [f for f in self.fetches.values() if f.state != "downloading"]
        for f in sorted(done, key=lambda f: f.created)[:max(0, len(done) - KEEP)]:
            del self.fetches[f.id]

    async def _run(self, f: Fetch, host: str) -> None:
        import httpx

        tmp: Path = self.files.directory / f"fetch-{f.id}.partial"
        try:
            hasher, head = hashlib.sha256(), b""
            url = f.url
            token = self.tokens.get(_family(host) or "")
            timeout = httpx.Timeout(60.0, read=300.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                for hop in range(10):
                    headers = {"User-Agent": "mlxcomfy-gpu"}
                    if hop == 0 and token:
                        headers["Authorization"] = f"Bearer {token}"
                    async with client.stream("GET", url, headers=headers) as r:
                        if r.status_code in REDIRECTS and "location" in r.headers:
                            url = urljoin(url, r.headers["location"])
                            if urlsplit(url).scheme != "https" and urlsplit(url).hostname not in self.extra_hosts:
                                raise FetchError("redirected to a non-https link")
                            continue
                        if r.status_code in (401, 403):
                            raise FetchError(self._login_hint(host))
                        if r.status_code != 200:
                            raise FetchError(f"HTTP {r.status_code} from {urlsplit(url).hostname}")
                        if (length := r.headers.get("content-length")) and int(length) > self.max_bytes:
                            raise FetchError(f"the file is {int(length) / 1e9:.1f} GB; the limit is "
                                             f"{self.max_bytes / 1e9:.0f} GB")
                        f.total = int(length) if length else f.total
                        with open(tmp, "wb") as out:
                            async for chunk in r.aiter_bytes(256 << 10):
                                if len(head) < 9:
                                    head += chunk[:9 - len(head)]
                                    if len(head) == 9 and not looks_like_safetensors(head):
                                        raise FetchError("the link isn't a .safetensors file")
                                await asyncio.to_thread(_write, out, hasher, chunk)
                                f.bytes += len(chunk)
                                if f.bytes > self.max_bytes:
                                    raise FetchError(f"the file is over {self.max_bytes / 1e9:.0f} GB")
                        break
                else:
                    raise FetchError("too many redirects")
            if len(head) < 9:
                raise FetchError("the link isn't a .safetensors file")
            sha = hasher.hexdigest()
            if f.sha256 and sha != f.sha256:
                raise FetchError("the download doesn't match the file's published SHA-256; try again")
            name = f"{sha}.{f.ext}"
            await self.files.adopt(tmp, name)
            f.name, f.total, f.state = name, f.bytes, "done"
        except asyncio.CancelledError:
            f.state = "cancelled"
        except FetchError as e:
            f.state, f.error = "failed", str(e)
        except Exception as e:  # noqa: BLE001 — network errors, a full disk…
            f.state, f.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _login_hint(host: str) -> str:
        if _family(host) == "civitai":
            return ("Civitai wants an API key for this file: create one at civitai.com/user/account, then "
                    "`modal secret create civitai CIVITAI_TOKEN=…` and deploy again (a GPU box: set CIVITAI_TOKEN)")
        if _family(host) == "huggingface":
            return ("this Hugging Face repo is gated: accept its licence with the account whose token is in the "
                    "huggingface secret (a GPU box: HF_TOKEN)")
        return f"{host} refused the download (it needs a login)"


def _write(out, hasher, chunk: bytes) -> None:
    out.write(chunk)
    hasher.update(chunk)


def looks_like_safetensors(head: bytes) -> bool:
    """An 8-byte little-endian header length, then the header's JSON object."""
    if len(head) < 9:
        return False
    n = int.from_bytes(head[:8], "little")
    return 1 < n < 100_000_000 and head[8:9] == b"{"
