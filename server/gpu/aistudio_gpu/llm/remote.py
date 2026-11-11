"""Reading parts of a big remote file without downloading it: a seekable file object over HTTP Range requests, for
GGUF headers and the config.json inside a zipped model folder (``zipfile`` reads the central directory at the end,
then just the members it needs).

Redirects (Hugging Face → its CDN, Civitai → a signed link) are followed once and the final URL kept; the login is
only sent to the site itself, never to the CDN.
"""

from __future__ import annotations

import io
import os
from urllib.parse import urljoin, urlsplit

import httpx

BLOCK = 1 << 20  # read granularity
REDIRECTS = (301, 302, 303, 307, 308)


def site_of(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    if host == "civitai.com" or host.endswith(".civitai.com"):
        return "civitai"
    if host in ("huggingface.co", "hf.co") or host == urlsplit(os.environ.get("HF_ENDPOINT", "")).hostname:
        return "huggingface"
    return None


def auth_headers(url: str, tokens: dict) -> dict:
    token = tokens.get(site_of(url) or "")
    return {"Authorization": f"Bearer {token}"} if token else {}


class RangeError(OSError):
    pass


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file. Blocks are cached, so small reads near each other cost one
    request. ``client`` is a sync httpx client (tests pass one with a MockTransport)."""

    def __init__(self, url: str, tokens: dict | None = None, client: httpx.Client | None = None,
                 max_bytes: int = 64 << 20):
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=False)
        self._own = client is None
        self.tokens = tokens or {}
        self.fetched = 0
        self.max_bytes = max_bytes  # a guard: inspecting never reads more than this
        self.blocks: dict[int, bytes] = {}
        self.pos = 0
        self.url, self.size = self._resolve(url)

    def _resolve(self, url: str) -> tuple[str, int]:
        headers = {"User-Agent": "aistudio-gpu", "Range": "bytes=0-0", **auth_headers(url, self.tokens)}
        for _ in range(10):
            r = self.client.get(url, headers=headers)
            if r.status_code in REDIRECTS and "location" in r.headers:
                url = urljoin(url, r.headers["location"])
                headers = {"User-Agent": "aistudio-gpu", "Range": "bytes=0-0"}  # no login past the site
                continue
            if r.status_code in (401, 403):
                raise PermissionError("the site wants a login for this file (a Hugging Face or Civitai token)")
            if r.status_code == 404:
                raise FileNotFoundError(url)
            if r.status_code != 206:
                raise RangeError(f"the server doesn't serve parts of the file (HTTP {r.status_code})")
            total = (r.headers.get("content-range") or "").rpartition("/")[2]
            if not total.isdigit():
                raise RangeError("the server didn't say how big the file is")
            return url, int(total)
        raise RangeError("too many redirects")

    def _block(self, i: int) -> bytes:
        if i not in self.blocks:
            start = i * BLOCK
            end = min(self.size, start + BLOCK) - 1
            if self.fetched + (end - start + 1) > self.max_bytes:
                raise RangeError(f"needed more than {self.max_bytes >> 20} MB to read the file's header")
            r = self.client.get(self.url, headers={"User-Agent": "aistudio-gpu", "Range": f"bytes={start}-{end}"})
            if r.status_code != 206:
                raise RangeError(f"HTTP {r.status_code} reading part of the file")
            self.blocks[i] = r.content
            self.fetched += len(r.content)
        return self.blocks[i]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.size - self.pos))
        out, done = memoryview(b), 0
        while done < n:
            i, off = divmod(self.pos + done, BLOCK)
            chunk = self._block(i)[off:off + n - done]
            if not chunk:
                break
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
        self.pos += done
        return done

    def close(self) -> None:
        if self._own:
            self.client.close()
        super().close()
