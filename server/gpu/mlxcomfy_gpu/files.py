"""Content-addressed input uploads, with the same rules as the Mac server (``FileStore.swift``):

- names are ``<sha256>`` or ``<sha256>.<ext>`` (lowercase hex, extension ≤ 8 letters/digits);
- ``HEAD`` → 200 when stored, else 404 with ``X-Upload-Offset`` (bytes of a partial upload);
- ``PUT`` streams to ``<name>.partial`` (resuming at ``Content-Range: bytes <start>-…/<total>``), checks the
  SHA-256 against the name and renames: 201 complete, 202 partial, 409 offset mismatch (with the expected offset).

With ``max_bytes`` the store keeps at most that much: after each new file it deletes the least recently used ones
(a file's modification time is bumped whenever a job uses it) and partial uploads untouched for a day. On Modal the
directory is a Volume, so uploads (LoRAs above all) outlive the front desk's container; ``on_stored`` commits it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path
from typing import AsyncIterator, Callable

NAME = re.compile(r"^[0-9a-f]{64}(\.[a-z0-9]{1,8})?$")
STALE_PARTIAL = 24 * 3600  # seconds before an abandoned partial upload is deleted


def is_name(name: str) -> bool:
    return bool(NAME.match(name))


class OffsetMismatch(Exception):
    def __init__(self, expected: int):
        super().__init__(f"upload offset mismatch, expected {expected}")
        self.expected = expected


class HashMismatch(Exception):
    pass


class FileStore:
    def __init__(self, directory: Path, max_bytes: int | None = None, on_stored: Callable[[], None] | None = None):
        self.directory, self.max_bytes, self.on_stored = directory, max_bytes, on_stored
        directory.mkdir(parents=True, exist_ok=True)

    def touch(self, name: str) -> None:
        """Marks a file as just used (pruning keeps the most recently used ones)."""
        try:
            os.utime(self.path(name))
        except OSError:
            pass

    def prune(self, keep: str | None = None) -> list[str]:
        """Deletes stale partial uploads, then the least recently used files beyond ``max_bytes``; returns the
        deleted names."""
        now, files, deleted = time.time(), [], []
        for p in self.directory.iterdir():
            try:
                st = p.stat()
            except OSError:
                continue
            if p.name.endswith(".partial"):
                if now - st.st_mtime > STALE_PARTIAL:
                    p.unlink(missing_ok=True)
                    deleted.append(p.name)
            elif is_name(p.name):
                files.append((st.st_mtime, st.st_size, p))
        if self.max_bytes is not None:
            total = sum(size for _, size, _ in files)
            for _, size, p in sorted(files, key=lambda f: f[0]):
                if total <= self.max_bytes:
                    break
                if p.name == keep:
                    continue
                p.unlink(missing_ok=True)
                deleted.append(p.name)
                total -= size
        return deleted

    def path(self, name: str) -> Path:
        if not is_name(name):
            raise ValueError(f"bad file name: {name}")
        return self.directory / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def partial_size(self, name: str) -> int:
        p = self.path(name).with_name(name + ".partial")
        return p.stat().st_size if p.exists() else 0

    async def delete(self, name: str) -> bool:
        """Removes a stored file (and any partial upload of it); True when there was one."""
        path = self.path(name)
        existed = path.exists()
        path.unlink(missing_ok=True)
        path.with_name(name + ".partial").unlink(missing_ok=True)
        if existed and self.on_stored:
            await asyncio.to_thread(self.on_stored)
        return existed

    async def adopt(self, path: Path, name: str) -> None:
        """Takes a file written elsewhere in this directory (a server-side download) into the store as ``name``."""
        os.replace(path, self.path(name))
        self.prune(keep=name)
        if self.on_stored:
            await asyncio.to_thread(self.on_stored)

    async def receive(self, name: str, offset: int, total: int | None, body: AsyncIterator[bytes]) -> bool:
        """Writes the body at ``offset``; returns True once the whole file is stored and verified."""
        final = self.path(name)
        if final.exists():
            async for _ in body:
                pass
            return True
        partial = final.with_name(name + ".partial")
        have = partial.stat().st_size if partial.exists() else 0
        if offset != have:
            raise OffsetMismatch(have)
        hasher = hashlib.sha256() if offset == 0 else None
        written = offset
        with open(partial, "r+b" if offset else "wb") as f:
            f.seek(offset)
            async for chunk in body:
                if chunk:
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    written += len(chunk)
        if total is not None and written < total:
            return False
        if hasher is None:  # resumed: hash the whole file
            hasher = hashlib.sha256()
            with open(partial, "rb") as f:
                for block in iter(lambda: f.read(4 << 20), b""):
                    hasher.update(block)
        if not name.startswith(hasher.hexdigest()):
            partial.unlink(missing_ok=True)
            raise HashMismatch(f"{name}: content does not match its name")
        os.replace(partial, final)
        self.prune(keep=name)
        if self.on_stored:
            await asyncio.to_thread(self.on_stored)
        return True
