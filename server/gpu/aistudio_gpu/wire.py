"""Job events on the wire: the same 20-byte little-endian records as the Mac server's protocol
(``Sources/AIStudioProtocol/WireEvent.swift``), so the app decodes both with one decoder.

    kind u8 | flags u8 | node u16 | run u32 | seq u32 | step u16 | total u16 | payloadLen u32 | payload…

A job is one "run" with no nodes (``node`` is always ``NO_NODE``). Several records can share one WebSocket frame.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum

HEADER = struct.Struct("<BBHIIHHI")
HEADER_SIZE = HEADER.size  # 20
NO_NODE = 0xFFFF


class Kind(IntEnum):
    STARTED = 1
    PROGRESS = 4
    # Payload: seconds f64, GPU seconds f64, estimated cost in USD f64 (-1 = unknown).
    FINISHED = 7
    CANCELLED = 8
    # Payload: JSON {"name", "kind", "bytes", "isPreview", "node"} (Swift's OutputDescriptor).
    OUTPUT_READY = 9
    # Payload: JPEG bytes; step/total are the sampler's.
    PREVIEW = 10
    # Payload: UTF-8 message.
    RUN_FAILED = 11
    # Payload: UTF-8 status line ("Starting GPU 0:41"); empty clears it.
    STATUS = 12


def encode(kind: int, run: int, seq: int = 0, step: int = 0, total: int = 0, payload: bytes = b"",
           node: int = NO_NODE, flags: int = 0) -> bytes:
    return HEADER.pack(int(kind), flags, node, run & 0xFFFFFFFF, seq & 0xFFFFFFFF, min(step, 0xFFFF),
                       min(total, 0xFFFF), len(payload)) + payload


@dataclass
class Record:
    kind: int
    flags: int
    node: int
    run: int
    seq: int
    step: int
    total: int
    payload: bytes


def decode_all(data: bytes) -> list[Record]:
    out: list[Record] = []
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        if len(view) - offset < HEADER_SIZE:
            raise ValueError("truncated record header")
        kind, flags, node, run, seq, step, total, length = HEADER.unpack_from(view, offset)
        start = offset + HEADER_SIZE
        if len(view) - start < length:
            raise ValueError("truncated record payload")
        out.append(Record(kind, flags, node, run, seq, step, total, bytes(view[start:start + length])))
        offset = start + length
    return out


def finished_payload(seconds: float, gpu_seconds: float, cost_usd: float | None) -> bytes:
    return struct.pack("<ddd", seconds, gpu_seconds, -1.0 if cost_usd is None else cost_usd)


def output_payload(name: str, kind: str, size: int) -> bytes:
    return json.dumps({"name": name, "kind": kind, "bytes": size, "isPreview": False, "node": 0},
                      separators=(",", ":")).encode()
