from aistudio_gpu.wire import Kind, decode_all, encode, finished_payload

# Same bytes as Tests/AIStudioProtocolTests/WireEventTests.swift `golden` and web/dashboard/src/lib/protocol.test.ts:
# a nodeFinished record (node 3, run 7, seq 42, step 5/8, 1.5 s, peak 1024 B), then a preview (node 1, run 7, 2/8).
GOLDEN = ("0500030007000000" "2a00000005000800" "10000000" "000000000000f83f" "0004000000000000"
          "0a00010007000000" "0000000002000800" "02000000" "ffd8")


def test_golden_bytes_match_swift_and_typescript():
    import struct

    frame = encode(5, run=7, seq=42, step=5, total=8, node=3, payload=struct.pack("<dq", 1.5, 1024))
    frame += encode(Kind.PREVIEW, run=7, step=2, total=8, node=1, payload=b"\xff\xd8")
    assert frame.hex() == GOLDEN
    a, b = decode_all(bytes.fromhex(GOLDEN))
    assert (a.kind, a.node, a.run, a.seq, a.step, a.total) == (5, 3, 7, 42, 5, 8)
    assert (b.kind, b.payload) == (Kind.PREVIEW, b"\xff\xd8")


def test_status_record_and_finished_payload():
    rec = encode(Kind.STATUS, run=9, seq=1, payload="Starting GPU 0:12".encode())
    # Golden bytes for the status kind (12), shared with the Swift and TypeScript tests.
    assert rec.hex() == "0c00ffff09000000" "0100000000000000" "11000000" + "Starting GPU 0:12".encode().hex()
    (r,) = decode_all(rec)
    assert r.kind == Kind.STATUS and r.payload.decode() == "Starting GPU 0:12"
    import struct

    assert struct.unpack("<ddd", finished_payload(12.5, 10.0, None)) == (12.5, 10.0, -1.0)
