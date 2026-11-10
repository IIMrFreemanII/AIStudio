import { describe, expect, it } from 'vitest';
import { FrameBatcher } from './frame';
import { NO_NODE,
  decodeAll, decodeHistory, decodeSample, Kind, METRIC_COUNT, payloadPeak, payloadSeconds, SeriesRing, WireError,
} from './protocol';

// Same bytes as Tests/MLXComfyProtocolTests/WireEventTests.swift `golden`: a nodeFinished record (node 3, run 7,
// seq 42, step 5/8, 1.5 s, peak 1024 B) followed by a preview record (node 1, run 7, step 2/8, payload ff d8).
const GOLDEN =
  '0500030007000000' + '2a00000005000800' + '10000000' + '000000000000f83f' + '0004000000000000' +
  '0a00010007000000' + '0000000002000800' + '02000000' + 'ffd8';

function hex(s: string): ArrayBuffer {
  const b = new Uint8Array(s.length / 2);
  for (let i = 0; i < b.length; i++) b[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16);
  return b.buffer;
}

describe('wire records', () => {
  it('decodes a status record like the Swift and Python encoders write it', () => {
    // Same bytes as WireEventTests.goldenStatus and server/gpu/tests/test_wire.py.
    const [e] = decodeAll(hex('0c00ffff090000000100000000000000110000005374617274696e672047505520303a3132'));
    expect(e).toMatchObject({ kind: Kind.nodeStatus, node: NO_NODE, run: 9, seq: 1 });
    expect(new TextDecoder().decode(e.payload)).toBe('Starting GPU 0:12');
  });

  it('decodes the golden frame', () => {
    const [a, b] = decodeAll(hex(GOLDEN));
    expect(a).toMatchObject({ kind: Kind.nodeFinished, flags: 0, node: 3, run: 7, seq: 42, step: 5, total: 8 });
    expect(payloadSeconds(a.payload)).toBe(1.5);
    expect(payloadPeak(a.payload)).toBe(1024);
    expect(b).toMatchObject({ kind: Kind.preview, node: 1, run: 7, step: 2, total: 8 });
    expect([...b.payload]).toEqual([0xff, 0xd8]);
  });

  it('payloads are views into the frame, not copies', () => {
    const frame = hex(GOLDEN);
    const [, b] = decodeAll(frame);
    expect(b.payload.buffer).toBe(frame);
  });

  it('rejects truncated frames', () => {
    expect(() => decodeAll(hex(GOLDEN.slice(0, 30)))).toThrow(WireError);
    expect(() => decodeAll(hex(GOLDEN.slice(0, GOLDEN.length - 2)))).toThrow(WireError);
  });

  it('peak -1 means unknown', () => {
    const p = new Uint8Array(16);
    new DataView(p.buffer).setBigInt64(8, -1n, true);
    expect(payloadPeak(p)).toBeNull();
  });
});

describe('metrics', () => {
  it('decodes a sample into a reused buffer', () => {
    const p = new Uint8Array(8 + METRIC_COUNT * 4);
    const v = new DataView(p.buffer);
    v.setFloat64(0, 1700000000.5, true);
    for (let i = 0; i < METRIC_COUNT; i++) v.setFloat32(8 + i * 4, i * 1.5, true);
    const into = new Float32Array(METRIC_COUNT);
    expect(decodeSample(p, into)).toBe(1700000000.5);
    expect(into[3]).toBe(4.5);
  });

  it('decodes a history (struct of arrays)', () => {
    const m = 2, n = 3;
    const p = new Uint8Array(8 + n * 8 + m * n * 4);
    const v = new DataView(p.buffer);
    v.setUint16(0, m, true);
    v.setUint16(2, n, true);
    for (let i = 0; i < n; i++) v.setFloat64(8 + i * 8, 100 + i, true);
    for (let c = 0; c < m; c++) for (let i = 0; i < n; i++) v.setFloat32(8 + n * 8 + (c * n + i) * 4, c * 10 + i, true);
    const h = decodeHistory(p);
    expect([...h.times]).toEqual([100, 101, 102]);
    expect([...h.columns[1]]).toEqual([10, 11, 12]);
  });

  it('ring keeps the newest samples and merges a history without duplicates', () => {
    const ring = new SeriesRing(4, 1);
    for (let t = 1; t <= 6; t++) ring.push(t, new Float32Array([t * 10]));
    expect([...ring.timeView()]).toEqual([3, 4, 5, 6]);
    expect([...ring.view(0)]).toEqual([30, 40, 50, 60]);
    ring.merge(new Float64Array([5, 6, 7]), [new Float32Array([0, 0, 70])]);
    expect([...ring.timeView()]).toEqual([4, 5, 6, 7]);
    expect([...ring.view(0)]).toEqual([40, 50, 60, 70]);
    ring.load(new Float64Array([1, 2]), [new Float32Array([1, 2])]);
    expect(ring.count).toBe(2);
  });
});

describe('frame batching', () => {
  it('runs one flush per frame however many requests arrive', () => {
    const queued: (() => void)[] = [];
    let flushes = 0;
    const b = new FrameBatcher(() => flushes++, (cb) => queued.push(cb));
    for (let i = 0; i < 1000; i++) b.request();
    expect(queued.length).toBe(1);
    queued.shift()!();
    expect(flushes).toBe(1);
    b.request();
    expect(queued.length).toBe(1);
  });
});
