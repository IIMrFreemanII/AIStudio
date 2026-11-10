// The server's binary records (see Sources/MLXComfyProtocol/WireEvent.swift): a fixed 20-byte little-endian header
//   kind u8 | flags u8 | node u16 | run u32 | seq u32 | step u16 | total u16 | payloadLen u32
// followed by the payload. Several records share one WebSocket frame. Payloads are views into the frame (no copy).

export const HEADER_SIZE = 20;
export const NO_NODE = 0xffff;

export const Kind = {
  started: 1,
  nodeStarted: 2,
  nodeCached: 3,
  progress: 4,
  nodeFinished: 5,
  nodeFailed: 6,
  finished: 7,
  cancelled: 8,
  outputReady: 9,
  preview: 10,
  runFailed: 11,
  nodeStatus: 12,
  metrics: 20,
  metricsHistory: 21,
} as const;
export type KindValue = (typeof Kind)[keyof typeof Kind];

export interface WireEvent {
  kind: number;
  flags: number;
  node: number;
  run: number;
  seq: number;
  step: number;
  total: number;
  payload: Uint8Array;
}

export class WireError extends Error {}

/** Every record in a frame, in order. */
export function decodeAll(frame: ArrayBuffer): WireEvent[] {
  const view = new DataView(frame);
  const out: WireEvent[] = [];
  let off = 0;
  while (off < frame.byteLength) {
    if (frame.byteLength - off < HEADER_SIZE) throw new WireError('truncated header');
    const len = view.getUint32(off + 16, true);
    const start = off + HEADER_SIZE;
    if (frame.byteLength - start < len) throw new WireError('truncated payload');
    out.push({
      kind: view.getUint8(off),
      flags: view.getUint8(off + 1),
      node: view.getUint16(off + 2, true),
      run: view.getUint32(off + 4, true),
      seq: view.getUint32(off + 8, true),
      step: view.getUint16(off + 12, true),
      total: view.getUint16(off + 14, true),
      payload: new Uint8Array(frame, start, len),
    });
    off = start + len;
  }
  return out;
}

function dv(p: Uint8Array): DataView {
  return new DataView(p.buffer, p.byteOffset, p.byteLength);
}

/** `nodeFinished` / `finished`: seconds (f64) and, for nodes, peak memory (i64, -1 = unknown). */
export function payloadSeconds(p: Uint8Array): number {
  return p.byteLength >= 8 ? dv(p).getFloat64(0, true) : 0;
}

export function payloadPeak(p: Uint8Array): number | null {
  if (p.byteLength < 16) return null;
  const v = dv(p).getBigInt64(8, true);
  return v < 0n ? null : Number(v);
}

const utf8 = new TextDecoder();
export function payloadText(p: Uint8Array): string {
  return utf8.decode(p);
}

// MARK: metrics

/** Column order of the metrics payloads (Sources/MLXComfyProtocol/Metrics.swift `MetricID`). */
export const Metric = {
  memoryActive: 0,
  memoryCache: 1,
  memoryPeak: 2,
  gpuUtilization: 3,
  cpuProcess: 4,
  eventsPerSecond: 5,
  previewsPerSecond: 6,
  bytesOutPerSecond: 7,
  bytesInPerSecond: 8,
  eventLatencyP50: 9,
  eventLatencyP99: 10,
  queueLength: 11,
} as const;
export const METRIC_COUNT = 12;

/** `metrics`: time f64, then one f32 per metric. Writes into `into` (reused, no allocation). */
export function decodeSample(p: Uint8Array, into: Float32Array): number {
  const v = dv(p);
  const n = Math.min(into.length, (p.byteLength - 8) >> 2);
  for (let i = 0; i < n; i++) into[i] = v.getFloat32(8 + i * 4, true);
  return v.getFloat64(0, true);
}

/** `metricsHistory`: m u16 | n u16 | reserved u32 | times f64×n | columns f32×(m×n). */
export function decodeHistory(p: Uint8Array): { times: Float64Array; columns: Float32Array[] } {
  const v = dv(p);
  const m = v.getUint16(0, true);
  const n = v.getUint16(2, true);
  if (p.byteLength !== 8 + n * 8 + m * n * 4) throw new WireError('bad history size');
  const times = new Float64Array(n);
  for (let i = 0; i < n; i++) times[i] = v.getFloat64(8 + i * 8, true);
  const base = 8 + n * 8;
  const columns: Float32Array[] = [];
  for (let c = 0; c < m; c++) {
    const col = new Float32Array(n);
    for (let i = 0; i < n; i++) col[i] = v.getFloat32(base + (c * n + i) * 4, true);
    columns.push(col);
  }
  return { times, columns };
}

/**
 * Fixed-capacity time series as typed arrays (struct of arrays): `times` and one column per metric, oldest first.
 * Charts read `subarray(0, count)` views, so a new sample never allocates.
 */
export class SeriesRing {
  readonly times: Float64Array;
  readonly columns: Float32Array[];
  count = 0;

  constructor(readonly capacity = 600, readonly metrics = METRIC_COUNT) {
    this.times = new Float64Array(capacity);
    this.columns = Array.from({ length: metrics }, () => new Float32Array(capacity));
  }

  push(time: number, sample: Float32Array): void {
    if (this.count === this.capacity) {
      this.times.copyWithin(0, 1);
      for (const c of this.columns) c.copyWithin(0, 1);
      this.count--;
    }
    const i = this.count++;
    this.times[i] = time;
    for (let m = 0; m < this.metrics; m++) this.columns[m][i] = sample[m] ?? 0;
  }

  /** Replaces the contents with a server history (newest `capacity` samples). */
  load(times: Float64Array, columns: Float32Array[]): void {
    const n = Math.min(times.length, this.capacity);
    const from = times.length - n;
    this.times.set(times.subarray(from));
    for (let m = 0; m < this.metrics; m++) {
      const src = columns[m];
      if (src) this.columns[m].set(src.subarray(from));
      else this.columns[m].fill(0, 0, n);
    }
    this.count = n;
  }

  /** Appends samples newer than what it holds (after a reconnect's history). */
  merge(times: Float64Array, columns: Float32Array[]): void {
    const last = this.count ? this.times[this.count - 1] : -Infinity;
    const sample = new Float32Array(this.metrics);
    for (let i = 0; i < times.length; i++) {
      if (times[i] <= last) continue;
      for (let m = 0; m < this.metrics; m++) sample[m] = columns[m]?.[i] ?? 0;
      this.push(times[i], sample);
    }
  }

  view(metric: number): Float32Array {
    return this.columns[metric].subarray(0, this.count);
  }

  timeView(): Float64Array {
    return this.times.subarray(0, this.count);
  }
}

// MARK: JSON messages (Sources/MLXComfyProtocol/Messages.swift)

export type RunStatus = 'waiting' | 'running' | 'done' | 'failed' | 'cancelled';

export interface OutputDescriptor { name: string; kind: 'image' | 'video' | 'audio'; bytes: number; isPreview: boolean; node: number }
export interface RunSummary {
  run: number; workflow: string; client: string; status: RunStatus; created: number;
  started?: number; finished?: number; seconds?: number; peakMemory?: number; error?: string;
  nodes: string[]; nodeTypes: string[]; outputs: OutputDescriptor[];
}
export interface InstalledModel { name: string; bytes: number; loaded: boolean }
export interface LoadedModelInfo { key: string; name: string; bytes: number; pinned: boolean; lastUsed: number }
export interface ServerInfo {
  protocolVersion: number; serverVersion: string; hostName: string; chip: string; osVersion: string; gpu: string;
  memory: { total: number; active: number; cache: number; peak: number; budget: number };
  nodeTypes: string[]; models: InstalledModel[];
}
export interface ClientInfo { session: string; name: string; role: 'app' | 'dashboard' | 'bench'; address: string; connected: number; bytesIn: number; bytesOut: number; previewHz: number }
export interface TransferInfo { id: string; name: string; direction: 'upload' | 'download'; bytes: number; total?: number; bytesPerSecond: number }
export interface ServerSettings { budgetFraction: number; previewHz: number; previewMaxSide: number; host: string; port: number; dataDirectory: string; modelsDirectory: string }
export interface DashboardState {
  info: ServerInfo; started: number; thermalState: string; queue: RunSummary[]; history: RunSummary[];
  loadedModels: LoadedModelInfo[]; clients: ClientInfo[]; transfers: TransferInfo[]; inputCacheBytes: number;
  disks: { path: string; free: number; total: number }[]; settings: ServerSettings;
}
export interface LogLine { time: number; level: 'debug' | 'info' | 'warning' | 'error'; message: string }

export type ServerMessage =
  | { type: 'welcome'; protocolVersion: number; serverVersion: string; session: string }
  | { type: 'state'; state: DashboardState }
  | { type: 'log'; log: LogLine }
  | { type: 'error'; message: string }
  | { type: 'pong'; id: number }
  | { type: string; [k: string]: unknown };

export const PROTOCOL_VERSION = 1;
