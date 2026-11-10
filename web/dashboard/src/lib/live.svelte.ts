import { FrameBatcher } from './frame';
import {
  decodeAll, decodeHistory, decodeSample, Kind, METRIC_COUNT, NO_NODE, payloadSeconds, payloadText,
  PROTOCOL_VERSION, SeriesRing, type DashboardState, type LogLine, type ServerMessage,
} from './protocol';

/** Per-node state of a run, as flat arrays indexed by node (no per-event objects). */
export const NodeState = { pending: 0, running: 1, cached: 2, done: 3, failed: 4 } as const;

export class RunLive {
  readonly state: Uint8Array;
  readonly step: Uint16Array;
  readonly total: Uint16Array;
  readonly seconds: Float32Array;
  readonly errors = new Map<number, string>();
  current = NO_NODE;
  finished = false;

  constructor(readonly run: number, readonly nodeCount: number) {
    this.state = new Uint8Array(nodeCount);
    this.step = new Uint16Array(nodeCount);
    this.total = new Uint16Array(nodeCount);
    this.seconds = new Float32Array(nodeCount);
  }
}

export type Connection = 'connecting' | 'open' | 'closed' | 'unauthorized';

/**
 * The dashboard's live data. Binary records update typed arrays immediately; Svelte sees a new `revision` at most
 * once per animation frame, so a burst of sampler steps costs one render.
 */
export class Live {
  connection = $state<Connection>('connecting');
  state = $state<DashboardState | null>(null);
  logs = $state<LogLine[]>([]);
  /** Bumped once per frame when runs, previews or metrics changed. */
  revision = $state(0);
  previewURL = $state<string | null>(null);
  previewRun = $state<number | null>(null);
  previewStep = $state<[number, number]>([0, 0]);
  serverError = $state<string | null>(null);

  readonly metrics = new SeriesRing(600, METRIC_COUNT);
  readonly runs = new Map<number, RunLive>();
  private sample = new Float32Array(METRIC_COUNT);
  private socket: WebSocket | null = null;
  private retry = 0;
  private stopped = false;
  private pendingLogs: LogLine[] = [];
  private pendingPreview: Uint8Array | null = null;
  private readonly batch = new FrameBatcher(() => this.flush());

  connect(): void {
    this.stopped = false;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/v1/ws`);
    ws.binaryType = 'arraybuffer';
    this.socket = ws;
    this.connection = 'connecting';
    ws.onopen = () => {
      this.retry = 0;
      ws.send(JSON.stringify({
        type: 'hello',
        hello: { protocolVersion: PROTOCOL_VERSION, role: 'dashboard', name: dashboardName(), previewHz: 2, previewMaxSide: 512 },
      }));
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') this.onText(e.data);
      else this.onBinary(e.data as ArrayBuffer);
    };
    ws.onclose = () => {
      if (this.socket !== ws) return;
      this.socket = null;
      if (this.stopped) return;
      // A refused upgrade looks like a close: check whether the session is still valid.
      fetch('/v1/state', { credentials: 'same-origin' }).then((r) => {
        if (r.status === 401) { this.connection = 'unauthorized'; return; }
        this.connection = 'closed';
        const delay = Math.min(10_000, 500 * 2 ** this.retry++);
        setTimeout(() => !this.stopped && this.connect(), delay);
      }, () => {
        this.connection = 'closed';
        setTimeout(() => !this.stopped && this.connect(), Math.min(10_000, 500 * 2 ** this.retry++));
      });
    };
  }

  close(): void {
    this.stopped = true;
    this.socket?.close();
    this.socket = null;
  }

  private onText(text: string): void {
    let msg: ServerMessage;
    try { msg = JSON.parse(text); } catch { return; }
    switch (msg.type) {
      case 'welcome': this.connection = 'open'; break;
      case 'state': this.state = (msg as { state: DashboardState }).state; this.syncRuns(); break;
      case 'log': this.pendingLogs.push((msg as { log: LogLine }).log); this.batch.request(); break;
      case 'error': this.serverError = String((msg as { message: string }).message); break;
    }
  }

  /** Creates live state for runs the server lists (their node tables come with the state). */
  private syncRuns(): void {
    const s = this.state;
    if (!s) return;
    for (const r of [...s.queue, ...s.history]) {
      if (!this.runs.has(r.run)) this.runs.set(r.run, new RunLive(r.run, r.nodes.length));
    }
    const keep = new Set([...s.queue, ...s.history].map((r) => r.run));
    for (const id of this.runs.keys()) if (!keep.has(id)) this.runs.delete(id);
  }

  private run(id: number): RunLive | undefined {
    let r = this.runs.get(id);
    if (!r) {
      const summary = this.state?.queue.find((q) => q.run === id) ?? this.state?.history.find((q) => q.run === id);
      if (!summary) return undefined;
      r = new RunLive(id, summary.nodes.length);
      this.runs.set(id, r);
    }
    return r;
  }

  private onBinary(frame: ArrayBuffer): void {
    let events;
    try { events = decodeAll(frame); } catch { return; }
    for (const e of events) {
      switch (e.kind) {
        case Kind.metrics: {
          const t = decodeSample(e.payload, this.sample);
          this.metrics.push(t, this.sample);
          break;
        }
        case Kind.metricsHistory: {
          const h = decodeHistory(e.payload);
          if (this.metrics.count === 0) this.metrics.load(h.times, h.columns);
          else this.metrics.merge(h.times, h.columns);
          break;
        }
        case Kind.preview:
          this.pendingPreview = e.payload.slice();
          this.previewRun = e.run;
          this.previewStep = [e.step, e.total];
          break;
        default: {
          const r = this.run(e.run);
          if (!r) break;
          const n = e.node;
          const valid = n !== NO_NODE && n < r.nodeCount;
          switch (e.kind) {
            case Kind.nodeStarted: if (valid) { r.state[n] = NodeState.running; r.current = n; } break;
            case Kind.nodeCached: if (valid) r.state[n] = NodeState.cached; break;
            case Kind.progress:
              if (valid) { r.state[n] = NodeState.running; r.step[n] = e.step; r.total[n] = e.total; r.current = n; }
              break;
            case Kind.nodeFinished: if (valid) { r.state[n] = NodeState.done; r.seconds[n] = payloadSeconds(e.payload); } break;
            case Kind.nodeFailed: if (valid) { r.state[n] = NodeState.failed; r.errors.set(n, payloadText(e.payload)); } break;
            case Kind.finished: case Kind.cancelled: case Kind.runFailed: r.finished = true; r.current = NO_NODE; break;
          }
        }
      }
    }
    this.batch.request();
  }

  private flush(): void {
    if (this.pendingLogs.length) {
      const next = this.logs.concat(this.pendingLogs);
      this.logs = next.length > 1000 ? next.slice(next.length - 1000) : next;
      this.pendingLogs = [];
    }
    if (this.pendingPreview) {
      const url = URL.createObjectURL(new Blob([this.pendingPreview as BlobPart], { type: 'image/jpeg' }));
      if (this.previewURL) URL.revokeObjectURL(this.previewURL);
      this.previewURL = url;
      this.pendingPreview = null;
    }
    this.revision++;
  }
}

function dashboardName(): string {
  const ua = navigator.userAgent;
  const browser = /Firefox\//.test(ua) ? 'Firefox' : /Edg\//.test(ua) ? 'Edge' : /Chrome\//.test(ua) ? 'Chrome' : /Safari\//.test(ua) ? 'Safari' : 'Browser';
  return `Dashboard (${browser})`;
}
