import Foundation
import NIOCore
import Hummingbird
import HummingbirdWebSocket
import AIStudioProtocol

/// Fans run events out to connected clients.
///
/// Hot path: a record is encoded once, appended to each interested client's pending buffer, and the client's writer
/// is signalled. Writers coalesce: they flush at most every ~16 ms unless a state change (started, finished, failed)
/// asks for an immediate flush. Previews never queue: each client holds only the latest frame per run and takes it
/// when its own preview rate allows. Publishing never blocks on the network.
final class Hub: @unchecked Sendable {
    final class Client: @unchecked Sendable {
        let session: String
        let address: String
        let connected = Date().timeIntervalSince1970
        var name = "client"
        var role: ClientRole = .app
        var previewHz: Double = 4
        var previewMaxSide = 512
        /// Runs whose events this client gets (dashboards get every run).
        var runs: Set<UInt32> = []
        var pending = Data()
        /// When the oldest pending record was published (for the latency metric).
        var pendingSince: UInt64 = 0
        var urgent = false
        var previews: [UInt32: Data] = [:]
        var lastPreview: UInt64 = 0
        var texts: [String] = []
        var bytesIn = 0
        var bytesOut = 0
        let signal: AsyncStream<Void>.Continuation
        let signals: AsyncStream<Void>

        init(session: String, address: String) {
            self.session = session
            self.address = address
            (signals, signal) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        }

        func wants(_ run: UInt32) -> Bool { role == .dashboard || runs.contains(run) }
    }

    /// The last records of a run, for clients re-attaching after a dropped connection.
    struct Replay {
        var seq: UInt32 = 0
        var records: [(seq: UInt32, data: Data)] = []
    }

    static let flushInterval: UInt64 = 16_000_000
    static let replayCapacity = 4096
    static let replayRuns = 64

    private let lock = NSLock()
    private var clients: [String: Client] = [:]
    private var replays: [UInt32: Replay] = [:]
    private var replayOrder: [UInt32] = []

    // Counters for the metrics sampler (read and reset once per second).
    private var events = 0
    private var previewsSent = 0
    private var bytesOut = 0
    private var bytesIn = 0
    private var latency = LatencyWindow()

    // MARK: clients

    func add(_ client: Client) { lock.withLock { clients[client.session] = client } }
    func remove(_ session: String) { lock.withLock { _ = clients.removeValue(forKey: session) } }

    func update(_ session: String, _ body: (Client) -> Void) {
        lock.withLock { if let c = clients[session] { body(c) } }
    }

    var clientInfos: [ClientInfo] {
        lock.withLock {
            clients.values.map {
                ClientInfo(session: $0.session, name: $0.name, role: $0.role, address: $0.address, connected: $0.connected,
                           bytesIn: $0.bytesIn, bytesOut: $0.bytesOut, previewHz: $0.previewHz)
            }.sorted { $0.connected < $1.connected }
        }
    }

    var hasDashboards: Bool { lock.withLock { clients.values.contains { $0.role == .dashboard } } }

    /// The highest preview rate and size any client wants for `run` (0 Hz: nobody is looking).
    func previewDemand(run: UInt32) -> (hz: Double, maxSide: Int) {
        lock.withLock {
            var hz = 0.0, side = 0
            for c in clients.values where c.wants(run) && c.previewHz > 0 {
                hz = max(hz, c.previewHz)
                side = max(side, c.previewMaxSide)
            }
            return (hz, side)
        }
    }

    // MARK: publishing

    /// Assigns the run's next sequence number, keeps the record for replay and queues it for interested clients.
    func publish(_ event: WireEvent, urgent: Bool = false) {
        var e = event
        let now = DispatchTime.now().uptimeNanoseconds
        lock.lock()
        var replay = replays[e.run] ?? Replay()
        if replays[e.run] == nil {
            replayOrder.append(e.run)
            if replayOrder.count > Self.replayRuns { replays.removeValue(forKey: replayOrder.removeFirst()) }
        }
        replay.seq &+= 1
        e.seq = replay.seq
        let data = e.encoded()
        replay.records.append((e.seq, data))
        if replay.records.count > Self.replayCapacity { replay.records.removeFirst(replay.records.count / 4) }
        replays[e.run] = replay
        events += 1
        var wake: [Client] = []
        for c in clients.values where c.wants(e.run) {
            if c.pending.isEmpty { c.pendingSince = now }
            c.pending.append(data)
            if urgent { c.urgent = true }
            wake.append(c)
        }
        lock.unlock()
        for c in wake { c.signal.yield() }
    }

    /// The latest preview of a run; replaces any frame a client hasn't taken yet.
    func publishPreview(run: UInt32, node: UInt16, step: Int, total: Int, jpeg: Data) {
        let data = WireEvent(kind: .preview, node: node, run: run, step: UInt16(clamping: step),
                             total: UInt16(clamping: total), payload: jpeg).encoded()
        lock.lock()
        var wake: [Client] = []
        for c in clients.values where c.wants(run) && c.previewHz > 0 {
            c.previews[run] = data
            wake.append(c)
        }
        lock.unlock()
        for c in wake { c.signal.yield() }
    }

    /// Sends a record to dashboards only (metrics), without a sequence number or replay.
    func publishToDashboards(_ event: WireEvent) {
        let data = event.encoded()
        let now = DispatchTime.now().uptimeNanoseconds
        lock.lock()
        var wake: [Client] = []
        for c in clients.values where c.role == .dashboard {
            if c.pending.isEmpty { c.pendingSince = now }
            c.pending.append(data)
            wake.append(c)
        }
        lock.unlock()
        for c in wake { c.signal.yield() }
    }

    func send(_ message: ServerMessage, to session: String) {
        guard let text = try? message.jsonString() else { return }
        let c: Client? = lock.withLock {
            // Control replies (pong, submitted, text deltas) skip the coalescing delay.
            clients[session].map { $0.texts.append(text); $0.urgent = true; return $0 }
        }
        c?.signal.yield()
    }

    func sendToDashboards(_ message: ServerMessage) {
        guard let text = try? message.jsonString() else { return }
        let targets: [Client] = lock.withLock {
            let ds = clients.values.filter { $0.role == .dashboard }
            ds.forEach { $0.texts.append(text) }
            return ds
        }
        targets.forEach { $0.signal.yield() }
    }

    /// Subscribes a client to a run and queues the records it missed after `lastSeq`.
    func subscribe(_ session: String, run: UInt32, after lastSeq: UInt32) {
        let c: Client? = lock.withLock {
            guard let c = clients[session] else { return nil }
            c.runs.insert(run)
            if let r = replays[run] {
                if c.pending.isEmpty { c.pendingSince = DispatchTime.now().uptimeNanoseconds }
                for rec in r.records where rec.seq > lastSeq { c.pending.append(rec.data) }
                c.urgent = true
            }
            return c
        }
        c?.signal.yield()
    }

    func countIn(_ session: String, bytes: Int) {
        lock.withLock {
            bytesIn += bytes
            clients[session]?.bytesIn += bytes
        }
    }

    func countTransfer(inBytes: Int = 0, outBytes: Int = 0) {
        lock.withLock {
            bytesIn += inBytes
            bytesOut += outBytes
        }
    }

    /// (events, previews, bytes out, bytes in, latency p50, p99) since the last call.
    func takeCounters() -> (Int, Int, Int, Int, Float, Float) {
        lock.withLock {
            let (p50, p99) = latency.percentiles()
            defer { events = 0; previewsSent = 0; bytesOut = 0; bytesIn = 0; latency.reset() }
            return (events, previewsSent, bytesOut, bytesIn, p50, p99)
        }
    }

    // MARK: writer

    /// Drains one client's queue into its socket until the connection closes.
    func pump(_ client: Client, outbound: WebSocketOutboundWriter) async {
        var lastFlush: UInt64 = 0
        for await _ in client.signals {
            // Coalesce bursts (sampler steps, node states) into one frame per interval.
            let urgent = lock.withLock { client.urgent }
            let now = DispatchTime.now().uptimeNanoseconds
            if !urgent, now &- lastFlush < Self.flushInterval {
                try? await Task.sleep(nanoseconds: Self.flushInterval - (now &- lastFlush))
            }
            let (texts, frame, since, previewsTaken, retryAfter) = take(client)
            do {
                for t in texts { try await outbound.write(.text(t)) }
                if !frame.isEmpty { try await outbound.write(.binary(ByteBuffer(bytes: frame))) }
            } catch {
                return
            }
            let done = DispatchTime.now().uptimeNanoseconds
            lastFlush = done
            let sent = texts.reduce(0) { $0 + $1.utf8.count } + frame.count
            lock.withLock {
                client.bytesOut += sent
                bytesOut += sent
                previewsSent += previewsTaken
                if since > 0 { latency.add(Float(done &- since) / 1e6) }
            }
            if let retryAfter {
                // A preview arrived faster than this client's rate: come back for it.
                Task {
                    try? await Task.sleep(nanoseconds: retryAfter)
                    client.signal.yield()
                }
            }
        }
    }

    private func take(_ c: Client) -> ([String], Data, UInt64, Int, UInt64?) {
        lock.lock()
        defer { lock.unlock() }
        let texts = c.texts
        c.texts.removeAll(keepingCapacity: true)
        var frame = c.pending
        c.pending = Data()
        let since = frame.isEmpty ? 0 : c.pendingSince
        c.urgent = false
        var taken = 0
        var retry: UInt64?
        if !c.previews.isEmpty, c.previewHz > 0 {
            let now = DispatchTime.now().uptimeNanoseconds
            let interval = UInt64(1e9 / c.previewHz)
            if now &- c.lastPreview >= interval {
                for (_, p) in c.previews { frame.append(p) }
                taken = c.previews.count
                c.previews.removeAll(keepingCapacity: true)
                c.lastPreview = now
            } else {
                retry = interval - (now &- c.lastPreview)
            }
        }
        return (texts, frame, since, taken, retry)
    }
}
