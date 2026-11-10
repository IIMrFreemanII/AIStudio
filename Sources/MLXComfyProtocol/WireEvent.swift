import Foundation

/// Version of the remote protocol; the server refuses clients with a different major.
public enum RemoteProtocol {
    public static let version = 1
    /// Default TCP port of `mlxcomfy-cli serve`.
    public static let defaultPort = 8188
}

/// The hot-path record: everything a run streams (progress, node states, previews, metrics) is one of these —
/// a fixed 20-byte little-endian header and an optional payload. Several records can share one WebSocket frame.
///
/// ```
/// kind u8 | flags u8 | node u16 | run u32 | seq u32 | step u16 | total u16 | payloadLen u32 | payload…
/// ```
///
/// Node ids are indices into the node table returned with `submitted`; strings appear only in payloads of rare
/// records (errors, output descriptions, logs).
public struct WireEvent: Equatable, Sendable {
    public enum Kind: UInt8, Sendable {
        case started = 1
        case nodeStarted = 2
        case nodeCached = 3
        case progress = 4
        /// Payload: seconds f64, peak memory i64 (-1 = unknown).
        case nodeFinished = 5
        /// Payload: UTF-8 message.
        case nodeFailed = 6
        /// Payload: seconds f64.
        case finished = 7
        case cancelled = 8
        /// Payload: JSON ``OutputDescriptor``.
        case outputReady = 9
        /// Payload: JPEG bytes. `step`/`total` are the sampler's.
        case preview = 10
        /// The run could not start or failed outside a node. Payload: UTF-8 message.
        case runFailed = 11
        /// What a node is doing besides stepping ("Starting GPU 0:41"). Payload: UTF-8; empty clears it.
        case nodeStatus = 12
        /// Dashboard: one sample per metric. Payload: time f64, then `MetricID.allCases.count` × f32.
        case metrics = 20
        /// Dashboard: the whole ring buffer. Payload: see ``MetricsHistory``.
        case metricsHistory = 21
    }

    public static let headerSize = 20
    /// Sentinel node index for records not about one node.
    public static let noNode: UInt16 = .max

    public var kind: Kind
    public var flags: UInt8 = 0
    public var node: UInt16 = WireEvent.noNode
    public var run: UInt32
    public var seq: UInt32 = 0
    public var step: UInt16 = 0
    public var total: UInt16 = 0
    public var payload: Data = Data()

    public init(kind: Kind, flags: UInt8 = 0, node: UInt16 = WireEvent.noNode, run: UInt32, seq: UInt32 = 0,
                step: UInt16 = 0, total: UInt16 = 0, payload: Data = Data()) {
        self.kind = kind
        self.flags = flags
        self.node = node
        self.run = run
        self.seq = seq
        self.step = step
        self.total = total
        self.payload = payload
    }

    public var encodedSize: Int { Self.headerSize + payload.count }

    /// Appends the record to `out` (no intermediate allocation besides `out` growing).
    public func encode(into out: inout Data) {
        withUnsafeTemporaryAllocation(byteCount: Self.headerSize, alignment: 4) { h in
            h.storeBytes(of: kind.rawValue, toByteOffset: 0, as: UInt8.self)
            h.storeBytes(of: flags, toByteOffset: 1, as: UInt8.self)
            h.storeBytes(of: node.littleEndian, toByteOffset: 2, as: UInt16.self)
            h.storeBytes(of: run.littleEndian, toByteOffset: 4, as: UInt32.self)
            h.storeBytes(of: seq.littleEndian, toByteOffset: 8, as: UInt32.self)
            h.storeBytes(of: step.littleEndian, toByteOffset: 12, as: UInt16.self)
            h.storeBytes(of: total.littleEndian, toByteOffset: 14, as: UInt16.self)
            h.storeBytes(of: UInt32(payload.count).littleEndian, toByteOffset: 16, as: UInt32.self)
            out.append(contentsOf: UnsafeRawBufferPointer(h))
        }
        out.append(payload)
    }

    public func encoded() -> Data {
        var d = Data(capacity: encodedSize)
        encode(into: &d)
        return d
    }

    /// Decodes every record in a frame. Records with an unknown kind are skipped (newer servers may add kinds).
    public static func decodeAll(_ data: Data) throws -> [WireEvent] {
        var events: [WireEvent] = []
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var offset = 0
            while offset < raw.count {
                guard raw.count - offset >= headerSize else { throw WireError.truncated }
                let kindRaw = raw[offset]
                let flags = raw[offset + 1]
                let node = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 2, as: UInt16.self))
                let run = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self))
                let seq = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8, as: UInt32.self))
                let step = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 12, as: UInt16.self))
                let total = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 14, as: UInt16.self))
                let length = Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 16, as: UInt32.self)))
                let start = offset + headerSize
                guard raw.count - start >= length else { throw WireError.truncated }
                if let kind = Kind(rawValue: kindRaw) {
                    let payload = length == 0 ? Data() : Data(bytes: raw.baseAddress! + start, count: length)
                    events.append(WireEvent(kind: kind, flags: flags, node: node, run: run, seq: seq, step: step,
                                            total: total, payload: payload))
                }
                offset = start + length
            }
        }
        return events
    }

    // MARK: typed payloads

    public static func float64(_ v: Double) -> Data { withUnsafeBytes(of: v.bitPattern.littleEndian) { Data($0) } }

    public static func nodeFinishedPayload(seconds: Double, peakMemory: Int?) -> Data {
        var d = float64(seconds)
        withUnsafeBytes(of: Int64(peakMemory ?? -1).littleEndian) { d.append(contentsOf: $0) }
        return d
    }

    public var seconds: Double? {
        guard payload.count >= 8 else { return nil }
        return payload.withUnsafeBytes { Double(bitPattern: UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
    }

    public var peakMemory: Int? {
        guard payload.count >= 16 else { return nil }
        let v = payload.withUnsafeBytes { Int64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: Int64.self)) }
        return v < 0 ? nil : Int(v)
    }

    public var message: String { String(decoding: payload, as: UTF8.self) }
}

public enum WireError: Error, Equatable {
    case truncated
}

/// What a finished Save/Preview node produced; sent with ``WireEvent/Kind/outputReady`` so the client can download
/// it while the rest of the run continues.
public struct OutputDescriptor: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case image, video, audio }
    /// The file's name inside the run (the download path is `/v1/runs/<run>/outputs/<name>`).
    public var name: String
    public var kind: Kind
    public var bytes: Int
    /// Written by a Preview node: shown, not saved to history.
    public var isPreview: Bool
    /// The node that wrote it (index into the node table).
    public var node: Int

    public init(name: String, kind: Kind, bytes: Int, isPreview: Bool, node: Int) {
        self.name = name
        self.kind = kind
        self.bytes = bytes
        self.isPreview = isPreview
        self.node = node
    }
}
