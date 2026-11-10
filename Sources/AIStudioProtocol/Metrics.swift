import Foundation

/// The time series the server samples once per second. The raw value is the column index in every metrics payload,
/// so new metrics are only ever appended.
public enum MetricID: Int, CaseIterable, Sendable {
    /// Bytes.
    case memoryActive = 0
    case memoryCache
    case memoryPeak
    /// Percent 0…100.
    case gpuUtilization
    /// Percent of one core (the server process).
    case cpuProcess
    case eventsPerSecond
    case previewsPerSecond
    /// Bytes per second.
    case bytesOutPerSecond
    case bytesInPerSecond
    /// Milliseconds from an event happening to its frame being written.
    case eventLatencyP50
    case eventLatencyP99
    case queueLength

    public static let count = allCases.count
}

/// Fixed-capacity time series, one contiguous `Float32` column per metric (struct of arrays). Pushing a sample is a
/// handful of stores; nothing allocates after `init`.
public struct MetricsRing: Sendable {
    public let capacity: Int
    public let metrics: Int
    /// `metrics × capacity`, column-major (metric m's samples at m*capacity…).
    public private(set) var values: [Float]
    public private(set) var times: [Double]
    /// Index the next sample goes to.
    public private(set) var head = 0
    public private(set) var count = 0

    public init(capacity: Int = 600, metrics: Int = MetricID.count) {
        self.capacity = capacity
        self.metrics = metrics
        values = [Float](repeating: 0, count: capacity * metrics)
        times = [Double](repeating: 0, count: capacity)
    }

    public mutating func push(time: Double, _ sample: [Float]) {
        precondition(sample.count == metrics)
        values.withUnsafeMutableBufferPointer { v in
            for m in 0..<metrics { v[m * capacity + head] = sample[m] }
        }
        times[head] = time
        head = (head + 1) % capacity
        count = min(count + 1, capacity)
    }

    /// Samples oldest first: (times, one array per metric).
    public func chronological() -> (times: [Double], columns: [[Float]]) {
        let start = (head - count + capacity) % capacity
        var t = [Double]()
        t.reserveCapacity(count)
        for i in 0..<count { t.append(times[(start + i) % capacity]) }
        let columns = (0..<metrics).map { m -> [Float] in
            var col = [Float]()
            col.reserveCapacity(count)
            for i in 0..<count { col.append(values[m * capacity + (start + i) % capacity]) }
            return col
        }
        return (t, columns)
    }

    public var latest: [Float]? {
        guard count > 0 else { return nil }
        let i = (head - 1 + capacity) % capacity
        return (0..<metrics).map { values[$0 * capacity + i] }
    }
}

/// Binary payloads of the dashboard's metrics records.
public enum MetricsCodec {
    /// `metricsHistory`: `metricCount u16 | sampleCount u16 | reserved u32 | times f64×n | columns f32×(m×n)`.
    public static func history(_ ring: MetricsRing) -> Data {
        let (times, columns) = ring.chronological()
        var d = Data(capacity: 8 + times.count * 8 + columns.count * times.count * 4)
        append(UInt16(ring.metrics), to: &d)
        append(UInt16(times.count), to: &d)
        append(UInt32(0), to: &d)
        for t in times { append(t.bitPattern, to: &d) }
        for col in columns { for v in col { append(v.bitPattern, to: &d) } }
        return d
    }

    /// `metrics`: `time f64 | value f32 × metricCount`.
    public static func sample(time: Double, _ values: [Float]) -> Data {
        var d = Data(capacity: 8 + values.count * 4)
        append(time.bitPattern, to: &d)
        for v in values { append(v.bitPattern, to: &d) }
        return d
    }

    public static func decodeSample(_ d: Data) -> (time: Double, values: [Float])? {
        guard d.count >= 8, (d.count - 8) % 4 == 0 else { return nil }
        return d.withUnsafeBytes { raw in
            let t = Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self)))
            let n = (d.count - 8) / 4
            let v = (0..<n).map {
                Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 8 + $0 * 4, as: UInt32.self)))
            }
            return (t, v)
        }
    }

    public static func decodeHistory(_ d: Data) -> (times: [Double], columns: [[Float]])? {
        guard d.count >= 8 else { return nil }
        return d.withUnsafeBytes { raw -> (times: [Double], columns: [[Float]])? in
            let m = Int(UInt16(littleEndian: raw.loadUnaligned(as: UInt16.self)))
            let n = Int(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: 2, as: UInt16.self)))
            guard d.count == 8 + n * 8 + m * n * 4 else { return nil }
            let times = (0..<n).map {
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: 8 + $0 * 8, as: UInt64.self)))
            }
            let base = 8 + n * 8
            let columns = (0..<m).map { col in
                (0..<n).map { i in
                    Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: base + (col * n + i) * 4,
                                                                             as: UInt32.self)))
                }
            }
            return (times, columns)
        }
    }

    private static func append<T: FixedWidthInteger>(_ v: T, to d: inout Data) {
        withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
    }
}

/// p50/p99 over a fixed window of latency samples (milliseconds), without sorting on every insert.
public struct LatencyWindow: Sendable {
    private var samples: [Float]
    private var next = 0
    private var filled = 0

    public init(capacity: Int = 1024) { samples = [Float](repeating: 0, count: capacity) }

    public mutating func add(_ ms: Float) {
        samples[next] = ms
        next = (next + 1) % samples.count
        filled = min(filled + 1, samples.count)
    }

    /// (p50, p99) of the samples since the last `reset`, or zeros.
    public func percentiles() -> (Float, Float) {
        guard filled > 0 else { return (0, 0) }
        let sorted = samples.prefix(filled).sorted()
        func p(_ q: Double) -> Float { sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * q + 0.5))] }
        return (p(0.5), p(0.99))
    }

    public mutating func reset() { next = 0; filled = 0 }
}
