import Foundation

/// What a stage is about to spend, shown before it runs: one line per kind of work.
public struct CostEstimate: Sendable, Equatable {
    public struct Line: Sendable, Equatable, Identifiable {
        public var id: String { label }
        public var label: String
        /// How many jobs or requests.
        public var count: Int
        /// "L40S", "H100"…
        public var gpu: String
        /// GPU seconds, all of them together (cold starts included).
        public var seconds: Double
        public var usd: Double

        public init(label: String, count: Int, gpu: String, seconds: Double, usd: Double) {
            self.label = label
            self.count = count
            self.gpu = gpu
            self.seconds = seconds
            self.usd = usd
        }
    }

    public var title: String
    public var lines: [Line]
    /// Said under the lines: what the figure leaves out or rests on.
    public var notes: [String]

    public init(title: String, lines: [Line], notes: [String] = []) {
        self.title = title
        self.lines = lines
        self.notes = notes
    }

    public var usd: Double { lines.reduce(0) { $0 + $1.usd } }
    public var seconds: Double { lines.reduce(0) { $0 + $1.seconds } }

    /// "$0.04", "<$0.01", "$1.20".
    public static func format(_ usd: Double) -> String {
        if usd > 0 && usd < 0.01 { return "<$0.01" }
        return String(format: "$%.2f", usd)
    }
}

public enum MovieCost {
    /// Modal's on-demand prices, USD per GPU-second (the server's `GPU_PRICES`, checked 2026-11-10): for an estimate
    /// when the server hasn't said its own price yet (a cold model).
    public static let gpuPrices: [String: Double] = [
        "B200": 0.001736, "H200": 0.001261, "H100": 0.001097, "A100-80GB": 0.000694, "A100-40GB": 0.000583,
        "L40S": 0.000542, "A10": 0.000306, "L4": 0.000222, "T4": 0.000164,
    ]

    /// The price of `count` GPUs of a kind ("H100", "h100:2" style names too); nil when it's not a known one.
    public static func price(gpu: String, count: Int = 1) -> Double? {
        let name = gpu.split(separator: ":").first.map(String.init)?.uppercased() ?? gpu
        let key = gpuPrices.keys.first { $0.uppercased() == name }
        return key.flatMap { gpuPrices[$0] }.map { $0 * Double(max(count, 1)) }
    }

    /// LLM requests on a GPU server's model. A cold model starts first (its usual cold start, billed); the GPU then
    /// stays up for the server's idle window after the last request, which is billed too.
    public static func llm(label: String, requests: Int, gpu: String, usdPerSecond: Double, warm: Bool,
                           coldStartSeconds: Double?, secondsPerRequest: Double = 45, idleSeconds: Double = 300)
        -> CostEstimate.Line
    {
        let start = warm ? 0 : (coldStartSeconds ?? 120)
        let seconds = start + Double(requests) * secondsPerRequest + idleSeconds
        return .init(label: label, count: requests, gpu: gpu, seconds: seconds, usd: seconds * usdPerSecond)
    }
}

extension MovieCost {
    /// Image jobs on a GPU server's task. A cold worker loads its models first (billed); after the last job it stays
    /// up for the server's idle window, which is billed too.
    public static func images(label: String, task: String, jobs: Int, megapixels: Double, gpu: String,
                              usdPerSecond: Double, timings: GPUTimings, warm: Bool, idleSeconds: Double)
        -> CostEstimate.Line
    {
        let start = warm ? 0 : timings.loadSeconds(task: task)
        let seconds = start + Double(jobs) * timings.seconds(task: task, megapixels: megapixels) + idleSeconds
        return .init(label: label, count: jobs, gpu: gpu, seconds: seconds, usd: seconds * usdPerSecond)
    }
}

/// How long a server task's jobs take, learned from the jobs that ran (GPU seconds per job, and the model loading
/// of a cold start), so the next estimate is closer. Until a task has run, conservative defaults measured or guessed
/// for its fast mode. Kept in the library (`gpu-timings.json`).
public struct GPUTimings: Codable, Sendable, Equatable {
    public struct Entry: Codable, Sendable, Equatable {
        /// GPU seconds per job (a running average).
        public var seconds: Double
        /// At this size.
        public var megapixels: Double
        public var count: Int
    }

    public struct Load: Codable, Sendable, Equatable {
        public var seconds: Double
        public var count: Int
    }

    public var jobs: [String: Entry] = [:]
    public var loads: [String: Load] = [:]
    /// When each task last finished a job: its worker is likely still warm within the server's idle window.
    public var lastJob: [String: Date] = [:]

    public init() {}

    /// The newest jobs count most: the average covers about the last 10.
    static let window = 10

    /// Defaults per task (fast mode on an H100 for the Qwen models): GPU seconds per job at a size, and loading.
    /// The Qwen ones as measured on Modal on 2026-11-15 (4.2 s and 6.9 s per job, 52 s and 45 s cold loads), with
    /// some room.
    public static let defaults: [String: (seconds: Double, megapixels: Double, load: Double)] = [
        "qwen-image": (5, 1.5, 60),
        "qwen-image-edit": (8, 1.0, 60),
        "sdxl": (5, 1.0, 15),
        "sd35-large": (3, 1.0, 40),
        "flux1-dev": (20, 1.0, 45),
        // A clip's "megapixels" are megapixels × frames (`MovieShots.work`): 832×480 × 81 frames ≈ 32.3. Measured
        // on Modal's H100 on 2026-11-10: 24 s a clip with fast on, about 90 s to load.
        "wan22-i2v": (24, 32.3, 90),
    ]

    /// Jobs of a task are bucketed by size (half-megapixel steps), since time grows with the pixel count.
    static func key(_ task: String, _ megapixels: Double) -> String {
        "\(task)@\(String(format: "%.1f", max(0.5, (megapixels * 2).rounded() / 2)))"
    }

    public mutating func record(task: String, megapixels: Double, gpuSeconds: Double, loadSeconds: Double,
                                at date: Date = Date()) {
        let k = Self.key(task, megapixels)
        var e = jobs[k] ?? Entry(seconds: gpuSeconds, megapixels: megapixels, count: 0)
        e.count += 1
        e.seconds += (gpuSeconds - e.seconds) / Double(min(e.count, Self.window))
        e.megapixels = megapixels
        jobs[k] = e
        if loadSeconds >= 1 {
            var l = loads[task] ?? Load(seconds: loadSeconds, count: 0)
            l.count += 1
            l.seconds += (loadSeconds - l.seconds) / Double(min(l.count, Self.window))
            loads[task] = l
        }
        lastJob[task] = date
    }

    /// GPU seconds per job at this size: learned at this size, else scaled from the task's nearest learned size or
    /// its default (time ∝ pixels), else a minute.
    public func seconds(task: String, megapixels: Double) -> Double {
        if let e = jobs[Self.key(task, megapixels)] { return e.seconds }
        let learned = jobs.filter { $0.key.hasPrefix(task + "@") }.values
            .min { abs($0.megapixels - megapixels) < abs($1.megapixels - megapixels) }
        if let e = learned { return e.seconds * megapixels / max(e.megapixels, 0.1) }
        if let d = Self.defaults[task] { return d.seconds * megapixels / d.megapixels }
        return 60
    }

    public func loadSeconds(task: String) -> Double {
        loads[task]?.seconds ?? Self.defaults[task]?.load ?? 120
    }

    /// Whether the task's worker probably hasn't shut down yet.
    public func isWarm(task: String, idleSeconds: Double, now: Date = Date()) -> Bool {
        lastJob[task].map { now.timeIntervalSince($0) < idleSeconds - 15 } ?? false
    }

    public static func load(_ url: URL) -> GPUTimings {
        (try? Data(contentsOf: url)).flatMap { try? JSONDecoder().decode(GPUTimings.self, from: $0) } ?? GPUTimings()
    }

    public func save(_ url: URL) throws {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try e.encode(self).write(to: url, options: .atomic)
    }
}
