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

    /// One way to run the work: on one GPU per kind of job, or several at once (each extra GPU loads the model and
    /// idles too, so it costs more and finishes sooner).
    public struct Plan: Sendable, Equatable, Identifiable {
        public var id: Int { parallel }
        /// Jobs of a kind sent at once (1: one GPU each).
        public var parallel: Int
        public var lines: [Line]
        /// Roughly how long until everything is done.
        public var wallSeconds: Double

        public init(parallel: Int, lines: [Line], wallSeconds: Double) {
            self.parallel = parallel
            self.lines = lines
            self.wallSeconds = wallSeconds
        }

        public var usd: Double { lines.reduce(0) { $0 + $1.usd } }
    }

    public var title: String
    /// The chosen plan's lines.
    public var lines: [Line]
    /// Said under the lines: what the figure leaves out or rests on.
    public var notes: [String]
    /// Ways to run it to choose from (empty: just `lines`).
    public var plans: [Plan] = []

    public init(title: String, lines: [Line], notes: [String] = []) {
        self.title = title
        self.lines = lines
        self.notes = notes
    }

    /// With a choice of plans, `chosen` first.
    public init(title: String, plans: [Plan], chosen: Int, notes: [String] = []) {
        self.title = title
        self.plans = plans
        self.lines = plans.first { $0.parallel == chosen }?.lines ?? plans.first?.lines ?? []
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
    /// How long a batch takes from Run to done: the model load when a GPU must start (any extra GPU is cold), then
    /// the jobs in rounds of `containers`.
    public static func wallSeconds(jobs: Int, containers: Int, perJob: Double, load: Double, warm: Bool) -> Double {
        let k = max(1, min(containers, jobs))
        let start = warm && k == 1 ? 0 : load
        return start + (Double(jobs) / Double(k)).rounded(.up) * perJob
    }

    /// Whether to run on several GPUs by default: only when it saves real waiting (5 minutes or more); otherwise one
    /// GPU, which skips the extra GPUs' loads and idle time.
    public static func prefersParallel(single: CostEstimate.Plan, parallel: CostEstimate.Plan) -> Bool {
        single.wallSeconds - parallel.wallSeconds >= 300
    }

    /// The GPUs a batch of `jobs` will use when the server runs `parallel` jobs of a kind at once and `busy` of them
    /// are taken already (other work): the free ones, at most `limit` and the jobs; with none free, one (the jobs
    /// wait for a busy GPU, whose model is loaded) and `waits`.
    public static func gpus(jobs: Int, parallel: Int, busy: Int, limit: Int) -> (count: Int, waits: Bool) {
        let free = max(0, parallel - busy)
        if free == 0 { return (1, true) }
        return (max(1, min(jobs, free, limit)), false)
    }

    /// `containers`: how many GPUs the jobs spread over (they run at once); each loads the model when it's cold
    /// (one is already up when `warm`) and stays up `idleSeconds` after its last job.
    public static func images(label: String, task: String, jobs: Int, megapixels: Double, gpu: String,
                              usdPerSecond: Double, timings: GPUTimings, warm: Bool, idleSeconds: Double,
                              containers: Int = 1)
        -> CostEstimate.Line
    {
        let gpus = max(1, min(containers, jobs))
        let loads = Double(warm ? gpus - 1 : gpus)
        let seconds = loads * timings.loadSeconds(task: task)
            + Double(jobs) * timings.seconds(task: task, megapixels: megapixels) + Double(gpus) * idleSeconds
        return .init(label: label, count: jobs, gpu: gpus > 1 ? "\(gpus)× \(gpu)" : gpu, seconds: seconds,
                     usd: seconds * usdPerSecond)
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

    /// Defaults per task: GPU seconds per job at a size, and loading. The Qwen ones on an H100, from Modal on 2026-11-15
    /// (fast: 4.2 s and 6.9 s per job, 52 s and 45 s cold loads), with some room. Qwen-Image is for its full mode, as
    /// Movie Studio's sheets and plates run it (50 steps with CFG, ≈ 6× fast's GPU seconds: a guess until learned).
    public static let defaults: [String: (seconds: Double, megapixels: Double, load: Double)] = [
        "qwen-image": (30, 1.5, 60),
        "qwen-image-edit": (8, 1.0, 60),
        "sdxl": (5, 1.0, 15),
        "sd35-large": (3, 1.0, 40),
        "flux1-dev": (20, 1.0, 45),
        // Full Chroma1-HD (40 steps with guidance), as character photos run it, measured on Modal's L40S on 2026-11-25:
        // ≈ 50–57 s a job at 832×1216, a 26 s cold load. Flash (fast) took 10 s, so learned timings bring it down.
        "chroma": (55, 1.0, 30),
        // A clip's "megapixels" are megapixels × frames (`MovieShots.work`): 832×480 × 81 frames ≈ 32.3. Measured
        // on Modal's H100 on 2026-11-10: 24 s a clip with fast on, about 90 s to load.
        "wan22-i2v": (24, 32.3, 90),
        // S2V as measured on Modal's H100 on 2026-11-15 with flash-attn (832×448, one 80-frame pass): 28 s of text,
        // audio and VAE encoding, 9.8 s a step (both guidance passes), so ≈ 430 s for the default 40 steps; a 77 s
        // cold load. The rest are guesses until measured; the audio models' work is seconds of audio
        // (`MovieVoices.work`).
        "wan22-s2v": (450, 32.3, 90),
        // The distilled lip-sync models: guesses (8 and 4 steps of a 14–16B DiT) until measured.
        "longcat-avatar": (90, 32.3, 120),
        "infinitetalk": (60, 32.3, 150),
        "qwen3-tts": (6, 4, 60),
        "breeze-tts": (6, 4, 60),
        // MiniMax H3, fast (8 steps) on an H200: a 5 s 1344×768 clip is 124 frames ≈ 128; loading ≈ 140 GB. Guesses
        // until the first runs are learned.
        "h3-video": (120, 128, 420),
        "h3-references": (150, 128, 420),
        "stable-audio-3": (4, 5, 40),
        "ace-step-xl": (25, 30, 90),
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
