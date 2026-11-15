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
