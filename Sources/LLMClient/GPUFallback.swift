import Foundation

/// A way to run a model: a GPU kind and how many.
public struct GPUPreset: Codable, Hashable, Sendable {
    public var gpu: String
    public var count: Int

    public init(gpu: String, count: Int) {
        self.gpu = gpu
        self.count = count
    }

    public init(_ option: GPUOption) { self.init(gpu: option.gpu, count: option.count) }
    public init(_ model: LLMModel) { self.init(gpu: model.gpu, count: model.count) }

    public var label: String { count > 1 ? "\(count)× \(gpu)" : gpu }
}

/// What a chat stream does when its model waits too long for a GPU: after `limit` seconds of "starting", it asks
/// `switchGPU` to move the model to another GPU (the label of the new one, or nil to keep waiting) and sends the
/// request again; once the reply begins (or the stream ends), `settle` puts the model's own GPU back for next time.
public struct GPUWaitPolicy: Sendable {
    public var limit: Double
    public var switchGPU: @Sendable (_ model: String) async -> String?
    public var settle: @Sendable (_ model: String) async -> Void
    /// Whether the model was just moved to another GPU (by this chat or another): a start that ends with "stopped
    /// before it was ready" then means that move, and the request goes again.
    public var switchedRecently: @Sendable (_ model: String) async -> Bool

    public init(limit: Double = 120, switchGPU: @escaping @Sendable (_ model: String) async -> String?,
                settle: @escaping @Sendable (_ model: String) async -> Void,
                switchedRecently: @escaping @Sendable (_ model: String) async -> Bool = { _ in false }) {
        self.limit = limit
        self.switchGPU = switchGPU
        self.settle = settle
        self.switchedRecently = switchedRecently
    }

    /// The server's error for a start that was stopped before it was ready.
    public static func isStoppedStart(_ error: Error) -> Bool {
        "\(error)".contains("stopped before it was ready") || error.localizedDescription.contains("stopped before it was ready")
    }
}

/// Moving a model to its next GPU preset when its own GPU doesn't come (pure: the app does the server calls).
public enum GPUFallback {
    /// The next preset to try: the first in the list that isn't the GPU in use and hasn't been tried in this start.
    public static func next(in list: [GPUPreset], current: GPUPreset, tried: Set<GPUPreset>) -> GPUPreset? {
        list.first { $0 != current && !tried.contains($0) }
    }

    /// The settings that move `model` to `preset`. Fast start and GPU snapshots work on one GPU only, so a
    /// multi-GPU preset turns them off for that start.
    public static func changes(to preset: GPUPreset, from model: LLMModel) -> [String: JSONValue] {
        var c: [String: JSONValue] = ["gpu": .string(preset.gpu), "count": .number(Double(preset.count))]
        if preset.count > 1 {
            if model.loadFormat == "processed" { c["loadFormat"] = .string("auto") }
            if model.snapshot == true { c["snapshot"] = .bool(false) }
        }
        return c
    }

    /// The settings that put `original` back (its GPU, and the start-up options a switch may have turned off).
    public static func restore(_ original: LLMModel) -> [String: JSONValue] {
        var c: [String: JSONValue] = ["gpu": .string(original.gpu), "count": .number(Double(original.count))]
        if let f = original.loadFormat { c["loadFormat"] = .string(f) }
        if let s = original.snapshot { c["snapshot"] = .bool(s) }
        return c
    }

    /// Whether a start that has waited since `since` has waited too long at `now`.
    public static func overdue(since: Date?, now: Date, limit: Double) -> Bool {
        guard let since else { return false }
        return now.timeIntervalSince(since) >= limit
    }

    /// The saved lists, `"serverID|modelID": [presets]`, as JSON (what the app keeps in its settings).
    public static func decode(_ json: String?) -> [String: [GPUPreset]] {
        guard let d = json?.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: [GPUPreset]].self, from: d)) ?? [:]
    }

    public static func encode(_ lists: [String: [GPUPreset]]) -> String {
        (try? JSONEncoder().encode(lists)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }
}
