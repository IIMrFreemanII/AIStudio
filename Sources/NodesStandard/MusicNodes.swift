import Foundation
import GraphRuntime
import MLXDiffusionCore

/// Musical keys offered by the Text to Music node ("auto" lets the model or its planner choose).
public enum MusicKeys {
    public static let all: [String] = ["auto"] + ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B"]
        .flatMap { ["\($0) major", "\($0) minor"] }
    public static let timeSignatures = ["auto", "2", "3", "4", "6"]
    /// Vocal languages the ACE-Step planner and DiT know ("auto" = detect from the lyrics).
    public static let languages = ["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi", "nl",
                                   "pl", "sv", "tr", "uk", "vi", "id", "th", "yue"]
}

/// Generates a song from a style description and lyrics with a music model (ACE-Step).
public struct TextToMusicNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "TextToMusic", displayName: "Text to Music", category: "audio",
        inputs: [
            InputSpec("model", .model),
            InputSpec("tags", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("lyrics", .string, default: .string("[Instrumental]"), widget: .text(multiline: true)),
            InputSpec("seconds", .float, default: .double(30), widget: .number(min: 10, max: 240, step: 1)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            // 0 = the model's default.
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 20, step: 1)),
            InputSpec("planner", .bool, default: .bool(true), widget: .toggle),
            // 0 = auto.
            InputSpec("bpm", .int, default: .int(0), widget: .number(min: 0, max: 300, step: 1)),
            InputSpec("key", .string, default: .string("auto"), widget: .choice(MusicKeys.all)),
            InputSpec("time_signature", .string, default: .string("auto"), widget: .choice(MusicKeys.timeSignatures)),
            InputSpec("language", .string, default: .string("auto"), widget: .choice(MusicKeys.languages)),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: GenerativeModel = try inputs.object("model")
        guard let audio = model as? AudioModel else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "\(model.displayName) is not an audio model")
        }
        func auto(_ s: String) -> String { s == "auto" ? "" : s }
        let steps = try inputs.int("steps"), bpm = try inputs.int("bpm")
        let request = AudioRequest(
            prompt: try inputs.string("tags"), seconds: try inputs.float("seconds"),
            seed: UInt64(truncatingIfNeeded: try inputs.int("seed")), steps: steps > 0 ? steps : nil,
            lyrics: try inputs.string("lyrics"), bpm: bpm > 0 ? bpm : nil, key: auto(try inputs.string("key")),
            timeSignature: auto(try inputs.string("time_signature")), language: auto(try inputs.string("language")),
            planner: try inputs.bool("planner"))
        let clip = try audio.generateAudio(request) { step, total in context.progress(step: step, total: total) }
        return ["audio": .object(clip)]
    }
}
