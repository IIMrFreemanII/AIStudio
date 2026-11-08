import Foundation
import GraphRuntime
import MLXDiffusionCore

/// Kokoro's English voices: `a` = American, `b` = British; `f`/`m` = female/male. Grades from the model's VOICES.md.
public enum SpeechVoices {
    public struct Voice: Hashable, Sendable {
        public let id: String
        public let grade: String
        public var name: String { id.split(separator: "_").last.map { $0.prefix(1).uppercased() + $0.dropFirst() } ?? id }
        public var accent: String { id.hasPrefix("b") ? "British" : "American" }
        public var gender: String { id.dropFirst().hasPrefix("f") ? "female" : "male" }
        public var label: String { "\(name) (\(accent) \(gender))" }
    }

    public static let all: [Voice] = [
        ("af_heart", "A"), ("af_bella", "A-"), ("af_nicole", "B-"), ("af_aoede", "C+"), ("af_kore", "C+"),
        ("af_sarah", "C+"), ("af_alloy", "C"), ("af_nova", "C"), ("af_sky", "C-"), ("af_jessica", "D"), ("af_river", "D"),
        ("am_fenrir", "C+"), ("am_michael", "C+"), ("am_puck", "C+"), ("am_echo", "D"), ("am_eric", "D"), ("am_liam", "D"),
        ("am_onyx", "D"), ("am_santa", "D-"), ("am_adam", "F+"),
        ("bf_emma", "B-"), ("bf_isabella", "C"), ("bf_alice", "D"), ("bf_lily", "D"),
        ("bm_fable", "C"), ("bm_george", "C"), ("bm_lewis", "D+"), ("bm_daniel", "D"),
    ].map { Voice(id: $0.0, grade: $0.1) }

    public static let defaultVoice = "af_heart"
}

/// Speaks text with a speech model (Kokoro): one voice, adjustable speed. Long text is split at sentence boundaries.
public struct TextToSpeechNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "TextToSpeech", displayName: "Text to Speech", category: "audio",
        inputs: [
            InputSpec("model", .model),
            InputSpec("text", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("voice", .string, default: .string(SpeechVoices.defaultVoice),
                      widget: .choice(SpeechVoices.all.map(\.id))),
            InputSpec("speed", .float, default: .double(1), widget: .number(min: 0.5, max: 2, step: 0.05)),
            // Seeds the vocoder's noise source (tiny effect; kept for reproducibility).
            InputSpec("seed", .int, default: .int(0), widget: .seed),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: GenerativeModel = try inputs.object("model")
        guard let audio = model as? AudioModel else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "\(model.displayName) is not an audio model")
        }
        let request = AudioRequest(prompt: try inputs.string("text"), seed: UInt64(truncatingIfNeeded: try inputs.int("seed")),
                                   voice: try inputs.string("voice"), speed: try inputs.float("speed"))
        let clip = try audio.generateAudio(request) { step, total in context.progress(step: step, total: total) }
        return ["audio": .object(clip)]
    }
}
