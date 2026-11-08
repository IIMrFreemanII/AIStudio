import Foundation
import MLX

/// A model that can also write text, e.g. through the language model it uses as a text encoder.
/// The prompt assistant (Enhance / Describe) works with any model that conforms.
public protocol TextGenerating: AnyObject {
    /// Chat-style generation. `images` are RGBA `[H, W, 4]` in `[0, 1]`. `onText` receives the full text so far
    /// after every token; return `false` to stop. Returns the final text.
    func generateText(system: String, user: String, images: [MLXArray], options: TextGenerationOptions,
                      onText: (String) -> Bool) throws -> String
}

public struct TextGenerationOptions: Sendable {
    public var maxTokens: Int
    /// 0 = greedy (deterministic).
    public var temperature: Float
    public var seed: UInt64
    public init(maxTokens: Int = 320, temperature: Float = 0.7, seed: UInt64 = 0) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.seed = seed
    }
}

/// What the prompt assistant is asked to do, and the instructions it gets for each task.
public enum PromptAssistant {
    public enum Mode: String, CaseIterable, Sendable {
        /// Rewrite a short idea (or, with an image, an edit instruction) into a detailed prompt.
        case enhance
        /// Write a prompt that would recreate the given image.
        case describe
        /// Follow a free-form instruction.
        case custom
    }

    public static let enhanceSystem = """
        You write prompts for a text-to-image model. Rewrite the user's idea as one detailed image prompt. \
        Describe the subject, setting, composition, lighting, colors, style and mood. Keep every detail the user \
        gave, and don't add text, logos or signatures unless asked. Reply with the prompt only: one paragraph \
        under 120 words, no preamble, no quotes, no lists.
        """

    public static let editSystem = """
        You write instructions for an image-editing model. The user gives an image and a short edit request. \
        Rewrite the request as one clear, specific instruction that names what in the image should change and \
        how, and says what must stay the same. Reply with the instruction only: one or two sentences, \
        no preamble, no quotes.
        """

    public static let describeSystem = """
        You write prompts for a text-to-image model. Describe the image as one detailed prompt that would \
        recreate it: the subject and their appearance, clothing and pose, the setting, composition and camera \
        angle, lighting, colors and style. Reply with the prompt only: one paragraph under 120 words, \
        no preamble, no quotes, no lists.
        """

    /// The system and user messages for a request.
    /// - Parameters:
    ///   - prompt: the current prompt (the idea to enhance, or extra guidance when describing).
    ///   - hasImage: whether an image is attached (enhance then rewrites an edit instruction).
    ///   - instruction: for `.custom`, the system instruction.
    ///   - overrides: user-edited system prompts by mode (`"enhance"`, `"edit"`, `"describe"`).
    public static func messages(mode: Mode, prompt: String, hasImage: Bool, instruction: String = "",
                                overrides: [String: String] = [:]) -> (system: String, user: String) {
        func system(_ key: String, _ fallback: String) -> String {
            let s = overrides[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return s.isEmpty ? fallback : s
        }
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        switch mode {
        case .enhance:
            return hasImage ? (system("edit", editSystem), p) : (system("enhance", enhanceSystem), p)
        case .describe:
            let user = p.isEmpty ? "Describe this image as a prompt."
                : "Describe this image as a prompt. Take this into account: \(p)"
            return (system("describe", describeSystem), user)
        case .custom:
            let i = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
            return (i.isEmpty ? "You are a helpful assistant." : i, p)
        }
    }

    /// Cleans model output into a prompt: trims, drops wrapping quotes and a leading "Prompt:" label.
    public static func clean(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for label in ["Prompt:", "prompt:", "Instruction:", "Edited prompt:", "Enhanced prompt:"]
        where s.hasPrefix(label) {
            s = String(s.dropFirst(label.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let quotes: [(Character, Character)] = [("\"", "\""), ("“", "”"), ("'", "'")]
        for (a, b) in quotes where s.count >= 2 && s.first == a && s.last == b {
            s = String(s.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }
}
