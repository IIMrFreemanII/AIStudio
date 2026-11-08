import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore

/// Writes or rewrites a prompt with the model's language model. Link `prompt` into Text Encode's `prompt`.
/// Thanks to the cache it only reruns when its own inputs change, so changing the seed of a Sampler downstream
/// doesn't rewrite the prompt.
public struct PromptAssistantNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "PromptAssistant", displayName: "Prompt Assistant", category: "conditioning",
        inputs: [
            InputSpec("model", .model),
            InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("image", .image, optional: true),
            InputSpec("mode", .string, default: .string(PromptAssistant.Mode.enhance.rawValue),
                      widget: .choice(PromptAssistant.Mode.allCases.map(\.rawValue))),
            // The system instruction in custom mode.
            InputSpec("instruction", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            InputSpec("temperature", .float, default: .double(0.7), widget: .number(min: 0, max: 1.5, step: 0.05)),
            InputSpec("max_tokens", .int, default: .int(320), widget: .number(min: 16, max: 2048, step: 16)),
        ],
        outputs: [OutputSpec("prompt", .string)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        guard let writer = model as? TextGenerating else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "\(model.displayName) can't generate text")
        }
        let mode = PromptAssistant.Mode(rawValue: try inputs.string("mode")) ?? .enhance
        let images = inputs.optionalObject("image", as: MLXArray.self).map { [$0] } ?? []
        if mode == .describe && images.isEmpty {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "describe mode needs an image")
        }
        let (system, user) = PromptAssistant.messages(mode: mode, prompt: try inputs.string("prompt"),
                                                      hasImage: !images.isEmpty,
                                                      instruction: try inputs.string("instruction"))
        let maxTokens = try inputs.int("max_tokens")
        let options = TextGenerationOptions(maxTokens: maxTokens, temperature: Float(try inputs.float("temperature")),
                                            seed: UInt64(truncatingIfNeeded: try inputs.int("seed")))
        var produced = 0
        let text = try writer.generateText(system: system, user: user, images: images, options: options) { partial in
            produced += 1
            context.progress(step: produced, total: maxTokens, preview: partial)
            return !Task.isCancelled
        }
        try context.checkCancellation()
        return ["prompt": .string(PromptAssistant.clean(text))]
    }
}
