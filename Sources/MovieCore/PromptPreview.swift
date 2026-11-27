import Foundation
import GraphRuntime

/// The Prompt Builder's preview: one picture of a written prompt on the model it was written for. The graph is that
/// model's Cloud GPU node and a Preview Image; inputs not bound here keep the node's defaults.
public enum PromptPreview {
    /// The picture's shape.
    public enum Aspect: String, CaseIterable, Codable, Sendable, Identifiable {
        case portrait, square, landscape
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .portrait: "Portrait"
            case .square: "Square"
            case .landscape: "Landscape"
            }
        }
    }

    /// The node type that draws `model`'s pictures; nil for the video models (the preview draws stills).
    public static func nodeType(_ model: PromptModel) -> String? {
        switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious: "RemoteSDXL"
        case .sd35: "RemoteSD35"
        case .flux: "RemoteFlux"
        case .chroma: "RemoteChroma"
        case .anima: "RemoteAnima"
        case .qwenImage: "RemoteQwenImage"
        case .minimaxImage: "RemoteH3Image"
        case .wan, .ltx23, .minimaxVideo: nil
        }
    }

    /// The server task of `model`'s node (for prices and timings).
    public static func task(_ model: PromptModel) -> String? {
        switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious: "sdxl"
        case .sd35: "sd35-large"
        case .flux: "flux1-dev"
        case .chroma: "chroma"
        case .anima: "anima"
        case .qwenImage: "qwen-image"
        case .minimaxImage: "h3-image"
        case .wan, .ltx23, .minimaxVideo: nil
        }
    }

    /// The picture size: around one megapixel.
    public static func size(_ model: PromptModel, aspect: Aspect) -> (width: Int, height: Int) {
        switch aspect {
        case .portrait: return (832, 1216)
        case .square: return (1024, 1024)
        case .landscape: return (1216, 832)
        }
    }

    /// The node takes a library fine-tune in its `model` input.
    public static func takesFineTune(_ model: PromptModel) -> Bool {
        switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious, .flux, .chroma, .anima, .qwenImage: true
        default: false
        }
    }

    /// The models whose node has a fast mode that skips guidance (Chroma Flash, SD 3.5 Large Turbo, Qwen-Image
    /// Lightning), so it ignores the negative prompt.
    public static let fastModels: Set<PromptModel> = [.chroma, .sd35, .qwenImage]

    /// Whether a preview runs `model`'s fast mode: always with the builder's Fast switch on; with it off only when
    /// there is no negative (a negative needs guidance, so it runs in full). nil: the model has no fast mode.
    public static func runsFast(_ model: PromptModel, negative: Bool, fast: Bool) -> Bool? {
        guard fastModels.contains(model) else { return nil }
        return fast || !negative
    }

    /// Slots: prompt, seed, width, height, server; negative where the model reads one; model (a fine-tune) where the
    /// node takes one. nil for the video models.
    ///
    /// Chroma, SD 3.5 and Qwen-Image default to their fast modes, which skip guidance and so ignore the negative
    /// prompt: with a negative they run in full, as the written prompt expects (docs/prompting.md), unless `fast`
    /// (the builder's Fast switch) keeps the fast mode anyway (`runsFast`).
    /// The most LoRAs one preview chains (the same as a character's photos).
    public static let maxLoRAs = 4

    /// `loras`: that many `RemoteLoRA` nodes chained into the node's `loras` input, with slots `lora1`…, and
    /// `lora1_strength`… (`loraValues`). The models without LoRAs (MiniMax H3 Image) ignore it.
    public static func template(_ model: PromptModel, negative: Bool, fast: Bool = false, loras: Int = 0) -> MovieTemplate? {
        guard let type = nodeType(model) else { return nil }
        var inputs: [String: NodeInput] = [
            "prompt": .value(.string("")), "seed": .value(.int(0)), "width": .value(.int(1024)),
            "height": .value(.int(1024)), "server": .value(.string("")),
        ]
        var slots: [String: MovieTemplate.Slot] = [:]
        var names = ["prompt", "seed", "width", "height", "server"]
        if model.takesNegative {
            inputs["negative"] = .value(.string(""))
            names.append("negative")
        }
        if let f = runsFast(model, negative: negative, fast: fast) { inputs["fast"] = .value(.bool(f)) }
        if takesFineTune(model) {
            inputs["model"] = .value(.string(""))
            names.append("model")
        }
        for n in names { slots[n] = .init(node: "image", input: n) }
        var nodes: [GraphNode] = []
        let n = takesLoRAs(model) ? min(max(loras, 0), maxLoRAs) : 0
        for k in stride(from: 1, through: n, by: 1) {
            var l: [String: NodeInput] = ["lora": .value(.string("")), "strength": .value(.double(1))]
            if k > 1 { l["loras"] = .link(node: "lora\(k - 1)", output: "loras") }
            nodes.append(GraphNode(id: "lora\(k)", type: "RemoteLoRA", inputs: l,
                                   position: GraphPoint(x: -320, y: 40 + Double(k - 1) * 160)))
            slots["lora\(k)"] = .init(node: "lora\(k)", input: "lora")
            slots["lora\(k)_strength"] = .init(node: "lora\(k)", input: "strength")
        }
        if n > 0 { inputs["loras"] = .link(node: "lora\(n)", output: "loras") }
        nodes += [
            GraphNode(id: "image", type: type, inputs: inputs, position: GraphPoint(x: 40, y: 40)),
            GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "image", output: "image")],
                      position: GraphPoint(x: 420, y: 60)),
        ]
        return MovieTemplate(name: "Prompt preview", graph: Graph(nodes: nodes), slots: slots)
    }

    /// The node takes LoRAs (every image model but MiniMax H3 Image).
    public static func takesLoRAs(_ model: PromptModel) -> Bool { nodeType(model) != nil && model != .minimaxImage }

    /// The values of LoRA slots `lora<first>`…: each library id and strength, in order.
    public static func loraValues(_ loras: [(id: String, strength: Double)], first: Int = 1) -> [String: JSONValue] {
        var v: [String: JSONValue] = [:]
        for (i, l) in loras.enumerated() {
            v["lora\(first + i)"] = .string(l.id)
            v["lora\(first + i)_strength"] = .double(l.strength)
        }
        return v
    }

    /// The values for one preview of `written` (the negative is left out where the model reads none).
    public static func values(_ written: WrittenPrompt, model: PromptModel, seed: Int, aspect: Aspect, server: String,
                              fineTune: String?) -> [String: JSONValue] {
        let (w, h) = size(model, aspect: aspect)
        var v: [String: JSONValue] = [
            "prompt": .string(written.prompt), "seed": .int(seed), "width": .int(w), "height": .int(h),
            "server": .string(server),
        ]
        if model.takesNegative { v["negative"] = .string(written.negative) }
        if takesFineTune(model) { v["model"] = .string(fineTune ?? "") }
        return v
    }
}
