import Foundation
import GraphRuntime
import MovieCore

/// What a character's photos are drawn with: the prompts (by the character's art style) and the SDXL graph.
public enum CharacterArt {
    public static let photoSize = (width: 832, height: 1216)
    public static let faceStrength = 0.6
    public static let templateName = "character-photo"
    public static let maxLoRAs = 4

    static let portraitRequest = "head-and-shoulders portrait, facing the camera, soft light, plain background"

    private static func clean(_ s: String) -> String {
        s.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            .trimmingCharacters(in: CharacterSet(charactersIn: " ,.;"))
    }

    /// A photo request as a caption: models sometimes start it with "You …" or "Me …", which the picture model would
    /// read as a second person.
    static func caption(_ request: String) -> String {
        var s = request
        for lead in ["you're ", "you are ", "you ", "i'm ", "i am ", "me ", "myself "] where s.lowercased().hasPrefix(lead) {
            s = String(s.dropFirst(lead.count))
            return s.prefix(1).uppercased() + s.dropFirst()
        }
        return s
    }

    private static func styleNegative(_ style: ArtModel.Style) -> String {
        switch style {
        case .photoreal: "cartoon, anime, illustration, painting, 3d render, deformed, extra fingers, blurry, watermark, text"
        case .anime: "lowres, bad anatomy, bad hands, extra digits, blurry, watermark, text, 3d, photo"
        }
    }

    /// Style negatives plus AgeGuard.negative.
    public static func negative(_ c: Character) -> String {
        styleNegative(c.artModel.style) + ", " + AgeGuard.negative
    }

    /// A photo the character sends: what it shows (the tag's text) in the chat's scene.
    public static func photoPrompt(_ c: Character, request: String, scene: Scenario?) -> String {
        prompt(c, request: request, scene: scene)
    }

    /// A portrait candidate for the character's face reference (no scene).
    public static func portraitPrompt(_ c: Character) -> String {
        prompt(c, request: portraitRequest, scene: nil)
    }

    private static func prompt(_ c: Character, request: String, scene: Scenario?) -> String {
        let gender = clean(c.gender), look = clean(c.look), request = caption(clean(request))
        let place = clean(scene?.place ?? ""), outfit = clean(scene?.outfit ?? ""), time = clean(scene?.time ?? "")
        switch c.artModel.style {
        case .photoreal:
            var out = "Photo of a \(c.age)-year-old adult" + (gender.isEmpty ? "" : " \(gender)")
            out += look.isEmpty ? "." : ", \(look)."
            if !request.isEmpty { out += " \(request)." }
            var parts: [String] = []
            if !place.isEmpty { parts.append("at \(place)") }
            if !outfit.isEmpty { parts.append("wearing \(outfit)") }
            if !time.isEmpty { parts.append(time) }
            if !parts.isEmpty {
                var sentence = parts.joined(separator: ", ")
                sentence = sentence.prefix(1).uppercased() + sentence.dropFirst()
                out += " \(sentence)."
            }
            return out + " Candid smartphone photo, natural light, sharp focus, detailed skin."
        case .anime:
            let tags = ["masterpiece", "best quality", "anime illustration",
                        gender.isEmpty ? "adult" : "adult \(gender)", "\(c.age) years old", look, request, place, outfit, time]
            return tags.filter { !$0.isEmpty }.joined(separator: ", ")
        }
    }

    /// Throws `CharacterArtError.youthTerm(term)` when AgeGuard finds a youth term in the request.
    public static func check(_ request: String) throws {
        if let term = AgeGuard.youthTerm(in: request) { throw CharacterArtError.youthTerm(term) }
    }

    /// The graph: Load Image (face) → SDXL (Cloud GPU) ← a chain of `loras` LoRA nodes → Preview Image.
    public static func template(loras: Int) -> MovieTemplate {
        let n = min(max(loras, 0), maxLoRAs)
        var nodes = [GraphNode(id: "face", type: "LoadImage", inputs: ["path": .value(.string(""))],
                               position: GraphPoint(x: 40, y: 40))]
        var slots: [String: MovieTemplate.Slot] = [:]
        var gen: [String: NodeInput] = [
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(photoSize.width)), "height": .value(.int(photoSize.height)),
            "server": .value(.string("")), "model": .value(.string("")), "face_strength": .value(.double(faceStrength)),
            "steps": .value(.int(30)), "guidance": .value(.double(6)), "sampler": .value(.string("dpmpp-2m-karras")),
            "face": .link(node: "face", output: "image"),
        ]
        for k in stride(from: 1, through: n, by: 1) {
            var inputs: [String: NodeInput] = ["lora": .value(.string("")), "strength": .value(.double(1))]
            if k > 1 { inputs["loras"] = .link(node: "lora\(k - 1)", output: "loras") }
            nodes.append(GraphNode(id: "lora\(k)", type: "RemoteLoRA", inputs: inputs,
                                   position: GraphPoint(x: 40, y: 200 + Double(k - 1) * 160)))
            slots["lora\(k)"] = .init(node: "lora\(k)", input: "lora")
            slots["lora\(k)_strength"] = .init(node: "lora\(k)", input: "strength")
        }
        if n > 0 { gen["loras"] = .link(node: "lora\(n)", output: "loras") }
        nodes.append(GraphNode(id: "image", type: "RemoteSDXL", inputs: gen, position: GraphPoint(x: 400, y: 60)))
        nodes.append(GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "image", output: "image")],
                               position: GraphPoint(x: 780, y: 80)))
        for name in ["prompt", "negative", "seed", "width", "height", "server", "model", "face_strength"] {
            slots[name] = .init(node: "image", input: name)
        }
        slots["face"] = .init(node: "face", input: "path")
        return MovieTemplate(name: templateName, graph: Graph(nodes: nodes), slots: slots)
    }

    /// The values to bind for one picture. `face` nil: the face node is removed (portrait candidates).
    public static func render(_ c: Character, prompt: String, seed: Int, server: String, face: URL?) -> MovieArt.Render {
        var values: [String: JSONValue] = [
            "prompt": .string(prompt), "negative": .string(negative(c)), "seed": .int(seed),
            "width": .int(photoSize.width), "height": .int(photoSize.height), "server": .string(server),
            "model": .string(c.artModel.variant ?? ""), "face_strength": .double(faceStrength),
        ]
        if let face { values["face"] = .string(face.path) }
        for (i, l) in c.loras.prefix(maxLoRAs).enumerated() {
            values["lora\(i + 1)"] = .string(l.lora)
            values["lora\(i + 1)_strength"] = .double(l.strength)
        }
        return MovieArt.Render(values: values, unusedPictures: face == nil ? ["face"] : [])
    }
}

public enum CharacterArtError: Error, CustomStringConvertible, Equatable {
    case youthTerm(String)

    public var description: String {
        switch self {
        case .youthTerm(let term): "That photo can't be taken: “\(term)” describes someone under 18."
        }
    }
}
