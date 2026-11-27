import Foundation
import GraphRuntime
import MovieCore

/// What a character's photos are drawn with: the prompts (by the character's art style) and the graph (SDXL, Chroma or
/// MiniMax H3 Image on a Cloud GPU server, by the character's art model).
public enum CharacterArt {
    public static let photoSize = (width: 832, height: 1216)
    public static let faceStrength = 0.6
    public static let templateName = "character-photo"
    public static let maxLoRAs = 4

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

    /// What the app knows about the character's image model beyond the character: the fine-tune's Civitai base model
    /// ("Pony", "Illustrious", …, which sets the SDXL quality tags) and the LoRAs' trigger words.
    public struct Context: Equatable, Sendable {
        public var baseModel: String?
        public var triggerWords: [String]
        /// The user turned the 18+ options on: a revealing photo shows the character's intimate picks.
        public var adult: Bool
        public init(baseModel: String? = nil, triggerWords: [String] = [], adult: Bool = false) {
            self.baseModel = baseModel
            self.triggerWords = triggerWords
            self.adult = adult
        }
    }

    /// The prompt dialect of the character's model (docs/prompting.md).
    public static func promptModel(_ c: Character, context: Context = .init()) -> PromptModel {
        switch c.artModel.family {
        case .sdxl: PromptModel.sdxl(baseModel: context.baseModel)
        case .chroma: .chroma
        case .minimax: .minimaxImage
        }
    }

    /// Style negatives plus AgeGuard.negative, the body picks' opposites and the fine-tune's quality negatives.
    public static func negative(_ c: Character, context: Context = .init()) -> String {
        let model = promptModel(c, context: context)
        var s = PromptSpec()
        s.avoid = avoid(c, parts: CharacterOptions.lookParts(c.traits, gender: c.gender, booru: model.readsBooru))
        return PromptWriter.write(s, for: model.takesNegative ? model : .sdxl).negative
    }

    private static func avoid(_ c: Character, parts: CharacterOptions.LookParts) -> [String] {
        (styleNegative(c.artModel.style) + ", " + AgeGuard.negative).split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) } + parts.avoid
    }

    /// A photo the character sends: what it shows (the tag's text) in the chat's scene.
    public static func photoPrompt(_ c: Character, request: String, scene: Scenario?, context: Context = .init()) -> String {
        PromptWriter.write(spec(c, request: request, scene: scene, context: context), for: promptModel(c, context: context)).prompt
    }

    /// A portrait candidate for the character's face reference (no scene).
    public static func portraitPrompt(_ c: Character, context: Context = .init()) -> String {
        var s = spec(c, request: "", scene: nil, context: context)
        s.shot = "head-and-shoulders portrait"
        s.action = "facing the camera"
        s.place = "in front of a plain background"
        s.lighting = ["soft light"]
        return PromptWriter.write(s, for: promptModel(c, context: context)).prompt
    }

    /// The character's photo as a model-neutral description: who (age, gender, looks, body), the clothing picks unless
    /// the request or scene names an outfit, what the photo shows, the scene's place, outfit and time.
    public static func spec(_ c: Character, request: String, scene: Scenario?, context: Context = .init()) -> PromptSpec {
        let place = clean(scene?.place ?? ""), outfit = clean(scene?.outfit ?? ""), time = clean(scene?.time ?? "")
        let dressed = !outfit.isEmpty || setsOutfit(request)
        let gender = clean(c.gender)
        let cover = coverage(request)
        let model = promptModel(c, context: context)
        let parts = CharacterOptions.lookParts(c.traits, gender: gender, withClothes: !dressed, coverage: cover,
                                               adult: context.adult, booru: model.readsBooru)
        var s = PromptSpec()
        switch c.artModel.style {
        case .photoreal:
            s.medium = model.readsBooru ? PromptCatalog.booruPhoto : "candid smartphone photo"
            s.lighting = ["natural light", "sharp focus", "detailed skin"]
        case .anime:
            // Plain SDXL has no quality tags of its own; anime fine-tunes bring theirs (PromptModel.qualityTags).
            s.medium = model.dialect == .tags && model.qualityTags.isEmpty ? "masterpiece, best quality, anime illustration"
                : "anime illustration"
        }
        s.subject = "a \(c.age)-year-old adult " + (gender.isEmpty ? "person" : gender)
        switch gender.lowercased() {
        case "woman", "female", "girl": s.pronoun = .she
        case "man", "male": s.pronoun = .he
        default: s.pronoun = .they
        }
        s.action = caption(clean(request))
        s.details = parts.details
        s.body = parts.body
        let free = clean(c.look)
        if !free.isEmpty { s.details.append(PromptDetail(free)) }
        s.outfit = outfit.isEmpty ? parts.outfit : [PromptDetail(outfit)]
        s.place = place
        if !time.isEmpty { s.time = [time] }
        s.avoid = avoid(c, parts: parts)
        s.without = parts.without
        s.triggerWords = context.triggerWords
        s.coverage = cover
        return s
    }

    private static let coverageWords: [(PromptSpec.Coverage, NSRegularExpression)] = ([
        (.impliedNude, "implied nud\\w*|covered (?:only )?(?:by|with|in) (?:a |the )?(?:sheet|bedsheet|towel)"),
        (.nude, "nude|naked|no clothes|nothing on|undressed"),
        (.topless, "topless|bare[- ]chested|shirtless"),
        (.lingerie, "lingerie|negligee"),
        (.underwear, "underwear|panties|bra and panties|boxers|briefs"),
        (.swimwear, "bikini|swimsuit|swimwear|swim trunks"),
    ] as [(PromptSpec.Coverage, String)]).map { ($0.0, try! NSRegularExpression(pattern: "\\b(?:\($0.1))\\b", options: .caseInsensitive)) }

    /// How much the request has the character wear: the first match of nude, topless, lingerie, underwear and swimwear
    /// words (implied nudity before nude); dressed when none.
    public static func coverage(_ request: String) -> PromptSpec.Coverage {
        let range = NSRange(request.startIndex..., in: request)
        return coverageWords.first { $0.1.firstMatch(in: request, range: range) != nil }?.0 ?? .dressed
    }

    private static let outfitWords: NSRegularExpression = try! NSRegularExpression(
        pattern: "\\b(wearing|wears|dressed|dress|dresses|gown|outfit|clothes|shirt|t-shirt|top|blouse|sweater|hoodie|"
            + "jacket|coat|blazer|suit|jeans|pants|trousers|shorts|skirt|leggings|bikini|swimsuit|lingerie|pajamas|"
            + "robe|uniform|costume|naked|nude|topless)\\b", options: .caseInsensitive)

    /// The text names clothes (so the clothing style picks are left out of that photo's look).
    public static func setsOutfit(_ text: String) -> Bool {
        outfitWords.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    /// How much of the head around a detected face goes into the face reference (side of the square, in face widths).
    public static let faceCropScale = 1.3

    /// The face reference's crop of a portrait: a square around the detected face (`face`: Vision's normalized box,
    /// origin bottom-left), `faceCropScale` times its size and nudged up for hair, in pixels with the origin top-left,
    /// kept inside the picture. The whole portrait would also carry its clothes and background into every photo
    /// (the face adapter copies what it sees), so outfits asked for in a chat lose to the portrait's.
    public static func faceCrop(face: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
        let w = Double(imageWidth), h = Double(imageHeight)
        let fw = face.width * w, fh = face.height * h
        var side = max(fw, fh) * faceCropScale
        side = min(side, w, h)
        let cx = face.midX * w
        let cy = (1 - face.midY) * h - fh * 0.15
        var x = cx - side / 2, y = cy - side / 2
        x = min(max(x, 0), w - side)
        y = min(max(y, 0), h - side)
        return CGRect(x: x.rounded(), y: y.rounded(), width: side.rounded(), height: side.rounded())
    }

    /// Throws `CharacterArtError.youthTerm(term)` when AgeGuard finds a youth term in the request.
    public static func check(_ request: String) throws {
        if let term = AgeGuard.youthTerm(in: request) { throw CharacterArtError.youthTerm(term) }
    }

    /// The server task that draws a family's pictures (for prices and timings).
    public static func task(_ family: ArtModel.Family) -> String {
        switch family {
        case .sdxl: "sdxl"
        case .chroma: "chroma"
        case .minimax: "h3-image"
        }
    }

    /// MiniMax H3 Image keeps the face by editing: the face reference goes in as <Picture 1>, and the prompt says what
    /// it is and what is kept and what changes (its official prompt guide: give each reference a role).
    public static let minimaxFaceLead = "<Picture 1> shows the face of the person in this photo: the face and hair are "
        + "exactly as in <Picture 1>, while the body, clothes, pose and background are as described here. "
    /// The same, when the picture is about a different hairstyle (the creator's hairstyle examples).
    public static let minimaxFaceOnlyLead = "<Picture 1> shows the face of the person in this photo: the face is exactly "
        + "as in <Picture 1>, while the hair, body, clothes, pose and background are as described here. "

    /// The graph: Load Image (face) → SDXL or Chroma (Cloud GPU) ← a chain of `loras` LoRA nodes → Preview Image.
    /// Chroma runs in full (40 steps with guidance), not Flash: only guidance uses the negative prompt, which holds
    /// AgeGuard's terms.
    ///
    /// MiniMax H3 Image: Load Image (face) → its picture 1; no negative prompt (AgeGuard's age and youth-term checks
    /// still apply to the prompt), no fine-tunes or LoRAs.
    public static func template(_ family: ArtModel.Family = .sdxl, loras: Int) -> MovieTemplate {
        if family == .minimax { return minimaxTemplate() }
        let n = min(max(loras, 0), maxLoRAs)
        var nodes = [GraphNode(id: "face", type: "LoadImage", inputs: ["path": .value(.string(""))],
                               position: GraphPoint(x: 40, y: 40))]
        var slots: [String: MovieTemplate.Slot] = [:]
        var gen: [String: NodeInput] = [
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(photoSize.width)), "height": .value(.int(photoSize.height)),
            "server": .value(.string("")), "model": .value(.string("")), "face_strength": .value(.double(faceStrength)),
            "face": .link(node: "face", output: "image"),
        ]
        switch family {
        case .sdxl:
            gen["steps"] = .value(.int(30))
            gen["guidance"] = .value(.double(6))
            gen["sampler"] = .value(.string("dpmpp-2m-karras"))
        case .chroma:
            gen["fast"] = .value(.bool(false))
            gen["steps"] = .value(.int(0))  // automatic: 40 at guidance 3
            gen["guidance"] = .value(.double(0))
        case .minimax:
            break  // minimaxTemplate()
        }
        for k in stride(from: 1, through: n, by: 1) {
            var inputs: [String: NodeInput] = ["lora": .value(.string("")), "strength": .value(.double(1))]
            if k > 1 { inputs["loras"] = .link(node: "lora\(k - 1)", output: "loras") }
            nodes.append(GraphNode(id: "lora\(k)", type: "RemoteLoRA", inputs: inputs,
                                   position: GraphPoint(x: 40, y: 200 + Double(k - 1) * 160)))
            slots["lora\(k)"] = .init(node: "lora\(k)", input: "lora")
            slots["lora\(k)_strength"] = .init(node: "lora\(k)", input: "strength")
        }
        if n > 0 { gen["loras"] = .link(node: "lora\(n)", output: "loras") }
        let type = family == .chroma ? "RemoteChroma" : "RemoteSDXL"
        nodes.append(GraphNode(id: "image", type: type, inputs: gen, position: GraphPoint(x: 400, y: 60)))
        nodes.append(GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "image", output: "image")],
                               position: GraphPoint(x: 780, y: 80)))
        for name in ["prompt", "negative", "seed", "width", "height", "server", "model", "face_strength"] {
            slots[name] = .init(node: "image", input: name)
        }
        slots["face"] = .init(node: "face", input: "path")
        return MovieTemplate(name: templateName, graph: Graph(nodes: nodes), slots: slots)
    }

    private static func minimaxTemplate() -> MovieTemplate {
        let gen: [String: NodeInput] = [
            "prompt": .value(.string("")), "seed": .value(.int(0)), "width": .value(.int(photoSize.width)),
            "height": .value(.int(photoSize.height)), "server": .value(.string("")), "canvas_mp": .value(.double(4)),
            "image1": .link(node: "face", output: "image"),
        ]
        let nodes = [
            GraphNode(id: "face", type: "LoadImage", inputs: ["path": .value(.string(""))], position: GraphPoint(x: 40, y: 40)),
            GraphNode(id: "image", type: "RemoteH3Image", inputs: gen, position: GraphPoint(x: 400, y: 60)),
            GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "image", output: "image")],
                      position: GraphPoint(x: 780, y: 80)),
        ]
        var slots: [String: MovieTemplate.Slot] = ["face": .init(node: "face", input: "path")]
        for name in ["prompt", "seed", "width", "height", "server"] { slots[name] = .init(node: "image", input: name) }
        return MovieTemplate(name: templateName, graph: Graph(nodes: nodes), slots: slots)
    }

    /// The values to bind for one picture. `face` nil: the face node is removed (portrait candidates).
    /// `extraNegative` is added to the character's negative prompt (the option example photos use it); `negative`
    /// replaces it (the Prompt Builder's preview writes its own). `keepHair`
    /// false: on MiniMax the face reference keeps only the face (a hairstyle example).
    public static func render(_ c: Character, prompt: String, seed: Int, server: String, face: URL?, keepHair: Bool = true,
                              extraNegative: String = "", negative written: String? = nil,
                              context: Context = .init()) -> MovieArt.Render {
        let base = written ?? negative(c, context: context)
        let negative = extraNegative.isEmpty ? base : base + ", " + extraNegative
        let lead = c.artModel.family == .minimax && face != nil ? (keepHair ? minimaxFaceLead : minimaxFaceOnlyLead) : ""
        var values: [String: JSONValue] = [
            "prompt": .string(lead + prompt), "negative": .string(negative), "seed": .int(seed),
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
