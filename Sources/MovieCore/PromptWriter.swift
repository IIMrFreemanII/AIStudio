import Foundation

/// The models the app writes prompts for, and how each one reads a prompt (docs/prompting.md has the rules and sources).
public enum PromptModel: String, CaseIterable, Codable, Sendable, Identifiable {
    case sd15, sdxl, sdxlPony, sdxlIllustrious, sd35, flux, chroma, anima, qwenImage, minimaxImage
    case wan, ltx23, minimaxVideo

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .sd15: "SD 1.5"
        case .sdxl: "SDXL"
        case .sdxlPony: "SDXL · Pony"
        case .sdxlIllustrious: "SDXL · Illustrious / NoobAI"
        case .sd35: "SD 3.5"
        case .flux: "FLUX.1"
        case .chroma: "Chroma"
        case .anima: "Anima"
        case .qwenImage: "Qwen-Image"
        case .minimaxImage: "MiniMax H3 Image"
        case .wan: "Wan 2.2 (video)"
        case .ltx23: "LTX-2.3 (video + sound)"
        case .minimaxVideo: "MiniMax H3 (video + sound)"
        }
    }

    /// How the prompt is written.
    public enum Dialect: Sendable { case tags, sentences, narrative, motion }

    public var dialect: Dialect {
        switch self {
        case .sd15, .sdxl, .sdxlPony, .sdxlIllustrious, .anima: .tags
        case .sd35, .flux, .chroma, .qwenImage: .sentences
        case .minimaxImage: .narrative
        case .wan, .ltx23, .minimaxVideo: .motion
        }
    }

    /// The model reads a negative prompt (classifier-free guidance, or NAG for Wan's fast mode). MiniMax (distilled,
    /// no CFG) and FLUX.1 dev (guidance-distilled) don't: their exclusions are written as sentences instead.
    /// Trained on Danbooru tags (Pony, Illustrious, Anima): a tag means what it says there.
    public var readsBooru: Bool { !qualityTags.isEmpty }

    public var takesNegative: Bool {
        switch self {
        case .flux, .minimaxImage, .minimaxVideo: false
        default: true
        }
    }

    /// `(words:1.2)` emphasis works (the SD 1.5 / SDXL workers parse A1111 weights).
    public var takesWeights: Bool { [.sd15, .sdxl, .sdxlPony, .sdxlIllustrious].contains(self) }

    /// The model makes sound too: the prompt describes it.
    public var hasSound: Bool { self == .ltx23 || self == .minimaxVideo }

    public var isVideo: Bool { dialect == .motion }

    /// Can draw legible text from the prompt (quoted words).
    public var drawsText: Bool { [.flux, .sd35, .qwenImage, .minimaxImage, .chroma].contains(self) }

    /// Quality tags a fine-tune family expects up front, and their negative counterpart.
    public var qualityTags: [String] {
        switch self {
        case .sdxlPony: ["score_9", "score_8_up", "score_7_up"]
        case .sdxlIllustrious: ["masterpiece", "best quality", "very aesthetic", "absurdres"]
        case .anima: ["masterpiece", "best quality"]
        default: []
        }
    }

    public var qualityNegative: [String] {
        switch self {
        case .sdxlPony: ["score_6", "score_5", "score_4"]
        case .sdxlIllustrious, .anima: ["worst quality", "low quality", "lowres"]
        default: []
        }
    }

    /// The SDXL flavour for a fine-tune from its Civitai base model ("Pony", "Illustrious", "NoobAI", "SDXL 1.0").
    public static func sdxl(baseModel: String?) -> PromptModel {
        let b = (baseModel ?? "").lowercased()
        if b.contains("pony") { return .sdxlPony }
        if b.contains("illustrious") || b.contains("noobai") || b.contains("noob ai") { return .sdxlIllustrious }
        return .sdxl
    }
}

/// One thing about the subject, in two forms: a short `tag` for tag-reading models (CLIP) and a fuller `text` for
/// models that read sentences. `weight` (> 1) is emphasis where the model takes weights.
public struct PromptDetail: Equatable, Sendable, Hashable, Codable {
    public var tag: String
    public var text: String
    public var weight: Double

    public init(tag: String, text: String? = nil, weight: Double = 1) {
        self.tag = tag
        self.text = text ?? tag
        self.weight = weight
    }

    public init(_ both: String, weight: Double = 1) { self.init(tag: both, text: both, weight: weight) }
}

/// What a picture or clip shows, independent of any model. `PromptWriter.write` turns it into one model's prompt.
public struct PromptSpec: Equatable, Sendable, Codable {
    public enum Pronoun: String, Sendable, Codable { case she, he, they, it }
    /// A reference picture (`<Picture 1>` on MiniMax; the face adapter elsewhere, where the prompt doesn't mention it).
    public enum Reference: String, Sendable, Codable { case none, faceAndHair, faceOnly }
    /// How much a person wears (18+ choices beyond dressed): replaces the outfit. Revealing coverage adds an adult
    /// cue to the body, and to the closing sentences of models without a negative prompt.
    public enum Coverage: String, Sendable, Codable, CaseIterable {
        case dressed, swimwear, lingerie, underwear, topless, impliedNude, nude

        /// The short tag the tag models read ("" when dressed).
        public var tag: String {
            switch self {
            case .dressed: ""
            case .swimwear: "swimsuit"
            case .lingerie: "lingerie"
            case .underwear: "only underwear"
            case .topless: "topless"
            case .impliedNude: "implied nudity, covered by a bedsheet"
            case .nude: "fully nude, naked"
            }
        }

        /// The parts shown bare: topless shows the chest, nude everything; implied nudity covers it.
        public var showsChest: Bool { self == .topless || self == .nude }
        public var showsAll: Bool { self == .nude }
        /// Bare skin where an adult cue belongs.
        public var revealing: Bool { self == .topless || self == .impliedNude || self == .nude }
    }

    /// The adult cue revealing pictures carry (in the body, and as a sentence where there is no negative prompt).
    public static let adultBody = "an adult body with mature proportions"
    public static let adultSentence = "The person is clearly a grown adult"

    /// What the picture is, e.g. "candid smartphone photo", "35mm film photo", "anime illustration".
    public var medium = ""
    /// Who or what, with its count and age when a person: "a 30-year-old adult man".
    public var subject = ""
    public var pronoun = Pronoun.it
    /// Framing: "full body", "head-and-shoulders portrait", "close-up".
    public var shot = ""
    /// What the subject is doing (or, for video, what happens).
    public var action = ""
    /// Looks: face, hair, skin, extras.
    public var details: [PromptDetail] = []
    /// Body shape.
    public var body: [PromptDetail] = []
    /// What they wear.
    public var outfit: [PromptDetail] = []
    public var place = ""
    /// Time of day, season, weather.
    public var time: [String] = []
    /// Camera angle, lens, (video) movement.
    public var camera: [String] = []
    public var lighting: [String] = []
    public var mood: [String] = []
    public var palette: [String] = []
    public var composition: [String] = []
    /// Video with sound: what is heard.
    public var sound: [String] = []
    /// Exact words to show in the picture ("" = none; models that draw text are then told to draw none).
    public var text = ""
    public var reference = Reference.none
    /// Concepts to keep out: the negative prompt where the model reads one.
    public var avoid: [String] = []
    /// The same kept out in words, for models without a negative ("no visible abs").
    public var without: [String] = []
    /// LoRA trigger words: first in tag and sentence prompts (MiniMax and the video models take no LoRAs here).
    public var triggerWords: [String] = []
    /// Dressed (the outfit), or less (18+): replaces the outfit when not dressed.
    public var coverage = Coverage.dressed

    public init() {}

    /// The spec as the writers read it: revealing coverage adds the adult cue.
    var withCoverage: PromptSpec {
        guard coverage.revealing else { return self }
        var s = self
        if !s.body.contains(where: { $0.tag == Self.adultBody }) { s.body.append(PromptDetail(Self.adultBody)) }
        if !s.without.contains(Self.adultSentence) { s.without.append(Self.adultSentence) }
        return s
    }

    public init(from decoder: Decoder) throws {
        let k = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ key: CodingKeys, _ fallback: T) -> T { (try? k.decodeIfPresent(T.self, forKey: key)) ?? fallback }
        medium = v(.medium, medium); subject = v(.subject, subject); pronoun = v(.pronoun, pronoun)
        shot = v(.shot, shot); action = v(.action, action); details = v(.details, details); body = v(.body, body)
        outfit = v(.outfit, outfit); place = v(.place, place); time = v(.time, time); camera = v(.camera, camera)
        lighting = v(.lighting, lighting); mood = v(.mood, mood); palette = v(.palette, palette)
        composition = v(.composition, composition); sound = v(.sound, sound); text = v(.text, text)
        reference = v(.reference, reference); avoid = v(.avoid, avoid); without = v(.without, without)
        triggerWords = v(.triggerWords, triggerWords); coverage = v(.coverage, coverage)
    }
}

/// A model's prompt and negative ("" when the model takes none).
public struct WrittenPrompt: Equatable, Sendable {
    public var prompt: String
    public var negative: String
}

/// Writes a `PromptSpec` the way each model reads best (docs/prompting.md):
/// - **tags** (SD 1.5, SDXL and fine-tunes, Anima): a short comma list, most important first, because CLIP weighs the
///   start most and older workers cut at 77 tokens; emphasis as `(…:1.2)`; exclusions in the negative.
/// - **sentences** (Chroma, FLUX, SD 3.5, Qwen-Image; T5 or an LLM encoder): plain sentences describing what is
///   visible; exclusions in the negative, or as sentences when the model takes none (FLUX).
/// - **narrative** (MiniMax H3 Image, after its official prompt-writing guide): one paragraph, purpose and framing first,
///   then the subject, then style through visible light and colour; the reference picture's role spelled out; no
///   negative, so exclusions are closing sentences, including "no text" unless text is asked for.
/// - **motion** (Wan, LTX-2.3, MiniMax H3 video): one flowing paragraph, the shot and camera move, then the action as it
///   unfolds, the look, the place and light, and the sound for models that make it.
public enum PromptWriter {
    public static func write(_ spec: PromptSpec, for model: PromptModel) -> WrittenPrompt {
        let s = spec.withCoverage
        let prompt: String
        switch model.dialect {
        case .tags: prompt = tags(s, model)
        case .sentences: prompt = sentences(s, model)
        case .narrative: prompt = narrative(s, model)
        case .motion: prompt = motion(s, model)
        }
        return WrittenPrompt(prompt: prompt, negative: model.takesNegative ? negative(s, model) : "")
    }

    // MARK: Tags

    private static func tags(_ s: PromptSpec, _ m: PromptModel) -> String {
        var out: [String] = m.qualityTags + s.triggerWords.map(clean)
        out.append(clean(s.medium))
        out.append(clean(s.subject))
        out.append(clean(s.shot))
        out.append(lowerFirst(clean(s.action)))
        let outfit = s.coverage == .dressed ? s.outfit : [PromptDetail(s.coverage.tag, weight: 1.2)]
        for d in s.body + outfit + s.details { out.append(weighted(d, m)) }
        if !s.place.isEmpty { out.append(clean(s.place)) }
        out += (s.time + s.lighting + s.camera + s.mood + s.palette + s.composition).map(clean)
        if !s.text.isEmpty, m.drawsText { out.append("text \"\(clean(s.text))\"") }
        return unique(out).joined(separator: ", ")
    }

    private static func weighted(_ d: PromptDetail, _ m: PromptModel) -> String {
        let t = stripArticle(clean(d.tag))
        guard m.takesWeights, d.weight != 1, !t.isEmpty else { return t }
        return "(\(t.replacingOccurrences(of: "(", with: "\\(").replacingOccurrences(of: ")", with: "\\)")):"
            + String(format: "%.1f", d.weight) + ")"
    }

    // MARK: Sentences

    private static func sentences(_ s: PromptSpec, _ m: PromptModel) -> String {
        // LoRA trigger words lead, as they were trained (a LoRA learns them as a caption's first words).
        let triggers = s.triggerWords.map(clean).filter { !$0.isEmpty }
        var out: [String] = triggers.isEmpty ? [] : [triggers.joined(separator: ", ") + "."]
        out.append(opening(s))
        out += describe(s)
        out += look(s, m)
        if !m.takesNegative { out += s.without.map(sentence) }
        return out.filter { !$0.isEmpty }.joined(separator: " ")
    }

    /// "A candid smartphone photo, full body, of a 30-year-old adult man sitting on a bench."
    private static func opening(_ s: PromptSpec) -> String {
        let medium = clean(s.medium), shot = clean(s.shot), subject = clean(s.subject), action = clean(s.action)
        var head = ""
        if !medium.isEmpty { head = article(medium) + (shot.isEmpty ? "" : ", \(shot),") }
        else if !shot.isEmpty { head = article(shot) }
        var body = subject
        if !action.isEmpty { body += body.isEmpty ? action : joinAction(action) }
        if head.isEmpty { return sentence(body) }
        return sentence(body.isEmpty ? head : "\(head) of \(body)")
    }

    /// The subject's looks, body, outfit, place and time as sentences.
    private static func describe(_ s: PromptSpec) -> [String] {
        let (p, has, wears) = verbs(s.pronoun)
        var out: [String] = []
        let looks = s.details.map { clean($0.text) }.filter { !$0.isEmpty }
        if !looks.isEmpty { out.append(sentence("\(p) \(has) \(list(looks))")) }
        let body = s.body.map { clean($0.text) }.filter { !$0.isEmpty }
        if !body.isEmpty { out.append(sentence("\(p) \(has) \(list(body))")) }
        let clothes = s.outfit.map { clean($0.text) }.filter { !$0.isEmpty }
        if s.coverage != .dressed { out.append(coverageSentence(s.coverage, s.pronoun)) }
        else if !clothes.isEmpty { out.append(sentence("\(p) \(wears) \(list(clothes))")) }
        let place = clean(s.place), time = s.time.map(clean).filter { !$0.isEmpty }
        if !place.isEmpty || !time.isEmpty {
            var where_ = place.isEmpty ? "" : (startsWithPreposition(place) ? place : "at \(place)")
            if !time.isEmpty { where_ += (where_.isEmpty ? "" : ", ") + list(time) }
            out.append(sentence(where_))
        }
        return out
    }

    /// Light, camera, mood, colour, composition and any text, as short sentences.
    private static func look(_ s: PromptSpec, _ m: PromptModel) -> [String] {
        var out: [String] = []
        let lighting = s.lighting.map(clean).filter { !$0.isEmpty }
        if !lighting.isEmpty { out.append(sentence(list(lighting))) }
        let camera = s.camera.map(clean).filter { !$0.isEmpty }
        if !camera.isEmpty { out.append(sentence(list(camera))) }
        let mood = s.mood.map(clean).filter { !$0.isEmpty }
        if !mood.isEmpty { out.append(sentence("The mood is \(list(mood))")) }
        let palette = s.palette.map(clean).filter { !$0.isEmpty }
        if !palette.isEmpty { out.append(sentence("Colours: \(list(palette))")) }
        let comp = s.composition.map(clean).filter { !$0.isEmpty }
        if !comp.isEmpty { out.append(sentence(list(comp))) }
        if m.drawsText {
            let text = clean(s.text)
            if !text.isEmpty { out.append("The text \"\(text)\" appears in the picture, spelled exactly.") }
        }
        return out
    }

    // MARK: Narrative (MiniMax H3 Image)

    private static func narrative(_ s: PromptSpec, _ m: PromptModel) -> String {
        var out: [String] = []
        let possessive = possessive(s.pronoun)
        if s.reference != .none {
            out.append("<Picture 1> shows the face of \(clean(s.subject).isEmpty ? "the subject" : clean(s.subject)).")
        }
        var lead = s
        if s.reference != .none {
            // The subject was introduced with the reference: the picture shows "that same" person.
            lead.subject = "that same " + noun(of: clean(s.subject))
        }
        out.append(opening(lead))
        switch s.reference {
        case .faceAndHair:
            out.append(sentence("\(possessive.capitalizedFirst) face and hair are kept exactly as in <Picture 1>; "
                                + "\(possessive) body, clothes, pose and background are as described here"))
        case .faceOnly:
            out.append(sentence("\(possessive.capitalizedFirst) face is kept exactly as in <Picture 1>; "
                                + "\(possessive) hair, body, clothes, pose and background are as described here"))
        case .none: break
        }
        out += describe(s)
        out += look(s, m)
        if clean(s.text).isEmpty { out.append("The image contains no text, letters or numbers.") }
        out += s.without.map(sentence)
        return out.filter { !$0.isEmpty }.joined(separator: " ")
    }

    // MARK: Motion (video)

    private static func motion(_ s: PromptSpec, _ m: PromptModel) -> String {
        var out: [String] = []
        let shot = clean(s.shot), camera = s.camera.map(clean).filter { !$0.isEmpty }
        var frame = shot.isEmpty ? "" : article(shot)
        if !camera.isEmpty { frame += (frame.isEmpty ? "" : ", ") + list(camera) }
        let medium = clean(s.medium)
        if !medium.isEmpty { frame = frame.isEmpty ? article(medium) : "\(article(medium)), \(frame)" }
        if !frame.isEmpty { out.append(sentence(frame)) }
        let subject = clean(s.subject), action = clean(s.action)
        if !subject.isEmpty || !action.isEmpty {
            out.append(sentence(subject.isEmpty ? action : subject + joinAction(action)))
        }
        out += describe(s)
        var looks = s
        looks.camera = []  // already in the opening
        out += look(looks, m)
        if m.hasSound {
            let sound = s.sound.map(clean).filter { !$0.isEmpty }
            if !sound.isEmpty { out.append(sentence("Sound: \(list(sound))")) }
        }
        if !m.takesNegative { out += s.without.map(sentence) }
        return out.filter { !$0.isEmpty }.joined(separator: " ")
    }

    // MARK: Negative

    private static func negative(_ s: PromptSpec, _ m: PromptModel) -> String {
        unique(m.qualityNegative + s.avoid.map(clean)).joined(separator: ", ")
    }

    // MARK: Text helpers

    static func clean(_ s: String) -> String {
        s.split(whereSeparator: \.isWhitespace).joined(separator: " ").trimmingCharacters(in: CharacterSet(charactersIn: " ,.;"))
    }

    private static func unique(_ items: [String]) -> [String] {
        var seen = Set<String>(), out: [String] = []
        for i in items where !i.isEmpty && seen.insert(i.lowercased()).inserted { out.append(i) }
        return out
    }

    private static func stripArticle(_ s: String) -> String {
        for a in ["a ", "an ", "the "] where s.lowercased().hasPrefix(a) { return String(s.dropFirst(a.count)) }
        return s
    }

    /// "a candid photo" / "an anime illustration"; text that already starts with an article is left alone.
    static func article(_ s: String) -> String {
        let lower = s.lowercased()
        if ["a ", "an ", "the "].contains(where: lower.hasPrefix) { return s }
        let vowel = lower.first.map { "aeiou".contains($0) } ?? false
        return (vowel ? "an " : "a ") + s
    }

    /// The noun of "a 30-year-old adult man" for "that same …": drops a leading article.
    private static func noun(of subject: String) -> String {
        let s = stripArticle(subject)
        return s.isEmpty ? "subject" : s
    }

    /// The action after the subject: " sitting on a bench" / " sits on a bench" / " in a café"; a caption that starts
    /// with a noun phrase ("Selfie on a rooftop") follows a comma so it doesn't read as a verb.
    private static func joinAction(_ action: String) -> String {
        guard !action.isEmpty else { return "" }
        let first = action.split(separator: " ").first.map { String($0).lowercased() } ?? ""
        let verbish = first.hasSuffix("ing") || first.hasSuffix("s")
            || ["with", "in", "at", "on", "by", "under", "near", "inside", "outside", "beside"].contains(first)
        return (verbish ? " " : ", ") + lowerFirst(action)
    }

    private static func startsWithPreposition(_ s: String) -> Bool {
        let first = s.split(separator: " ").first.map { String($0).lowercased() } ?? ""
        return ["at", "in", "on", "by", "inside", "outside", "near", "under", "beside", "behind", "through", "along", "across"].contains(first)
    }

    private static func verbs(_ p: PromptSpec.Pronoun) -> (String, String, String) {
        switch p {
        case .she: ("She", "has", "wears")
        case .he: ("He", "has", "wears")
        case .they: ("They", "have", "wear")
        case .it: ("It", "has", "wears")
        }
    }

    /// "She is fully nude." and the like: coverage as a sentence (the outfit's "She wears …" would not read).
    private static func coverageSentence(_ c: PromptSpec.Coverage, _ p: PromptSpec.Pronoun) -> String {
        let (who, _, wears) = verbs(p)
        let isVerb = p == .they ? "are" : "is"
        switch c {
        case .dressed: return ""
        case .swimwear: return sentence("\(who) \(wears) a swimsuit")
        case .lingerie: return sentence("\(who) \(wears) lingerie")
        case .underwear: return sentence("\(who) \(wears) only underwear")
        case .topless: return sentence("\(who) \(isVerb) topless, bare-chested")
        case .impliedNude: return sentence("\(who) \(isVerb) nude but covered by a bedsheet, nothing explicit visible")
        case .nude: return sentence("\(who) \(isVerb) fully nude")
        }
    }

    private static func possessive(_ p: PromptSpec.Pronoun) -> String {
        switch p {
        case .she: "her"
        case .he: "his"
        case .they: "their"
        case .it: "its"
        }
    }

    /// "a, b and c".
    static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: ""
        case 1: items[0]
        default: items.dropLast().joined(separator: ", ") + " and " + items.last!
        }
    }

    /// Capitalized, ending in a period.
    static func sentence(_ s: String) -> String {
        let t = clean(s)
        if t.isEmpty { return "" }
        let body = t.capitalizedFirst
        return body.hasSuffix("?") || body.hasSuffix("!") || body.hasSuffix("\"") ? body : body + "."
    }

    private static func lowerFirst(_ s: String) -> String {
        guard let f = s.first else { return s }
        // A caption's capital ("Sitting on …") is lowered; acronyms ("LED …") are kept.
        if s.count > 1, f.isUppercase, s.dropFirst().first?.isLowercase == true { return f.lowercased() + s.dropFirst() }
        return s
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
