import CharacterCore
import Foundation
import MovieCore

/// What the Prompt Builder has picked: the model to write for, the subject (a person, with the character creator's
/// look options, or anything typed), what happens, and the catalog's chips. `written` turns it into that model's
/// prompt (PromptWriter; docs/prompting.md has the rules).
struct PromptBuilderDraft: Codable, Equatable {
    var model: PromptModel = .sdxl
    /// Free text: the whole subject, or for a person what they are ("a lighthouse keeper").
    var subject = ""
    var isPerson = true
    var gender = "woman"
    var age = 28
    /// The character creator's look picks (category id → option ids).
    var traits: [String: [String]] = [:]
    var action = ""
    var place = ""
    var text = ""
    var avoid = ""
    /// For MiniMax: <Picture 1> is the person's face.
    var faceReference = false
    var triggerWords = ""
    /// Catalog picks (group id → choice ids).
    var picks: [String: [String]] = [:]

    // The preview (PromptBuilderPreview).
    /// The saved character the look came from ("Start from Character"): the preview uses their face, LoRAs and
    /// fine-tune. Changing the draft never changes the character.
    var characterID: UUID?
    var characterName = ""
    /// The library fine-tune the preview draws with ("" = the base model).
    var variant = ""
    var aspect: PromptPreview.Aspect = .portrait
    var seed = Int.random(in: 0..<Int(Int32.max))
    /// The same seed for every preview, so a changed option shows only its own effect.
    var keepSeed = true

    static let storageKey = "promptBuilder.draft"

    init() {}

    /// Missing keys (a draft saved by an older build) keep their defaults.
    init(from decoder: Decoder) throws {
        let k = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ key: CodingKeys, _ fallback: T) -> T { (try? k.decodeIfPresent(T.self, forKey: key)) ?? fallback }
        model = v(.model, model)
        subject = v(.subject, subject)
        isPerson = v(.isPerson, isPerson)
        gender = v(.gender, gender)
        age = v(.age, age)
        traits = v(.traits, traits)
        action = v(.action, action)
        place = v(.place, place)
        text = v(.text, text)
        avoid = v(.avoid, avoid)
        faceReference = v(.faceReference, faceReference)
        triggerWords = v(.triggerWords, triggerWords)
        picks = v(.picks, picks)
        characterID = try? k.decodeIfPresent(UUID.self, forKey: .characterID)
        characterName = v(.characterName, characterName)
        variant = v(.variant, variant)
        aspect = v(.aspect, aspect)
        seed = v(.seed, seed)
        keepSeed = v(.keepSeed, keepSeed)
    }

    /// Takes a saved character's look, age, gender, image model and trigger words (`model`/`triggerWords` from
    /// `CharacterArt.promptModel` and `AppModel.artContext`). The scene, picks and Keep-out text stay.
    mutating func apply(character c: CharacterCore.Character, model: PromptModel, triggerWords words: [String]) {
        characterID = c.id
        characterName = c.name
        isPerson = true
        gender = c.gender.isEmpty ? "person" : c.gender
        age = max(c.age, AgeGuard.minimumAge)
        traits = c.traits
        subject = Self.clean(c.look)
        self.model = model
        variant = c.artModel.variant ?? ""
        triggerWords = words.joined(separator: ", ")
        // MiniMax keeps the face from <Picture 1>, which the preview fills with the character's face.
        faceReference = model == .minimaxImage
        if picks["medium"] == nil {
            picks["medium"] = [c.artModel.style == .anime ? "anime" : "candid-phone"]
        }
    }

    /// What differs from `old`, in words ("+ Freckles", "− Golden hour", "Seed 12 → 40"): under two previews compared.
    func changes(from old: PromptBuilderDraft) -> [String] {
        var out: [String] = []
        func field(_ name: String, _ a: String, _ b: String) {
            if a != b { out.append(b.isEmpty ? "− \(name)" : "\(name): \(b)") }
        }
        if model != old.model { out.append("Model: \(model.title)") }
        field("Subject", old.subject, subject)
        if gender != old.gender { out.append("Gender: \(gender)") }
        if age != old.age { out.append("Age \(old.age) → \(age)") }
        func labels(_ ids: Set<String>, _ label: (String) -> String?) -> [String] { ids.compactMap(label).sorted() }
        for id in Set(traits.keys).union(old.traits.keys).sorted() {
            let a = Set(old.traits[id] ?? []), b = Set(traits[id] ?? [])
            let cat = CharacterOptions.category(id)
            let label: (String) -> String? = { o in cat?.options.first { $0.id == o }?.label ?? o }
            out += labels(b.subtracting(a), label).map { "+ \($0)" }
            out += labels(a.subtracting(b), label).map { "− \($0)" }
        }
        for id in Set(picks.keys).union(old.picks.keys).sorted() {
            let a = Set(old.picks[id] ?? []), b = Set(picks[id] ?? [])
            let group = PromptCatalog.group(id)
            let label: (String) -> String? = { c in group?.choices.first { $0.id == c }?.label ?? c }
            out += labels(b.subtracting(a), label).map { "+ \($0)" }
            out += labels(a.subtracting(b), label).map { "− \($0)" }
        }
        field("Action", old.action, action)
        field("Place", old.place, place)
        field("Text", old.text, text)
        field("Keep out", old.avoid, avoid)
        field("Trigger words", old.triggerWords, triggerWords)
        if variant != old.variant { out.append(variant.isEmpty ? "Base model" : "Fine-tune changed") }
        if aspect != old.aspect { out.append(aspect.title) }
        if seed != old.seed { out.append("Seed \(old.seed) → \(seed)") }
        return out
    }

    /// Unlinks the character (the look picks stay).
    mutating func detachCharacter() {
        characterID = nil
        characterName = ""
    }

    /// The description the writers read.
    var spec: PromptSpec {
        var s = PromptSpec()
        let who = Self.clean(subject)
        if isPerson {
            let g = Self.clean(gender)
            s.subject = "a \(max(age, AgeGuard.minimumAge))-year-old adult " + (g.isEmpty ? "person" : g)
            switch g.lowercased() {
            case "woman": s.pronoun = .she
            case "man": s.pronoun = .he
            default: s.pronoun = .they
            }
            s.coverage = PromptCatalog.coverage(picks, adult: adult)
            let parts = CharacterOptions.lookParts(traits, gender: g, coverage: s.coverage, adult: adult)
            // Who they are: the race ("a kitsune with fox ears…"), then what they are ("a jazz singer").
            for part in [parts.race, who] where !part.isEmpty { s.subject += ", " + part }
            s.details = parts.details
            s.body = parts.body
            s.outfit = parts.outfit
            s.avoid = AgeGuard.negative.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } + parts.avoid
            s.without = parts.without
            if faceReference { s.reference = .faceAndHair }
        } else {
            s.subject = who
            s.pronoun = .it
        }
        s.action = Self.clean(action)
        s.place = Self.clean(place)
        s.text = Self.clean(text)
        s.avoid += Self.list(avoid)
        s.triggerWords = Self.list(triggerWords)
        PromptCatalog.apply(picks, to: &s, model: model, adult: adult && isPerson)
        return s
    }

    /// The 18+ options are on ("Show 18+ options"; automated runs set `adultOverride` instead of the user default).
    var adult: Bool { AppModel.adultOptions }
    nonisolated(unsafe) static var adultOverride: Bool?

    /// The prompt for the picked model; nil when a free-text field describes someone under 18.
    var written: WrittenPrompt? {
        if problem != nil { return nil }
        return PromptWriter.write(spec, for: model)
    }

    /// Why nothing can be written (a youth term in the text), else nil.
    var problem: String? {
        for (field, value) in [("Subject", subject), ("What happens", action), ("Place", place), ("Text", text)] {
            if let term = AgeGuard.youthTerm(in: value) { return "\(field) describes someone under 18 (“\(term)”)." }
        }
        return nil
    }

    private static func clean(_ s: String) -> String {
        s.split(whereSeparator: \.isWhitespace).joined(separator: " ").trimmingCharacters(in: CharacterSet(charactersIn: " ,.;"))
    }

    private static func list(_ s: String) -> [String] {
        s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }

    /// The prompt model of a node type (and, for SDXL, its fine-tune's Civitai base model); nil when the node takes no
    /// prompt the builder can write.
    static func promptModel(nodeType: String, baseModel: String?) -> PromptModel? {
        switch nodeType {
        case "RemoteSD15": .sd15
        case "RemoteSDXL": PromptModel.sdxl(baseModel: baseModel)
        case "RemoteSD35": .sd35
        case "RemoteFlux": .flux
        case "RemoteChroma": .chroma
        case "RemoteAnima": .anima
        case "RemoteQwenImage", "RemoteQwenImageEdit": .qwenImage
        case "RemoteH3Image": .minimaxImage
        case "RemoteWanTextToVideo", "RemoteWanImageToVideo", "RemoteWanSpeechToVideo": .wan
        case "RemoteLTX23": .ltx23
        case "RemoteH3Video", "RemoteH3References": .minimaxVideo
        default: nil
        }
    }
}

/// The node the Prompt Builder sheet writes into.
struct PromptBuilderTarget: Identifiable, Equatable {
    let id: String  // node id
    let model: PromptModel
    /// The node has a negative prompt input.
    let takesNegative: Bool
}
