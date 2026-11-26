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

    static let storageKey = "promptBuilder.draft"

    /// The description the writers read.
    var spec: PromptSpec {
        var s = PromptSpec()
        let who = Self.clean(subject)
        if isPerson {
            let g = Self.clean(gender)
            s.subject = "a \(max(age, AgeGuard.minimumAge))-year-old adult " + (g.isEmpty ? "person" : g)
                + (who.isEmpty ? "" : ", \(who)")
            switch g.lowercased() {
            case "woman": s.pronoun = .she
            case "man": s.pronoun = .he
            default: s.pronoun = .they
            }
            let parts = CharacterOptions.lookParts(traits, gender: g)
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
        PromptCatalog.apply(picks, to: &s, model: model)
        return s
    }

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
