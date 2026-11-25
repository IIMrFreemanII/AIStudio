import Agent
import Foundation
import LLMClient
import MovieCore

/// "Surprise Me" in the creator: a random character from the options catalog, and the free-text details (story,
/// personality, likes, how they text, a first message) a chat model writes to fit the picks.
public enum CharacterSurprise {
    static let names: [String: [String]] = [
        "woman": ["Ava", "Chloe", "Daria", "Elise", "Freya", "Isla", "Jade", "Lena", "Maya", "Nadia", "Nora", "Olivia",
                  "Priya", "Rosa", "Selene", "Tessa", "Vera", "Yuna", "Zoe", "Camille"],
        "man": ["Adrian", "Ben", "Caleb", "Dante", "Elias", "Felix", "Gabe", "Hugo", "Ivan", "Jonah", "Leon", "Marco",
                "Nico", "Omar", "Rafael", "Sam", "Theo", "Viktor", "Wes", "Kenji"],
        "non-binary": ["Alex", "Ari", "Blake", "Charlie", "Eden", "Jules", "Kai", "Quinn", "Remy", "Rowan", "Sage", "Sky"],
    ]
    public static let ages = 21...45

    /// A new random character in place of `c`: a name for its gender (the gender is kept when it's woman, man or
    /// non-binary, else one is picked), an age, and picks in every option category (one where only one fits, else one
    /// to three). The free-text fields are cleared so nothing left over contradicts the picks; its id, image model,
    /// portraits, voice samples and chat settings stay.
    public static func roll(_ c: Character, using rng: inout some RandomNumberGenerator) -> Character {
        var out = c
        let known = c.gender.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        out.gender = names[known] != nil ? known : ["woman", "man"].randomElement(using: &rng)!
        out.name = names[out.gender]!.randomElement(using: &rng)!
        out.age = Int.random(in: ages, using: &rng)
        var traits: [String: [String]] = [:]
        for category in CharacterOptions.all {
            let options = CharacterOptions.options(category, gender: out.gender)
            guard !options.isEmpty else { continue }
            let count = category.max == 1 ? 1 : Int.random(in: 1...min(category.max, 3), using: &rng)
            // Catalog order, so the picks read the same way the chips show them.
            let picked = Set(options.shuffled(using: &rng).prefix(count).map(\.id))
            traits[category.id] = options.map(\.id).filter { picked.contains($0) }
        }
        out.traits = traits
        out.look = ""
        out.personality = ""
        out.backstory = ""
        out.speakingStyle = ""
        out.likes = ""
        out.voice = ""
        out.greeting = ""
        return out
    }

    public static func roll(_ c: Character) -> Character {
        var rng = SystemRandomNumberGenerator()
        return roll(c, using: &rng)
    }
}

/// What a chat model writes to round a character out. Every field may be missing or empty.
public struct CharacterDetails: Codable, Equatable, Sendable {
    public var name: String?
    public var look: String?
    public var personality: String?
    public var backstory: String?
    public var speakingStyle: String?
    public var likes: String?
    public var greeting: String?

    public init(name: String? = nil, look: String? = nil, personality: String? = nil, backstory: String? = nil,
                speakingStyle: String? = nil, likes: String? = nil, greeting: String? = nil) {
        self.name = name
        self.look = look
        self.personality = personality
        self.backstory = backstory
        self.speakingStyle = speakingStyle
        self.likes = likes
        self.greeting = greeting
    }

    enum CodingKeys: String, CodingKey {
        case name, look, personality, backstory, likes, greeting
        case speakingStyle = "speaking_style"
    }

    /// The character with these details as its free text; the name too when `name` is true. Empty fields leave the
    /// character's own text.
    public func apply(to c: Character, name useName: Bool) -> Character {
        var out = c
        func clean(_ s: String?) -> String? {
            let t = (s ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : t
        }
        if useName, let n = clean(name), n.count <= 30 { out.name = n }
        if let v = clean(look) { out.look = v }
        if let v = clean(personality) { out.personality = v }
        if let v = clean(backstory) { out.backstory = v }
        if let v = clean(speakingStyle) { out.speakingStyle = v }
        if let v = clean(likes) { out.likes = v }
        if let v = clean(greeting) { out.greeting = v }
        return out
    }
}

/// Asks a chat model for a character's details from its picks.
public enum CharacterWriter {
    public static let system = """
        You help create a fictional adult companion character for a chat app. From the character's picked traits, \
        write the details that make them feel like a real person, consistent with every pick. Reply with only JSON, \
        no other text, in this shape:
        {"name": "...", "look": "...", "personality": "...", "backstory": "...", "speaking_style": "...", "likes": "...", "greeting": "..."}
        - name: a first name that suits them (keep the given one unless asked for a new one).
        - look: one or two small distinctive details the picks don't say (a scar, a favourite necklace), under 15 words.
        - personality: what they're like beyond the picks, one sentence.
        - backstory: two or three sentences: where they live, their job, something that shaped them. They are an adult.
        - speaking_style: how they text, one short phrase.
        - likes: four to six specific things they love, comma-separated.
        - greeting: their first text message to the user, in their voice, one or two sentences.
        Don't repeat the picks word for word; build on them.
        """

    /// The request. Its first line is "Task: character details".
    public static func user(_ c: Character, newName: Bool) -> String {
        var out = "Task: character details\n"
        out += "Name: \(c.name.isEmpty ? "(none yet)" : c.name)\(newName ? " (suggest a new one)" : "")\n"
        out += "Age: \(c.age)\nGender: \(c.gender)\n"
        for (label, text) in [("Look", c.fullLook), ("Personality", c.fullPersonality), ("Backstory", c.fullBackstory),
                              ("Texting style", c.fullSpeakingStyle), ("Habits", c.habits), ("Interests", c.fullLikes),
                              ("Voice", c.fullVoice)] where !text.isEmpty {
            out += "\(label): \(text)\n"
        }
        return out
    }

    /// Parses a reply: strips ``` fences, takes the first `{` to the last `}`, decodes it.
    public static func parse(_ text: String) throws -> CharacterDetails {
        let s = text.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        guard let open = s.firstIndex(of: "{"), let close = s.lastIndex(of: "}"), open < close else {
            throw MoviePlannerError.noJSON
        }
        do {
            return try JSONDecoder().decode(CharacterDetails.self, from: Data(s[open...close].utf8))
        } catch {
            throw MoviePlannerError.invalid("\(error)")
        }
    }

    /// Asks the model (one retry on a parse error); returns the details and the summed cost.
    public static func run(client: any ChatStreaming, model: String, character: Character, newName: Bool,
                           onProgress: @escaping @Sendable (MoviePlanner.Progress) -> Void = { _ in })
        async throws -> (details: CharacterDetails, cost: Double)
    {
        let box = CostBox()
        let details: CharacterDetails = try await MoviePlanner.ask(
            client: client, model: model, system: system, user: user(character, newName: newName), maxTokens: 700,
            onProgress: onProgress, onCost: { box.add($0) }, parse: { try parse($0) })
        return (details, box.total)
    }
}
