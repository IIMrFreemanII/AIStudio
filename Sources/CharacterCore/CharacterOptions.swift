import Foundation

/// Predefined options for creating a character: the user taps chips instead of typing.
/// Picks are stored on `Character.traits` (category id → option ids) and turned into prompt text here.
public enum CharacterOptions {
    public enum Group: String, CaseIterable, Sendable { case basics, look, personality, habits, voice }

    public struct Option: Identifiable, Equatable, Sendable, Hashable {
        public let id: String
        public let label: String
        public let phrase: String
        /// "man" or "woman": shown only for that gender; nil = everyone.
        public let onlyFor: String?
        public init(_ id: String, _ label: String, _ phrase: String, onlyFor: String? = nil) {
            self.id = id
            self.label = label
            self.phrase = phrase
            self.onlyFor = onlyFor
        }
    }

    public struct Category: Identifiable, Equatable, Sendable {
        public let id: String
        public let title: String
        public let group: Group
        /// 1 = pick one.
        public let max: Int
        public let options: [Option]
        public init(_ id: String, _ title: String, _ group: Group, max: Int, _ options: [Option]) {
            self.id = id
            self.title = title
            self.group = group
            self.max = max
            self.options = options
        }
    }

    private typealias O = Option

    public static let all: [Category] = [
        Category("occupation", "Occupation", .basics, max: 1, [
            O("barista", "Barista", "works as a barista"), O("artist", "Artist", "works as an artist"),
            O("nurse", "Nurse", "works as a nurse"), O("architect", "Architect", "works as an architect"),
            O("musician", "Musician", "works as a musician"), O("chef", "Chef", "works as a chef"),
            O("photographer", "Photographer", "works as a photographer"),
            O("fitness-coach", "Fitness coach", "works as a fitness coach"),
            O("software-developer", "Software developer", "works as a software developer"),
            O("writer", "Writer", "works as a writer"), O("lawyer", "Lawyer", "works as a lawyer"),
            O("bartender", "Bartender", "works as a bartender"),
        ]),
        Category("met", "How you met", .basics, max: 1, [
            O("dating-app", "Dating app", "you met on a dating app"), O("coworkers", "Coworkers", "you are coworkers"),
            O("neighbors", "Neighbors", "you are neighbors"), O("old-friends", "Old friends", "you are old friends"),
            O("at-a-bar", "At a bar", "you met at a bar"), O("at-the-gym", "At the gym", "you met at the gym"),
            O("blind-date", "Blind date", "you met on a blind date"),
            O("online-gaming", "Online gaming", "you met while gaming online"),
        ]),
        Category("hairColor", "Hair color", .look, max: 1, [
            O("black", "Black", "black hair"), O("dark-brown", "Dark brown", "dark brown hair"),
            O("brown", "Brown", "brown hair"), O("auburn", "Auburn", "auburn hair"), O("red", "Red", "red hair"),
            O("blonde", "Blonde", "blonde hair"), O("platinum", "Platinum", "platinum blonde hair"),
            O("gray", "Gray", "silver-gray hair"), O("pink", "Pink", "pink hair"), O("blue", "Blue", "blue hair"),
        ]),
        Category("hairStyle", "Hair style", .look, max: 1, [
            O("short", "Short", "short hair"), O("long-wavy", "Long wavy", "long wavy hair"),
            O("shoulder-straight", "Shoulder-length", "shoulder-length straight hair"),
            O("curly", "Curly", "curly hair"), O("bob", "Bob", "a sleek bob"), O("buzz-cut", "Buzz cut", "a buzz cut"),
            O("braids", "Braids", "braids"), O("ponytail", "Ponytail", "a ponytail"),
            O("tousled", "Tousled", "tousled hair"),
        ]),
        Category("eyes", "Eyes", .look, max: 1, [
            O("brown", "Brown", "brown eyes"), O("hazel", "Hazel", "hazel eyes"), O("green", "Green", "green eyes"),
            O("blue", "Blue", "blue eyes"), O("gray", "Gray", "gray eyes"), O("amber", "Amber", "amber eyes"),
            O("dark", "Dark", "dark eyes"),
        ]),
        Category("skin", "Skin", .look, max: 1, [
            O("fair", "Fair", "fair skin"), O("light", "Light", "light skin"), O("olive", "Olive", "olive skin"),
            O("tan", "Tan", "tan skin"), O("brown", "Brown", "brown skin"), O("dark", "Dark", "dark skin"),
        ]),
        Category("body", "Body", .look, max: 1, [
            O("slim", "Slim", "a slim build"), O("athletic", "Athletic", "an athletic build"),
            O("average", "Average", "an average build"), O("curvy", "Curvy", "a curvy figure"),
            O("muscular", "Muscular", "a muscular build"), O("plus-size", "Plus-size", "a plus-size figure"),
            O("broad-shouldered", "Broad-shouldered", "broad shoulders"),
        ]),
        Category("height", "Height", .look, max: 1, [
            O("short", "Short", "short height"), O("average", "Average", "average height"), O("tall", "Tall", "tall"),
        ]),
        Category("clothes", "Style", .look, max: 2, [
            O("casual", "Casual", "dresses in casual clothes"), O("elegant", "Elegant", "dresses elegantly"),
            O("sporty", "Sporty", "dresses in sporty clothes"), O("streetwear", "Streetwear", "wears streetwear"),
            O("boho", "Boho", "dresses in boho style"), O("business", "Business", "dresses in business attire"),
            O("punk", "Punk / edgy", "dresses in an edgy, punk style"), O("cozy", "Cozy knits", "wears cozy knits"),
        ]),
        Category("extras", "Extras", .look, max: 4, [
            O("freckles", "Freckles", "freckles"), O("glasses", "Glasses", "glasses"),
            O("tattoos", "Tattoos", "tattoos"), O("piercings", "Piercings", "piercings"),
            O("dimples", "Dimples", "dimples"), O("beard", "Beard", "a beard", onlyFor: "man"),
            O("stubble", "Stubble", "stubble", onlyFor: "man"),
            O("bold-makeup", "Bold makeup", "bold makeup", onlyFor: "woman"),
            O("scar", "Scar", "a small scar"), O("warm-smile", "Warm smile", "a warm smile"),
        ]),
        Category("traits", "Traits", .personality, max: 4, [
            O("playful", "Playful", "playful"), O("shy", "Shy", "shy"), O("confident", "Confident", "confident"),
            O("caring", "Caring", "caring"), O("sarcastic", "Sarcastic", "sarcastic"),
            O("adventurous", "Adventurous", "adventurous"), O("romantic", "Romantic", "romantic"),
            O("mysterious", "Mysterious", "mysterious"), O("bubbly", "Bubbly", "bubbly"), O("calm", "Calm", "calm"),
            O("ambitious", "Ambitious", "ambitious"), O("nerdy", "Nerdy", "nerdy"),
        ]),
        Category("flirting", "Flirting", .personality, max: 1, [
            O("subtle", "Subtle", "flirts subtly"), O("sweet", "Sweet", "flirts sweetly"),
            O("bold", "Bold", "flirts boldly"), O("teasing", "Teasing", "flirts by teasing"),
            O("shy", "Shy", "flirts shyly"),
        ]),
        Category("texting", "Texting style", .personality, max: 2, [
            O("short-sweet", "Short and sweet", "writes short, sweet messages"),
            O("emojis", "Lots of emojis", "uses lots of emojis"), O("long", "Long messages", "writes long messages"),
            O("witty", "Witty", "writes with wit"), O("poetic", "Poetic", "writes poetically"),
            O("pet-names", "Uses pet names", "uses pet names"),
        ]),
        Category("rhythm", "Daily rhythm", .habits, max: 1, [
            O("early-riser", "Early riser", "early riser"), O("night-owl", "Night owl", "night owl"),
        ]),
        Category("habits", "Habits", .habits, max: 4, [
            O("coffee-first", "Coffee first thing", "has coffee first thing in the morning"),
            O("gym-regular", "Gym regular", "goes to the gym regularly"),
            O("morning-runs", "Morning runs", "goes for morning runs"), O("yoga", "Yoga", "does yoga"),
            O("cooks-nightly", "Cooks every night", "cooks every night"),
            O("reads-before-bed", "Reads before bed", "reads before bed"), O("gamer", "Gamer", "plays video games"),
            O("plays-instrument", "Plays an instrument", "plays an instrument"),
            O("good-morning-texts", "Good-morning texts", "sends good-morning texts"),
            O("double-texts", "Double-texts", "double-texts"), O("voice-notes", "Voice notes", "sends voice notes"),
            O("journals", "Journals", "keeps a journal"),
        ]),
        Category("hobbies", "Interests", .habits, max: 5, [
            O("music", "Music", "music"), O("movies", "Movies", "movies"), O("travel", "Travel", "travel"),
            O("hiking", "Hiking", "hiking"), O("art", "Art", "art"), O("books", "Books", "books"),
            O("cooking", "Cooking", "cooking"), O("fashion", "Fashion", "fashion"), O("gaming", "Gaming", "gaming"),
            O("dancing", "Dancing", "dancing"), O("photography", "Photography", "photography"),
            O("tech", "Tech", "tech"),
        ]),
        Category("voiceTone", "Tone", .voice, max: 2, [
            O("warm", "Warm", "warm"), O("soft", "Soft", "soft"), O("husky", "Husky", "husky"),
            O("bright", "Bright", "bright"), O("smoky", "Smoky", "smoky"), O("crisp", "Crisp", "crisp"),
            O("sultry", "Sultry", "sultry"), O("gentle", "Gentle", "gentle"), O("playful", "Playful", "playful"),
        ]),
        Category("voicePitch", "Pitch", .voice, max: 1, [
            O("low", "Low", "low-pitched"), O("medium", "Medium", "medium pitch"), O("high", "High", "high-pitched"),
        ]),
        Category("voicePace", "Pace", .voice, max: 1, [
            O("slow", "Slow", "speaks slowly"), O("relaxed", "Relaxed", "relaxed pace"),
            O("quick", "Quick", "speaks quickly"),
        ]),
        Category("accent", "Accent", .voice, max: 1, [
            O("neutral-american", "Neutral American", "a neutral American accent"),
            O("british", "British", "a British accent"), O("australian", "Australian", "an Australian accent"),
            O("irish", "Irish", "an Irish accent"), O("scottish", "Scottish", "a Scottish accent"),
            O("french", "French", "a French accent"), O("italian", "Italian", "an Italian accent"),
            O("spanish", "Spanish", "a Spanish accent"), O("german", "German", "a German accent"),
            O("russian", "Russian", "a Russian accent"),
        ]),
    ]

    public static func categories(_ group: Group) -> [Category] { all.filter { $0.group == group } }

    public static func category(_ id: String) -> Category? { all.first { $0.id == id } }

    /// The category's options shown for this gender text. "man" and "woman" (lowercased, trimmed) hide options meant
    /// for the other one; any other gender text shows all options.
    public static func options(_ category: Category, gender: String) -> [Option] {
        let g = gender.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard g == "man" || g == "woman" else { return category.options }
        return category.options.filter { $0.onlyFor == nil || $0.onlyFor == g }
    }

    private static func phrases(_ traits: [String: [String]], in cats: [Category]) -> [String] {
        cats.flatMap { cat -> [String] in
            let picked = Set(traits[cat.id] ?? [])
            return cat.options.filter { picked.contains($0.id) }.map(\.phrase)
        }
    }

    /// Picked phrases of a group in catalog order (not pick order). Unknown ids are ignored.
    public static func phrases(_ traits: [String: [String]], group: Group) -> [String] {
        phrases(traits, in: categories(group))
    }

    /// Same, for a list of category ids (in the order given).
    public static func phrases(_ traits: [String: [String]], categories ids: [String]) -> [String] {
        phrases(traits, in: ids.compactMap { category($0) })
    }

    /// Tapping a chip: a picked option is removed; a single-pick category replaces; a full one drops its oldest pick.
    public static func toggle(_ traits: [String: [String]], category id: String, option: String) -> [String: [String]] {
        guard let cat = category(id) else { return traits }
        var out = traits
        var list = out[id] ?? []
        if let i = list.firstIndex(of: option) {
            list.remove(at: i)
        } else if cat.max == 1 {
            list = [option]
        } else {
            if list.count >= cat.max { list.removeFirst(list.count - cat.max + 1) }
            list.append(option)
        }
        out[id] = list.isEmpty ? nil : list
        return out
    }

    fileprivate static func join(_ phrases: [String], _ free: String) -> String {
        var text = free.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = text.last, ".;,".contains(last) { text.removeLast() }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return (phrases + [text]).filter { !$0.isEmpty }.joined(separator: ", ")
    }
}

extension Character {
    public var fullLook: String { CharacterOptions.join(CharacterOptions.phrases(traits, group: .look), look) }
    public var fullPersonality: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["traits", "flirting"]), personality)
    }
    public var fullSpeakingStyle: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["texting"]), speakingStyle)
    }
    public var habits: String { CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["rhythm", "habits"]), "") }
    public var fullLikes: String { CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["hobbies"]), likes) }
    public var fullBackstory: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["occupation", "met"]), backstory)
    }
    public var fullVoice: String { CharacterOptions.join(CharacterOptions.phrases(traits, group: .voice), voice) }
}
