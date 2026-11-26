import Foundation

/// Predefined options for creating a character: the user taps chips instead of typing.
/// Picks are stored on `Character.traits` (category id → option ids) and turned into prompt text here.
public enum CharacterOptions {
    /// `intimate`: 18+ body details, shown only with the adult options on and written only into pictures that show
    /// that part (`lookParts`); never in `fullLook`, voices or the chat model's looks.
    public enum Group: String, CaseIterable, Sendable { case basics, look, personality, habits, voice, intimate }

    /// How the look group is laid out in the creator, the editor and the Prompt Builder.
    public enum Section: String, CaseIterable, Sendable {
        case hair, face, body, style
        public var title: String {
            switch self {
            case .hair: "Hair"
            case .face: "Face"
            case .body: "Body"
            case .style: "Style"
            }
        }
    }

    /// The look section of a look category (nil for the other groups).
    public static func section(of category: String) -> Section? { sections[category] }

    private static let sections: [String: Section] = [
        "hairColor": .hair, "hairStyle": .hair,
        "eyes": .face, "skin": .face, "faceShape": .face, "eyeShape": .face, "eyebrows": .face, "nose": .face,
        "lips": .face, "jaw": .face, "heritage": .face, "makeup": .face, "facialHair": .face, "extras": .face,
        "body": .body, "height": .body, "muscle": .body, "shoulders": .body, "bust": .body, "bustShape": .body,
        "butt": .body, "figure": .body, "bodyHair": .body, "bodyMarks": .body,
        "clothes": .style,
    ]

    /// The detailed face and body categories: Surprise Me fills each only sometimes, so a random character doesn't
    /// get twenty details.
    public static let detailCategories: Set<String> = [
        "faceShape", "eyeShape", "eyebrows", "nose", "lips", "jaw", "makeup", "facialHair", "muscle", "shoulders",
        "bustShape", "bodyHair", "bodyMarks",
    ]

    /// The 18+ categories, by what a picture must show for them to be written.
    public static let chestCategories = ["nipples"]
    public static let intimateCategories = ["nipples", "pubicHair", "penisSize", "circumcision"]

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
        // Broad regions, each followed by its finer origins (up to 2 picks, for mixed heritage).
        Category("heritage", "Heritage", .look, max: 2, [
            O("east-asian", "East Asian", "of East Asian descent"),
            O("japanese", "Japanese", "of Japanese descent"), O("korean", "Korean", "of Korean descent"),
            O("chinese", "Chinese", "of Chinese descent"),
            O("southeast-asian", "Southeast Asian", "of Southeast Asian descent"),
            O("filipino", "Filipino", "of Filipino descent"), O("vietnamese", "Vietnamese", "of Vietnamese descent"),
            O("thai", "Thai", "of Thai descent"), O("indonesian", "Indonesian", "of Indonesian descent"),
            O("south-asian", "South Asian", "of South Asian descent"),
            O("indian", "Indian", "of Indian descent"), O("pakistani", "Pakistani", "of Pakistani descent"),
            O("black", "Black", "Black, of African descent"),
            O("west-african", "West African", "of West African descent"),
            O("east-african", "East African", "of East African descent"),
            O("caribbean", "Caribbean", "of Afro-Caribbean descent"),
            O("latin", "Latina / Latino", "of Latin American descent"),
            O("mexican", "Mexican", "of Mexican descent"), O("brazilian", "Brazilian", "of Brazilian descent"),
            O("colombian", "Colombian", "of Colombian descent"),
            O("middle-eastern", "Middle Eastern", "of Middle Eastern descent"),
            O("arab", "Arab", "of Arab descent"), O("persian", "Persian", "of Persian descent"),
            O("turkish", "Turkish", "of Turkish descent"),
            O("white", "White European", "of European descent"),
            O("nordic", "Nordic", "of Nordic descent"), O("slavic", "Slavic", "of Slavic descent"),
            O("mediterranean", "Mediterranean", "of Mediterranean descent"),
            O("british-irish", "British / Irish", "of British or Irish descent"),
            O("indigenous-american", "Indigenous American", "of Indigenous American descent"),
            O("polynesian", "Polynesian", "of Polynesian descent"),
            O("mixed", "Mixed", "of mixed heritage"),
        ]),
        Category("faceShape", "Face shape", .look, max: 1, [
            O("oval", "Oval", "an oval face"), O("round", "Round", "a round face"),
            O("heart", "Heart", "a heart-shaped face"), O("square", "Square", "a square face"),
            O("long", "Long", "a long face"), O("diamond", "Diamond", "a diamond-shaped face"),
        ]),
        Category("eyeShape", "Eye shape", .look, max: 1, [
            O("almond", "Almond", "almond-shaped eyes"), O("round", "Round", "round eyes"),
            O("hooded", "Hooded", "hooded eyes"), O("upturned", "Upturned", "upturned cat-like eyes"),
            O("downturned", "Downturned", "downturned eyes"), O("monolid", "Monolid", "monolid eyes"),
            O("deep-set", "Deep-set", "deep-set eyes"),
        ]),
        Category("eyebrows", "Eyebrows", .look, max: 1, [
            O("thick", "Thick", "thick full eyebrows"), O("thin", "Thin", "thin eyebrows"),
            O("arched", "Arched", "high arched eyebrows"), O("straight", "Straight", "straight eyebrows"),
            O("bushy", "Bushy", "bushy eyebrows", onlyFor: "man"),
        ]),
        Category("nose", "Nose", .look, max: 1, [
            O("straight", "Straight", "a straight nose"), O("narrow", "Small, narrow", "a small narrow nose"),
            O("roman", "Roman", "a Roman nose with a high bridge"), O("wide", "Wide", "a wide nose"),
            O("upturned", "Upturned", "a slightly upturned nose"), O("long", "Long", "a long nose"),
        ]),
        Category("lips", "Lips", .look, max: 1, [
            O("full", "Full", "full lips"), O("thin", "Thin", "thin lips"),
            O("cupids-bow", "Cupid's bow", "lips with a defined cupid's bow"), O("wide", "Wide", "a wide mouth"),
            O("pouty", "Pouty", "pouty lips"),
        ]),
        Category("jaw", "Cheeks and jaw", .look, max: 2, [
            O("high-cheekbones", "High cheekbones", "high cheekbones"), O("hollow-cheeks", "Hollow cheeks", "hollow cheeks"),
            O("soft-cheeks", "Soft cheeks", "soft rounded cheeks"), O("sharp-jaw", "Sharp jawline", "a sharp jawline"),
            O("square-jaw", "Square jaw", "a square jaw"), O("soft-jaw", "Soft jaw", "a soft jawline"),
            O("strong-chin", "Strong chin", "a strong chin"), O("cleft-chin", "Cleft chin", "a cleft chin"),
        ]),
        Category("makeup", "Makeup", .look, max: 2, [
            O("none", "No makeup", "no makeup, bare skin"), O("natural", "Natural", "natural light makeup"),
            O("soft-glam", "Soft glam", "soft glam makeup"), O("smoky", "Smoky eyes", "smoky eye makeup"),
            O("winged", "Winged liner", "winged eyeliner"), O("red-lips", "Red lipstick", "red lipstick"),
            O("glossy", "Glossy lips", "glossy lips"),
        ]),
        Category("facialHair", "Facial hair", .look, max: 1, [
            O("clean-shaven", "Clean-shaven", "clean-shaven", onlyFor: "man"),
            O("short-boxed", "Short boxed beard", "a short boxed beard", onlyFor: "man"),
            O("long-beard", "Long beard", "a long full beard", onlyFor: "man"),
            O("goatee", "Goatee", "a goatee", onlyFor: "man"), O("mustache", "Mustache", "a mustache", onlyFor: "man"),
        ]),
        Category("body", "Body", .look, max: 1, [
            O("slim", "Slim", "a slim build"), O("athletic", "Athletic", "an athletic build"),
            O("average", "Average", "an average build"), O("curvy", "Curvy", "a curvy figure"),
            O("muscular", "Muscular", "a muscular build"), O("plus-size", "Plus-size", "a plus-size figure"),
            O("broad-shouldered", "Broad-shouldered", "broad shoulders"),
            O("hourglass", "Hourglass", "an hourglass figure", onlyFor: "woman"),
            O("lean", "Lean", "a lean build"), O("toned", "Toned", "a toned build"), O("soft", "Soft", "a soft build"),
            O("chubby", "Chubby", "a chubby build"), O("stocky", "Stocky", "a stocky build", onlyFor: "man"),
            O("dad-bod", "Dad bod", "a dad bod", onlyFor: "man"),
            O("voluptuous", "Voluptuous", "a voluptuous figure", onlyFor: "woman"),
        ]),
        Category("height", "Height", .look, max: 1, [
            O("short", "Short", "short height"), O("average", "Average", "average height"), O("tall", "Tall", "tall"),
            O("very-tall", "Very tall", "very tall"),
        ]),
        Category("muscle", "Muscle tone", .look, max: 1, [
            O("soft", "Soft", "no muscle definition"), O("toned", "Toned", "lightly toned muscles"),
            O("defined", "Defined", "clearly defined muscles"), O("very-muscular", "Very muscular", "big, heavily built muscles"),
        ]),
        Category("shoulders", "Shoulders", .look, max: 1, [
            O("narrow", "Narrow", "narrow shoulders"), O("rounded", "Rounded", "soft rounded shoulders"),
            O("square", "Square", "square shoulders"), O("broad", "Broad", "very broad shoulders"),
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
            O("beauty-mark", "Beauty mark", "a beauty mark above the lip"), O("rosy-cheeks", "Rosy cheeks", "rosy cheeks"),
            O("gap-teeth", "Gap teeth", "a small gap between the front teeth"),
        ]),
        Category("bust", "Breasts", .look, max: 1, [
            O("small", "Small", "small breasts", onlyFor: "woman"),
            O("medium", "Medium", "medium breasts", onlyFor: "woman"),
            O("large", "Large", "large breasts", onlyFor: "woman"),
            O("very-large", "Very large", "very large breasts", onlyFor: "woman"),
        ]),
        Category("bustShape", "Breast shape", .look, max: 1, [
            O("round", "Round", "round breasts", onlyFor: "woman"),
            O("teardrop", "Teardrop", "teardrop-shaped breasts", onlyFor: "woman"),
            O("perky", "Perky", "perky breasts", onlyFor: "woman"),
            O("natural", "Soft, natural", "soft natural breasts", onlyFor: "woman"),
            O("wide-set", "Wide-set", "wide-set breasts", onlyFor: "woman"),
        ]),
        Category("butt", "Butt", .look, max: 1, [
            O("small", "Small", "a small butt"), O("perky", "Perky", "a perky butt"),
            O("round", "Round", "a round butt"), O("big", "Big", "a big butt"),
            O("heart", "Heart-shaped", "a heart-shaped butt"), O("athletic", "Athletic", "an athletic, muscular butt"),
        ]),
        Category("figure", "Figure details", .look, max: 4, [
            O("narrow-waist", "Narrow waist", "a narrow waist", onlyFor: "woman"),
            O("wide-hips", "Wide hips", "wide hips", onlyFor: "woman"),
            O("long-legs", "Long legs", "long legs"), O("thick-thighs", "Thick thighs", "thick thighs"),
            O("toned-abs", "Toned abs", "toned abs"), O("toned-arms", "Toned arms", "toned arms"),
            O("broad-chest", "Broad chest", "a broad chest", onlyFor: "man"),
            O("hairy-chest", "Hairy chest", "a hairy chest", onlyFor: "man"),
            O("flat-stomach", "Flat stomach", "a flat stomach"), O("soft-belly", "Soft belly", "a soft belly"),
            O("love-handles", "Love handles", "love handles"), O("slender-arms", "Slender arms", "slender arms"),
            O("muscular-legs", "Muscular legs", "muscular legs"), O("long-neck", "Long neck", "a long neck"),
            O("toned-back", "Toned back", "a toned back"),
        ]),
        Category("bodyHair", "Body hair", .look, max: 1, [
            O("smooth", "Smooth", "smooth hairless skin"), O("light", "Light", "light body hair"),
            O("natural", "Natural", "natural body hair"), O("hairy", "Hairy", "a hairy body", onlyFor: "man"),
        ]),
        Category("bodyMarks", "Body marks", .look, max: 3, [
            O("sleeve-tattoo", "Sleeve tattoo", "a full sleeve tattoo"), O("back-tattoo", "Back tattoo", "a large back tattoo"),
            O("small-tattoos", "Small tattoos", "a few small tattoos"),
            O("belly-piercing", "Belly piercing", "a belly button piercing"), O("tan-lines", "Tan lines", "tan lines"),
            O("freckled-shoulders", "Freckled shoulders", "freckled shoulders"),
            O("stretch-marks", "Stretch marks", "faint stretch marks"), O("birthmark", "Birthmark", "a birthmark"),
        ]),
        // 18+ (`Group.intimate`): only with the adult options on, and only in pictures that show that part.
        Category("nipples", "Nipples", .intimate, max: 2, [
            O("small", "Small", "small nipples"), O("puffy", "Puffy", "puffy nipples", onlyFor: "woman"),
            O("large-areolae", "Large areolae", "large areolae", onlyFor: "woman"),
            O("pink", "Light pink", "light pink nipples"), O("dark", "Dark", "dark nipples"),
            O("pierced", "Pierced", "pierced nipples"),
        ]),
        Category("pubicHair", "Pubic hair", .intimate, max: 1, [
            O("shaved", "Shaved", "a shaved pubic area"), O("trimmed", "Trimmed", "neatly trimmed pubic hair"),
            O("landing-strip", "Landing strip", "a landing strip of pubic hair", onlyFor: "woman"),
            O("natural", "Natural", "natural pubic hair"),
        ]),
        Category("penisSize", "Size", .intimate, max: 1, [
            O("average", "Average", "an average-size penis", onlyFor: "man"),
            O("large", "Large", "a large penis", onlyFor: "man"),
        ]),
        Category("circumcision", "Circumcision", .intimate, max: 1, [
            O("circumcised", "Circumcised", "a circumcised penis", onlyFor: "man"),
            O("uncut", "Uncut", "an uncircumcised penis", onlyFor: "man"),
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

    /// The look categories of one section, in catalog order.
    public static func categories(_ section: Section) -> [Category] { all.filter { sections[$0.id] == section } }

    public static func category(_ id: String) -> Category? { all.first { $0.id == id } }

    /// The category's options shown for this gender text. "man" and "woman" (lowercased, trimmed) hide options meant
    /// for the other one; any other gender text shows all options.
    public static func options(_ category: Category, gender: String) -> [Option] {
        let g = gender.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard g == "man" || g == "woman" else { return category.options }
        return category.options.filter { $0.onlyFor == nil || $0.onlyFor == g }
    }

    private static func phrases(_ traits: [String: [String]], in cats: [Category], gender: String?) -> [String] {
        cats.flatMap { cat -> [String] in
            let picked = Set(traits[cat.id] ?? [])
            let shown = gender.map { options(cat, gender: $0) } ?? cat.options
            return shown.filter { picked.contains($0.id) }.map(\.phrase)
        }
    }

    /// Picked phrases of a group in catalog order (not pick order). Unknown ids are ignored.
    public static func phrases(_ traits: [String: [String]], group: Group, gender: String? = nil) -> [String] {
        phrases(traits, in: categories(group), gender: gender)
    }

    /// Same, for a list of category ids (in the order given).
    public static func phrases(_ traits: [String: [String]], categories ids: [String], gender: String? = nil) -> [String] {
        phrases(traits, in: ids.compactMap { category($0) }, gender: gender)
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
    public var fullLook: String { CharacterOptions.join(CharacterOptions.phrases(traits, group: .look, gender: gender), look) }
    /// The look without the clothing style picks (a photo that asks for an outfit: the picks would fight it).
    public var lookWithoutClothes: String {
        let ids = CharacterOptions.categories(.look).map(\.id).filter { $0 != "clothes" }
        return CharacterOptions.join(CharacterOptions.phrases(traits, categories: ids, gender: gender), look)
    }
    public var fullPersonality: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["traits", "flirting"], gender: gender), personality)
    }
    public var fullSpeakingStyle: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["texting"], gender: gender), speakingStyle)
    }
    public var habits: String { CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["rhythm", "habits"], gender: gender), "") }
    public var fullLikes: String { CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["hobbies"], gender: gender), likes) }
    public var fullBackstory: String {
        CharacterOptions.join(CharacterOptions.phrases(traits, categories: ["occupation", "met"], gender: gender), backstory)
    }
    public var fullVoice: String { CharacterOptions.join(CharacterOptions.phrases(traits, group: .voice, gender: gender), voice) }
}
