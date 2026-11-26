import Foundation

/// How each option is pictured in the creator (photo, color swatch, symbol or nothing), a one-line hint for every
/// option, and the prompts used to make the example photos.
extension CharacterOptions {
    public enum Visual: Equatable, Sendable { case photo, swatch(hex: String), symbol(String), none }

    public static let photoCategories: Set<String> = ["hairStyle", "body", "bust", "butt", "figure", "clothes", "extras"]
    public static let swatchCategories: Set<String> = ["hairColor", "eyes", "skin"]
    public static let symbolCategories: Set<String> = ["occupation", "met", "rhythm", "habits", "hobbies"]

    /// .photo for photo categories; .swatch / .symbol from the tables below; .none otherwise (and for unknown ids).
    public static func visual(category: String, option: String) -> Visual {
        guard let cat = Self.category(category), cat.options.contains(where: { $0.id == option }) else { return .none }
        if photoCategories.contains(category) { return .photo }
        let key = "\(category)/\(option)"
        if swatchCategories.contains(category), let hex = swatchHex[key] { return .swatch(hex: hex) }
        if symbolCategories.contains(category), let name = symbolNames[key] { return .symbol(name) }
        return .none
    }

    /// One-line explanation of the option, without its label: sentence case, no trailing period, at most 60 characters.
    /// "" for an unknown id.
    public static func hint(category: String, option: String) -> String { hints["\(category)/\(option)"] ?? "" }

    /// The example photos to make: every option of every photo category, for each of its exampleGenders, in catalog order.
    public static var examples: [(category: String, option: String, gender: String)] {
        all.filter { photoCategories.contains($0.id) }.flatMap { cat in
            cat.options.flatMap { opt in
                exampleGenders(category: cat.id, option: opt.id).map { (category: cat.id, option: opt.id, gender: $0) }
            }
        }
    }

    /// ["woman", "man"], or just [onlyFor] when the option has one.
    public static func exampleGenders(category: String, option: String) -> [String] {
        guard let opt = Self.category(category)?.options.first(where: { $0.id == option }) else { return [] }
        if let only = opt.onlyFor { return [only] }
        return ["woman", "man"]
    }

    /// "<category>-<option>-<gender>.jpg", e.g. "hairStyle-long-wavy-woman.jpg".
    public static func exampleFile(category: String, option: String, gender: String) -> String {
        "\(category)-\(option)-\(gender).jpg"
    }

    /// Fixed per (category, gender): FNV-1a 32-bit hash of "category/gender", masked to 0x7FFF_FFFF.
    public static func exampleSeed(category: String, gender: String) -> Int {
        var hash: UInt32 = 2_166_136_261
        for byte in "\(category)/\(gender)".utf8 {
            hash ^= UInt32(byte)
            hash = hash &* 16_777_619
        }
        return Int(hash & 0x7FFF_FFFF)
    }

    /// The SDXL prompt for one example photo. "" for an unknown category or option. Options whose plain phrase barely
    /// changes the picture get a stronger description right after the subject (see `emphasis`); the phrase still follows.
    public static func examplePrompt(category: String, option: String, gender: String) -> String {
        guard photoCategories.contains(category),
              let phrase = Self.category(category)?.options.first(where: { $0.id == option })?.phrase else { return "" }
        // The models (the user's picks, scripts/option-models): a sexy 21-year-old woman with a platinum blonde bob, green
        // eyes and freckles; a handsome 30-year-old man with a dark brown quiff, green eyes and a short full beard. Both
        // look like attractive glamour models whatever body the option shows.
        let woman = gender == "woman"
        let hair = woman ? "platinum blonde hair in a sleek bob with bangs" : "dark brown hair in a textured quiff"
        var base = woman ? "fair skin, European features, a heart-shaped face, green eyes, light freckles, soft pink lips"
            : "a handsome face, green eyes, a short neat full beard"
        if woman && option != "bold-makeup" { base += ", natural light makeup" }
        base += woman ? ", sexy and alluring, confident seductive look" : ", sexy and attractive, confident smoldering look"
        let underwear = woman ? "wearing only a black lace lingerie set, bra and panties"
            : "shirtless, wearing only fitted black boxer briefs"
        let top = woman ? "fitted low-cut black top" : "fitted black t-shirt"
        var who = "Glamour photo of a \(exampleAge(gender: gender))-year-old adult \(gender)"
        if let e = emphasis(category: category, option: option, gender: gender) { who += " with \(e.text)" }
        who += ", \(base)"
        let body: String
        switch category {
        case "hairStyle":
            body = "\(who), \(woman ? "platinum blonde" : "dark brown") hair, \(phrase), head-and-shoulders portrait, \(top)"
        case "extras":
            let framing = switch option {
            case "tattoos": "waist-up portrait, arms visible, \(woman ? "black tank top" : "shirtless")"
            case "piercings": "close-up of the face and ears, \(top)"
            default: "close-up portrait of the face, \(top)"
            }
            body = "\(who), \(hair), \(phrase), \(framing)"
        case "body", "bust", "figure":
            body = "\(who), \(hair), \(phrase), full body, standing in a confident alluring pose, facing the camera, \(underwear)"
        case "butt":
            body = "\(who), \(hair), \(phrase), \(underwear), full body seen from behind, looking over the shoulder"
        case "clothes":
            body = "\(who), \(hair), \(phrase), full body, standing"
        default:
            return ""
        }
        return body + ", plain light gray studio background, flattering soft light."
    }

    /// The model's age in the example photos: 21 for the woman, 30 for the man.
    public static func exampleAge(gender: String) -> Int { gender == "woman" ? 21 : 30 }

    /// Extra negative-prompt words for an example: never nude for the body types (they're shown in underwear), plus
    /// what pushes away from the neighbouring options.
    public static func exampleNegative(category: String, option: String, gender: String) -> String {
        var words: [String] = []
        if ["body", "bust", "butt", "figure"].contains(category) {
            words.append("nude, naked, nipples, topless, explicit, shirt, t-shirt, shorts, sneakers, "
                         + "outdoors, street, building, cropped")
        }
        words.append("black and white, monochrome")
        if let e = emphasis(category: category, option: option, gender: gender), !e.negative.isEmpty {
            words.append(e.negative)
        }
        return words.joined(separator: ", ")
    }

    /// A stronger description (and what to keep out) for options whose phrase alone hardly shows; a gender-specific
    /// entry ("category/option/man") wins over the shared one.
    private static func emphasis(category: String, option: String, gender: String) -> (text: String, negative: String)? {
        emphases["\(category)/\(option)/\(gender)"] ?? emphases["\(category)/\(option)"]
    }

    private static let emphases: [String: (text: String, negative: String)] = [
        "hairStyle/short/woman": ("a short pixie cut", "long hair, shoulder-length hair"),
        "hairStyle/bob/man": ("a chin-length bob haircut", "full body, short hair"),
        "body/slim": ("a very slim, thin, slender body and a narrow frame", "muscular, abs, curvy, fat"),
        "body/athletic": ("an athletic, toned, fit body", "fat, skinny"),
        "body/average": ("an ordinary everyday body, soft untoned stomach", "abs, six-pack, muscular, fitness model, skinny"),
        "body/curvy": ("a curvy body with a full bust and full rounded hips", "skinny, abs, muscular"),
        "body/curvy/man": ("a soft, stocky body with a little belly", "abs, six-pack, muscular, fitness model"),
        "body/muscular": ("a very muscular bodybuilder physique, big muscles", "skinny, fat"),
        "body/plus-size": ("a plus-size, heavy body and a big belly", "skinny, abs, muscular"),
        "body/broad-shouldered": ("very broad shoulders and a wide V-shaped upper body", "narrow shoulders"),
        "body/hourglass": ("an hourglass figure: large bust, wide hips and a very narrow waist", "skinny, flat chest, abs"),
        "bust/small": ("small breasts, a flat chest, A cup", "large breasts, big breasts, cleavage"),
        "bust/medium": ("medium breasts, B cup", "huge breasts, flat chest"),
        "bust/large": ("large breasts, D cup, cleavage", "small breasts, flat chest"),
        "bust/very-large": ("huge breasts, an extremely large bust, G cup, deep cleavage", "small breasts, medium breasts, flat chest"),
        "butt/small": ("a small flat butt and narrow hips", "big butt, wide hips, round butt"),
        "butt/perky": ("a perky, lifted, firm butt", "flat butt, saggy butt"),
        "butt/round": ("a round, full bubble butt", "flat butt"),
        "butt/big": ("a very big, wide, thick butt and wide hips", "small butt, flat butt, skinny"),
        "figure/narrow-waist": ("a tiny, very narrow waist", "thick waist, belly"),
        "figure/wide-hips": ("very wide hips", "narrow hips"),
        "figure/long-legs": ("very long legs", "short legs, bodybuilder"),
        "figure/thick-thighs": ("very thick, full thighs", "skinny legs, thin thighs, bodybuilder"),
        "figure/toned-abs": ("visible six-pack abs", "belly, soft stomach"),
        "figure/toned-arms": ("toned, muscular arms", "thin arms"),
        "figure/broad-chest": ("a very broad, big, muscular chest", "narrow chest, skinny"),
        "figure/hairy-chest": ("a very hairy chest, thick chest hair", "hairless chest, smooth chest, bodybuilder"),
        "extras/freckles": ("lots of freckles across the nose and cheeks", ""),
        "extras/tattoos": ("sleeve tattoos covering both arms and a neck tattoo", ""),
        "extras/piercings": ("a big silver nose ring, a lip ring and several silver ear piercings", "no jewelry"),
        "extras/scar": ("a prominent scar across the face from the eyebrow down to the cheek", "smooth flawless skin"),
        "extras/bold-makeup": ("bold makeup: bright red lipstick and dark smoky eyes", "no makeup"),
    ]

    private static let swatchHex: [String: String] = [
        "hairColor/black": "#1B1B1D", "hairColor/dark-brown": "#3B2417", "hairColor/brown": "#6B4426",
        "hairColor/auburn": "#7B3A24", "hairColor/red": "#B2432A", "hairColor/blonde": "#D9B56A",
        "hairColor/platinum": "#EDE6D3", "hairColor/gray": "#B8BCC2", "hairColor/pink": "#F2A3C3",
        "hairColor/blue": "#6F9BE0",
        "eyes/brown": "#5A3825", "eyes/hazel": "#8A6A2F", "eyes/green": "#4F8A55", "eyes/blue": "#4A7FB8",
        "eyes/gray": "#8C98A3", "eyes/amber": "#B67A1E", "eyes/dark": "#231813",
        "skin/fair": "#F6DCCB", "skin/light": "#EBC3A3", "skin/olive": "#C9A074", "skin/tan": "#B07C52",
        "skin/brown": "#8A5A3A", "skin/dark": "#5B3A28",
    ]

    private static let symbolNames: [String: String] = [
        "occupation/barista": "cup.and.saucer", "occupation/artist": "paintpalette", "occupation/nurse": "cross.case",
        "occupation/architect": "building.columns", "occupation/musician": "music.note",
        "occupation/chef": "fork.knife", "occupation/photographer": "camera",
        "occupation/fitness-coach": "figure.strengthtraining.traditional",
        "occupation/software-developer": "laptopcomputer", "occupation/writer": "pencil",
        "occupation/lawyer": "scalemass", "occupation/bartender": "wineglass",
        "met/dating-app": "iphone", "met/coworkers": "briefcase", "met/neighbors": "house",
        "met/old-friends": "person.2", "met/at-a-bar": "wineglass", "met/at-the-gym": "dumbbell",
        "met/blind-date": "eyes", "met/online-gaming": "gamecontroller",
        "rhythm/early-riser": "sunrise", "rhythm/night-owl": "moon.stars",
        "habits/coffee-first": "mug", "habits/gym-regular": "dumbbell", "habits/morning-runs": "figure.run",
        "habits/yoga": "figure.yoga", "habits/cooks-nightly": "frying.pan", "habits/reads-before-bed": "book",
        "habits/gamer": "gamecontroller", "habits/plays-instrument": "guitars", "habits/good-morning-texts": "message",
        "habits/double-texts": "bubble.left.and.bubble.right", "habits/voice-notes": "waveform",
        "habits/journals": "book.closed",
        "hobbies/music": "music.note", "hobbies/movies": "film", "hobbies/travel": "airplane",
        "hobbies/hiking": "mountain.2", "hobbies/art": "paintpalette", "hobbies/books": "book",
        "hobbies/cooking": "frying.pan", "hobbies/fashion": "tshirt", "hobbies/gaming": "gamecontroller",
        "hobbies/dancing": "figure.dance", "hobbies/photography": "camera", "hobbies/tech": "cpu",
    ]

    private static let hints: [String: String] = [
        "occupation/barista": "Makes coffee in a cafe",
        "occupation/artist": "Paints, draws or makes art",
        "occupation/nurse": "Cares for sick people in a hospital",
        "occupation/architect": "Designs buildings",
        "occupation/musician": "Plays or writes music for a living",
        "occupation/chef": "Cooks in a restaurant kitchen",
        "occupation/photographer": "Takes photos for work",
        "occupation/fitness-coach": "Trains people to get fit",
        "occupation/software-developer": "Writes computer programs",
        "occupation/writer": "Writes books, stories or articles",
        "occupation/lawyer": "Works with laws and court cases",
        "occupation/bartender": "Mixes and serves drinks at a bar",
        "met/dating-app": "You matched on a dating app",
        "met/coworkers": "You work at the same place",
        "met/neighbors": "You live next door to each other",
        "met/old-friends": "You have known each other for years",
        "met/at-a-bar": "You started talking at a bar",
        "met/at-the-gym": "You met while working out",
        "met/blind-date": "Friends set you up without a photo or details",
        "met/online-gaming": "You played a video game together online",
        "hairColor/black": "Very dark, black",
        "hairColor/dark-brown": "Deep brown, almost black",
        "hairColor/brown": "Medium brown",
        "hairColor/auburn": "Reddish-brown",
        "hairColor/red": "Bright red or copper",
        "hairColor/blonde": "Light golden yellow",
        "hairColor/platinum": "Very pale, almost white blonde",
        "hairColor/gray": "Silver or light gray",
        "hairColor/pink": "Dyed light pink",
        "hairColor/blue": "Dyed blue",
        "hairStyle/short": "Cut close to the head",
        "hairStyle/long-wavy": "Long hair with soft waves",
        "hairStyle/shoulder-straight": "Straight hair that reaches the shoulders",
        "hairStyle/curly": "Hair with tight, bouncy curls",
        "hairStyle/bob": "Smooth, chin-length cut",
        "hairStyle/buzz-cut": "Hair shaved very short",
        "hairStyle/braids": "Hair woven into plaits",
        "hairStyle/ponytail": "Hair tied back in one tail",
        "hairStyle/tousled": "Messy, just-got-up look",
        "eyes/brown": "Classic brown",
        "eyes/hazel": "Green-brown mix",
        "eyes/green": "Clear green",
        "eyes/blue": "Light blue",
        "eyes/gray": "Cool silver-gray",
        "eyes/amber": "Golden, honey color",
        "eyes/dark": "So dark they look almost black",
        "skin/fair": "Very light, burns easily",
        "skin/light": "Light, with a pink or neutral tone",
        "skin/olive": "Light brown with a slight green-gold tone",
        "skin/tan": "Warm golden brown, like after sun",
        "skin/brown": "Medium to deep brown",
        "skin/dark": "Deep, rich dark brown",
        "body/slim": "Thin and light",
        "body/athletic": "Fit, with firm muscles",
        "body/average": "Neither thin nor heavy",
        "body/curvy": "Full chest and hips, smaller waist",
        "body/muscular": "Large, strong muscles",
        "body/plus-size": "Fuller, softer body",
        "body/broad-shouldered": "Wide shoulders, strong upper body",
        "body/hourglass": "Chest and hips about equal, narrow waist",
        "height/short": "Shorter than most people",
        "height/average": "About the same as most people",
        "height/tall": "Taller than most people",
        "clothes/casual": "Relaxed everyday clothes",
        "clothes/elegant": "Refined, dressy and polished",
        "clothes/sporty": "Athletic clothes and sneakers",
        "clothes/streetwear": "Urban style: hoodies, sneakers, caps",
        "clothes/boho": "Loose, earthy, flowy layers",
        "clothes/business": "Suits, blazers and smart shirts",
        "clothes/punk": "Dark, bold look with leather and chains",
        "clothes/cozy": "Soft, warm knitted sweaters",
        "extras/freckles": "Small brown dots on the skin",
        "extras/glasses": "Wears eyeglasses",
        "extras/tattoos": "Pictures inked on the skin",
        "extras/piercings": "Small rings or studs in ears or face",
        "extras/dimples": "Little dents in the cheeks when smiling",
        "extras/beard": "Full hair on the chin and cheeks",
        "extras/stubble": "Short, rough hair that just grew back",
        "extras/bold-makeup": "Strong colors, like red lips or dark eyes",
        "extras/scar": "A small mark on the face from an old cut",
        "extras/warm-smile": "A kind, friendly smile",
        "bust/small": "Small and light",
        "bust/medium": "Medium size",
        "bust/large": "Full and large",
        "bust/very-large": "Very full and large",
        "butt/small": "Small and flat",
        "butt/perky": "High and firm",
        "butt/round": "Full and rounded",
        "butt/big": "Large and full",
        "figure/narrow-waist": "Waist is much smaller than hips",
        "figure/wide-hips": "Hips wider than the waist",
        "figure/long-legs": "Legs look long for the body",
        "figure/thick-thighs": "Full, strong upper legs",
        "figure/toned-abs": "Flat stomach with visible muscle lines",
        "figure/toned-arms": "Firm arms with some muscle",
        "figure/broad-chest": "Wide, strong chest",
        "figure/hairy-chest": "Hair on the chest",
        "traits/playful": "Loves jokes and having fun",
        "traits/shy": "Quiet, a bit nervous with new people",
        "traits/confident": "Sure of themselves",
        "traits/caring": "Kind and looks after others",
        "traits/sarcastic": "Says the opposite to be funny",
        "traits/adventurous": "Loves trying new, exciting things",
        "traits/romantic": "Loves sweet gestures and deep feelings",
        "traits/mysterious": "Does not tell everything about themselves",
        "traits/bubbly": "Cheerful, full of energy",
        "traits/calm": "Relaxed, stays peaceful under stress",
        "traits/ambitious": "Works hard to reach big goals",
        "traits/nerdy": "Passionate about science, tech or geeky topics",
        "flirting/subtle": "Hints and small signals",
        "flirting/sweet": "Kind words and gentle compliments",
        "flirting/bold": "Direct, says what they want",
        "flirting/teasing": "Playful jokes to get your attention",
        "flirting/shy": "Nervous glances and blushing",
        "texting/short-sweet": "Brief, kind messages",
        "texting/emojis": "Adds many emoji to messages",
        "texting/long": "Writes long, detailed messages",
        "texting/witty": "Clever, funny replies",
        "texting/poetic": "Uses beautiful, artistic words",
        "texting/pet-names": "Calls you names like babe or honey",
        "rhythm/early-riser": "Wakes up early in the morning",
        "rhythm/night-owl": "Stays up late at night",
        "habits/coffee-first": "Needs coffee before anything else",
        "habits/gym-regular": "Works out at the gym often",
        "habits/morning-runs": "Runs outside early in the day",
        "habits/yoga": "Stretches and relaxes with yoga",
        "habits/cooks-nightly": "Makes dinner at home each evening",
        "habits/reads-before-bed": "Reads a book before sleeping",
        "habits/gamer": "Often plays video games",
        "habits/plays-instrument": "Plays guitar, piano or similar",
        "habits/good-morning-texts": "Writes to you first every morning",
        "habits/double-texts": "Sends a second message before you reply",
        "habits/voice-notes": "Sends short audio messages",
        "habits/journals": "Writes down thoughts in a notebook",
        "hobbies/music": "Listening to or making music",
        "hobbies/movies": "Watching films",
        "hobbies/travel": "Visiting new places",
        "hobbies/hiking": "Long walks in nature",
        "hobbies/art": "Painting, drawing and galleries",
        "hobbies/books": "Reading stories and novels",
        "hobbies/cooking": "Making tasty food",
        "hobbies/fashion": "Clothes and personal style",
        "hobbies/gaming": "Playing video games",
        "hobbies/dancing": "Moving to music",
        "hobbies/photography": "Taking pictures",
        "hobbies/tech": "Gadgets, apps and computers",
        "voiceTone/warm": "Friendly and comforting",
        "voiceTone/soft": "Quiet and gentle",
        "voiceTone/husky": "Low and slightly rough",
        "voiceTone/bright": "Light, clear and cheerful",
        "voiceTone/smoky": "Deep and breathy, a bit raspy",
        "voiceTone/crisp": "Sharp and very clear",
        "voiceTone/sultry": "Low, slow and charming",
        "voiceTone/gentle": "Calm and kind",
        "voiceTone/playful": "Teasing, with a smile in the voice",
        "voicePitch/low": "Deep voice",
        "voicePitch/medium": "Middle voice, not deep or high",
        "voicePitch/high": "Light, higher voice",
        "voicePace/slow": "Takes time between words",
        "voicePace/relaxed": "Easy, comfortable speed",
        "voicePace/quick": "Talks fast",
        "accent/neutral-american": "Standard American English",
        "accent/british": "English as spoken in England",
        "accent/australian": "English as spoken in Australia",
        "accent/irish": "English as spoken in Ireland",
        "accent/scottish": "English as spoken in Scotland",
        "accent/french": "English with a French sound",
        "accent/italian": "English with an Italian sound",
        "accent/spanish": "English with a Spanish sound",
        "accent/german": "English with a German sound",
        "accent/russian": "English with a Russian sound",
    ]
}
