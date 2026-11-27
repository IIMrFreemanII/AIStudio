import Foundation
import MovieCore

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
            cat.options.filter { !unpictured.contains("\(cat.id)/\($0.id)") }.flatMap { opt in
                exampleGenders(category: cat.id, option: opt.id).map { (category: cat.id, option: opt.id, gender: $0) }
            }
        }
    }

    /// Options of photo categories with no example photo drawn yet (shown as text chips; drawn when asked for).
    public static let unpictured: Set<String> = [
        "body/lean", "body/toned", "body/soft", "body/chubby", "body/stocky", "body/dad-bod", "body/voluptuous",
        "butt/heart", "butt/athletic", "figure/flat-stomach", "figure/soft-belly", "figure/love-handles",
        "figure/slender-arms", "figure/muscular-legs", "figure/long-neck", "figure/toned-back",
        "extras/beauty-mark", "extras/rosy-cheeks", "extras/gap-teeth",
    ]

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

    /// The example photo of one option as a model-neutral description (nil for an unknown category or option). Options
    /// whose plain phrase barely changes the picture get a stronger description (see `emphasis`); the phrase stays too.
    public static func exampleSpec(category: String, option: String, gender: String) -> PromptSpec? {
        guard photoCategories.contains(category),
              let phrase = Self.category(category)?.options.first(where: { $0.id == option })?.phrase else { return nil }
        // The models (the user's picks, scripts/option-models): a sexy 21-year-old woman with a platinum blonde bob, green
        // eyes and freckles; a handsome 30-year-old man with a dark brown quiff, green eyes and a short full beard. Both
        // look like attractive models whatever body the option shows.
        let woman = gender == "woman"
        var s = PromptSpec()
        s.medium = "studio photo"
        s.subject = "a \(exampleAge(gender: gender))-year-old adult \(gender)"
        s.pronoun = woman ? .she : .he
        s.place = "in front of a plain light gray studio background"
        s.lighting = ["flattering soft light"]
        var face = woman ? ["fair skin", "European features", "a heart-shaped face", "green eyes", "light freckles", "soft pink lips"]
            : ["a handsome face", "green eyes", "a short neat full beard"]
        if woman && option != "bold-makeup" { face.append("natural light makeup") }
        face.append(woman ? "a sexy, alluring, confident seductive look" : "a sexy, attractive, confident smoldering look")
        let hair = woman ? "platinum blonde hair in a sleek bob with bangs" : "dark brown hair in a textured quiff"
        let underwear = woman ? "only a black lace lingerie set, bra and panties" : "only fitted black boxer briefs, shirtless"
        let top = woman ? "a fitted low-cut black top" : "a fitted black t-shirt"
        let e = emphasis(category: category, option: option, gender: gender)
        // The option itself: its stronger description and its phrase, emphasized where the model takes weights.
        let picked = PromptDetail(tag: phrase, text: e.map { "\($0.text), \(phrase)" } ?? phrase, weight: 1.3)
        s.details = face.map { PromptDetail($0) }
        switch category {
        case "hairStyle":
            s.shot = "head-and-shoulders portrait"
            s.details.append(PromptDetail(woman ? "platinum blonde hair" : "dark brown hair"))
            s.details.append(picked)
            s.outfit = [PromptDetail(top)]
        case "extras":
            s.details.append(PromptDetail(hair))
            s.details.append(picked)
            switch option {
            case "tattoos":
                s.shot = "waist-up portrait, arms visible"
                s.outfit = [PromptDetail(woman ? "a black tank top" : "no shirt")]
            case "piercings":
                s.shot = "close-up of the face and ears"
                s.outfit = [PromptDetail(top)]
            default:
                s.shot = "close-up portrait of the face"
                s.outfit = [PromptDetail(top)]
            }
        case "body", "bust", "figure":
            s.shot = "full body"
            s.action = "standing in a confident alluring pose, facing the camera"
            s.details.append(PromptDetail(hair))
            s.body = [picked]
            s.outfit = [PromptDetail(underwear)]
        case "butt":
            s.shot = "full body seen from behind"
            s.action = "looking over the shoulder"
            s.details.append(PromptDetail(hair))
            s.body = [picked]
            s.outfit = [PromptDetail(underwear)]
        case "clothes":
            s.shot = "full body"
            s.action = "standing"
            s.details.append(PromptDetail(hair))
            s.outfit = [picked]
        default:
            return nil
        }
        s.avoid = exampleNegative(category: category, option: option, gender: gender)
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        if let w = without(category: category, option: option, gender: gender) { s.without = [w] }
        return s
    }

    /// The example photo's prompt for one model ("" for an unknown category or option). The bundled set is drawn by
    /// MiniMax H3 Image.
    public static func examplePrompt(category: String, option: String, gender: String,
                                     model: PromptModel = .minimaxImage) -> String {
        exampleSpec(category: category, option: option, gender: gender).map { PromptWriter.write($0, for: model).prompt } ?? ""
    }

    /// The picked look options as picture details (docs/prompting.md): looks, body and outfit. Each uses the chip's
    /// phrase as the short tag and the stronger visual description (if any) as the text. Also what to keep out of the
    /// body, both as negative words and in words for models without a negative. Clothing picks are left out when
    /// `withClothes` is false (the photo names its own outfit). With two clothing picks each is named by its tag,
    /// because two full outfits would fight.
    public struct LookParts: Equatable, Sendable {
        public var details: [PromptDetail] = []
        public var body: [PromptDetail] = []
        public var outfit: [PromptDetail] = []
        public var avoid: [String] = []
        public var without: [String] = []
    }

    public static let identityCategories = ["hairColor", "hairStyle", "eyes", "skin", "extras", "heritage", "faceShape",
                                            "eyeShape", "eyebrows", "nose", "lips", "jaw", "makeup", "facialHair"]
    public static let bodyCategories = ["body", "height", "muscle", "shoulders", "bust", "bustShape", "butt", "figure",
                                        "bodyHair", "bodyMarks"]

    /// `coverage` and `adult`: with the 18+ options on, a picture that shows the chest (topless, nude) adds the
    /// nipple picks, and a nude one every intimate pick; they go with the body.
    /// `booru`: the model reads Danbooru tags (`PromptModel.readsBooru`), so body and clothing picks use those tags.
    public static func lookParts(_ traits: [String: [String]], gender: String, withClothes: Bool = true,
                                 coverage: PromptSpec.Coverage = .dressed, adult: Bool = false,
                                 booru: Bool = false) -> LookParts {
        var parts = LookParts()
        func picked(_ id: String) -> [Option] {
            guard let cat = category(id) else { return [] }
            let ids = Set(traits[id] ?? [])
            return options(cat, gender: gender).filter { ids.contains($0.id) }
        }
        let g = gender.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        for id in identityCategories {
            parts.details += picked(id).map { PromptDetail($0.phrase) }
        }
        var avoid: [(term: String, from: Int)] = []  // `from`: the body detail it came from
        for id in bodyCategories {
            for o in picked(id) {
                let e = emphasis(category: id, option: o.id, gender: g)
                let tag = booru ? booruBodyTags["\(id)/\(o.id)"] ?? o.phrase : o.phrase
                parts.body.append(PromptDetail(tag: tag, text: e?.text ?? o.phrase, weight: 1.2))
                if let e, !e.negative.isEmpty {
                    avoid += e.negative.split(separator: ",").map { ($0.trimmingCharacters(in: .whitespaces), parts.body.count - 1) }
                }
                if let w = without(category: id, option: o.id, gender: g) { parts.without.append(w) }
            }
        }
        if adult {
            let shown = coverage.showsAll ? intimateCategories : coverage.showsChest ? chestCategories : []
            for id in shown { parts.body += picked(id).map { PromptDetail(tag: $0.phrase, text: $0.phrase, weight: 1.1) } }
        }
        if withClothes && coverage == .dressed {
            let clothes = picked("clothes")
            for o in clothes {
                let tag = (booru ? booruClothesTags[o.id] : nil) ?? clothesTags[o.id] ?? o.phrase
                let text = clothes.count == 1 ? (emphasis(category: "clothes", option: o.id, gender: g)?.text ?? tag) : tag
                parts.outfit.append(PromptDetail(tag: tag, text: text, weight: 1.2))
            }
        }
        // A word to avoid that another pick asks for (toned abs with an average body) is dropped. A pick's own text
        // may name it ("no visible abs"), so only the other picks count.
        let others = parts.details.map { ($0.tag + " " + $0.text).lowercased() }
            + parts.outfit.map { ($0.tag + " " + $0.text).lowercased() }
        var seen = Set<String>()
        parts.avoid = avoid.filter { a in
            let t = a.term.lowercased()
            let asked = others + parts.body.enumerated().filter { $0.offset != a.from }
                .map { ($0.element.tag + " " + $0.element.text).lowercased() }
            return !t.isEmpty && !asked.contains { $0.contains(t) } && seen.insert(t).inserted
        }.map(\.term)
        return parts
    }

    /// The clothing styles' chip phrases are about the person ("dresses in boho style"). These are the short tags the
    /// picture models read.
    private static let clothesTags: [String: String] = [
        "casual": "casual clothes", "elegant": "elegant outfit", "sporty": "sporty clothes", "streetwear": "streetwear",
        "boho": "boho style clothes", "business": "business attire", "punk": "edgy punk style outfit",
        "cozy": "cozy knitwear",
    ]

    /// Danbooru tags for body picks whose plain phrase the tag-trained models misread: on Anima "athletic build,
    /// clearly defined muscles" drew an extreme bodybuilder in a bikini.
    private static let booruBodyTags: [String: String] = [
        "body/athletic": "toned", "muscle/toned": "toned", "muscle/defined": "toned, abs",
        "muscle/very-muscular": "muscular",
    ]
    /// Danbooru clothing tags: "casual clothes" says too little there, and Anima undressed her.
    private static let booruClothesTags: [String: String] = [
        "casual": "t-shirt, jeans, sneakers", "sporty": "sportswear", "streetwear": "hoodie, streetwear",
        "cozy": "sweater",
    ]

    /// An option's exclusion in words, for models without a negative prompt (MiniMax).
    private static func without(category: String, option: String, gender: String) -> String? {
        withouts["\(category)/\(option)/\(gender)"] ?? withouts["\(category)/\(option)"]
    }

    private static let withouts: [String: String] = [
        "body/slim": "no visible muscle definition",
        "body/average": "no visible abs",
        "body/average/man": "no visible abs and no muscle definition",
        "body/curvy/man": "no visible abs",
        "body/plus-size": "no visible abs",
        "body/soft": "no visible abs",
        "body/chubby": "no visible abs",
        "body/dad-bod": "no visible abs",
        "muscle/soft": "no muscle definition",
        "figure/soft-belly": "no visible abs",
        "bodyHair/smooth": "no body hair",
        "clothes/streetwear/man": "no denim",
        "clothes/boho/man": "no denim",
    ]

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
        // MiniMax has no negative prompt, and "glamour" pulls the man towards a fitness model: say what he isn't
        "body/slim/man": ("a skinny, scrawny, lanky body: bony narrow shoulders, thin bony arms, a flat stomach with "
                          + "no abs at all and no muscle definition, visible ribs and collarbones, a long thin neck",
                          "muscular, abs, six-pack, bodybuilder, broad shoulders"),
        "body/athletic": ("an athletic, toned, fit body", "fat, skinny"),
        // Not "soft untoned stomach": with "no visible abs" MiniMax drew a big bare belly.
        "body/average": ("an average, healthy everyday body, neither skinny nor athletic", "abs, six-pack, muscular, fitness model, skinny"),
        "body/average/man": ("a dad bod: an ordinary out-of-shape body with a soft round belly that hangs a little over "
                             + "the waistband, a soft chest, no visible abs at all and no muscle definition", "abs, six-pack, muscular, fitness model, skinny"),
        "body/curvy": ("a curvy body with a full bust and full rounded hips", "skinny, abs, muscular"),
        "body/curvy/man": ("a soft, stocky body with a little belly", "abs, six-pack, muscular, fitness model"),
        "body/muscular": ("a very muscular bodybuilder physique, big muscles", "skinny, fat"),
        "body/plus-size": ("a plus-size, heavy body and a big belly", "skinny, abs, muscular"),
        "body/broad-shouldered": ("very broad shoulders and a wide V-shaped upper body", "narrow shoulders"),
        "body/hourglass": ("an hourglass figure: large bust, wide hips and a very narrow waist", "skinny, flat chest, abs"),
        "bust/small": ("small breasts, a small chest, A cup", "large breasts, big breasts, cleavage"),
        "bust/medium": ("medium breasts, B cup", "huge breasts, flat chest"),
        "bust/large": ("large breasts, D cup, cleavage", "small breasts, flat chest"),
        "bust/very-large": ("huge breasts, an extremely large bust, G cup, deep cleavage", "small breasts, medium breasts, flat chest"),
        "butt/small": ("a small flat butt and narrow hips", "big butt, wide hips, round butt"),
        "butt/perky": ("a perky, lifted, firm butt", "flat butt, saggy butt"),
        "butt/round": ("a round, full bubble butt", "flat butt"),
        "butt/big": ("a very big, wide, thick butt and wide hips", "small butt, flat butt, skinny"),
        "figure/narrow-waist": ("a very narrow, cinched waist", "thick waist, belly"),
        "figure/wide-hips": ("very wide hips", "narrow hips"),
        "figure/long-legs": ("very long legs", "short legs, bodybuilder"),
        "figure/thick-thighs": ("very thick, full thighs", "skinny legs, thin thighs, bodybuilder"),
        "figure/toned-abs": ("visible six-pack abs", "belly, soft stomach"),
        "figure/toned-arms": ("toned, muscular arms", "thin arms"),
        "figure/broad-chest": ("a very broad, big, muscular chest", "narrow chest, skinny"),
        "figure/hairy-chest": ("a very hairy chest, thick chest hair", "hairless chest, smooth chest, bodybuilder"),
        "body/lean": ("a lean, slim body with a little visible muscle", "fat, bodybuilder"),
        "body/toned": ("a toned, fit, firm body without bulk", "fat, bodybuilder"),
        "body/soft": ("a soft, gently curved body with no visible muscle", "abs, six-pack, muscular"),
        "body/chubby": ("a chubby, soft, round body with some extra weight", "skinny, abs, muscular"),
        "body/stocky": ("a stocky, solid, thick-set body", "skinny, lanky"),
        "body/dad-bod": ("a dad bod: an ordinary body with a soft round belly and no visible abs", "abs, six-pack, muscular, fitness model"),
        "body/voluptuous": ("a voluptuous body with a full bust, wide hips and soft curves", "skinny, flat chest, abs"),
        "muscle/soft": ("no visible muscle definition", "abs, six-pack, muscular"),
        "muscle/defined": ("clearly defined, visible muscles", "fat, soft body"),
        "muscle/very-muscular": ("big, heavily built bodybuilder muscles", "skinny"),
        "figure/flat-stomach": ("a flat, firm stomach", "belly"),
        "figure/soft-belly": ("a soft, slightly rounded belly", "six-pack abs, flat stomach"),
        "figure/love-handles": ("a little softness at the waist, love handles", "six-pack abs"),
        "bodyHair/smooth": ("smooth, completely hairless skin", "body hair, chest hair"),
        "bodyHair/hairy": ("a lot of body hair on the chest, belly and arms", "smooth hairless chest"),
        "clothes/casual": ("casual clothes: a soft t-shirt, relaxed jeans and white sneakers", ""),
        "clothes/elegant": ("an elegant outfit: a fitted satin cocktail dress and heels", ""),
        "clothes/elegant/man": ("an elegant outfit: a tailored dark suit, a crisp white shirt and polished shoes", ""),
        "clothes/sporty": ("sporty clothes: high-waisted athletic leggings, a fitted sports top and running shoes", ""),
        "clothes/sporty/man": ("sporty clothes: a zip-up track jacket, joggers and running shoes", ""),
        "clothes/streetwear": ("streetwear: an oversized hoodie, baggy cargo pants and chunky sneakers", "denim jacket"),
        "clothes/boho": ("boho style: a flowy floral maxi dress, layered necklaces and leather sandals", "denim jacket"),
        "clothes/business": ("business attire: a tailored blazer, a silk blouse and slim trousers", ""),
        "clothes/business/man": ("business attire: a navy suit, a white shirt and a silk tie", ""),
        "clothes/punk": ("an edgy punk style: a black leather jacket, ripped black jeans, studs and combat boots", ""),
        "clothes/cozy": ("cozy knits: a chunky oversized knit sweater and soft knit lounge pants", ""),
        "clothes/streetwear/man": ("streetwear: an oversized plain black hoodie with no print or text, a silver chain, baggy olive cargo "
                                   + "pants and chunky white high-top sneakers, no jacket, no denim",
                                   "denim, jeans, denim jacket, shirt"),
        "clothes/boho/man": ("boho style: a loose white linen shirt open at the chest, layered wooden bead necklaces "
                             + "and leather bracelets, wide light beige linen trousers and leather sandals, no denim",
                             "denim, jeans, denim jacket, t-shirt"),
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
        "heritage/east-asian": "Chinese, Japanese or Korean roots",
        "heritage/southeast-asian": "Thai, Vietnamese or Filipino roots, say",
        "heritage/south-asian": "Indian, Pakistani or Bangladeshi roots",
        "heritage/black": "African or Caribbean roots",
        "heritage/latin": "Latin American roots",
        "heritage/middle-eastern": "Arab, Persian or Turkish roots, say",
        "heritage/white": "European roots",
        "heritage/mixed": "Roots in more than one place",
        "faceShape/oval": "Balanced, a little longer than wide",
        "faceShape/round": "Soft and about as wide as long",
        "faceShape/heart": "Wide forehead, narrow pointed chin",
        "faceShape/square": "Strong, angular jaw and forehead",
        "faceShape/long": "Clearly longer than wide",
        "faceShape/diamond": "Wide cheekbones, narrow forehead and chin",
        "eyeShape/almond": "Oval eyes with slightly pointed corners",
        "eyeShape/round": "Big, open round eyes",
        "eyeShape/hooded": "A fold of skin over the eyelid",
        "eyeShape/upturned": "Outer corners lift up, cat-like",
        "eyeShape/downturned": "Outer corners tilt down, soft look",
        "eyeShape/monolid": "No visible crease on the eyelid",
        "eyeShape/deep-set": "Eyes sit deeper under the brow",
        "eyebrows/thick": "Full, dense brows",
        "eyebrows/thin": "Fine, slim brows",
        "eyebrows/arched": "Brows with a high curve",
        "eyebrows/straight": "Flat brows with little arch",
        "eyebrows/bushy": "Big, untamed brows",
        "nose/straight": "A straight bridge",
        "nose/narrow": "A small, slim nose",
        "nose/roman": "A high, slightly curved bridge",
        "nose/wide": "A broad nose",
        "nose/upturned": "The tip points up a little",
        "nose/long": "A long, elegant nose",
        "lips/full": "Plump lips",
        "lips/thin": "Slim lips",
        "lips/cupids-bow": "A clear double curve on the upper lip",
        "lips/wide": "A broad smile",
        "lips/pouty": "Lips that push forward a little",
        "jaw/high-cheekbones": "Cheekbones sit high and stand out",
        "jaw/hollow-cheeks": "Cheeks dip in under the cheekbones",
        "jaw/soft-cheeks": "Full, soft cheeks",
        "jaw/sharp-jaw": "A crisp, defined jawline",
        "jaw/square-jaw": "A wide, angular jaw",
        "jaw/soft-jaw": "A gentle, rounded jawline",
        "jaw/strong-chin": "A chin that stands out",
        "jaw/cleft-chin": "A small dimple in the chin",
        "makeup/none": "Bare face, no makeup",
        "makeup/natural": "Light makeup that looks barely there",
        "makeup/soft-glam": "Polished, warm and glowing",
        "makeup/smoky": "Dark, blended eye shadow",
        "makeup/winged": "Eyeliner that flicks out at the corners",
        "makeup/red-lips": "Classic red lips",
        "makeup/glossy": "Shiny lip gloss",
        "facialHair/clean-shaven": "No facial hair",
        "facialHair/short-boxed": "A short, neatly shaped beard",
        "facialHair/long-beard": "A long, full beard",
        "facialHair/goatee": "Hair on the chin only",
        "facialHair/mustache": "Hair above the upper lip only",
        "body/lean": "Slim with a little muscle",
        "body/toned": "Fit and firm, not bulky",
        "body/soft": "Gentle curves, no muscle showing",
        "body/chubby": "Round and soft, carries some weight",
        "body/stocky": "Short, solid and thick-set",
        "body/dad-bod": "Ordinary build with a bit of a belly",
        "body/voluptuous": "Full, generous curves",
        "height/very-tall": "Much taller than most people",
        "muscle/soft": "No muscle definition showing",
        "muscle/toned": "Muscles show a little",
        "muscle/defined": "Muscles clearly visible",
        "muscle/very-muscular": "Big, bulky muscles",
        "shoulders/narrow": "Slim, narrow shoulders",
        "shoulders/rounded": "Soft, sloping shoulders",
        "shoulders/square": "Straight, squared-off shoulders",
        "shoulders/broad": "Very wide shoulders",
        "bustShape/round": "Full and round",
        "bustShape/teardrop": "Fuller at the bottom",
        "bustShape/perky": "Lifted and firm",
        "bustShape/natural": "Soft with a natural shape",
        "bustShape/wide-set": "More space between the breasts",
        "butt/heart": "Full at the bottom, like an upside-down heart",
        "butt/athletic": "Firm and muscular",
        "figure/flat-stomach": "A flat, firm belly",
        "figure/soft-belly": "A soft, slightly round belly",
        "figure/love-handles": "A little softness at the sides",
        "figure/slender-arms": "Slim arms",
        "figure/muscular-legs": "Strong, muscular legs",
        "figure/long-neck": "A long, graceful neck",
        "figure/toned-back": "Defined back muscles",
        "bodyHair/smooth": "No body hair",
        "bodyHair/light": "A little body hair",
        "bodyHair/natural": "Body hair left as it grows",
        "bodyHair/hairy": "Lots of body hair",
        "bodyMarks/sleeve-tattoo": "Tattoos cover a whole arm",
        "bodyMarks/back-tattoo": "A big tattoo across the back",
        "bodyMarks/small-tattoos": "A few small tattoos here and there",
        "bodyMarks/belly-piercing": "A ring or stud in the belly button",
        "bodyMarks/tan-lines": "Paler skin where swimwear was",
        "bodyMarks/freckled-shoulders": "Freckles across the shoulders",
        "bodyMarks/stretch-marks": "Faint lines on hips or thighs",
        "bodyMarks/birthmark": "A small mark on the skin since birth",
        "extras/beauty-mark": "A small dark mole above the lip",
        "extras/rosy-cheeks": "Naturally pink cheeks",
        "extras/gap-teeth": "A small gap between the front teeth",
        "nipples/small": "Small nipples",
        "nipples/puffy": "Nipples and areolae that stand out",
        "nipples/large-areolae": "Wide areolae",
        "nipples/pink": "Light pink nipples",
        "nipples/dark": "Darker nipples",
        "nipples/pierced": "Nipple piercings",
        "pubicHair/shaved": "No pubic hair",
        "pubicHair/trimmed": "Short, neat pubic hair",
        "pubicHair/landing-strip": "A narrow strip of pubic hair",
        "pubicHair/natural": "Pubic hair left as it grows",
        "penisSize/average": "Average size",
        "penisSize/large": "Larger than average",
        "circumcision/circumcised": "No foreskin",
        "circumcision/uncut": "With foreskin",
        "heritage/japanese": "Roots in Japan",
        "heritage/korean": "Roots in Korea",
        "heritage/chinese": "Roots in China",
        "heritage/filipino": "Roots in the Philippines",
        "heritage/vietnamese": "Roots in Vietnam",
        "heritage/thai": "Roots in Thailand",
        "heritage/indonesian": "Roots in Indonesia",
        "heritage/indian": "Roots in India",
        "heritage/pakistani": "Roots in Pakistan",
        "heritage/west-african": "Nigerian, Ghanaian or Senegalese roots, say",
        "heritage/east-african": "Ethiopian, Kenyan or Somali roots, say",
        "heritage/caribbean": "Jamaican, Haitian or Trinidadian roots, say",
        "heritage/mexican": "Roots in Mexico",
        "heritage/brazilian": "Roots in Brazil",
        "heritage/colombian": "Roots in Colombia",
        "heritage/arab": "Roots in the Arab world",
        "heritage/persian": "Iranian roots",
        "heritage/turkish": "Roots in Turkey",
        "heritage/nordic": "Scandinavian or Finnish roots",
        "heritage/slavic": "Polish, Russian or Ukrainian roots, say",
        "heritage/mediterranean": "Italian, Greek or Spanish roots, say",
        "heritage/british-irish": "Roots in Britain or Ireland",
        "heritage/indigenous-american": "Native roots in the Americas",
        "heritage/polynesian": "Hawaiian, Samoan or Maori roots, say",
    ]
}
