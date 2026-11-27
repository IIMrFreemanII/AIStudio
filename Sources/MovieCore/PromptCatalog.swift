import Foundation

/// The Prompt Builder's choices: groups of chips, each adding words to one part of a `PromptSpec`.
public enum PromptCatalog {
    /// Which part of the spec a group fills.
    public enum Field: String, Sendable, Codable { case medium, shot, camera, lighting, time, mood, palette, composition, place, sound, action, coverage }

    public struct Choice: Identifiable, Equatable, Sendable, Hashable {
        public let id: String
        public let label: String
        /// The words it adds (the writer places them).
        public let words: String
        /// Other words for the models trained on Danbooru tags (Pony, Illustrious, Anima), where a tag means what
        /// it says there ("cowboy shot" is a framing; photo SDXL draws a cowboy).
        public let booru: String?
        public init(_ id: String, _ label: String, _ words: String? = nil, booru: String? = nil) {
            self.id = id
            self.label = label
            self.words = words ?? label.lowercased()
            self.booru = booru
        }

        /// The words for `model`.
        public func words(for model: PromptModel) -> String { model.readsBooru ? booru ?? words : words }
    }

    public struct Group: Identifiable, Equatable, Sendable {
        public let id: String
        public let title: String
        public let field: Field
        /// How many may be picked (1 = pick one).
        public let max: Int
        /// Only for video models, or only for those with sound.
        public let videoOnly: Bool
        public let soundOnly: Bool
        /// 18+: shown only when the user turned the adult options on.
        public let adult: Bool
        public let choices: [Choice]
        public init(_ id: String, _ title: String, _ field: Field, max: Int, videoOnly: Bool = false, soundOnly: Bool = false,
                    adult: Bool = false, _ choices: [Choice]) {
            self.id = id
            self.title = title
            self.field = field
            self.max = max
            self.videoOnly = videoOnly
            self.soundOnly = soundOnly
            self.adult = adult
            self.choices = choices
        }
    }

    private typealias C = Choice

    /// A candid photo for the tag-trained models: "smartphone" drew a phone camera screen around the picture on Anima
    /// and a phone in her hand on Illustrious.
    public static let booruPhoto = "realistic, photo (medium)"

    public static let groups: [Group] = [
        Group("medium", "Medium", .medium, max: 1, [
            C("candid-phone", "Candid phone photo", "candid smartphone photo", booru: booruPhoto),
            C("studio-photo", "Studio photo", "studio photo"),
            C("film-35mm", "35mm film photo", "35mm film photo"),
            C("polaroid", "Polaroid", "instant polaroid photo"),
            C("editorial", "Fashion editorial", "fashion editorial photo"),
            C("street", "Street photo", "street photography"),
            C("documentary", "Documentary", "documentary photo"),
            C("cinematic-still", "Film still", "cinematic film still"),
            C("product", "Product shot", "product photo"),
            C("food", "Food photo", "food photography"),
            C("architecture", "Architecture", "architectural photo"),
            C("landscape-photo", "Landscape photo", "landscape photograph"),
            C("macro", "Macro photo", "macro photograph"),
            C("bw-photo", "Black-and-white photo", "black-and-white photo"),
            C("anime", "Anime", "anime illustration"),
            C("manga", "Manga", "manga panel, black and white ink"),
            C("digital-painting", "Digital painting", "digital painting"),
            C("oil", "Oil painting", "oil painting on canvas"),
            C("watercolor", "Watercolour", "watercolour painting"),
            C("ink", "Ink drawing", "ink drawing"),
            C("pencil", "Pencil sketch", "pencil sketch"),
            C("concept-art", "Concept art", "concept art"),
            C("comic", "Comic book", "comic book art"),
            C("pixel", "Pixel art", "pixel art"),
            C("3d-render", "3D render", "3D render"),
            C("claymation", "Claymation", "claymation figure"),
            C("isometric", "Isometric", "isometric illustration"),
            C("vector", "Flat vector", "flat vector illustration"),
            C("poster", "Poster", "graphic poster design"),
        ]),
        Group("shot", "Shot", .shot, max: 1, [
            C("extreme-close-up", "Extreme close-up"), C("close-up", "Close-up"),
            C("head-shoulders", "Head and shoulders", "head-and-shoulders portrait"),
            C("medium-close", "Medium close-up"), C("medium", "Medium shot"), C("cowboy", "Mid-thigh up", "medium full shot, framed from mid-thigh up", booru: "cowboy shot"),
            C("full-body", "Full body"), C("wide", "Wide shot"), C("extreme-wide", "Extreme wide shot"),
            C("establishing", "Establishing shot"), C("over-shoulder", "Over the shoulder", "over-the-shoulder shot"),
            C("pov", "POV", "first-person point of view"), C("selfie", "Selfie", "selfie, arm's length"),
            C("mirror-selfie", "Mirror selfie"), C("two-shot", "Two-shot"), C("insert", "Detail insert", "detail insert shot"),
        ]),
        Group("angle", "Camera angle", .camera, max: 1, [
            C("eye-level", "Eye level"), C("low-angle", "Low angle", "low-angle shot"), C("high-angle", "High angle", "high-angle shot"),
            C("birds-eye", "Bird's-eye view"), C("worms-eye", "Worm's-eye view"), C("dutch", "Dutch angle", "dutch angle, tilted horizon"),
            C("top-down", "Top-down", "top-down flat lay"), C("profile", "Profile", "side profile view"),
            C("three-quarter", "Three-quarter view"), C("from-behind", "From behind", "seen from behind"),
            C("aerial", "Aerial drone view"),
        ]),
        Group("lens", "Lens and focus", .camera, max: 3, [
            C("ultra-wide", "Ultra-wide 14mm", "14mm ultra-wide lens"), C("wide-24", "Wide 24mm", "24mm lens"),
            C("normal-50", "Normal 50mm", "50mm lens"), C("portrait-85", "Portrait 85mm", "85mm portrait lens"),
            C("tele-200", "Telephoto 200mm", "200mm telephoto lens, compressed background"),
            C("fisheye", "Fisheye"), C("tilt-shift", "Tilt-shift", "tilt-shift miniature effect"),
            C("shallow-dof", "Shallow depth of field", "shallow depth of field, soft bokeh background"),
            C("deep-focus", "Deep focus", "deep focus, everything sharp"), C("motion-blur", "Motion blur"),
            C("long-exposure", "Long exposure"), C("film-grain", "Film grain"), C("anamorphic", "Anamorphic flare", "anamorphic lens flare"),
        ]),
        Group("movement", "Camera movement", .camera, max: 2, videoOnly: true, [
            C("static", "Static", "static camera, locked off"), C("slow-push", "Slow push in", "slow dolly push-in"),
            C("pull-out", "Pull out", "slow dolly pull-out"), C("pan-left", "Pan left", "slow pan left"),
            C("pan-right", "Pan right", "slow pan right"), C("tilt-up", "Tilt up"), C("tilt-down", "Tilt down"),
            C("tracking", "Tracking", "tracking shot following the subject"), C("orbit", "Orbit", "camera orbits around the subject"),
            C("crane-up", "Crane up", "crane shot rising up"), C("handheld", "Handheld", "handheld camera, slight shake"),
            C("zoom-in", "Zoom in", "slow zoom in"), C("whip-pan", "Whip pan"), C("fpv", "FPV drone", "fast FPV drone flight"),
            C("rack-focus", "Rack focus", "rack focus from foreground to subject"),
        ]),
        Group("lighting", "Lighting", .lighting, max: 3, [
            C("natural", "Natural light"), C("soft-window", "Soft window light"), C("golden-hour", "Golden hour", "warm golden-hour sunlight"),
            C("blue-hour", "Blue hour", "cool blue-hour light"), C("overcast", "Overcast", "soft overcast light"),
            C("harsh-sun", "Harsh midday sun", "harsh midday sun, hard shadows"), C("backlit", "Backlit", "backlit, glowing rim light"),
            C("rim", "Rim light"), C("studio-softbox", "Studio softbox", "large softbox key light"),
            C("rembrandt", "Rembrandt", "Rembrandt lighting"), C("butterfly", "Butterfly", "butterfly lighting"),
            C("split", "Split light", "split lighting, half the face in shadow"), C("low-key", "Low key", "low-key lighting, deep shadows"),
            C("high-key", "High key", "high-key lighting, bright and airy"), C("neon", "Neon", "neon light, pink and cyan glow"),
            C("candle", "Candlelight", "warm flickering candlelight"), C("fireplace", "Firelight", "warm firelight"),
            C("moonlight", "Moonlight", "cold moonlight"), C("flash", "Direct flash", "direct on-camera flash"),
            C("volumetric", "Volumetric", "volumetric light rays through haze"), C("practical", "Practical lamps", "warm practical lamps in frame"),
            C("silhouette", "Silhouette", "silhouette against a bright background"), C("dappled", "Dappled", "dappled light through leaves"),
        ]),
        Group("time", "Time of day", .time, max: 1, [
            C("dawn", "Dawn"), C("sunrise", "Sunrise"), C("morning", "Morning"), C("noon", "Noon"), C("afternoon", "Afternoon"),
            C("sunset", "Sunset"), C("dusk", "Dusk"), C("evening", "Evening"), C("night", "Night"), C("midnight", "Midnight"),
        ]),
        Group("weather", "Weather and season", .time, max: 2, [
            C("clear", "Clear sky"), C("cloudy", "Cloudy"), C("rain", "Rain", "light rain, wet surfaces"),
            C("storm", "Storm", "thunderstorm, dramatic clouds"), C("fog", "Fog", "thick fog"), C("mist", "Morning mist"),
            C("snow", "Snow", "falling snow"), C("wind", "Windy", "strong wind"), C("heat-haze", "Heat haze"),
            C("spring", "Spring", "spring, blossoms"), C("summer", "Summer"), C("autumn", "Autumn", "autumn, falling leaves"),
            C("winter", "Winter"),
        ]),
        Group("mood", "Mood", .mood, max: 3, [
            C("cozy", "Cozy"), C("romantic", "Romantic"), C("intimate", "Intimate"), C("playful", "Playful"), C("joyful", "Joyful"),
            C("calm", "Calm"), C("dreamy", "Dreamy"), C("nostalgic", "Nostalgic"), C("melancholic", "Melancholic"),
            C("mysterious", "Mysterious"), C("tense", "Tense"), C("eerie", "Eerie"), C("epic", "Epic"), C("gritty", "Gritty"),
            C("sensual", "Sensual"), C("whimsical", "Whimsical"), C("lonely", "Lonely"), C("energetic", "Energetic"),
        ]),
        Group("palette", "Colours", .palette, max: 2, [
            C("warm", "Warm tones"), C("cool", "Cool tones"), C("teal-orange", "Teal and orange"), C("pastel", "Pastel"),
            C("muted", "Muted", "muted, desaturated colours"), C("vibrant", "Vibrant", "vibrant saturated colours"),
            C("monochrome", "Monochrome"), C("earthy", "Earth tones"), C("neon-palette", "Neon", "neon pink and electric blue"),
            C("gold-black", "Black and gold"), C("red-accent", "Red accent", "mostly neutral with a single red accent"),
            C("sepia", "Sepia"), C("jewel", "Jewel tones", "emerald, sapphire and ruby jewel tones"),
            C("kodak", "Kodak Portra", "Kodak Portra film colours"), C("cinestill", "CineStill", "CineStill 800T colours, red halation"),
            C("bleach", "Bleach bypass", "bleach-bypass look, low saturation, high contrast"),
        ]),
        Group("composition", "Composition", .composition, max: 2, [
            C("thirds", "Rule of thirds", "subject on the rule-of-thirds line"), C("centered", "Centered", "centered symmetrical composition"),
            C("leading-lines", "Leading lines"), C("negative-space", "Negative space", "lots of negative space"),
            C("frame-in-frame", "Frame within a frame"), C("foreground", "Foreground layer", "out-of-focus foreground element"),
            C("reflection", "Reflection", "reflection in water or glass"), C("silhouette-comp", "Against the sky", "subject against the open sky"),
            C("crowded", "Busy background", "busy, detailed background"), C("minimal", "Minimal", "minimal, clean background"),
        ]),
        Group("place", "Place", .place, max: 1, [
            C("bedroom", "Bedroom", "a cozy bedroom"), C("kitchen", "Kitchen", "a sunlit kitchen"), C("living-room", "Living room", "a living room"),
            C("bathroom", "Bathroom", "a steamy bathroom"), C("cafe", "Café", "a small café"), C("bar", "Bar", "a dim cocktail bar"),
            C("restaurant", "Restaurant", "a candlelit restaurant"), C("office", "Office", "a modern office"),
            C("gym", "Gym", "a gym"), C("library", "Library", "an old library"), C("studio-gray", "Plain studio", "a plain light gray studio background"),
            C("rooftop", "Rooftop", "a city rooftop"), C("street", "City street", "a busy city street"), C("alley", "Neon alley", "a rainy neon-lit alley"),
            C("subway", "Subway", "a subway car"), C("beach", "Beach", "a sandy beach"), C("pool", "Pool", "the edge of a swimming pool"),
            C("forest", "Forest", "a forest"), C("mountains", "Mountains", "a mountain ridge"), C("lake", "Lake", "a quiet lake"),
            C("desert", "Desert", "a desert"), C("field", "Flower field", "a field of wildflowers"), C("snowy-woods", "Snowy woods", "snowy woods"),
            C("car", "In a car", "inside a car"), C("train", "Train window", "by a train window"), C("hotel", "Hotel room", "a hotel room"),
            C("concert", "Concert", "a crowded concert"), C("space", "Space station", "a space station"), C("castle", "Castle hall", "a castle hall"),
        ]),
        Group("sound", "Sound", .sound, max: 3, soundOnly: true, [
            C("ambience-city", "City ambience", "city traffic ambience"), C("ambience-nature", "Nature", "birdsong and wind in the trees"),
            C("rain-sound", "Rain", "rain falling"), C("waves", "Waves", "ocean waves"), C("crowd", "Crowd murmur"),
            C("cafe-sound", "Café chatter", "café chatter and clinking cups"), C("music-soft", "Soft music", "soft background music"),
            C("music-upbeat", "Upbeat music"), C("footsteps", "Footsteps"), C("fire-crackle", "Fire crackling"),
            C("silence", "Near silence", "near silence, room tone"), C("dialogue", "Speaks", "the person speaks the line clearly"),
            C("laugh", "Laughter", "soft laughter"), C("wind-sound", "Wind", "howling wind"),
        ]),
        // A person's coverage (PromptSpec.Coverage; choice ids are its raw values). Nothing picked = dressed.
        Group("nudity", "Nudity (18+)", .coverage, max: 1, adult: true, [
            C("swimwear", "Swimwear", PromptSpec.Coverage.swimwear.tag), C("lingerie", "Lingerie", PromptSpec.Coverage.lingerie.tag),
            C("underwear", "Underwear", PromptSpec.Coverage.underwear.tag), C("topless", "Topless", PromptSpec.Coverage.topless.tag),
            C("impliedNude", "Implied nude", PromptSpec.Coverage.impliedNude.tag), C("nude", "Fully nude", PromptSpec.Coverage.nude.tag),
        ]),
    ]

    public static func group(_ id: String) -> Group? { groups.first { $0.id == id } }

    /// The groups that apply to a model (camera moves only for video, sound only where the model makes it).
    /// `adult`: include the 18+ groups (Nudity).
    public static func groups(for model: PromptModel, adult: Bool = false) -> [Group] {
        groups.filter { (!$0.videoOnly || model.isVideo) && (!$0.soundOnly || model.hasSound) && (!$0.adult || adult) }
    }

    /// The picked coverage (dressed when nothing is picked, or when the 18+ options are off).
    public static func coverage(_ picks: [String: [String]], adult: Bool) -> PromptSpec.Coverage {
        guard adult, let id = picks["nudity"]?.first else { return .dressed }
        return PromptSpec.Coverage(rawValue: id) ?? .dressed
    }

    /// Adds the picked choices (group id → choice ids, in catalog order) to `spec`. Unknown ids are ignored; a
    /// single-value field (medium, shot, place, action) keeps what the spec already has unless it is empty.
    public static func apply(_ picks: [String: [String]], to spec: inout PromptSpec, model: PromptModel, adult: Bool = false) {
        for g in groups(for: model, adult: adult) {
            let ids = Set(picks[g.id] ?? [])
            let words = g.choices.filter { ids.contains($0.id) }.map { $0.words(for: model) }
            guard !words.isEmpty else { continue }
            switch g.field {
            case .medium: if spec.medium.isEmpty { spec.medium = words.joined(separator: ", ") }
            case .shot: if spec.shot.isEmpty { spec.shot = words.joined(separator: ", ") }
            case .place: if spec.place.isEmpty { spec.place = words.joined(separator: ", ") }
            case .action: if spec.action.isEmpty { spec.action = words.joined(separator: ", ") }
            case .camera: spec.camera += words
            case .lighting: spec.lighting += words
            case .time: spec.time += words
            case .mood: spec.mood += words
            case .palette: spec.palette += words
            case .composition: spec.composition += words
            case .sound: spec.sound += words
            case .coverage: if spec.coverage == .dressed { spec.coverage = coverage(picks, adult: adult) }
            }
        }
    }

    /// Tapping a chip: removes a picked one; a pick-one group replaces; a full group drops its oldest pick.
    public static func toggle(_ picks: [String: [String]], group id: String, choice: String) -> [String: [String]] {
        guard let g = group(id) else { return picks }
        var out = picks
        var list = out[id] ?? []
        if let i = list.firstIndex(of: choice) { list.remove(at: i) }
        else if g.max == 1 { list = [choice] }
        else {
            list.append(choice)
            if list.count > g.max { list.removeFirst(list.count - g.max) }
        }
        out[id] = list.isEmpty ? nil : list
        return out
    }
}
