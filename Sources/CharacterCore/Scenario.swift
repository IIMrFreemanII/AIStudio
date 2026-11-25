import Foundation

/// Where and when a chat happens: it sets the scene in the prompt and in photos.
public struct Scenario: Codable, Identifiable, Equatable, Sendable, Hashable {
    public var id: String
    public var title: String
    public var place: String
    public var outfit: String
    public var time: String
    public var mood: String

    public init(id: String, title: String, place: String, outfit: String = "", time: String = "", mood: String = "") {
        self.id = id
        self.title = title
        self.place = place
        self.outfit = outfit
        self.time = time
        self.mood = mood
    }

    /// A custom scene that brings an album photo's look back.
    public static func like(_ item: AlbumItem, base: Scenario?, date: String) -> Scenario {
        Scenario(id: "album-\(item.id.uuidString)", title: "Like the photo of \(date)",
                 place: base?.place ?? "the same place as in that photo",
                 outfit: item.prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                 time: base?.time ?? "", mood: base?.mood ?? "")
    }

    public static let defaultID = "texting"

    /// The built-in scenarios.
    public static let builtIns: [Scenario] = [
        Scenario(id: "texting", title: "Texting from home", place: "at home, on the sofa, texting",
                 outfit: "comfortable everyday clothes", time: "whenever it is now", mood: "relaxed"),
        Scenario(id: "coffee", title: "Coffee date", place: "a small cozy coffee shop with big windows",
                 outfit: "a casual stylish outfit", time: "a sunny late morning", mood: "warm and curious"),
        Scenario(id: "dinner", title: "Dinner date", place: "a candle-lit restaurant with soft music",
                 outfit: "an elegant dinner outfit", time: "evening", mood: "charming and a little nervous"),
        Scenario(id: "beach", title: "Day at the beach", place: "a sunny beach with white sand and calm waves",
                 outfit: "a summer outfit and sunglasses", time: "a bright afternoon", mood: "playful and carefree"),
        Scenario(id: "walk", title: "Evening walk", place: "a quiet city park path under street lamps",
                 outfit: "a light jacket and sneakers", time: "dusk, the air is cool", mood: "calm and talkative"),
        Scenario(id: "home", title: "Cozy night in", place: "a warm living room with blankets and a movie on",
                 outfit: "soft pajamas and a big sweater", time: "late evening", mood: "cozy and affectionate"),
    ]
}
