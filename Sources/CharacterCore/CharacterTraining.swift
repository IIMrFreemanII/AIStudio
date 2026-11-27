import Foundation

/// A character LoRA: about 40 face-referenced pictures of the character in varied framings, angles, outfits, places and
/// light, of which the user keeps 15–30; then a LoRA trained on them on the GPU (`train-sdxl-lora` or
/// `train-h3-image-lora`). Captions describe what changes between pictures and leave the person out, so the LoRA
/// learns the face, hair and body under the character's trigger word.
public enum CharacterTraining {
    public static let pictureCount = 40
    public static let minPicked = 15
    public static let maxPicked = 30

    /// The server task that trains a LoRA for the family (Chroma has none yet).
    public static func task(_ family: ArtModel.Family) -> String? {
        switch family {
        case .sdxl: "train-sdxl-lora"
        case .minimax: "train-h3-image-lora"
        case .chroma: nil
        }
    }

    /// A rare word for the character that starts every caption and later every photo prompt: up to six letters of
    /// the name and three characters of the id, such as `zqmia_3f9`.
    public static func trigger(for c: Character) -> String {
        let letters = c.name.lowercased().unicodeScalars.filter { (97...122).contains($0.value) }  // a…z
        let name = String(String.UnicodeScalarView(letters.prefix(6)))
        return "zq" + (name.isEmpty ? "char" : name) + "_" + c.id.uuidString.lowercased().prefix(3)
    }

    static let framings = ["close-up portrait", "head-and-shoulders portrait", "upper-body shot",
                           "medium shot from the waist up", "full-body shot"]
    static let angles = ["facing the camera", "in three-quarter view", "in profile", "seen from slightly above",
                         "seen from slightly below"]
    static let expressions = ["smiling", "laughing", "with a calm, neutral expression", "looking thoughtful"]
    static let places = ["in a sunlit park", "at a café table", "on a city street", "at the beach",
                         "in a cozy living room", "in a studio with a plain grey background", "in a forest",
                         "on a rooftop at dusk"]
    static let lights = ["soft daylight", "golden hour light", "overcast light", "warm indoor light"]

    static func outfits(_ gender: String) -> [String] {
        let g = gender.lowercased()
        let woman = g == "woman" || g == "female"
        let man = g == "man" || g == "male"
        return ["wearing a plain white t-shirt and jeans", "wearing a fitted black turtleneck",
                "wearing a casual hoodie", "wearing a smart blazer", "wearing fitted activewear",
                woman ? "wearing a one-piece swimsuit" : man ? "wearing swim shorts" : "wearing swimwear",
                "wearing a cozy knit sweater",
                woman ? "wearing a light summer dress" : "wearing a linen shirt"]
    }

    /// What each training picture shows: a deterministic, varied mix (all dressed; fitted clothes and swimwear show
    /// the body shape). Nothing about the person's own look, which the photo prompt adds.
    public static func requests(count: Int = pictureCount, gender: String) -> [String] {
        let outfits = outfits(gender)
        return (0..<max(count, 0)).map { k in
            [framings[k % framings.count], angles[(k * 2 + k / 5) % angles.count],
             expressions[(k * 3) % expressions.count], outfits[(k * 5 + k / 8) % outfits.count],
             places[(k * 7 + k / 8) % places.count], lights[(k + k / 4) % lights.count]].joined(separator: ", ")
        }
    }

    /// The caption the LoRA trains with: the trigger, who (a woman, a man, a person) and what the picture shows.
    public static func caption(request: String, trigger: String, gender: String) -> String {
        let who: String
        switch gender.lowercased() {
        case "woman", "female": who = "a woman"
        case "man", "male": who = "a man"
        default: who = "a person"
        }
        return "\(trigger), \(who), \(request)"
    }
}

/// A character's training pictures and LoRA, kept in `<character>/training/training.json` (apart from the character,
/// so an open editor's save can't overwrite a training that finished meanwhile).
public struct TrainingSet: Codable, Equatable, Sendable {
    public enum Status: String, Codable, Sendable { case empty, drawing, ready, training, done, failed }

    public struct Picture: Codable, Equatable, Sendable, Identifiable {
        public var file: String
        public var request: String
        public var picked: Bool
        public var id: String { file }
        public init(file: String, request: String, picked: Bool = true) {
            self.file = file
            self.request = request
            self.picked = picked
        }
    }

    public var pictures: [Picture] = []
    public var status: Status = .empty
    /// The family the pictures were drawn on (training targets it).
    public var family: ArtModel.Family?
    public var trigger: String?
    /// The library id of the trained LoRA.
    public var loraID: String?
    public var error: String?
    public var spentUSD: Double = 0

    public init() {}

    public var picked: [Picture] { pictures.filter(\.picked) }
    public var canTrain: Bool { (CharacterTraining.minPicked...CharacterTraining.maxPicked).contains(picked.count) }
}

extension CharacterStore {
    public func trainingFolder(_ id: UUID) -> URL { folder(id).appendingPathComponent("training", isDirectory: true) }

    public func training(_ id: UUID) -> TrainingSet {
        let url = trainingFolder(id).appendingPathComponent("training.json")
        guard let data = try? Data(contentsOf: url), let set = try? JSONDecoder().decode(TrainingSet.self, from: data)
        else { return TrainingSet() }
        return set
    }

    public func saveTraining(_ set: TrainingSet, for id: UUID) throws {
        let folder = trainingFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(set).write(to: folder.appendingPathComponent("training.json"), options: .atomic)
    }
}

/// A plain ZIP archive (stored, no compression: the pictures are PNGs already), enough for a training upload.
public enum ZipArchive {
    public static func write(_ entries: [(name: String, data: Data)], to url: URL) throws {
        var out = Data()
        var central = Data()
        for e in entries {
            let name = Data(e.name.utf8)
            let crc = crc32(e.data)
            let offset = UInt32(out.count)
            out.append(le32(0x0403_4b50))
            out.append(le16(20)); out.append(le16(0)); out.append(le16(0))  // version, flags, stored
            out.append(le16(0)); out.append(le16(0x21))  // time, date (1980-01-01)
            out.append(le32(crc)); out.append(le32(UInt32(e.data.count))); out.append(le32(UInt32(e.data.count)))
            out.append(le16(UInt16(name.count))); out.append(le16(0))
            out.append(name)
            out.append(e.data)
            central.append(le32(0x0201_4b50))
            central.append(le16(20)); central.append(le16(20)); central.append(le16(0)); central.append(le16(0))
            central.append(le16(0)); central.append(le16(0x21))
            central.append(le32(crc)); central.append(le32(UInt32(e.data.count))); central.append(le32(UInt32(e.data.count)))
            central.append(le16(UInt16(name.count))); central.append(le16(0)); central.append(le16(0))
            central.append(le16(0)); central.append(le16(0)); central.append(le32(0)); central.append(le32(offset))
            central.append(name)
        }
        let start = UInt32(out.count)
        out.append(central)
        out.append(le32(0x0605_4b50)); out.append(le16(0)); out.append(le16(0))
        out.append(le16(UInt16(entries.count))); out.append(le16(UInt16(entries.count)))
        out.append(le32(UInt32(central.count))); out.append(le32(start)); out.append(le16(0))
        try out.write(to: url, options: .atomic)
    }

    static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    public static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        data.withUnsafeBytes { buf in
            for b in buf.bindMemory(to: UInt8.self) { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        }
        return c ^ 0xFFFF_FFFF
    }

    private static func le16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    private static func le32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
}
