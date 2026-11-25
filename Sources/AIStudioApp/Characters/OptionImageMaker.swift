import AppKit
import CharacterCore
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The example photos on the creator's look chips: made once on a GPU server (SDXL, photoreal) and bundled with the
/// app in `Resources/OptionImages`.
enum OptionImageMaker {
    static let maxPixelSize = 352
    static let quality = 0.8

    /// Draws every example (or only those of `categories`) into `out`, skipping files already there. Returns the files
    /// it made and their summed cost; a failed example is reported through `say` and skipped.
    @MainActor
    static func make(_ model: AppModel, server: String, out: URL, categories: Set<String>? = nil,
                     limit: Int? = nil, say: (String) -> Void) async -> (made: [URL], cost: Double) {
        let fm = FileManager.default
        try? fm.createDirectory(at: out, withIntermediateDirectories: true)
        var made: [URL] = []
        var cost = 0.0
        var todo = CharacterOptions.examples.filter { categories?.contains($0.category) ?? true }
        if let limit { todo = Array(todo.prefix(limit)) }
        for (i, ex) in todo.enumerated() {
            let file = out.appendingPathComponent(CharacterOptions.exampleFile(category: ex.category, option: ex.option,
                                                                              gender: ex.gender))
            if fm.fileExists(atPath: file.path) { continue }
            // A throwaway character: photoreal SDXL, no portraits (so no face reference).
            var c = AppModel.Companion(name: "Example", age: 30, gender: ex.gender)
            c.artModel = ArtModel()
            let prompt = CharacterOptions.examplePrompt(category: ex.category, option: ex.option, gender: ex.gender)
            let full = fm.temporaryDirectory.appendingPathComponent("option-example-\(UUID().uuidString).png")
            defer { try? fm.removeItem(at: full) }
            do {
                let spent = try await model.drawPhoto(
                    c, prompt: prompt, server: server, to: full, title: "option example",
                    seed: CharacterOptions.exampleSeed(category: ex.category, gender: ex.gender),
                    extraNegative: CharacterOptions.exampleNegative(category: ex.category, option: ex.option,
                                                                    gender: ex.gender)) { _ in }
                cost += spent ?? 0
                try shrink(full, to: file)
                made.append(file)
                say("\(i + 1)/\(todo.count) \(file.lastPathComponent) $\(String(format: "%.4f", spent ?? 0))")
            } catch {
                say("FAIL \(file.lastPathComponent): \(error)")
            }
        }
        return (made, cost)
    }

    /// A JPEG of `source` whose longer side is `maxPixelSize`.
    static func shrink(_ source: URL, to dest: URL) throws {
        guard let src = CGImageSourceCreateWithURL(source as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
                  kCGImageSourceCreateThumbnailWithTransform: true,
              ] as CFDictionary),
              let out = CGImageDestinationCreateWithURL(dest as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(out, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(out) else { throw CocoaError(.fileWriteUnknown) }
    }
}

/// The bundled example photos, by option and gender.
enum OptionImages {
    private static let folder = Bundle.module.url(forResource: "OptionImages", withExtension: nil)
    private static let cache = NSCache<NSString, NSImage>()

    /// The file for this option: a man's or a woman's own; for any other gender the woman's, else the man's. Nil when
    /// none is bundled.
    static func url(category: String, option: String, gender: String) -> URL? {
        guard let folder else { return nil }
        let g = gender.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let order = g == "man" || g == "woman" ? [g] : ["woman", "man"]
        let fm = FileManager.default
        for candidate in order {
            let file = folder.appendingPathComponent(
                CharacterOptions.exampleFile(category: category, option: option, gender: candidate))
            if fm.fileExists(atPath: file.path) { return file }
        }
        return nil
    }

    static func image(category: String, option: String, gender: String) -> NSImage? {
        guard let file = url(category: category, option: option, gender: gender) else { return nil }
        if let hit = cache.object(forKey: file.path as NSString) { return hit }
        guard let image = NSImage(contentsOf: file) else { return nil }
        cache.setObject(image, forKey: file.path as NSString)
        return image
    }
}
