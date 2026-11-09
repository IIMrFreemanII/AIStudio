import Foundation

/// What an input asset is, from its extension (and masks from the `masks/` folder).
public enum AssetKind: String, CaseIterable, Sendable, Identifiable {
    case image, mask, video, audio
    public var id: String { rawValue }

    public static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "webp", "bmp", "gif"]
    public static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]
    public static let audioExtensions: Set<String> = ["wav", "m4a", "mp3", "aif", "aiff", "caf", "flac", "aac"]

    /// The Load node that reads this kind.
    public var loadNodeType: String {
        switch self {
        case .image: return "LoadImage"
        case .mask: return "LoadMask"
        case .video: return "LoadVideo"
        case .audio: return "LoadAudio"
        }
    }
}

/// A file in the input folder.
public struct AssetItem: Identifiable, Hashable, Sendable {
    public var url: URL
    public var kind: AssetKind
    public var modified: Date
    /// The path Load nodes store: relative to the input folder, e.g. `fox.png` or `masks/mask-1.png`.
    public var relativePath: String
    public var id: URL { url }
    public var name: String { url.lastPathComponent }
}

/// The input folder, like ComfyUI's `input/`: files you import (or drop on the graph) are copied here, and Load nodes
/// refer to them by a path relative to it, so workflows don't break when the originals move.
public struct AssetStore: Sendable {
    public let root: URL
    public var useTrash = true

    public init(root: URL, useTrash: Bool = true) {
        self.root = root.standardizedFileURL
        self.useTrash = useTrash
    }

    public var masks: URL { root.appendingPathComponent("masks", isDirectory: true) }

    public static func kind(of url: URL) -> AssetKind? {
        let ext = url.pathExtension.lowercased()
        if AssetKind.imageExtensions.contains(ext) {
            return url.deletingLastPathComponent().lastPathComponent == "masks" ? .mask : .image
        }
        if AssetKind.videoExtensions.contains(ext) { return .video }
        if AssetKind.audioExtensions.contains(ext) { return .audio }
        return nil
    }

    /// True when `url` is inside the input folder.
    public func contains(_ url: URL) -> Bool {
        url.standardizedFileURL.path.hasPrefix(root.path + "/")
    }

    /// The Load-node path for a file in the input folder (nil if it isn't in it).
    public func relativePath(_ url: URL) -> String? {
        let p = url.standardizedFileURL.path
        guard p.hasPrefix(root.path + "/") else { return nil }
        return String(p.dropFirst(root.path.count + 1))
    }

    /// Every asset in the folder and its subfolders (hidden files skipped), newest first.
    public func list() -> [AssetItem] {
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
                                                     options: [.skipsHiddenFiles]) else { return [] }
        var items: [AssetItem] = []
        for case let url as URL in e {
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .isRegularFileKey])
            guard values?.isRegularFile == true, let kind = Self.kind(of: url), let rel = relativePath(url) else { continue }
            items.append(AssetItem(url: url.standardizedFileURL, kind: kind, modified: values?.contentModificationDate ?? .distantPast,
                                   relativePath: rel))
        }
        return items.sorted { ($0.modified, $0.relativePath) > ($1.modified, $1.relativePath) }
    }

    /// `<dir>/<name>`, or `<stem> (1).<ext>`, `<stem> (2).<ext>`… when taken.
    public func uniqueURL(for name: String, in dir: URL) -> URL {
        let base = URL(fileURLWithPath: name)
        let stem = base.deletingPathExtension().lastPathComponent, ext = base.pathExtension
        var candidate = dir.appendingPathComponent(name)
        var n = 1
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = dir.appendingPathComponent(ext.isEmpty ? "\(stem) (\(n))" : "\(stem) (\(n)).\(ext)")
            n += 1
        }
        return candidate.standardizedFileURL
    }

    /// Copies a file into the input folder (or `masks/`) and returns the copy. A file already in the folder, or one
    /// identical to a file there of the same name, is returned as it is.
    @discardableResult
    public func importFile(_ url: URL, asMask: Bool = false) throws -> URL {
        if contains(url) { return url.standardizedFileURL }
        guard Self.kind(of: url) != nil else { throw CocoaError(.fileReadUnsupportedScheme) }
        let dir = asMask ? masks : root
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // The same file imported again (same name and contents) is reused rather than copied as "name (1)".
        let same = dir.appendingPathComponent(url.lastPathComponent)
        if FileManager.default.contentsEqual(atPath: same.path, andPath: url.path) { return same.standardizedFileURL }
        let target = uniqueURL(for: url.lastPathComponent, in: dir)
        try FileManager.default.copyItem(at: url, to: target)
        return target
    }

    /// Renames an asset (keeping its extension unless `name` has one).
    @discardableResult
    public func rename(_ url: URL, to name: String) throws -> URL {
        var n = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !n.isEmpty else { return url }
        if URL(fileURLWithPath: n).pathExtension.isEmpty { n += "." + url.pathExtension }
        guard n != url.lastPathComponent else { return url }
        let target = uniqueURL(for: n, in: url.deletingLastPathComponent())
        try FileManager.default.moveItem(at: url, to: target)
        return target
    }

    public func delete(_ url: URL) throws {
        if useTrash { try FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        else { try FileManager.default.removeItem(at: url) }
    }
}
