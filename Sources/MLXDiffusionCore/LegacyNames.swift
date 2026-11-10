import Foundation

/// The project was called MLXComfyUI before it was renamed AIStudio. Data written under the old names (folders,
/// model manifests and caches, metadata keys, Keychain items, settings) is moved or read through here, so an
/// existing install keeps its models, library and tokens.
public enum LegacyNames {
    /// The old folder name in Application Support, Documents and Pictures.
    public static let folderName = "MLXComfyUI"
    public static let newFolderName = "AIStudio"
    /// The key a graph was embedded under in saved images, videos and sounds.
    public static let graphMetadataKey = "mlxcomfy.graph"
    /// Prefix of the Keychain services (`MLXComfyUI.huggingface` and so on).
    public static let keychainPrefix = "MLXComfyUI."
    /// The app's settings domains: the bundled app's identifier, and the executable name when run unbundled.
    public static let defaultsDomains = ["local.mlxcomfyui", "MLXComfyApp"]

    /// Old and new names of the files and folders inside a model folder.
    static let modelFolderItems = [("mlxcomfy.json", "aistudio.json"), (".mlxcomfy-cache", ".aistudio-cache"),
                                   ("mlxcomfy-link.json", "aistudio-link.json")]

    /// The folders under the old name: `~/Library/Application Support/MLXComfyUI`, `~/Documents/MLXComfyUI` and
    /// `~/Pictures/MLXComfyUI`, each with the AIStudio folder it moves to.
    public static var folderMoves: [(old: URL, new: URL)] {
        let fm = FileManager.default
        return [FileManager.SearchPathDirectory.applicationSupportDirectory, .documentDirectory, .picturesDirectory]
            .compactMap { fm.urls(for: $0, in: .userDomainMask).first }
            .map { ($0.appendingPathComponent(folderName), $0.appendingPathComponent(newFolderName)) }
    }

    /// Moves each old folder to its new name and leaves a symbolic link at the old path, so absolute paths saved
    /// before the rename (in workflows, history and settings) keep working. Runs once per process; a folder whose
    /// new name already exists is left alone. Called by the default paths of the models, library and server folders.
    public static func migrateFolders() { _ = migrated }

    private static let migrated: Bool = {
        let fm = FileManager.default
        for (old, new) in folderMoves {
            guard let attrs = try? fm.attributesOfItem(atPath: old.path),
                  attrs[.type] as? FileAttributeType == .typeDirectory,
                  !fm.fileExists(atPath: new.path) else { continue }
            do {
                try fm.moveItem(at: old, to: new)
                try? fm.createSymbolicLink(at: old, withDestinationURL: new)
            } catch {
                FileHandle.standardError.write(Data("Couldn't move \(old.path) to \(new.path): \(error)\n".utf8))
            }
        }
        // Models in the default folder are upgraded now; folders elsewhere when they're next loaded.
        let models = folderMoves[0].new.appendingPathComponent("models")
        for dir in (try? fm.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)) ?? [] {
            upgradeModelFolder(dir)
        }
        return true
    }()

    /// Renames `mlxcomfy.json`, `.mlxcomfy-cache` and `mlxcomfy-link.json` in a model folder (and in its direct
    /// subfolders, where multi-model repos keep theirs) to the AIStudio names. Read-only folders are left as they are.
    public static func upgradeModelFolder(_ dir: URL) {
        let fm = FileManager.default
        var dirs = [dir]
        if let subs = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey]) {
            dirs += subs.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        }
        for d in dirs {
            for (old, new) in modelFolderItems {
                let from = d.appendingPathComponent(old), to = d.appendingPathComponent(new)
                guard fm.fileExists(atPath: from.path), !fm.fileExists(atPath: to.path) else { continue }
                try? fm.moveItem(at: from, to: to)
            }
        }
    }

    /// `path` with an old folder prefix (`…/Application Support/MLXComfyUI/…` and so on) replaced by the new one.
    public static func upgradedPath(_ path: String) -> String {
        for (old, new) in folderMoves where path == old.path || path.hasPrefix(old.path + "/") {
            return new.path + path.dropFirst(old.path.count)
        }
        return path
    }

    /// The `AISTUDIO_` environment variable `name`, or its `MLXCOMFY_` predecessor.
    public static func environment(_ name: String) -> String? {
        let env = ProcessInfo.processInfo.environment
        if let v = env[name] { return v }
        guard name.hasPrefix("AISTUDIO_") else { return nil }
        return env["MLXCOMFY_" + name.dropFirst("AISTUDIO_".count)]
    }
}
