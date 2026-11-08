import Foundation

/// Where model folders live, and how a model path in a graph or on the command line is found.
public enum ModelPaths {
    /// `~/Library/Application Support/MLXComfyUI/models`: where the Models window downloads to by default.
    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLXComfyUI/models")
    }

    private static let lock = NSLock()
    private static var _directory: URL?

    /// The models folder searched by `resolve`. The app sets it to the folder chosen in the Models window.
    public static var directory: URL {
        get { lock.withLock { _directory } ?? defaultDirectory }
        set { lock.withLock { _directory = newValue } }
    }

    /// A model folder for `path`: the path itself (absolute, `~`, or relative to the working directory) when it
    /// exists, otherwise the folder of the same name in `directory`. So `qwen-turbo-mlx` finds
    /// `<models>/qwen-turbo-mlx`, and graphs saved with an old absolute path still load after the folder moved.
    public static func resolve(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        let direct = (expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded)
            : URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(expanded))
            .standardizedFileURL
        if FileManager.default.fileExists(atPath: direct.path) { return direct }
        let name = direct.lastPathComponent
        guard !name.isEmpty, name != "/" else { return direct }
        let candidate = directory.appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : direct
    }
}
