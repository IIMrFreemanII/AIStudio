import Foundation

/// Where the user's workflows, presets and input assets live, and where open tabs are kept between launches.
///
/// The library is a visible folder (`~/Documents/AIStudio` by default) so it is easy to browse, back up or sync;
/// the session (open tabs and their unsaved drafts) is app state and lives in Application Support.
public struct LibraryPaths: Sendable, Equatable {
    public var root: URL
    public var sessionDirectory: URL

    public var workflows: URL { root.appendingPathComponent("workflows", isDirectory: true) }
    public var presets: URL { root.appendingPathComponent("presets", isDirectory: true) }
    public var input: URL { root.appendingPathComponent("input", isDirectory: true) }
    /// Saved chats: `<id>.json`, and `<id>/` for their attachments.
    public var chats: URL { root.appendingPathComponent("chats", isDirectory: true) }

    public static var defaultRoot: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AIStudio", isDirectory: true)
    }

    public static var defaultSessionDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AIStudio/session", isDirectory: true)
    }

    /// A custom `root` without a session directory keeps its session inside it (`.session/`), so a throwaway library
    /// (smoke tests) never touches the real open tabs.
    public init(root: URL? = nil, sessionDirectory: URL? = nil) {
        let r = (root ?? Self.defaultRoot).standardizedFileURL
        self.root = r
        self.sessionDirectory = sessionDirectory
            ?? (root == nil ? Self.defaultSessionDirectory : r.appendingPathComponent(".session", isDirectory: true))
    }

    /// Creates the folders that don't exist yet.
    public func ensure() {
        for dir in [workflows, presets, input, chats, sessionDirectory] {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }
}
