import Foundation

/// A flow script saved as a `.js` file.
public struct SavedFlow: Sendable, Equatable {
    public var name: String
    public var meta: FlowMeta
    public var url: URL
    public var personal: Bool

    public init(name: String, meta: FlowMeta, url: URL, personal: Bool) {
        self.name = name
        self.meta = meta
        self.url = url
        self.personal = personal
    }
}

/// Saved flows: `<project>/.aistudio/flows/*.js` and the personal folder.
public enum FlowStore {
    /// Tests point this at a temporary folder.
    nonisolated(unsafe) public static var personalOverride: URL?

    /// ~/Library/Application Support/AIStudio/flows
    public static var personalFolder: URL {
        if let personalOverride { return personalOverride }
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("AIStudio/flows", isDirectory: true)
    }

    /// The project's folder first, then the personal one.
    public static func folders(root: URL?) -> [(URL, personal: Bool)] {
        var out: [(URL, personal: Bool)] = []
        if let root { out.append((root.appendingPathComponent(".aistudio/flows", isDirectory: true), false)) }
        out.append((personalFolder, true))
        return out
    }

    /// The `*.js` files whose meta parses; the name is the file name without `.js`; a project flow wins over a
    /// personal one with the same name. Sorted by name.
    public static func list(root: URL?) -> [SavedFlow] {
        var seen = Set<String>()
        var out: [SavedFlow] = []
        for (folder, personal) in folders(root: root) {
            let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            for url in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where url.pathExtension == "js" {
                let name = url.deletingPathExtension().lastPathComponent
                guard !seen.contains(name), let text = try? String(contentsOf: url, encoding: .utf8),
                      let meta = try? FlowScript.meta(text) else { continue }
                seen.insert(name)
                out.append(SavedFlow(name: name, meta: meta, url: url, personal: personal))
            }
        }
        return out.sorted { $0.name < $1.name }
    }

    public static func load(_ name: String, root: URL?) -> String? {
        guard valid(name) else { return nil }
        for (folder, _) in folders(root: root) {
            if let text = try? String(contentsOf: folder.appendingPathComponent(name + ".js"), encoding: .utf8) { return text }
        }
        return nil
    }

    private static func valid(_ name: String) -> Bool {
        !name.isEmpty && !name.contains("/") && !name.hasPrefix(".")
    }

    @discardableResult
    public static func save(_ name: String, script: String, root: URL?, personal: Bool) throws -> URL {
        guard valid(name) else { throw ToolError("\"\(name)\" isn't a usable flow name") }
        let folder: URL
        if personal || root == nil {
            folder = personalFolder
        } else {
            folder = root!.appendingPathComponent(".aistudio/flows", isDirectory: true)
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent(name + ".js")
        try script.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    /// For the system prompt: "" when there are none.
    public static func promptSection(_ flows: [SavedFlow]) -> String {
        guard !flows.isEmpty else { return "" }
        let lines = flows.map { f -> String in
            let d = f.meta.description.replacingOccurrences(of: "\n", with: " ")
            return "- \(f.name): \(d.count > 200 ? String(d.prefix(200)) + "…" : d)"
        }
        return "Saved flows (run with flow {name}):\n" + lines.joined(separator: "\n")
    }
}
