import Foundation
import GraphRuntime

/// A saved workflow file in the library.
public struct WorkflowEntry: Identifiable, Hashable, Sendable {
    public var url: URL
    public var modified: Date
    public var id: URL { url }
    /// The file name without `.json`, shown as the workflow's name.
    public var name: String { url.deletingPathExtension().lastPathComponent }
}

/// A folder of workflows, with its subfolders.
public struct WorkflowFolder: Identifiable, Hashable, Sendable {
    public var url: URL
    public var folders: [WorkflowFolder]
    public var workflows: [WorkflowEntry]
    public var id: URL { url }
    public var name: String { url.lastPathComponent }

    /// Every workflow in this folder and below.
    public var allWorkflows: [WorkflowEntry] { workflows + folders.flatMap(\.allWorkflows) }
}

/// The workflows folder: `.json` graphs in nested folders.
public struct WorkflowStore: Sendable {
    public let root: URL
    /// Deletes move files to the Trash; tests delete them outright.
    public var useTrash = true

    public init(root: URL, useTrash: Bool = true) {
        self.root = root.standardizedFileURL
        self.useTrash = useTrash
    }

    // MARK: listing

    public func tree() -> WorkflowFolder { folder(at: root) }

    public func all() -> [WorkflowEntry] { tree().allWorkflows }

    private func folder(at url: URL) -> WorkflowFolder {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
                                                 options: [.skipsHiddenFiles])) ?? []
        var folders: [WorkflowFolder] = []
        var workflows: [WorkflowEntry] = []
        for item in items {
            let values = try? item.resourceValues(forKeys: [.isDirectoryKey, .contentModificationDateKey])
            if values?.isDirectory == true {
                folders.append(folder(at: item))
            } else if item.pathExtension.lowercased() == "json" {
                workflows.append(WorkflowEntry(url: item.standardizedFileURL, modified: values?.contentModificationDate ?? .distantPast))
            }
        }
        let byName: (String, String) -> Bool = { $0.localizedStandardCompare($1) == .orderedAscending }
        return WorkflowFolder(url: url.standardizedFileURL,
                              folders: folders.sorted { byName($0.name, $1.name) },
                              workflows: workflows.sorted { byName($0.name, $1.name) })
    }

    // MARK: names

    /// A file-system-safe workflow name: no path separators or colons, no leading dots, trimmed, never empty.
    public static func sanitize(_ name: String) -> String {
        var s = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        while s.hasPrefix(".") { s.removeFirst() }
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.lowercased().hasSuffix(".json") { s = String(s.dropLast(5)) }
        return s.isEmpty ? "Untitled" : String(s.prefix(120))
    }

    /// `<folder>/<name>.json`, or `<name> 2.json`, `<name> 3.json`… when taken (`excluding` is the file being renamed).
    public func uniqueURL(name: String, in folder: URL? = nil, excluding: URL? = nil) -> URL {
        let dir = folder ?? root
        let base = Self.sanitize(name)
        var n = 1
        while true {
            let candidate = dir.appendingPathComponent(n == 1 ? "\(base).json" : "\(base) \(n).json").standardizedFileURL
            if candidate == excluding?.standardizedFileURL || !FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            n += 1
        }
    }

    // MARK: editing

    /// Writes a new workflow named `name` (made unique) and returns its file.
    @discardableResult
    public func create(_ graph: Graph, name: String, in folder: URL? = nil) throws -> URL {
        let url = uniqueURL(name: name, in: folder)
        try write(graph, to: url)
        return url
    }

    public func write(_ graph: Graph, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try graph.jsonData().write(to: url, options: .atomic)
    }

    /// Renames a workflow file or a folder; returns the new location.
    @discardableResult
    public func rename(_ url: URL, to name: String) throws -> URL {
        let isFolder = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        let dir = url.deletingLastPathComponent()
        let target: URL
        if isFolder {
            target = uniqueFolderURL(name: name, in: dir, excluding: url)
        } else {
            target = uniqueURL(name: name, in: dir, excluding: url)
        }
        guard target.standardizedFileURL != url.standardizedFileURL else { return url }
        try FileManager.default.moveItem(at: url, to: target)
        return target
    }

    /// Copies a workflow next to itself as "<name> 2".
    @discardableResult
    public func duplicate(_ url: URL) throws -> URL {
        // "Fox 2" duplicates to the next free "Fox N", not "Fox 2 2".
        var base = url.deletingPathExtension().lastPathComponent
        if let r = base.range(of: #" \d+$"#, options: .regularExpression) { base.removeSubrange(r) }
        let target = uniqueURL(name: base, in: url.deletingLastPathComponent())
        try FileManager.default.copyItem(at: url, to: target)
        return target
    }

    /// Moves a workflow (or folder) into `folder`, renaming it if the name is taken there.
    @discardableResult
    public func move(_ url: URL, to folder: URL) throws -> URL {
        guard url.deletingLastPathComponent().standardizedFileURL != folder.standardizedFileURL else { return url }
        let isFolder = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        if isFolder, folder.standardizedFileURL.path.hasPrefix(url.standardizedFileURL.path + "/") {
            throw CocoaError(.fileWriteInvalidFileName)  // into itself
        }
        let target = isFolder ? uniqueFolderURL(name: url.lastPathComponent, in: folder)
            : uniqueURL(name: url.deletingPathExtension().lastPathComponent, in: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: url, to: target)
        return target
    }

    @discardableResult
    public func createFolder(named name: String, in parent: URL? = nil) throws -> URL {
        let url = uniqueFolderURL(name: name, in: parent ?? root)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    public func delete(_ url: URL) throws {
        if useTrash { try FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        else { try FileManager.default.removeItem(at: url) }
    }

    func uniqueFolderURL(name: String, in dir: URL, excluding: URL? = nil) -> URL {
        let base = Self.sanitize(name)
        var n = 1
        while true {
            let candidate = dir.appendingPathComponent(n == 1 ? base : "\(base) \(n)", isDirectory: true).standardizedFileURL
            if candidate.path == excluding?.standardizedFileURL.path || !FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            n += 1
        }
    }
}
