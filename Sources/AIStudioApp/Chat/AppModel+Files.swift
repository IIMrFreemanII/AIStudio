import AIStudioLibrary
import Agent
import AppKit
import Foundation
import LLMClient
import UniformTypeIdentifiers

/// The side panel's Files tab: the project's files (what `list_files` sees: git's tracked and untracked files without
/// ignored ones), a previewed file, and links to files in replies.
extension AppModel {
    /// Lists the project's files again (a listing in progress runs once more when it ends).
    func refreshProjectFiles(_ doc: ChatDocument) {
        guard doc.isCoding, let root = doc.projectURL else { return }
        if doc.filesRefreshing {
            doc.filesRefreshAgain = true
            return
        }
        doc.filesRefreshing = true
        Task { @MainActor [weak doc] in
            repeat {
                doc?.filesRefreshAgain = false
                let paths = await ProjectFiles.list(in: root)
                let (tree, index) = await Task.detached {
                    let t = FileTree(paths: paths)
                    return (t, ProjectFileIndex(root: root.standardizedFileURL.path, paths: t.files))
                }.value
                guard let doc else { return }
                if doc.fileTree != tree {
                    doc.fileTree = tree
                    doc.fileIndex = index
                }
            } while doc?.filesRefreshAgain == true
            doc?.filesRefreshing = false
        }
    }

    /// After a tool call that may have changed files: list them again (if they were listed) and re-read the preview.
    func projectFilesChanged(_ doc: ChatDocument) {
        doc.filesRevision += 1
        if doc.fileTree != nil { refreshProjectFiles(doc) }
    }

    /// Opens the Files tab on a project file (a path relative to the project, or absolute inside it), its folders
    /// open, at a line.
    func showProjectFile(_ doc: ChatDocument, path: String, line: Int? = nil) {
        guard let rel = relativeProjectPath(doc, path) else { return }
        doc.previewPath = rel
        doc.previewLine = line
        doc.fileFilter = ""
        doc.expandedFolders.formUnion(FileTree.parents(of: rel))
        if doc.sidePanel != .files { showSidePanel(doc, .files) } else if doc.fileTree == nil { refreshProjectFiles(doc) }
    }

    /// A path as the project's relative path, when it's inside the project.
    func relativeProjectPath(_ doc: ChatDocument, _ path: String) -> String? {
        guard let root = doc.projectURL?.standardizedFileURL.path else { return nil }
        var p = path
        if p.hasPrefix("file://"), let u = URL(string: p) { p = u.path }
        if p.hasPrefix("/") {
            let std = URL(fileURLWithPath: p).standardizedFileURL.path
            guard std.hasPrefix(root + "/") else { return nil }
            return String(std.dropFirst(root.count + 1))
        }
        if p.hasPrefix("./") { p = String(p.dropFirst(2)) }
        return p.isEmpty ? nil : p
    }

    /// Adds `path` (in backticks) to the message being written.
    func mentionInDraft(_ doc: ChatDocument, _ path: String) {
        let sep = doc.draft.isEmpty || doc.draft.hasSuffix(" ") || doc.draft.hasSuffix("\n") ? "" : " "
        doc.draft += sep + "`\(path)` "
    }

    /// A link the chat's text was clicked on: a project file (`aistudio-file:`) opens in the Files tab, a media file
    /// in the viewer. Returns false for links the system should open.
    func openChatLink(_ url: URL, in doc: ChatDocument) -> Bool {
        if url.scheme == ChatLinks.scheme {
            guard let c = URLComponents(url: url, resolvingAgainstBaseURL: false),
                  let path = c.queryItems?.first(where: { $0.name == "path" })?.value else { return true }
            let line = c.queryItems?.first(where: { $0.name == "line" })?.value.flatMap(Int.init)
            if doc.isCoding { showProjectFile(doc, path: path, line: line) }
            return true
        }
        // Web links in a coding chat open in the side panel's Browser tab (⌘-click: the default browser).
        if doc.isCoding, ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
           !NSEvent.modifierFlags.contains(.command) {
            openWebPage(doc, url)
            return true
        }
        if url.isFileURL {
            if doc.isCoding, relativeProjectPath(doc, url.path) != nil {
                showProjectFile(doc, path: url.path)
            } else {
                chatViewer = ChatViewerItem(urls: [url], index: 0)
            }
            return true
        }
        return false
    }
}

/// Links to project files in a chat's text: `aistudio-file://open?path=Sources/a.swift&line=42`.
enum ChatLinks {
    static let scheme = "aistudio-file"

    static func url(path: String, line: Int?) -> URL? {
        var c = URLComponents()
        c.scheme = scheme
        c.host = "open"
        c.queryItems = [URLQueryItem(name: "path", value: path)] + (line.map { [URLQueryItem(name: "line", value: "\($0)")] } ?? [])
        return c.url
    }

    private static let pathLike = try! NSRegularExpression(pattern: #"(?<![\w/.-])[\w./-]*\w\.[A-Za-z0-9]+(?::\d+)?"#)

    /// Paths in plain text (`src/greet.py`, `README.md`, `a.swift:42`) that name one of the project's files, with
    /// their links.
    static func barePaths(in text: String, files: ProjectFileIndex) -> [(Range<String.Index>, URL)] {
        guard text.contains(".") else { return [] }
        let ns = text as NSString
        return pathLike.matches(in: text, range: NSRange(location: 0, length: ns.length)).compactMap { m in
            guard let r = Range(m.range, in: text), let hit = resolve(String(text[r]), in: files),
                  let url = url(path: hit.path, line: hit.line) else { return nil }
            return (r, url)
        }
    }

    /// `Sources/a.swift`, `a.swift:42`, `./x.md:3:5`, `a.swift#L42` as a path and a line, when the path is one of the
    /// project's files (or the file name of exactly one of them).
    static func resolve(_ text: String, in files: ProjectFileIndex) -> (path: String, line: Int?)? {
        var s = text.trimmingCharacters(in: .whitespaces)
        guard !s.isEmpty, s.count < 300, !s.contains(" "), !s.contains("://") else { return nil }
        if s.hasPrefix("./") { s = String(s.dropFirst(2)) }
        var line: Int?
        if let r = s.range(of: "#L"), let n = Int(s[r.upperBound...].prefix { $0.isNumber }) {
            line = n
            s = String(s[..<r.lowerBound])
        }
        let parts = s.split(separator: ":", omittingEmptySubsequences: false)
        if parts.count >= 2, let n = Int(parts[1]) {
            line = n
            s = String(parts[0])
        }
        if files.paths.contains(s) { return (s, line) }
        if !s.contains("/"), let only = files.byName[s], only.count == 1 { return (only[0], line) }
        // An absolute path inside the project.
        if s.hasPrefix(files.root + "/") {
            let rel = String(s.dropFirst(files.root.count + 1))
            if files.paths.contains(rel) { return (rel, line) }
        }
        return nil
    }
}

/// A project's files for resolving paths in text: by path, and by file name.
struct ProjectFileIndex: Equatable, Sendable {
    let root: String
    let paths: Set<String>
    let byName: [String: [String]]
    /// Changes with the files (part of the key replies' parsed Markdown is cached under).
    let stamp: Int

    init(root: String, paths: Set<String>) {
        self.root = root
        self.paths = paths
        stamp = paths.hashValue ^ root.hashValue
        var names: [String: [String]] = [:]
        for p in paths { names[(p as NSString).lastPathComponent, default: []].append(p) }
        byName = names
    }

    /// The project files an "@word" in the composer may mean, best first: the name starts with it, then the name
    /// contains it, then the path does, then the path has its letters in order. Shorter paths first among equals; with
    /// nothing typed yet, the shortest paths.
    func matches(_ query: String, limit: Int = 8) -> [String] {
        let q = query.lowercased()
        func subsequence(_ text: String) -> Bool {
            var it = q.makeIterator()
            var want = it.next()
            for c in text where c == want { want = it.next() }
            return want == nil
        }
        var scored: [(score: Int, path: String)] = []
        for p in paths {
            let lower = p.lowercased()
            let name = (lower as NSString).lastPathComponent
            let score: Int
            if q.isEmpty || name.hasPrefix(q) { score = 0 }
            else if name.contains(q) { score = 1 }
            else if lower.contains(q) { score = 2 }
            else if subsequence(lower) { score = 3 }
            else { continue }
            scored.append((score, p))
        }
        return scored.sorted {
            ($0.score, $0.path.count, $0.path) < ($1.score, $1.path.count, $1.path)
        }.prefix(limit).map(\.path)
    }
}

/// Icons for files by kind, cached by extension (the explorer asks for one per row).
@MainActor enum FileIcons {
    private static var cache: [String: NSImage] = [:]

    static func icon(for path: String, folder: Bool = false) -> NSImage {
        let ext = folder ? "/" : (path as NSString).pathExtension.lowercased()
        if let i = cache[ext] { return i }
        let type: UTType = folder ? .folder : (UTType(filenameExtension: ext) ?? .data)
        let i = NSWorkspace.shared.icon(for: type)
        cache[ext] = i
        return i
    }
}
