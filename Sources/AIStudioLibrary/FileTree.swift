import Foundation

/// A project's files as a folder tree, for the coding chat's file explorer, and flattened into the rows shown
/// (folders first, then files, each by name; only open folders' contents).
public struct FileTree: Sendable, Equatable {
    public struct Node: Sendable, Equatable {
        public var name: String
        /// Relative to the project ("" for the root).
        public var path: String
        public var isFolder: Bool
        public var children: [Node]
    }

    public struct Row: Sendable, Equatable, Identifiable {
        public var id: String { path }
        public var path: String
        public var name: String
        public var depth: Int
        public var isFolder: Bool
        /// A folder's row: shown open (in the expanded set, or while filtering).
        public var open: Bool
    }

    public var root: Node
    /// Every file's path, for lookups (is this path in the project?).
    public var files: Set<String>

    /// From relative file paths (`a/b/c.swift`); folders come from the paths.
    public init(paths: [String]) {
        final class Box {
            var name: String, path: String
            var folders: [String: Box] = [:]
            var files: [String] = []
            init(_ name: String, _ path: String) { self.name = name; self.path = path }
        }
        let top = Box("", "")
        var all = Set<String>()
        for p in paths {
            let parts = p.split(separator: "/").map(String.init)
            guard !parts.isEmpty else { continue }
            all.insert(parts.joined(separator: "/"))
            var cur = top
            for f in parts.dropLast() {
                let path = cur.path.isEmpty ? f : cur.path + "/" + f
                if let next = cur.folders[f] { cur = next } else {
                    let next = Box(f, path)
                    cur.folders[f] = next
                    cur = next
                }
            }
            cur.files.append(parts.last!)
        }
        func order(_ a: String, _ b: String) -> Bool {
            a.localizedStandardCompare(b) == .orderedAscending
        }
        func node(_ b: Box) -> Node {
            let folders = b.folders.values.sorted { order($0.name, $1.name) }.map(node)
            let files = Set(b.files).sorted(by: order).map {
                Node(name: $0, path: b.path.isEmpty ? $0 : b.path + "/" + $0, isFolder: false, children: [])
            }
            return Node(name: b.name, path: b.path, isFolder: true, children: folders + files)
        }
        root = node(top)
        files = all
    }

    /// The rows shown: the open folders' contents. With a filter (case-insensitive, matched against the path), only
    /// matching files and the folders holding them, all open.
    public func rows(expanded: Set<String>, filter: String = "") -> [Row] {
        let q = filter.trimmingCharacters(in: .whitespaces).lowercased()
        var out: [Row] = []
        func matches(_ n: Node) -> Bool {
            n.isFolder ? n.children.contains(where: matches) : n.path.lowercased().contains(q)
        }
        func walk(_ n: Node, _ depth: Int) {
            for c in n.children {
                if !q.isEmpty, !matches(c) { continue }
                let open = c.isFolder && (!q.isEmpty || expanded.contains(c.path))
                out.append(Row(path: c.path, name: c.name, depth: depth, isFolder: c.isFolder, open: open))
                if open { walk(c, depth + 1) }
            }
        }
        walk(root, 0)
        return out
    }

    /// The folders holding a path ("a", "a/b" for "a/b/c.swift"): open them to show it.
    public static func parents(of path: String) -> [String] {
        let parts = path.split(separator: "/").dropLast()
        return parts.indices.map { parts[parts.startIndex...$0].joined(separator: "/") }
    }
}
