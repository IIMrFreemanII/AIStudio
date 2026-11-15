import Foundation

public struct FileTabList: Codable, Equatable, Sendable {
    public private(set) var tabs: [String]          // project-relative paths, in order
    public private(set) var transient: String?      // the one tab a single click opened (replaced by the next), nil if none
    public var active: String?

    public init(tabs: [String] = [], active: String? = nil) {
        self.tabs = tabs
        self.active = active
        self.transient = nil
    }

    /// Opens `path` and makes it active. Already open: just activates it (and pins it if `pin`). Not open and !pin:
    /// replaces the current transient tab in place (same index) if there is one, else is inserted right after the active
    /// tab (or at the end) and becomes the transient one. pin: inserted after the active tab, not transient.
    public mutating func open(_ path: String, pin: Bool) {
        defer { active = path }
        if tabs.contains(path) {
            if pin, transient == path { transient = nil }
            return
        }
        if !pin, let t = transient, let i = tabs.firstIndex(of: t) {
            tabs[i] = path
            transient = path
            return
        }
        tabs.insert(path, at: insertionIndex())
        if !pin { transient = path }
    }

    /// The transient tab becomes a normal one (on edit or double-click). No-op otherwise.
    public mutating func pin(_ path: String) {
        if transient == path { transient = nil }
    }

    /// Closes a tab; if it was active, the tab to its right becomes active, else the one to its left, else nil.
    public mutating func close(_ path: String) {
        guard let i = tabs.firstIndex(of: path) else { return }
        tabs.remove(at: i)
        if transient == path { transient = nil }
        if active == path {
            active = i < tabs.count ? tabs[i] : (i > 0 ? tabs[i - 1] : nil)
        }
    }

    /// A file was renamed or moved: its tab keeps its place. If `new` is already open, the old tab is dropped instead.
    public mutating func rename(_ old: String, to new: String) {
        guard old != new, let i = tabs.firstIndex(of: old) else { return }
        if tabs.contains(new) { tabs.remove(at: i) } else { tabs[i] = new }
        if transient == old { transient = new }
        if active == old { active = new }
    }

    /// Drops tabs whose paths aren't in `existing` (deleted files), fixing `active` as close() does.
    public mutating func keepOnly(_ existing: Set<String>) {
        if let a = active, let i = tabs.firstIndex(of: a), !existing.contains(a) {
            active = tabs[(i + 1)...].first(where: existing.contains) ?? tabs[..<i].last(where: existing.contains)
        }
        tabs.removeAll { !existing.contains($0) }
        if let t = transient, !tabs.contains(t) { transient = nil }
        if let a = active, !tabs.contains(a) { active = nil }
    }

    public func isTransient(_ path: String) -> Bool { transient == path }

    private func insertionIndex() -> Int {
        if let a = active, let i = tabs.firstIndex(of: a) { return i + 1 }
        return tabs.count
    }
}
