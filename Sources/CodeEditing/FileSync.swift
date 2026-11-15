import Foundation
import CryptoKit

public struct FileStamp: Equatable, Sendable, Codable {
    public var mtime: Date
    public var size: Int
    public var inode: UInt64

    public init(mtime: Date, size: Int, inode: UInt64) {
        self.mtime = mtime; self.size = size; self.inode = inode
    }

    /// The file's stamp via FileManager attributes; nil if it's missing.
    public static func of(_ url: URL) -> FileStamp? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let mtime = attrs[.modificationDate] as? Date else { return nil }
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        let inode = (attrs[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        return FileStamp(mtime: mtime, size: size, inode: inode)
    }
}

public struct FileSnapshot: Equatable, Sendable {
    public var stamp: FileStamp
    public var hash: String
    public var text: String

    /// `hash` is the sha256 hex of the text's UTF-8.
    public init(stamp: FileStamp, text: String) {
        self.stamp = stamp
        self.text = text
        self.hash = FileSnapshot.hash(of: text)
    }

    public static func hash(of text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

public enum SyncAction: Equatable, Sendable {
    case none                    // disk is what we based on
    case adoptStamp(FileStamp)   // the stamp moved but the content is the same (touch, our own write echo)
    case reload(String)          // the buffer is clean (or equals the disk): take the disk text
    case merged(String)          // dirty buffer, disjoint changes: the three-way merge result
    case conflict(String)        // dirty buffer, overlapping changes: the disk text, for the user to choose
    case deleted                 // the file is gone
}

public enum FileSync {
    /// base = what the buffer was loaded/saved from; disk = current stamp (nil: missing) and, when the stamp differs,
    /// the disk text; buffer = the editor's current text; dirty = it has unsaved edits.
    public static func decide(base: FileSnapshot, diskStamp: FileStamp?, diskText: String?, buffer: String, dirty: Bool) -> SyncAction {
        guard let diskStamp else { return .deleted }
        if diskStamp == base.stamp { return .none }
        guard let diskText else { return .none }
        if FileSnapshot.hash(of: diskText) == base.hash { return .adoptStamp(diskStamp) }
        if diskText == buffer || !dirty { return .reload(diskText) }
        if let merged = LineMerge.merge(base: base.text, mine: buffer, theirs: diskText) { return .merged(merged) }
        return .conflict(diskText)
    }

    /// Whether a save may overwrite the disk: the stamp still equals the base's (a missing file: yes, saving recreates it).
    public static func canSave(base: FileSnapshot, diskStamp: FileStamp?) -> Bool {
        diskStamp == nil || diskStamp == base.stamp
    }
}
