import Foundation

public enum EditorMode: Equatable, Sendable {
    case editable            // highlighted and editable
    case plainEditable       // editable, no highlighting, wrapped (a line over maxLineLength)
    case readOnly(String)    // shown in full, plain, not editable; the reason for a bar
    case partial(String)     // only the first `editableBytes` shown, read-only; the note
    case binary              // not text (shown by Quick Look)
}

public enum EditorLimits {
    public static let editableBytes = 2 << 20       // 2 MB
    public static let readableBytes = 16 << 20      // 16 MB
    public static let maxLineLength = 20_000        // UTF-16 units

    /// `head` = the first bytes of the file (up to editableBytes + 1, or the whole file if smaller); `size` = its full size.
    /// For `.readOnly` with a size between 2 and 16 MB the caller reads the whole file.
    public static func mode(head: Data, size: Int) -> EditorMode {
        if head.prefix(8192).contains(0) { return .binary }
        if size > readableBytes {
            return .partial("Showing the first 2 MB of \(sizeString(size)).")
        }
        if size > editableBytes {
            return .readOnly("This file is \(sizeString(size)), too big to edit here.")
        }
        guard let text = String(data: head, encoding: .utf8) else {
            return .readOnly("This file isn't UTF-8 text: read-only.")
        }
        var run = 0
        for unit in text.utf16 {
            if unit == 10 || unit == 13 { run = 0; continue }
            run += 1
            if run > maxLineLength { return .plainEditable }
        }
        return .editable
    }

    private static func sizeString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
