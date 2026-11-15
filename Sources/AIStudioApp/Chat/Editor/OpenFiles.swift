import CodeEditing
import Foundation
import Observation

/// A coding chat's open files: the tabs (saved with the chat) and an editable, watched buffer per text file, made when
/// the file is first shown.
@MainActor @Observable
final class OpenFiles {
    var list = FileTabList()
    /// Made on demand (also while a view's body runs), so not observed.
    @ObservationIgnored private(set) var buffers: [String: FileBuffer] = [:]
    /// Set up for each new buffer (the owner wires its callbacks).
    @ObservationIgnored var wire: ((String, FileBuffer) -> Void)?

    /// The editable, watched buffer for a project file (text and Markdown only; nil for images, PDFs…), made on first use.
    func buffer(_ rel: String, root: URL) -> FileBuffer? {
        if let b = buffers[rel] { return b }
        let url = root.appendingPathComponent(rel)
        let kind = PreviewKind.of(url)
        guard kind == .text || kind == .markdown else { return nil }
        let b = FileBuffer(url: url, relPath: rel, editable: true, watch: true)
        buffers[rel] = b
        wire?(rel, b)
        return b
    }

    /// Closes the file's buffer (saving it) and drops it.
    func remove(_ rel: String) {
        buffers.removeValue(forKey: rel)?.close()
    }

    /// Closes every buffer (saving them) and drops them; the tabs stay.
    func removeAll() {
        let all = buffers.values
        buffers = [:]
        all.forEach { $0.close() }
    }

    var all: [FileBuffer] { Array(buffers.values) }
}
