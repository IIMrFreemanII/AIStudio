import Foundation
import LLMClient
import SwiftUI

/// The text in your message's bubble. A long one (a paste) shows its first lines with a fade and "Show more · 4,512
/// lines · 270 KB"; opened, it scrolls in a box (at most `CollapsedText.inlineLimit` characters laid out). Open shows
/// all of it in the full-window viewer, with line numbers, loaded a part at a time.
struct MessageBubbleText: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let turn: ChatTurn

    var body: some View {
        if let preview = CollapsedText.message(turn.text) {
            let key = "msg:\(turn.id.uuidString)"
            let open = doc.openRuns.contains(key)
            VStack(alignment: .leading, spacing: 8) {
                if open {
                    let big = turn.text.utf16.count > CollapsedText.inlineLimit
                    ScrollView {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(big ? String(turn.text.prefix(CollapsedText.inlineLimit)) : turn.text)
                                .font(.mgBody).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            if big {
                                let rest = String(turn.text.dropFirst(CollapsedText.inlineLimit))
                                Text("… \(CollapsedText.size(rest)) more · Open to see all")
                                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            }
                        }
                    }
                    .frame(maxHeight: 420)
                    .scrollBounceBehavior(.basedOnSize)
                    .mgScrollIndicators()
                } else {
                    Text(preview.text)
                        .font(.mgBody).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        // The text itself fades out (the bubble's colour is see-through: a coloured fade over it
                        // would show as a lighter band).
                        .mask(LinearGradient(stops: [.init(color: .black, location: 0), .init(color: .black, location: 0.6),
                                                     .init(color: .black.opacity(0.15), location: 1)],
                                             startPoint: .top, endPoint: .bottom))
                }
                HStack(spacing: 10) {
                    Button {
                        withAnimation(.easeOut(duration: 0.15)) { doc.toggleOpen(key) }
                    } label: {
                        Label(open ? "Show less" : "Show more · \(preview.lines.formatted()) lines · \(CollapsedText.size(turn.text))",
                              systemImage: open ? "chevron.up" : "chevron.down")
                            .font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgTestAnchor("message-more:\(turn.id.uuidString)")
                    Spacer(minLength: 8)
                    Button { openFull() } label: {
                        Label("Open", systemImage: "arrow.up.left.and.arrow.down.right").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Show the whole message in the viewer")
                    .mgTestAnchor("message-open:\(turn.id.uuidString)")
                }
            }
            .frame(maxWidth: 640)
        } else {
            Text(turn.text).font(.mgBody).textSelection(.enabled)
        }
    }

    /// Writes the message to the chat's files folder once and shows it full-window.
    func openFull() {
        let folder = model.chatStore.filesFolder(doc.id)
        let url = folder.appendingPathComponent("message-\(turn.id.uuidString).txt")
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try? turn.text.write(to: url, atomically: true, encoding: .utf8)
        }
        model.chatViewer = ChatViewerItem(urls: [url], index: 0, titles: ["Your message"])
    }
}

/// The first lines of a long text, shown collapsed (a plan, a long message). Only the preview is laid out, so a long
/// text doesn't slow the chat down; the text is scanned once for its line count, never split whole.
enum CollapsedText {
    struct Preview {
        /// The lines shown.
        var text: String
        /// The text's lines in all, and how many aren't shown.
        var lines: Int
        var hidden: Int
    }

    /// Nil when the text is short enough to show whole: no more than `overLines` lines and `overChars` characters
    /// (UTF-16 units, cheap to count). Otherwise its first `showLines` lines, at most `showChars` characters, cut at a
    /// word, without a heading or blank line at the end.
    static func of(_ text: String, overLines: Int, overChars: Int, showLines: Int, showChars: Int) -> Preview? {
        let lines = lineCount(text)
        guard lines > overLines || text.utf16.count > overChars else { return nil }
        var shown: [Substring] = []
        var chars = 0
        var rest = text[...]
        while shown.count < showLines, !rest.isEmpty {
            let end = rest.firstIndex(of: "\n") ?? rest.endIndex
            let l = rest[..<end]
            rest = end < rest.endIndex ? rest[rest.index(after: end)...] : rest[rest.endIndex...]
            if chars + l.utf16.count > showChars {
                // A very long line (one huge paragraph): its start, cut at a word.
                let head = l.prefix(max(0, showChars - chars))
                let cut = head.lastIndex(of: " ").map { head[..<$0] } ?? head
                if !cut.isEmpty { shown.append(Substring(cut + "…")) }
                break
            }
            shown.append(l)
            chars += l.utf16.count + 1
        }
        // Not cut right after a heading: it would end the preview with a title and nothing under it.
        while let last = shown.last, shown.count > 1,
              last.trimmingCharacters(in: .whitespaces).hasPrefix("#") || last.trimmingCharacters(in: .whitespaces).isEmpty {
            shown.removeLast()
        }
        return Preview(text: shown.joined(separator: "\n"), lines: lines, hidden: max(0, lines - shown.count))
    }

    /// A long message you sent: collapsed above 12 lines or 1,200 characters, about 8 lines shown.
    static func message(_ text: String) -> Preview? {
        of(text, overLines: 12, overChars: 1200, showLines: 8, showChars: 900)
    }

    /// The most of a message laid out inline when it's opened; the rest is in the full-window viewer.
    static let inlineLimit = 20_000

    static func lineCount(_ text: String) -> Int {
        var n = 1
        for b in text.utf8 where b == 10 { n += 1 }
        return n
    }

    /// "270 KB", for a footer.
    static func size(_ text: String) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(text.utf8.count), countStyle: .file)
    }
}
