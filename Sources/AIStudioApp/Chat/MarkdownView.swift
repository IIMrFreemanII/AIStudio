import AIStudioLibrary
import AppKit
import SwiftUI

/// A model's reply as Markdown: the blocks from `Markdown.blocks`, inline styles from `AttributedString(markdown:)`,
/// selectable text, code blocks with a Copy button.
///
/// Runs of prose (paragraphs, headings, quotes) are one `Text` each, not one per paragraph: a long reply was dozens
/// of views, each laid out, measured and given its own selection overlay, which made opening and
/// scrolling a long chat slow. As one text, a selection also spans paragraphs. Parsed replies are cached by text.
struct MarkdownView: View {
    let text: String
    @Environment(\.chatFiles) private var files

    var body: some View {
        let segments = MarkdownSegments.of(text, files: files.index)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(segments.indices, id: \.self) { i in
                switch segments[i] {
                case .prose(let a):
                    Text(a).font(.mgBody).fixedSize(horizontal: false, vertical: true)
                case .block(let block):
                    MarkdownBlockView(block: block)
                }
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A reply split into prose (one attributed text per run of paragraphs, headings and quotes) and the blocks that stay
/// views of their own (lists, code with its Copy button, tables, rules).
enum MarkdownSegment {
    case prose(AttributedString)
    case block(MarkdownBlock)
}

enum MarkdownSegments {
    private final class Box {
        let segments: [MarkdownSegment]
        init(_ s: [MarkdownSegment]) { segments = s }
    }

    /// Bounded by size: a streaming reply adds an entry per update.
    private static let cache: NSCache<NSString, Box> = {
        let c = NSCache<NSString, Box>()
        c.totalCostLimit = 16 << 20
        return c
    }()

    /// `files`: a coding chat's project files; paths to them in the text become links (`ChatLinks`).
    @MainActor static func of(_ text: String, files: ProjectFileIndex? = nil) -> [MarkdownSegment] {
        let key = (files.map { "\($0.stamp)\u{1}" } ?? "") + text
        if let hit = cache.object(forKey: key as NSString) { return hit.segments }
        let segments = make(Markdown.blocks(text), files: files)
        cache.setObject(Box(segments), forKey: key as NSString, cost: text.utf8.count * 4)
        return segments
    }

    @MainActor static func make(_ blocks: [MarkdownBlock], files: ProjectFileIndex? = nil) -> [MarkdownSegment] {
        var out: [MarkdownSegment] = []
        var run: [MarkdownBlock] = []
        func flush() {
            if !run.isEmpty { out.append(.prose(prose(run, files: files))) }
            run = []
        }
        for b in blocks {
            switch b {
            case .heading, .paragraph, .quote: run.append(b)
            // A list keeps its marker column: SwiftUI's Text ignores a paragraph's head indent, so as merged text a
            // wrapped item's second line started under the marker.
            case .list, .code, .table, .rule, .media:
                flush()
                out.append(.block(b))
            }
        }
        flush()
        return out
    }

    /// Column `c` of a table's rows as one text, a line per row (a missing cell is an empty line).
    @MainActor static func column(_ rows: [[String]], _ c: Int, files: ProjectFileIndex? = nil) -> AttributedString {
        var out = AttributedString()
        for (i, row) in rows.enumerated() {
            if i > 0 { out += AttributedString("\n") }
            out += inlineMarkdown(c < row.count ? row[c] : "", files: files)
        }
        return out
    }

    /// Paragraphs as one attributed string: a short empty line between blocks (as the old 10 pt spacing), headings in
    /// their fonts, quote bars and text in the secondary colour.
    @MainActor static func prose(_ blocks: [MarkdownBlock], files: ProjectFileIndex? = nil) -> AttributedString {
        var out = AttributedString()
        var gap = AttributedString("\n\n")
        gap.font = .system(size: 5)
        for (i, b) in blocks.enumerated() {
            if i > 0 { out += gap }
            switch b {
            case .heading(let level, let text):
                var a = inlineMarkdown(text, files: files)
                a.font = level <= 1 ? .mgTitle3 : level == 2 ? .mgHeadline : .mgBody.weight(.semibold)
                out += a
            case .paragraph(let text):
                out += inlineMarkdown(text, files: files)
            case .quote(let text):
                for (j, l) in text.components(separatedBy: "\n").enumerated() {
                    if j > 0 { out += AttributedString("\n") }
                    var bar = AttributedString("┃\u{2002}")
                    bar.foregroundColor = MG.separator
                    var body = inlineMarkdown(l, files: files)
                    body.foregroundColor = MG.secondaryLabel
                    out += bar + body
                }
            case .list, .code, .table, .rule, .media:
                break
            }
        }
        return out
    }
}

/// Inline Markdown (emphasis, code spans, links); falls back to the plain text. With a project's files, code spans,
/// relative links and bare paths in the text naming one of them (`Sources/a.swift`, `a.swift:42`) link to it
/// (`ChatLinks`).
func inlineMarkdown(_ s: String, files: ProjectFileIndex? = nil) -> AttributedString {
    let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace,
                                                          failurePolicy: .returnPartiallyParsedIfPossible)
    var a = (try? AttributedString(markdown: s, options: options)) ?? AttributedString(s)
    var bare: [(Range<AttributedString.Index>, URL)] = []
    for run in a.runs {
        let code = run.inlinePresentationIntent?.contains(.code) == true
        if code { a[run.range].backgroundColor = MG.fill }
        guard let files else { continue }
        if let link = run.link {
            // A relative link (`[the view](Sources/View.swift)`): to the project file.
            if link.scheme == nil, let hit = ChatLinks.resolve(link.relativeString.removingPercentEncoding ?? link.relativeString, in: files) {
                a[run.range].link = ChatLinks.url(path: hit.path, line: hit.line)
            }
        } else if code, let hit = ChatLinks.resolve(String(a[run.range].characters), in: files) {
            a[run.range].link = ChatLinks.url(path: hit.path, line: hit.line)
        } else if !code {
            // Small models rarely put paths in backticks: "the bug is in src/greet.py" links too.
            let text = String(a[run.range].characters)
            for (r, url) in ChatLinks.barePaths(in: text, files: files) {
                let lo = a.characters.index(run.range.lowerBound, offsetBy: text.distance(from: text.startIndex, to: r.lowerBound))
                let hi = a.characters.index(lo, offsetBy: text.distance(from: r.lowerBound, to: r.upperBound))
                bare.append((lo..<hi, url))
            }
        }
    }
    for (r, url) in bare { a[r].link = url }
    return a
}

/// What a chat's Markdown resolves paths against: the coding chat's project files (links to them), its folder
/// (relative images), the chat's files folder (attachments).
struct ChatFilesContext {
    var index: ProjectFileIndex?
    var folder: URL?
    var root: URL?

    /// An image's or video's `src`: a web URL, a file URL, an absolute path, or a path relative to the project (else
    /// the chat's folder).
    func resolve(_ src: String) -> URL? {
        if let u = URL(string: src), let scheme = u.scheme?.lowercased() {
            return ["http", "https", "file"].contains(scheme) ? u : nil
        }
        let p = src.removingPercentEncoding ?? src
        if p.hasPrefix("/") { return URL(fileURLWithPath: p) }
        if p.hasPrefix("~/") { return URL(fileURLWithPath: (p as NSString).expandingTildeInPath) }
        if let root { return root.appendingPathComponent(p.hasPrefix("./") ? String(p.dropFirst(2)) : p) }
        return folder?.appendingPathComponent(p)
    }
}

private struct ChatFilesKey: EnvironmentKey {
    static let defaultValue = ChatFilesContext()
}

extension EnvironmentValues {
    var chatFiles: ChatFilesContext {
        get { self[ChatFilesKey.self] }
        set { self[ChatFilesKey.self] = newValue }
    }
    /// Runs shell commands in the chat's terminal: set by coding chats, so their ```bash blocks get a Run button.
    @Entry var chatRunCommand: (@MainActor (String) -> Void)? = nil
}

struct MarkdownBlockView: View {
    let block: MarkdownBlock
    @Environment(\.chatFiles) private var files

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(inlineMarkdown(text, files: files.index))
                .font(level <= 1 ? .mgTitle3 : level == 2 ? .mgHeadline : .mgBody.weight(.semibold))
                .padding(.top, level <= 2 ? 4 : 0)
        case .paragraph(let text):
            Text(inlineMarkdown(text, files: files.index)).font(.mgBody).fixedSize(horizontal: false, vertical: true)
        case .code(let language, let text, let closed):
            CodeBlockView(language: language, code: text, closed: closed)
        case .list(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Group {
                            if let checked = item.checked {
                                Image(systemName: checked ? "checkmark.square" : "square")
                            } else if let n = item.number {
                                Text("\(n).")
                            } else {
                                Text(item.depth == 0 ? "•" : "◦")
                            }
                        }
                        .font(.mgBody).foregroundStyle(MG.secondaryLabel)
                        .frame(minWidth: 14, alignment: .trailing)
                        .textSelection(.disabled)  // a selection overlay each costs; the item text stays selectable
                        Text(inlineMarkdown(item.text, files: files.index)).font(.mgBody).fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.depth) * 18)
                }
            }
        case .quote(let text):
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 1.5).fill(MG.separator).frame(width: 3)
                Text(inlineMarkdown(text, files: files.index)).font(.mgBody).foregroundStyle(MG.secondaryLabel)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .table(let header, let rows):
            // Each column's body is one text (a line per row), not a text per cell: every selectable text gets its
            // own selection overlay, and a cell each made a long table slow to show. Cells don't wrap here (the
            // table scrolls sideways), so the rows line up across columns.
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        ForEach(Array(header.enumerated()), id: \.offset) { _, h in
                            Text(inlineMarkdown(h)).font(.mgCallout.weight(.semibold))
                        }
                    }
                    Divider()
                    if !rows.isEmpty {
                        GridRow {
                            ForEach(0..<max(header.count, rows.map(\.count).max() ?? 0), id: \.self) { c in
                                Text(MarkdownSegments.column(rows, c, files: files.index)).font(.mgCallout).lineSpacing(6)
                            }
                        }
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
            }
        case .rule:
            MGDivider()
        case .media(let alt, let src):
            ChatMediaBlock(alt: alt, url: files.resolve(src), src: src)
        }
    }
}

struct CodeBlockView: View {
    let language: String
    let code: String
    /// False while the reply is still streaming the block: no Run for a half-written command.
    var closed = true
    @Environment(\.chatRunCommand) private var runCommand
    @State private var copied = false
    @State private var sent = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer()
                if closed, let runCommand, let commands = ShellBlock.commands(language: language, code: code) {
                    Button {
                        runCommand(commands)
                        sent = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { sent = false }
                    } label: {
                        Label(sent ? "Sent" : "Run", systemImage: sent ? "checkmark" : "play").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Run in this chat's terminal")
                    .mgTestAnchor("code-run:" + (commands.components(separatedBy: "\n").first ?? ""))
                }
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.mgFootnote)
                }
                .buttonStyle(.mg(.borderless))
                if ChatQuoting.active {
                    Button { ChatQuoting.post(code) } label: {
                        Label("Reply", systemImage: "arrowshape.turn.up.left").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Reply to this code")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            MGDivider()
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code).font(.mgMono).fixedSize(horizontal: true, vertical: true).padding(10)
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
    }
}
