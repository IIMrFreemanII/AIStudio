import AVKit
import AppKit
import PDFKit
import QuickLookUI
import SwiftUI
import UniformTypeIdentifiers

/// What kind of viewer a file gets.
enum PreviewKind: Equatable {
    case image, video, audio, pdf, markdown, text, other

    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "webm", "avi", "mkv"]

    /// By the file's type; files of no known kind are read as text when they have no NUL bytes.
    static func of(_ url: URL) -> PreviewKind {
        let ext = url.pathExtension.lowercased()
        if ["md", "markdown", "mdown"].contains(ext) { return .markdown }
        if videoExtensions.contains(ext) { return .video }
        if let t = UTType(filenameExtension: ext) {
            if t.conforms(to: .pdf) { return .pdf }
            if t.conforms(to: .image) { return .image }
            if t.conforms(to: .movie) || t.conforms(to: .video) { return .video }
            if t.conforms(to: .audio) { return .audio }
            if t.conforms(to: .rtf) || t.conforms(to: .rtfd) || t.conforms(to: .webArchive) { return .other }
            if t.conforms(to: .text) || t.conforms(to: .sourceCode) || t.conforms(to: .json)
                || t.conforms(to: .xml) || t.conforms(to: .propertyList) { return .text }
            if t.conforms(to: .archive) || t.conforms(to: .executable) || t.conforms(to: .presentation)
                || t.conforms(to: .spreadsheet) { return .other }
        }
        return looksLikeText(url) ? .text : .other
    }

    /// No NUL byte in its first 8 KB (an empty file counts as text).
    static func looksLikeText(_ url: URL) -> Bool {
        guard let h = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? h.close() }
        let head = (try? h.read(upToCount: 8192)) ?? Data()
        return !head.contains(0)
    }
}

/// A file shown read-only: images, videos, audio, PDFs, Markdown (rendered or as source), text and code with line
/// numbers, anything else through Quick Look. The header names it and has its actions.
struct FilePreview: View {
    let url: URL
    /// The name shown (an attachment's original name; its copy has a generated one).
    var title: String?
    /// A line to show and highlight (text files).
    var line: Int?
    /// Changes when the file may have changed: it's read again.
    var revision = 0
    /// Adds the file to the message being written (project files).
    var onMention: (() -> Void)?
    var onClose: (() -> Void)?
    /// The full-window viewer draws its own dark backdrop: no header background there.
    var dark = false
    @State private var source = false

    var body: some View {
        let kind = PreviewKind.of(url)
        VStack(spacing: 0) {
            header(kind)
            MGDivider()
            Group {
                if !FileManager.default.fileExists(atPath: url.path) {
                    MGEmptyState(symbol: "questionmark.folder", title: "File not found", message: url.path)
                } else {
                    switch kind {
                    case .image: ImagePreview(url: url, revision: revision)
                    case .video: ChatVideoPlayer(url: url).padding(8)
                    case .audio: AudioResultView(url: url)
                    case .pdf: PDFPreview(url: url, revision: revision)
                    case .markdown:
                        if source { TextPreview(url: url, line: line, revision: revision) } else {
                            MarkdownFilePreview(url: url, revision: revision)
                        }
                    case .text: TextPreview(url: url, line: line, revision: revision)
                    case .other: QuickLookPreview(url: url, revision: revision)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .id(url)  // a fresh viewer per file; the pane itself stays (a split keeps its size)
        }
    }

    func header(_ kind: PreviewKind) -> some View {
        HStack(spacing: 6) {
            Image(nsImage: FileIcons.icon(for: title ?? url.path)).resizable().frame(width: 16, height: 16)
            Text(title ?? url.lastPathComponent).font(.mgCallout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: 4)
            if kind == .markdown {
                MGSegmentedPicker(selection: $source, options: [(false, "Preview"), (true, "Source")])
                    .fixedSize()
            }
            Group {
                if let onMention {
                    Button(action: onMention) { Image(systemName: "text.badge.plus") }
                        .mgHelp("Mention in the message")
                }
                Button { NSWorkspace.shared.open(url) } label: { Image(systemName: "arrow.up.forward.app") }.accessibilityLabel("Open in Default App")
                    .mgHelp("Open in the default app")
                Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }.accessibilityLabel("Reveal in Finder")
                    .mgHelp("Reveal in Finder")
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(url.path, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                    .mgHelp("Copy the path")
                if let onClose {
                    Button(action: onClose) { Image(systemName: "xmark") }.mgHelp("Close the preview")
                }
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.small)
        }
        .padding(.horizontal, 10)
        .frame(height: 34)
        .background(dark ? Color.clear : MG.controlBackground)
    }
}

// MARK: text

/// Text or code: monospaced, with line numbers, selectable; lazily laid out in chunks so a long file opens fast.
/// At most the first 2 MB are shown.
struct TextPreview: View {
    let url: URL
    var line: Int?
    var revision = 0
    @State private var chunks: [Chunk] = []
    @State private var note: String?
    @State private var loaded = false

    nonisolated static let limit = 2 << 20
    nonisolated static let chunkLines = 200

    struct Chunk: Identifiable {
        let id: Int  // the first line's number
        let numbers: String
        let text: String
        let highlighted: Bool
    }

    var body: some View {
        GeometryReader { geo in
        ScrollViewReader { proxy in
            ScrollView([.vertical, .horizontal]) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if let note {
                        Text(note).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).padding(8)
                    }
                    ForEach(chunks) { c in
                        HStack(alignment: .top, spacing: 10) {
                            Text(c.numbers).foregroundStyle(MG.tertiaryLabel).multilineTextAlignment(.trailing)
                                .frame(minWidth: 28, alignment: .trailing)
                                .textSelection(.disabled)
                            Text(c.text).textSelection(.enabled)
                        }
                        .font(.mgMono)
                        .fixedSize(horizontal: true, vertical: true)
                        .padding(.horizontal, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(c.highlighted ? MG.accent.opacity(0.18) : .clear)
                        .id(c.id)
                    }
                }
                .padding(.vertical, 6)
                // At least as wide as the pane: short lines start at its left edge (not centred).
                .frame(minWidth: geo.size.width, alignment: .leading)
            }
            .mgScrollIndicators()
            .overlay { if !loaded { MGSpinner() } }
            .task(id: "\(url.path)|\(revision)|\(line ?? 0)") {
                let (c, n) = await Task.detached { [url, line] in Self.load(url, highlight: line) }.value
                chunks = c
                note = n
                loaded = true
                if let line {
                    try? await Task.sleep(nanoseconds: 50_000_000)
                    proxy.scrollTo(line, anchor: .center)
                }
            }
        }
        }
    }

    /// The file's lines in chunks; the highlighted line is a chunk of its own (so it can be marked and scrolled to).
    nonisolated static func load(_ url: URL, highlight: Int?) -> ([Chunk], String?) {
        guard let h = try? FileHandle(forReadingFrom: url) else { return ([], "Couldn't read the file.") }
        defer { try? h.close() }
        let data = (try? h.read(upToCount: limit + 1)) ?? Data()
        let cut = data.count > limit
        let text = String(decoding: cut ? data.prefix(limit) : data, as: UTF8.self)
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" && lines.count > 1 { lines.removeLast() }
        var out: [Chunk] = []
        var i = 0
        func add(_ from: Int, _ to: Int, _ hl: Bool) {
            guard from < to else { return }
            out.append(Chunk(id: from + 1, numbers: (from + 1...to).map(String.init).joined(separator: "\n"),
                             text: lines[from..<to].joined(separator: "\n"), highlighted: hl))
        }
        let target = highlight.map { min(max($0, 1), lines.count) - 1 }
        while i < lines.count {
            var end = min(i + chunkLines, lines.count)
            if let t = target, t >= i, t < end {
                add(i, t, false)
                add(t, t + 1, true)
                i = t + 1
                end = min(i + chunkLines, lines.count)
            }
            add(i, end, false)
            i = end
        }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? data.count
        let note = cut ? "Showing the first 2 MB of \(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))." : nil
        return (out, note)
    }
}

/// A Markdown file rendered (as a reply is).
struct MarkdownFilePreview: View {
    let url: URL
    var revision = 0
    @State private var text = ""

    var body: some View {
        ScrollView {
            MarkdownView(text: text).padding(14)
        }
        .mgScrollIndicators()
        .task(id: "\(url.path)|\(revision)") {
            text = await Task.detached { [url] in
                guard let d = try? Data(contentsOf: url) else { return "" }
                return String(decoding: d.prefix(TextPreview.limit), as: UTF8.self)
            }.value
        }
    }
}

// MARK: media

/// An image fitted to the space; pinch or double-click to zoom, drag to move.
struct ImagePreview: View {
    let url: URL
    var revision = 0
    @State private var image: NSImage?
    @State private var zoom: CGFloat = 1
    @State private var zoomStart: CGFloat?
    @State private var offset: CGSize = .zero
    @State private var offsetStart: CGSize?

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(zoom > 2 ? .none : .high)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: image.size.width, maxHeight: image.size.height)  // small images aren't blown up
                    .scaleEffect(zoom)
                    .offset(offset)
                    .padding(12)
                    .gesture(MagnifyGesture()
                        .onChanged { v in
                            if zoomStart == nil { zoomStart = zoom }
                            zoom = min(max(zoomStart! * v.magnification, 0.25), 16)
                        }
                        .onEnded { _ in zoomStart = nil })
                    .simultaneousGesture(DragGesture()
                        .onChanged { v in
                            guard zoom != 1 else { return }
                            if offsetStart == nil { offsetStart = offset }
                            offset = CGSize(width: offsetStart!.width + v.translation.width,
                                            height: offsetStart!.height + v.translation.height)
                        }
                        .onEnded { _ in offsetStart = nil })
                    .onTapGesture(count: 2) {
                        withAnimation(.easeOut(duration: 0.15)) {
                            if zoom == 1 { zoom = 2 } else { zoom = 1; offset = .zero }
                        }
                    }
            } else {
                MGSpinner()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
        .task(id: "\(url.path)|\(revision)") {
            zoom = 1
            offset = .zero
            image = NSImage(contentsOf: url)
        }
    }
}

/// A video with the system's controls (no autoplay).
struct ChatVideoPlayer: View {
    let url: URL
    @State private var player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
            .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            .task(id: url) {
                let p = AVPlayer(url: url)
                p.actionAtItemEnd = .pause
                player = p
            }
            .onDisappear { player?.pause() }
    }
}

struct PDFPreview: NSViewRepresentable {
    let url: URL
    var revision = 0

    func makeNSView(context: Context) -> PDFView {
        let v = PDFView()
        v.autoScales = true
        v.displayMode = .singlePageContinuous
        v.backgroundColor = .clear
        v.document = PDFDocument(url: url)
        context.coordinator.loaded = "\(url.path)|\(revision)"
        return v
    }

    func updateNSView(_ v: PDFView, context: Context) {
        let key = "\(url.path)|\(revision)"
        if context.coordinator.loaded != key {
            v.document = PDFDocument(url: url)
            context.coordinator.loaded = key
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var loaded = "" }
}

/// Anything else, as Finder's Quick Look shows it (documents, archives' contents, fonts, 3D models…).
struct QuickLookPreview: NSViewRepresentable {
    let url: URL
    var revision = 0

    func makeNSView(context: Context) -> QLPreviewView {
        let v = QLPreviewView(frame: .zero, style: .normal) ?? QLPreviewView()
        v.autostarts = true
        v.previewItem = url as NSURL
        context.coordinator.loaded = "\(url.path)|\(revision)"
        return v
    }

    func updateNSView(_ v: QLPreviewView, context: Context) {
        let key = "\(url.path)|\(revision)"
        if context.coordinator.loaded != key {
            v.previewItem = url as NSURL
            v.refreshPreviewItem()
            context.coordinator.loaded = key
        }
    }

    static func dismantleNSView(_ v: QLPreviewView, coordinator: Coordinator) { v.close() }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var loaded = "" }
}
