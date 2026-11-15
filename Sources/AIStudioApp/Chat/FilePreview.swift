import AVKit
import AppKit
import CodeEditing
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
        // Extensions TypeScript shares with MPEG transport streams (which the system calls video): text is code.
        if ["ts", "mts", "cts"].contains(ext), looksLikeText(url) { return .text }
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
    /// An editable buffer for the file (the Files pane's tabs); without one the preview makes a read-only buffer.
    var buffer: FileBuffer?
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
                        if source { TextFilePreview(url: url, line: line, revision: revision, external: buffer) } else if let buffer {
                            // The buffer's text, so the rendering follows what is typed (`edits` is what changes).
                            MarkdownFilePreview(url: url, revision: revision, text: buffer.loaded ? { _ = buffer.edits; return buffer.text }() : nil)
                                .task { await buffer.load() }
                        } else {
                            MarkdownFilePreview(url: url, revision: revision)
                        }
                    case .text: TextFilePreview(url: url, line: line, revision: revision, external: buffer)
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

/// Text or code in the code editor (line numbers, syntax colours, find). Uses the given buffer, or makes a read-only
/// one for the file; it is read again when `revision` changes.
private struct TextFilePreview: View {
    let url: URL
    var line: Int?
    var revision = 0
    var external: FileBuffer?
    @State private var own: FileBuffer?
    @State private var seenRevision: Int?

    var body: some View {
        let buffer = external ?? own
        Group {
            if let buffer {
                CodeEditorView(buffer: buffer)
                    .task(id: line) { if let line { buffer.reveal(lines: line...line) } }
                    .task(id: revision) {
                        if let seen = seenRevision, seen != revision { await buffer.reloadFromDisk() }
                        seenRevision = revision
                    }
            } else {
                Color.clear
            }
        }
        .task {
            if external == nil, own == nil { own = FileBuffer(url: url, relPath: nil, editable: false, watch: false) }
        }
        .onDisappear { if external == nil { own?.close() } }
    }
}

/// A Markdown file rendered (as a reply is).
struct MarkdownFilePreview: View {
    let url: URL
    var revision = 0
    /// Rendered instead of reading the file (an editor buffer's text).
    var text: String?
    @State private var read = ""

    var body: some View {
        ScrollView {
            MarkdownView(text: text ?? read).padding(14)
        }
        .mgScrollIndicators()
        .task(id: "\(url.path)|\(revision)") {
            guard text == nil else { return }
            read = await Task.detached { [url] in
                guard let d = try? Data(contentsOf: url) else { return "" }
                return String(decoding: d.prefix(EditorLimits.editableBytes), as: UTF8.self)
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
