import AppKit
import ImageIO
import SwiftUI

/// Under the graph: the active tab's live preview while it runs, else its latest (or chosen) result, with players
/// for video and audio, A/B compare, and a strip of the tab's earlier results.
struct ResultsPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let doc = model.active
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text("Results").font(.mgHeadline)
                if let url = doc.displayedURL, doc.previewImage == nil {
                    Text(url.lastPathComponent).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                if let url = doc.displayedURL {
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }
                        .mgHelp("Reveal in Finder")
                }
                Button { model.openViewer() } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                    .mgHelp("Open in the viewer (double-click the result)")
                    .disabled(model.displayedResult == nil)
                Button { withAnimation { model.showResultsPane = false } } label: { Image(systemName: "chevron.down") }
                    .mgHelp("Hide the results pane (⌥⌘R)")
            }
            .buttonStyle(.mg(.icon))
            .padding(.leading, 16)
            .padding(.trailing, 10)
            .frame(height: 32)
            .background(MG.barOverContent)
            .overlay(alignment: .bottom) { MGDivider() }
            .overlay(alignment: .top) { MGDivider(gap: true) }
            ResultContent(doc: doc, padding: 12)
                .onTapGesture(count: 2) { model.openViewer() }
                .overlay(alignment: .bottom) { if doc.outputs.count > 1 { ResultStrip(doc: doc) } }
        }
        .background(MG.contentBackground)
    }
}

/// The result itself: live preview, media player, compare, image, or an empty state.
struct ResultContent: View {
    @Environment(AppModel.self) private var model
    let doc: WorkflowDocument
    var padding: CGFloat = 24

    var body: some View {
        ZStack {
            if doc.previewImage == nil, let url = doc.displayedMediaURL, doc.displayedMediaKind != .image {
                MediaCanvas(url: url, kind: doc.displayedMediaKind).id(url)
            } else if doc.previewImage == nil, let a = doc.displayedImage, let b = doc.compareImage {
                CompareView(a: a, b: b).id(doc.compareURL)
            } else if let image = doc.previewImage ?? doc.displayedImage {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(doc.previewImage != nil ? .medium : .high)
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .shadow(color: .black.opacity(0.25), radius: 10, y: 3)
                    .padding(padding)
                    .mgContextMenu {
                        if let url = doc.displayedURL {
                            MGMenuItem.button("Open in Viewer", shortcut: KeyboardShortcut("y")) { model.openViewer() }
                            MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                            MGMenuItem.button("Copy Image") { Self.copy(image) }
                        }
                    }
            } else {
                MGEmptyState(symbol: "photo.on.rectangle.angled", title: "No results yet",
                             message: "Press ⌘↩ to run this workflow. Results also show on their Save and Preview nodes.")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    static func copy(_ image: CGImage) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.writeObjects([NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))])
    }
}

/// Thumbnails of the tab's results, newest first; click to show one.
struct ResultStrip: View {
    @Environment(AppModel.self) private var model
    let doc: WorkflowDocument

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(doc.outputs.reversed().prefix(40)) { r in
                    ResultThumb(result: r)
                        .frame(width: 44, height: 44)
                        .overlay(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous).strokeBorder(
                            r.url != nil && r.url == doc.displayedURL ? MG.accent : MG.separator,
                            lineWidth: r.url != nil && r.url == doc.displayedURL ? 2 : 0.5))
                        .onTapGesture { model.show(r, in: doc) }
                        .onTapGesture(count: 2) { model.show(r, in: doc); model.openViewer() }
                }
            }
            .padding(6)
        }
        .mgGlass(.floatingPanel, radius: MGRadius.lg)
        .padding(8)
        .frame(maxWidth: 520)
    }
}

struct ResultThumb: View {
    let result: NodeResult

    var body: some View {
        Rectangle().fill(MG.fill)
            .overlay {
                if let img = result.image {
                    Image(decorative: img, scale: 1).resizable().aspectRatio(contentMode: .fill)
                } else if result.kind == .audio, let url = result.url {
                    AudioThumb(url: url)
                }
            }
            .overlay(alignment: .bottomLeading) {
                if result.kind == .video {
                    Image(systemName: "play.fill").font(.system(size: 7)).foregroundStyle(.white)
                        .padding(3).background(.black.opacity(0.55), in: Circle()).padding(3)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous))
    }
}

// MARK: - viewer

/// Full-window viewer: zoom and pan, video and audio playback, A/B compare, ← → through the tab's results (or the
/// history), Esc to close.
struct ResultViewer: View {
    @Environment(AppModel.self) private var model
    @State private var zoom: CGFloat = 1
    @State private var zoomStart: CGFloat?
    @State private var offset: CGSize = .zero
    @State private var offsetStart: CGSize?
    @State private var image: CGImage?
    @FocusState private var focused: Bool

    var body: some View {
        let items = model.viewerItems
        let current = model.viewerResult
        let index = current.flatMap { c in items.firstIndex { $0.id == c.id || ($0.url != nil && $0.url == c.url) } }
        ZStack {
            Color.black.opacity(0.92).ignoresSafeArea()
                .onTapGesture { close() }
            if let current {
                content(current)
            }
            VStack {
                HStack(spacing: 10) {
                    Text(current?.url?.lastPathComponent ?? "Preview").font(.mgHeadline).lineLimit(1)
                    if let index { Text("\(index + 1) of \(items.count)").foregroundStyle(MG.secondaryLabel) }
                    Spacer()
                    if current?.kind == .image {
                        Text("\(Int(zoom * 100))%").font(.mgSubheadline.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                        Button { withAnimation { zoom = 1; offset = .zero } } label: { Image(systemName: "arrow.down.right.and.arrow.up.left") }
                            .mgHelp("Fit (double-click)")
                    }
                    if let url = current?.url {
                        Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }
                            .mgHelp("Reveal in Finder")
                    }
                    Button { close() } label: { Image(systemName: "xmark") }
                        .mgHelp("Close (Esc)")
                }
                .buttonStyle(.mg(.icon))
                .padding(.horizontal, 14)
                .frame(height: 40)
                .mgGlass(.floatingPanel, radius: MGRadius.lg)
                .padding(12)
                Spacer()
                HStack {
                    navButton("chevron.left", enabled: (index ?? 0) > 0) { step(-1, items, index) }
                    Spacer()
                    navButton("chevron.right", enabled: index.map { $0 < items.count - 1 } ?? false) { step(1, items, index) }
                }
                .padding(.horizontal, 16)
                Spacer()
            }
        }
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onAppear { focused = true }
        .onKeyPress(.escape) { close(); return .handled }
        .onKeyPress(.leftArrow) { step(-1, items, index); return .handled }
        .onKeyPress(.rightArrow) { step(1, items, index); return .handled }
        .task(id: current?.id) {
            zoom = 1
            offset = .zero
            image = current?.image
            if image == nil, current?.kind == .image, let url = current?.url {
                image = await Task.detached { () -> CGImage? in
                    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
                    return CGImageSourceCreateImageAtIndex(src, 0, nil)
                }.value
            }
        }
    }

    @ViewBuilder func content(_ r: NodeResult) -> some View {
        switch r.kind {
        case .video, .audio:
            if let url = r.url { MediaCanvas(url: url, kind: r.kind).id(url).padding(40) }
        case .image:
            if let a = image, let b = model.compareImage, r.url == model.displayedURL {
                CompareView(a: a, b: b).padding(40)
            } else if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(zoom > 2 ? .none : .high)
                    .aspectRatio(contentMode: .fit)
                    .scaleEffect(zoom)
                    .offset(offset)
                    .padding(40)
                    .gesture(MagnifyGesture()
                        .onChanged { v in
                            if zoomStart == nil { zoomStart = zoom }
                            zoom = min(max(zoomStart! * v.magnification, 0.25), 16)
                        }
                        .onEnded { _ in zoomStart = nil })
                    .simultaneousGesture(DragGesture()
                        .onChanged { v in
                            if offsetStart == nil { offsetStart = offset }
                            offset = CGSize(width: offsetStart!.width + v.translation.width,
                                            height: offsetStart!.height + v.translation.height)
                        }
                        .onEnded { _ in offsetStart = nil })
                    .onTapGesture(count: 2) { withAnimation { if zoom == 1 { zoom = 2 } else { zoom = 1; offset = .zero } } }
            } else {
                MGSpinner(tint: .white)
            }
        }
    }

    func navButton(_ symbol: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 20, weight: .semibold)).foregroundStyle(MG.label)
                .frame(width: 44, height: 64).mgGlass(.floatingPanel, radius: MGRadius.lg)
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.25)
        .disabled(!enabled)
    }

    func step(_ delta: Int, _ items: [NodeResult], _ index: Int?) {
        guard let index, items.indices.contains(index + delta) else { return }
        model.viewerResult = items[index + delta]
    }

    func close() { withAnimation(.easeOut(duration: 0.15)) { model.viewerResult = nil } }
}
