import AppKit
import SwiftUI

/// Files shown full-window from a chat (a message's attachments, an image in a reply), and the one in front.
struct ChatViewerItem: Equatable {
    var urls: [URL]
    var index: Int
    /// The names shown, by index (nil: the files' own).
    var titles: [String]? = nil
}

/// A chat's attachments and media over the window: the file's preview, ← → between a message's files, Esc closes.
struct ChatMediaViewer: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool

    var body: some View {
        if let item = model.chatViewer, item.urls.indices.contains(item.index) {
            let url = item.urls[item.index]
            ZStack {
                Color.black.opacity(0.55).ignoresSafeArea()
                    .onTapGesture { close() }
                HStack(spacing: 12) {
                    nav("chevron.left", enabled: item.index > 0) { step(-1) }
                    VStack(spacing: 0) {
                        FilePreview(url: url, title: item.titles.flatMap { $0.indices.contains(item.index) ? $0[item.index] : nil },
                                    onClose: { close() })
                        if item.urls.count > 1 {
                            MGDivider()
                            Text("\(item.index + 1) of \(item.urls.count)").font(.mgFootnote.monospacedDigit())
                                .foregroundStyle(MG.secondaryLabel).frame(height: 24)
                        }
                    }
                    .background(MG.windowBackground)
                    .clipShape(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
                    .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
                    .shadow(color: .black.opacity(0.3), radius: 24, y: 8)
                    nav("chevron.right", enabled: item.index < item.urls.count - 1) { step(1) }
                }
                .padding(28)
            }
            .focusable()
            .focusEffectDisabled()
            .focused($focused)
            .onAppear { focused = true }
            .onKeyPress(.escape) { close(); return .handled }
            .onKeyPress(.leftArrow) { step(-1); return .handled }
            .onKeyPress(.rightArrow) { step(1); return .handled }
        }
    }

    func nav(_ symbol: String, enabled: Bool, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 18, weight: .semibold)).frame(width: 36, height: 36)
        }
        .buttonStyle(.mg(.icon))
        .opacity(enabled ? 1 : 0)
        .disabled(!enabled)
    }

    func step(_ d: Int) {
        guard var item = model.chatViewer else { return }
        let i = item.index + d
        guard item.urls.indices.contains(i) else { return }
        item.index = i
        model.chatViewer = item
    }

    func close() { model.chatViewer = nil }
}

/// An image or video in a reply (`![alt](src)` on a line of its own): shown inline; click an image for the viewer.
struct ChatMediaBlock: View {
    @Environment(AppModel.self) private var model
    let alt: String
    let url: URL?
    let src: String

    var isVideo: Bool { url.map { PreviewKind.videoExtensions.contains($0.pathExtension.lowercased()) } ?? false }

    var body: some View {
        if let url, !url.isFileURL || FileManager.default.fileExists(atPath: url.path) {
            VStack(alignment: .leading, spacing: 4) {
                if isVideo {
                    ChatVideoPlayer(url: url).aspectRatio(16 / 9, contentMode: .fit).frame(maxWidth: 560)
                } else if url.isFileURL {
                    LocalImage(url: url)
                        .onTapGesture { model.chatViewer = ChatViewerItem(urls: [url], index: 0) }
                        .mgHelp(alt.isEmpty ? url.lastPathComponent : alt)
                } else {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image): image.resizable().aspectRatio(contentMode: .fit)
                        case .failure: missing("Couldn't load the image: \(src)")
                        default: MGSpinner().frame(width: 80, height: 60)
                        }
                    }
                    .frame(maxWidth: 560, maxHeight: 360, alignment: .leading)
                    .onTapGesture { NSWorkspace.shared.open(url) }
                    .mgHelp(alt.isEmpty ? src : alt)
                }
                if !alt.isEmpty {
                    Text(alt).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
            }
        } else {
            missing("Image not found: \(src)")
        }
    }

    func missing(_ text: String) -> some View {
        Label(text, systemImage: "photo.badge.exclamationmark").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
    }
}

/// A local image, at most 560 × 360 (not blown up), loaded off the main thread.
private struct LocalImage: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                    .frame(maxWidth: min(560, image.size.width), maxHeight: min(360, image.size.height), alignment: .leading)
                    .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            } else {
                RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill).frame(width: 120, height: 80)
            }
        }
        .task(id: url) {
            image = await Task.detached { [url] in NSImage(contentsOf: url) }.value
        }
    }
}
