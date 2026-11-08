import AppKit
import SwiftUI

/// The image workspace: the latest result, or the live step preview while sampling.
struct CanvasView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor).ignoresSafeArea()
            if model.previewImage == nil, let a = model.displayedImage, let b = model.compareImage {
                CompareView(a: a, b: b).id(model.compareURL)
            } else if let image = model.previewImage ?? model.displayedImage {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(model.previewImage != nil ? .medium : .high)
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .shadow(color: .black.opacity(0.25), radius: 12, y: 4)
                    .padding(24)
                    .contextMenu {
                        if let url = model.displayedURL {
                            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                            Button("Copy Image") { copy(image) }
                        }
                    }
            } else {
                ContentUnavailableView {
                    Label("No image yet", systemImage: "photo.on.rectangle.angled")
                } description: {
                    Text("Write a prompt in the inspector and press ⌘↩ to generate.\n"
                         + "Drop a PNG made by MLXComfyUI here to restore its graph.")
                }
            }
        }
        .overlay(alignment: .bottom) { RunStatusPill().padding(.bottom, 20) }
    }

    func copy(_ image: CGImage) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.writeObjects([NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))])
    }
}

/// Floating "Sampler · step 3/8" status shown over the image and the graph while a job runs.
struct RunStatusPill: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.isRunning {
            let running = model.nodeStates.first { if case .running = $0.value { return true }; return false }
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                if let (id, state) = running, case .running(let step, let total) = state {
                    let name = model.displayName(flatID: id)
                    Text(total > 0 ? "\(name) · step \(step) of \(total)" : "\(name)…")
                        .font(.callout.monospacedDigit())
                } else {
                    Text("Starting…").font(.callout)
                }
                if model.pendingCount > 0 {
                    Text("+\(model.pendingCount) queued").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
        }
    }
}
