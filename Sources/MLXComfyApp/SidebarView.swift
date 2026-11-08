import AppKit
import ImageIO
import SwiftUI

struct SidebarView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        List {
            if !model.queue.isEmpty {
                Section {
                    ForEach(model.queue) { item in
                        QueueRow(item: item)
                    }
                    .onMove { model.movePending(from: $0, to: $1) }
                } header: {
                    HStack {
                        Text("Queue")
                        Spacer()
                        if model.queue.contains(where: \.isFinished) {
                            Button("Clear") { model.clearFinished() }
                                .buttonStyle(.plain).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Section("History") {
                if model.history.isEmpty {
                    Text("Generated images appear here.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                ForEach(model.history) { item in
                    HistoryRow(item: item, selected: model.displayedURL == item.url)
                        .contentShape(Rectangle())
                        .onTapGesture { model.show(item) }
                        .contextMenu {
                            Button("Show Image") { model.show(item) }
                            Button("Restore Graph") {
                                if model.open(item.url, undoManager: undoManager) { model.workspace = .graph }
                            }
                            Divider()
                            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                            Button("Move to Trash", role: .destructive) { model.trash(item) }
                        }
                        .draggable(item.url)
                }
            }
        }
        .listStyle(.sidebar)
    }
}

struct QueueRow: View {
    @Environment(AppModel.self) private var model
    let item: QueueItem

    var body: some View {
        HStack(spacing: 8) {
            statusIcon.frame(width: 16)
            VStack(alignment: .leading, spacing: 3) {
                Text(item.title).lineLimit(1).font(.callout)
                switch item.status {
                case .running(let step, let total) where total > 0:
                    ProgressView(value: Double(step), total: Double(total))
                        .progressViewStyle(.linear).controlSize(.small)
                    Text("Step \(step) of \(total)").font(.caption2).foregroundStyle(.secondary)
                case .running:
                    Text("Running…").font(.caption2).foregroundStyle(.secondary)
                case .done(let s):
                    Text(String(format: "%.1fs", s) + (item.seed.map { " · seed \($0)" } ?? ""))
                        .font(.caption2).foregroundStyle(.secondary)
                case .failed(let msg):
                    Text(msg).font(.caption2).foregroundStyle(.red).lineLimit(2)
                case .cancelled:
                    Text("Cancelled").font(.caption2).foregroundStyle(.secondary)
                case .pending:
                    Text(item.seed.map { "Waiting · seed \($0)" } ?? "Waiting")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            if case .running = item.status {
                Button { model.cancelCurrent() } label: { Image(systemName: "stop.circle") }
                    .buttonStyle(.plain).help("Cancel")
            } else {
                Button { model.removeFromQueue(item.id) } label: { Image(systemName: "xmark.circle") }
                    .buttonStyle(.plain).foregroundStyle(.secondary).help("Remove")
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder var statusIcon: some View {
        switch item.status {
        case .pending: Image(systemName: "clock").foregroundStyle(.secondary)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red)
        case .cancelled: Image(systemName: "minus.circle").foregroundStyle(.secondary)
        }
    }
}

struct HistoryRow: View {
    let item: HistoryItem
    let selected: Bool
    @State private var thumbnail: CGImage?
    @State private var prompt: String?

    var body: some View {
        HStack(spacing: 10) {
            Group {
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 1).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .frame(width: 52, height: 52)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 2))
            VStack(alignment: .leading, spacing: 2) {
                Text(prompt ?? item.url.deletingPathExtension().lastPathComponent)
                    .font(.callout).lineLimit(2)
                Text(item.date, format: .relative(presentation: .named))
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .task(id: item.url) {
            let url = item.url
            let (thumb, text) = await Task.detached(priority: .utility) { () -> (CGImage?, String?) in
                var thumb: CGImage?
                if let src = CGImageSourceCreateWithURL(url as CFURL, nil) {
                    thumb = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceThumbnailMaxPixelSize: 160,
                    ] as CFDictionary)
                }
                let g = AppModel.embeddedGraph(in: url)
                let text = g.flatMap { g -> String? in
                    guard let n = g.nodes.first(where: { $0.type == "TextEncode" }),
                          case .value(.string(let s))? = n.inputs["prompt"] else { return nil }
                    return s
                }
                return (thumb, text)
            }.value
            thumbnail = thumb
            prompt = text
        }
    }
}
