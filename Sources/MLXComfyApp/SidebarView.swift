import AppKit
import ImageIO
import SwiftUI

struct SidebarView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    @State private var query = ""
    @AppStorage("historyFavoritesOnly") private var favoritesOnly = false
    @AppStorage("historyGrid") private var gridLayout = false

    /// Queue rows: single jobs, and one collapsed row per XY plot.
    enum Row: Identifiable {
        case job(QueueItem)
        case plot(UUID, title: String)
        var id: String {
            switch self {
            case .job(let j): return j.id.uuidString
            case .plot(let g, _): return "plot-" + g.uuidString
            }
        }
    }

    var queueRows: [Row] {
        var seen = Set<UUID>()
        return model.queue.compactMap { item in
            guard let g = item.groupID else { return .job(item) }
            guard seen.insert(g).inserted else { return nil }
            let spec = model.plots[g]?.spec
            let axes = [spec?.x.input, spec?.y?.input].compactMap { $0 }.joined(separator: " × ")
            return .plot(g, title: "XY plot" + (axes.isEmpty ? "" : " · \(axes)"))
        }
    }

    var filteredHistory: [HistoryItem] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        return model.history.filter { item in
            (!favoritesOnly || item.isFavorite)
                && (q.isEmpty || (item.prompt ?? "").lowercased().contains(q)
                    || item.url.lastPathComponent.lowercased().contains(q))
        }
    }

    var body: some View {
        let rows = queueRows
        let hasPlots = rows.contains { if case .plot = $0 { return true }; return false }
        List {
            if !model.queue.isEmpty {
                Section {
                    ForEach(rows) { row in
                        switch row {
                        case .job(let item): QueueRow(item: item)
                        case .plot(let g, let title): PlotQueueRow(group: g, title: title)
                        }
                    }
                    // Reordering maps rows to queue items one-to-one only without plot groups.
                    .onMove(perform: hasPlots ? nil : { model.movePending(from: $0, to: $1) })
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
            Section {
                let items = filteredHistory
                if model.history.isEmpty {
                    Text("Generated images appear here.")
                        .font(.callout).foregroundStyle(.secondary)
                } else if items.isEmpty {
                    Text(favoritesOnly ? "No favorites match." : "No images match.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if gridLayout {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 72, maximum: 120), spacing: 6)], spacing: 6) {
                        ForEach(items) { item in
                            HistoryThumb(item: item, selected: model.displayedURL == item.url,
                                         compared: model.compareURL == item.url)
                                .aspectRatio(1, contentMode: .fit)
                                .modifier(HistoryActions(item: item))
                        }
                    }
                    .padding(.vertical, 4)
                } else {
                    ForEach(items) { item in
                        HistoryRow(item: item, selected: model.displayedURL == item.url,
                                   compared: model.compareURL == item.url)
                            .modifier(HistoryActions(item: item))
                    }
                }
            } header: {
                HStack(spacing: 8) {
                    Text("History")
                    Spacer()
                    Button { favoritesOnly.toggle() } label: {
                        Image(systemName: favoritesOnly ? "star.fill" : "star")
                    }
                    .buttonStyle(.plain).foregroundStyle(favoritesOnly ? Color.yellow : .secondary)
                    .help(favoritesOnly ? "Show all images" : "Show favorites only")
                    Button { gridLayout.toggle() } label: {
                        Image(systemName: gridLayout ? "list.bullet" : "square.grid.2x2")
                    }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
                    .help(gridLayout ? "Show as a list" : "Show as a grid")
                }
            }
        }
        .listStyle(.sidebar)
        .searchable(text: $query, placement: .sidebar, prompt: "Search prompts")
    }
}

/// Click to show, ⌘-click to compare with the shown image, and the context menu.
struct HistoryActions: ViewModifier {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    let item: HistoryItem

    func body(content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .onTapGesture {
                if NSEvent.modifierFlags.contains(.command) { model.compare(with: item.url) }
                else { model.endCompare(); model.show(item) }
            }
            .contextMenu {
                Button("Show Image") { model.endCompare(); model.show(item) }
                Button("Compare with Shown Image") { model.compare(with: item.url) }
                    .disabled(model.displayedImage == nil || model.displayedURL == item.url)
                Button("Restore Graph") {
                    if model.open(item.url, undoManager: undoManager) { model.workspace = .graph }
                }
                Button(item.isFavorite ? "Remove from Favorites" : "Add to Favorites") { model.toggleFavorite(item) }
                Divider()
                Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                Button("Move to Trash", role: .destructive) { model.trash(item) }
            }
            .draggable(item.url)
    }
}

/// One collapsed row for all cells of an XY plot.
struct PlotQueueRow: View {
    @Environment(AppModel.self) private var model
    let group: UUID
    let title: String

    var body: some View {
        let p = model.plotProgress(group)
        HStack(spacing: 8) {
            Group {
                if p.running { ProgressView().controlSize(.mini) }
                else if p.done == p.total { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                else { Image(systemName: "square.grid.3x3").foregroundStyle(.secondary) }
            }
            .frame(width: 16)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).lineLimit(1).font(.callout)
                ProgressView(value: Double(p.done), total: Double(max(p.total, 1)))
                    .progressViewStyle(.linear).controlSize(.small)
                Text("\(p.done) of \(p.total) images").font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if p.done < p.total {
                Button { model.cancelPlot(group) } label: { Image(systemName: "stop.circle") }
                    .buttonStyle(.plain).help("Stop the plot (finished cells are still saved as a grid)")
            } else {
                Button { model.queue.removeAll { $0.groupID == group } } label: { Image(systemName: "xmark.circle") }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
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
    var compared = false

    var body: some View {
        HStack(spacing: 10) {
            HistoryThumb(item: item, selected: selected, compared: compared, showStar: false)
                .frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.prompt.flatMap { $0.isEmpty ? nil : $0 } ?? item.url.deletingPathExtension().lastPathComponent)
                    .font(.callout).lineLimit(2)
                HStack(spacing: 4) {
                    if item.isFavorite { Image(systemName: "star.fill").foregroundStyle(.yellow) }
                    if compared { Text("B").fontWeight(.bold).foregroundStyle(Color.accentColor) }
                    Text(item.date, format: .relative(presentation: .named))
                }
                .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

/// A history thumbnail, loaded off the main thread.
struct HistoryThumb: View {
    let item: HistoryItem
    let selected: Bool
    var compared = false
    var showStar = true
    @State private var thumbnail: CGImage?

    var body: some View {
        Rectangle().fill(.quaternary)
            .overlay {
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 1).resizable().aspectRatio(contentMode: .fill)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(
                selected ? Color.accentColor : compared ? Color.orange : .clear, lineWidth: 2))
            .overlay(alignment: .topTrailing) {
                if showStar && item.isFavorite {
                    Image(systemName: "star.fill").font(.caption2).foregroundStyle(.yellow)
                        .shadow(radius: 1).padding(4)
                }
            }
            .task(id: item.url) {
                let url = item.url
                thumbnail = await Task.detached(priority: .utility) { () -> CGImage? in
                    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
                    return CGImageSourceCreateThumbnailAtIndex(src, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceThumbnailMaxPixelSize: 240,
                    ] as CFDictionary)
                }.value
            }
    }
}
