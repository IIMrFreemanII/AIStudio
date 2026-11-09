import AppKit
import ImageIO
import MLXDiffusionCore
import SwiftUI

/// Pending, running and finished jobs, each labelled with its workflow.
struct QueuePanel: View {
    @Environment(AppModel.self) private var model

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

    var body: some View {
        let rows = queueRows
        let hasPlots = rows.contains { if case .plot = $0 { return true }; return false }
        VStack(spacing: 0) {
            PanelHeader(title: "Queue") {
                if model.queue.contains(where: \.isFinished) {
                    Button("Clear Finished") { model.clearFinished() }.buttonStyle(.mg(.borderless)).controlSize(.small)
                }
                if model.pendingCount > 0 {
                    Button("Cancel All") { model.cancelAll() }.buttonStyle(.mg(.borderless, destructive: true)).controlSize(.small)
                }
            }
            List {
                if model.queue.isEmpty {
                    Text("Nothing queued. Press ⌘↩ to run the current workflow; results are in Assets › Output.")
                        .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        .padding(.horizontal, 16)
                        .mgListRow()
                }
                ForEach(rows) { row in
                    Group {
                        switch row {
                        case .job(let item): QueueRow(item: item)
                        case .plot(let g, let title): PlotQueueRow(group: g, title: title)
                        }
                    }
                    .mgRow(height: 38)
                    .mgListRow()
                }
                // Reordering maps rows to queue items one-to-one only without plot groups.
                .onMove(perform: hasPlots ? nil : { model.movePending(from: $0, to: $1) })
            }
            .mgList()
        }
    }
}

/// Everything generated (the output folder and its workflow subfolders): search, favorites, a workflow filter, and
/// a list or a grid.
struct OutputList: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    @State private var workflow: String? = nil
    @AppStorage("historyFavoritesOnly") private var favoritesOnly = false
    @AppStorage("historyGrid") private var gridLayout = false

    var filteredHistory: [HistoryItem] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        return model.history.filter { item in
            (!favoritesOnly || item.isFavorite)
                && (workflow == nil || item.workflow == workflow)
                && (q.isEmpty || (item.prompt ?? "").lowercased().contains(q)
                    || item.url.lastPathComponent.lowercased().contains(q)
                    || (item.workflow ?? "").lowercased().contains(q))
        }
    }

    var body: some View {
        let workflows = Array(Set(model.history.compactMap(\.workflow))).sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        VStack(spacing: 6) {
            HStack(spacing: 4) {
                MGSearchField(prompt: "Search prompts", text: $query)
                Button { favoritesOnly.toggle() } label: {
                    Image(systemName: favoritesOnly ? "star.fill" : "star")
                        .foregroundStyle(favoritesOnly ? MG.hueYellow : MG.secondaryLabel)
                }
                .buttonStyle(.mg(.icon))
                .mgHelp(favoritesOnly ? "Show everything" : "Show favorites only")
                Button { gridLayout.toggle() } label: { Image(systemName: gridLayout ? "list.bullet" : "square.grid.2x2") }
                    .buttonStyle(.mg(.icon))
                    .mgHelp(gridLayout ? "Show as a list" : "Show as a grid")
            }
            .padding(.horizontal, 12)
            MGPicker(selection: $workflow,
                     options: [(String?.none, "All workflows")] + workflows.map { (String?.some($0), $0) },
                     maxWidth: .infinity)
                .padding(.horizontal, 12)
            List {
                let items = filteredHistory
                if model.history.isEmpty {
                    Text("Generated images, videos and sounds appear here.")
                        .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        .padding(.horizontal, 16).mgListRow()
                } else if items.isEmpty {
                    Text(favoritesOnly ? "No favorites match." : "Nothing matches.")
                        .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        .padding(.horizontal, 16).mgListRow()
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
                    .padding(.horizontal, 12)
                    .mgListRow()
                } else {
                    ForEach(items) { item in
                        HistoryRow(item: item, selected: model.displayedURL == item.url,
                                   compared: model.compareURL == item.url)
                            .mgRow(selected: model.displayedURL == item.url, height: 60)
                            .modifier(HistoryActions(item: item))
                            .mgListRow()
                    }
                }
            }
            .mgList()
        }
    }
}

/// Click to show, ⌘-click to compare with the shown image, and the context menu.
struct HistoryActions: ViewModifier {
    @Environment(AppModel.self) private var model
    let item: HistoryItem

    func body(content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .onTapGesture(count: 2) { model.endCompare(); model.show(item); model.openViewer() }
            .onTapGesture {
                if NSEvent.modifierFlags.contains(.command) { model.compare(with: item.url) }
                else { model.endCompare(); model.show(item) }
            }
            .mgContextMenu {
                MGMenuItem.button("Show in Results") { model.endCompare(); model.show(item) }
                MGMenuItem.button("Open in Viewer") { model.endCompare(); model.show(item); model.openViewer() }
                MGMenuItem.button("Compare with Shown Image",
                                  enabled: model.displayedImage != nil && model.displayedURL != item.url) {
                    model.compare(with: item.url)
                }
                MGMenuItem.button("Restore Graph in New Tab") { model.open(item.url) }
                MGMenuItem.button("Copy to Assets › Input") { model.importAssets([item.url]) }
                MGMenuItem.button(item.isFavorite ? "Remove from Favorites" : "Add to Favorites") { model.toggleFavorite(item) }
                MGMenuItem.divider
                MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                MGMenuItem.button("Move to Trash", destructive: true) { model.trash(item) }
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
                if p.running { MGSpinner().controlSize(.mini) }
                else if p.done == p.total { Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success) }
                else { Image(systemName: "square.grid.3x3").foregroundStyle(.secondary) }
            }
            .frame(width: 16)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).lineLimit(1).font(.mgCallout)
                MGProgress(value: Double(p.done), total: Double(max(p.total, 1)))
                    .frame(maxWidth: .infinity)
                Text("\(p.done) of \(p.total) images").font(.mgFootnote).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if p.done < p.total {
                Button { model.cancelPlot(group) } label: { Image(systemName: "stop.circle") }
                    .buttonStyle(.mg(.icon, destructive: true)).mgHelp("Stop the plot (finished cells are still saved as a grid)")
            } else {
                Button { model.queue.removeAll { $0.groupID == group } } label: { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon)).controlSize(.small)
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
                Text(item.title).lineLimit(1).font(.mgCallout)
                if !item.workflow.isEmpty {
                    Text(item.workflow).font(.mgFootnote).foregroundStyle(.secondary).lineLimit(1)
                }
                switch item.status {
                case .running(let step, let total) where total > 0:
                    MGProgress(value: Double(step), total: Double(total))
                        .frame(maxWidth: .infinity)
                    Text("Step \(step) of \(total)").font(.mgFootnote).foregroundStyle(.secondary)
                case .running:
                    Text("Running…").font(.mgFootnote).foregroundStyle(.secondary)
                case .done(let s):
                    Text(String(format: "%.1fs", s) + (item.seed.map { " · seed \($0)" } ?? ""))
                        .font(.mgFootnote).foregroundStyle(.secondary)
                case .failed(let msg):
                    Text(msg).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(2)
                case .cancelled:
                    Text("Cancelled").font(.mgFootnote).foregroundStyle(.secondary)
                case .pending:
                    Text(item.seed.map { "Waiting · seed \($0)" } ?? "Waiting")
                        .font(.mgFootnote).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            if case .running = item.status {
                Button { model.cancelCurrent() } label: { Image(systemName: "stop.fill") }
                    .buttonStyle(.mg(.icon, destructive: true)).controlSize(.small).mgHelp("Cancel")
            } else {
                Button { model.removeFromQueue(item.id) } label: { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Remove")
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder var statusIcon: some View {
        switch item.status {
        case .pending: Image(systemName: "clock").foregroundStyle(.secondary)
        case .running: MGSpinner().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        case .failed: Image(systemName: "exclamationmark.circle.fill").foregroundStyle(MG.destructive)
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
                    .font(.mgCallout).lineLimit(2)
                HStack(spacing: 4) {
                    if item.isFavorite { Image(systemName: "star.fill").foregroundStyle(MG.hueYellow) }
                    if compared { Text("B").fontWeight(.bold).foregroundStyle(MG.accent) }
                    if let w = item.workflow { Text(w).lineLimit(1); Text("·") }
                    Text(item.date, format: .relative(presentation: .named))
                }
                .font(.mgFootnote).foregroundStyle(.secondary)
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
        Rectangle().fill(MG.fill)
            .overlay {
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 1).resizable().aspectRatio(contentMode: .fill)
                }
            }
            .overlay { if item.kind == .audio { AudioThumb(url: item.url) } }
            .overlay(alignment: .bottomLeading) {
                if item.kind == .video {
                    Image(systemName: "play.fill").font(.system(size: 8)).foregroundStyle(.white)
                        .padding(3).background(.black.opacity(0.55), in: Circle()).padding(4)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).strokeBorder(
                selected ? MG.accent : compared ? MG.warning : MG.separator, lineWidth: selected || compared ? 2 : 0.5))
            .overlay(alignment: .topTrailing) {
                if showStar && item.isFavorite {
                    Image(systemName: "star.fill").font(.caption2).foregroundStyle(MG.hueYellow)
                        .shadow(radius: 1).padding(4)
                }
            }
            .task(id: item.url) {
                let url = item.url
                if item.kind == .video { thumbnail = await MediaIO.posterFrame(url, maxSize: 240); return }
                if item.kind == .audio { return }
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
