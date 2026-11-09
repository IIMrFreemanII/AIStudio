import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The workflow tabs above the graph: click to switch, drag to reorder, × (or ⌘W) to close, + for a preset.
struct TabStrip: View {
    @Environment(AppModel.self) private var model
    @State private var dragging: UUID?

    var body: some View {
        HStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 2) {
                    ForEach(model.tabs) { doc in
                        TabButton(doc: doc, active: doc.id == model.activeTabID)
                            .onDrag {
                                dragging = doc.id
                                return NSItemProvider(object: doc.id.uuidString as NSString)
                            }
                            .onDrop(of: [.text], delegate: TabDropDelegate(target: doc.id, dragging: $dragging, model: model))
                    }
                }
                .padding(.horizontal, 6)
            }
            Button { model.showPresetGallery = true } label: {
                Image(systemName: "plus").frame(width: 26, height: 26)
            }
            .buttonStyle(.borderless)
            .help("New workflow from a preset (⌘N); ⌘T for a blank text-to-image tab")
            .padding(.trailing, 8)
        }
        .frame(height: 34)
        .background(.bar)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { model.newTab() }
    }
}

struct TabDropDelegate: DropDelegate {
    let target: UUID
    @Binding var dragging: UUID?
    let model: AppModel

    func dropEntered(info: DropInfo) {
        guard let d = dragging, d != target else { return }
        MainActor.assumeIsolated {
            withAnimation(.easeOut(duration: 0.15)) {
                let from = model.tabs.firstIndex { $0.id == d } ?? 0
                let to = model.tabs.firstIndex { $0.id == target } ?? 0
                model.moveTab(d, before: from < to ? model.tabs.dropFirst(to + 1).first?.id : target)
            }
        }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? { DropProposal(operation: .move) }

    func performDrop(info: DropInfo) -> Bool {
        dragging = nil
        return true
    }
}

struct TabButton: View {
    @Environment(AppModel.self) private var model
    let doc: WorkflowDocument
    let active: Bool
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 6) {
            if model.runningDocID == doc.id {
                ProgressView().controlSize(.mini)
            } else {
                Image(systemName: doc.fileURL == nil ? "doc" : "doc.text")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Text(doc.name).font(.system(size: 12, weight: active ? .semibold : .regular)).lineLimit(1)
            ZStack {
                if hovering || (active && !doc.isDirty) {
                    Button { TabCommands.close(doc, model: model) } label: {
                        Image(systemName: "xmark").font(.system(size: 9, weight: .bold))
                    }
                    .buttonStyle(.borderless)
                    .help("Close tab (⌘W)")
                } else if doc.isDirty {
                    Circle().fill(Color.secondary).frame(width: 7, height: 7).help("Unsaved changes")
                }
            }
            .frame(width: 14, height: 14)
        }
        .padding(.horizontal, 10)
        .frame(height: 26)
        .frame(maxWidth: 220)
        .background(RoundedRectangle(cornerRadius: 6).fill(active ? Color(nsColor: .controlBackgroundColor) : .clear))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(active ? Color.primary.opacity(0.12) : .clear))
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { model.activate(doc.id) }
        .help(doc.fileURL?.path ?? "Not saved yet")
        .contextMenu {
            Button("Close Tab") { TabCommands.close(doc, model: model) }
            Button("Close Other Tabs") {
                for other in model.tabs where other.id != doc.id { TabCommands.close(other, model: model) }
            }
            Divider()
            Button("Duplicate Tab") {
                model.openTab(doc.graph, name: doc.name, presetID: doc.presetID)
            }
            if let url = doc.fileURL {
                Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
        }
    }
}

/// Closing and saving tabs, shared by the tab strip and the menus.
@MainActor
enum TabCommands {
    /// Closes a tab, asking first when it has unsaved changes.
    static func close(_ doc: WorkflowDocument, model: AppModel) {
        guard doc.isDirty else { model.closeTab(doc.id); return }
        model.activate(doc.id)
        let alert = NSAlert()
        alert.messageText = "Save changes to “\(doc.name)”?"
        alert.informativeText = "Your changes are lost if you don't save them."
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Don't Save")
        alert.addButton(withTitle: "Cancel")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            if save(doc, model: model) { model.closeTab(doc.id) }
        case .alertSecondButtonReturn:
            model.closeTab(doc.id)
        default:
            break
        }
    }

    /// ⌘S: in place, or asks for a name (in the workflows folder) the first time.
    @discardableResult
    static func save(_ doc: WorkflowDocument, model: AppModel) -> Bool {
        if let url = doc.fileURL { return model.save(doc, to: url) }
        return saveAs(doc, model: model)
    }

    @discardableResult
    static func saveAs(_ doc: WorkflowDocument, model: AppModel) -> Bool {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.directoryURL = doc.fileURL?.deletingLastPathComponent() ?? model.library.workflows
        panel.nameFieldStringValue = doc.name + ".json"
        panel.message = "Workflows are kept in \(model.library.workflows.path) (subfolders are fine)."
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        return model.save(doc, to: url)
    }
}
