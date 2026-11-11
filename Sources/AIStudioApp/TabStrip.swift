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
                HStack(spacing: 4) {
                    ForEach(model.tabs) { doc in
                        TabButton(doc: doc, active: doc.id == model.activeTabID && model.activeChatID == nil)
                            .onDrag {
                                dragging = doc.id
                                return NSItemProvider(object: doc.id.uuidString as NSString)
                            }
                            .onDrop(of: [.text], delegate: TabDropDelegate(target: doc.id, dragging: $dragging, model: model))
                    }
                    let openChats = model.chats.filter(\.tabOpen)
                    if !openChats.isEmpty {
                        Rectangle().fill(MG.separator).frame(width: 1, height: 18).padding(.horizontal, 4)
                        ForEach(openChats) { chat in
                            ChatTabButton(chat: chat, active: chat.id == model.activeChatID)
                        }
                    }
                }
                .padding(.horizontal, 6)
            }
            Button { model.showPresetGallery = true } label: {
                Image(systemName: "plus").font(.system(size: 12, weight: .medium))
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.large)
            .mgHelp("New workflow from a preset (⌘N); ⌘T for a blank text-to-image tab")
            .padding(.trailing, 8)
        }
        // The design system's document tab bar: 40 tall, 28 pt pills, a hairline under it.
        .frame(height: 40)
        .mgChrome(MG.barTint, material: .headerView)
        .overlay(alignment: .bottom) { MGDivider() }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { model.newTab() }
    }
}

/// A chat's tab: its title, a spinner while a reply streams, × to close.
struct ChatTabButton: View {
    @Environment(AppModel.self) private var model
    let chat: ChatDocument
    let active: Bool
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 6) {
            if chat.isStreaming {
                MGSpinner().controlSize(.mini)
            } else {
                Image(systemName: "bubble.left").font(.system(size: 11))
                    .foregroundStyle(active ? MG.accent : MG.secondaryLabel)
            }
            Text(chat.title).font(.mg(size: 12.5, weight: active ? .medium : .regular)).lineLimit(1)
                .foregroundStyle(active ? MG.label : MG.secondaryLabel)
            if chat.isAgent, chat.todos.contains(where: { $0.status != "completed" }) {
                Text("\(chat.todos.filter { $0.status == "completed" }.count)/\(chat.todos.count)")
                    .font(.mgFootnote).monospacedDigit().foregroundStyle(MG.secondaryLabel)
            }
            ZStack {
                if hovering || active {
                    Button { model.closeChat(chat.id) } label: {
                        Image(systemName: "xmark").font(.system(size: 8, weight: .bold))
                    }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.mini)
                    .mgHelp("Close chat (⌘W)")
                }
            }
            .frame(width: 18, height: 18)
        }
        .padding(.leading, 10)
        .padding(.trailing, 5)
        .frame(height: 28)
        .frame(maxWidth: 220)
        .background {
            RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                .fill(active ? MG.selectedTab : hovering ? MG.hover : .clear)
                .mgShadow(active ? MGShadow.control : MGShadowSpec(x: 0, y: 0, blur: 0, color: .clear))
        }
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { model.activateChat(chat.id) }
        .mgHelp(chat.transcript.modelID.map { "Chat with \($0)" } ?? "Chat")
        .mgContextMenu {
            MGMenuItem.button("Close Chat", shortcut: KeyboardShortcut("w")) { model.closeChat(chat.id) }
            MGMenuItem.button("New Chat", shortcut: KeyboardShortcut("n", modifiers: [.command, .option])) { model.newChat() }
        }
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
                MGSpinner().controlSize(.mini)
            } else {
                Image(systemName: doc.fileURL == nil ? "doc" : "doc.text")
                    .font(.system(size: 11)).foregroundStyle(active ? MG.accent : MG.secondaryLabel)
            }
            Text(doc.name).font(.mg(size: 12.5, weight: active ? .medium : .regular)).lineLimit(1)
                .foregroundStyle(active ? MG.label : MG.secondaryLabel)
            ZStack {
                if hovering || (active && !doc.isDirty) {
                    Button { TabCommands.close(doc, model: model) } label: {
                        Image(systemName: "xmark").font(.system(size: 8, weight: .bold))
                    }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.mini)
                    .mgHelp("Close tab (⌘W)")
                } else if doc.isDirty {
                    Circle().fill(MG.secondaryLabel).frame(width: 7, height: 7).mgHelp("Unsaved changes")
                }
            }
            .frame(width: 18, height: 18)
        }
        .padding(.leading, 10)
        .padding(.trailing, 5)
        .frame(height: 28)
        .frame(maxWidth: 220)
        .background {
            // The selected tab is a raised pill; others show the hover fill.
            RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                .fill(active ? MG.selectedTab : hovering ? MG.hover : .clear)
                .mgShadow(active ? MGShadow.control : MGShadowSpec(x: 0, y: 0, blur: 0, color: .clear))
        }
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { model.activate(doc.id) }
        .mgHelp(doc.fileURL?.path ?? "Not saved yet")
        .mgContextMenu {
            MGMenuItem.button("Close Tab", shortcut: KeyboardShortcut("w")) { TabCommands.close(doc, model: model) }
            MGMenuItem.button("Close Other Tabs", enabled: model.tabs.count > 1) {
                TabCommands.close(model.tabs.filter { $0.id != doc.id }, model: model)
            }
            MGMenuItem.divider
            MGMenuItem.button("Duplicate Tab") { model.openTab(doc.graph, name: doc.name, presetID: doc.presetID) }
            if let url = doc.fileURL {
                MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
        }
    }
}

/// Closing and saving tabs, shared by the tab strip and the menus. Questions go through the window's presenter, so
/// they finish later: `then` runs once the tab is closed or saved.
@MainActor
enum TabCommands {
    /// Closes a tab, asking first when it has unsaved changes.
    static func close(_ doc: WorkflowDocument, model: AppModel, then: (() -> Void)? = nil) {
        guard doc.isDirty else { model.closeTab(doc.id); then?(); return }
        model.activate(doc.id)
        model.presenter.alert("Save changes to “\(doc.name)”?", message: "Your changes are lost if you don't save them.",
                              actions: [
                                  MGAlertAction(label: "Save") {
                                      save(doc, model: model) { saved in
                                          if saved { model.closeTab(doc.id); then?() }
                                      }
                                  },
                                  .destructive("Don't Save") { model.closeTab(doc.id); then?() },
                                  .cancel(),
                              ])
    }

    /// Closes tabs one after another (each dirty one asks); stops at the first Cancel.
    static func close(_ docs: [WorkflowDocument], model: AppModel) {
        guard let first = docs.first else { return }
        close(first, model: model) { close(Array(docs.dropFirst()), model: model) }
    }

    /// ⌘S: in place, or asks for a name (in the workflows folder) the first time.
    static func save(_ doc: WorkflowDocument, model: AppModel, then: ((Bool) -> Void)? = nil) {
        if let url = doc.fileURL { then?(model.save(doc, to: url)); return }
        saveAs(doc, model: model, then: then)
    }

    /// Save As: a name and a folder in the workflow library, or anywhere through Finder.
    static func saveAs(_ doc: WorkflowDocument, model: AppModel, then: ((Bool) -> Void)? = nil) {
        model.presenter.chooseFile(
            title: "Save Workflow",
            message: "Workflows are kept in the library; subfolders are fine.",
            roots: [MGFileRoot(title: "Workflows", url: model.library.workflows)],
            mode: .save(name: doc.name, type: .json),
            startIn: doc.fileURL?.deletingLastPathComponent(),
            finder: {
                let panel = NSSavePanel()
                panel.allowedContentTypes = [.json]
                panel.directoryURL = doc.fileURL?.deletingLastPathComponent() ?? model.library.workflows
                panel.nameFieldStringValue = doc.name + ".json"
                return panel.runModal() == .OK ? panel.url.map { [$0] } : nil
            },
            onChoose: { urls in
                guard let url = urls.first else { return }
                then?(model.save(doc, to: url))
            })
    }
}
