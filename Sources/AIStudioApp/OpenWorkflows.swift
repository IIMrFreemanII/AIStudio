import AppKit
import SwiftUI

/// The open workflows, at the top of the Workflows panel: click to bring one in front, × (or ⌘W) to close,
/// + for a preset. Unsaved ones live only here until they're saved.
struct OpenWorkflowsSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 4) {
            Text("OPEN").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
            Spacer()
            Button { model.showPresetGallery = true } label: { Image(systemName: "plus").font(.system(size: 11, weight: .medium)) }
                .buttonStyle(.mg(.icon))
                .controlSize(.small)
                .mgHelp("New workflow from a preset (⌘N); ⌘T for a blank text-to-image workflow")
                .accessibilityLabel("New workflow")
        }
        .padding(.leading, 16).padding(.trailing, 10).padding(.top, 4).padding(.bottom, 2)
        ForEach(model.tabs) { doc in OpenWorkflowRow(doc: doc) }
        MGDivider().padding(.horizontal, 12).padding(.vertical, 6)
    }
}

/// One open workflow: a spinner while it runs, its name, a dot for unsaved changes, × to close.
struct OpenWorkflowRow: View {
    @Environment(AppModel.self) private var model
    let doc: WorkflowDocument
    @State private var hovering = false

    private var active: Bool { doc.id == model.activeTabID && model.activeChatID == nil }

    var body: some View {
        HStack(spacing: 6) {
            if model.runningDocID == doc.id {
                MGSpinner().controlSize(.mini).frame(width: 16)
            } else {
                Image(systemName: doc.fileURL == nil ? "doc" : "doc.text")
                    .foregroundStyle(active ? MG.accent : MG.secondaryLabel).frame(width: 16)
            }
            Text(doc.name).lineLimit(1)
            Spacer(minLength: 0)
            ZStack {
                if hovering {
                    Button { TabCommands.close(doc, model: model) } label: {
                        Image(systemName: "xmark").font(.system(size: 8, weight: .bold))
                    }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.mini)
                    .mgHelp("Close (⌘W)")
                } else if doc.isDirty {
                    Circle().fill(MG.secondaryLabel).frame(width: 7, height: 7).mgHelp("Unsaved changes")
                }
            }
            .frame(width: 18, height: 18)
        }
        .mgRow(selected: active)
        .mgTestAnchor("open-workflow:\(doc.name)")
        .onHover { hovering = $0 }
        .onTapGesture { model.activate(doc.id) }
        .mgHelp(doc.fileURL?.path ?? "Not saved yet")
        .mgContextMenu {
            MGMenuItem.button("Close", shortcut: KeyboardShortcut("w")) { TabCommands.close(doc, model: model) }
            MGMenuItem.button("Close Others", enabled: model.tabs.count > 1) {
                TabCommands.close(model.tabs.filter { $0.id != doc.id }, model: model)
            }
            MGMenuItem.divider
            MGMenuItem.button(doc.fileURL == nil ? "Save…" : "Save") { TabCommands.save(doc, model: model) }
            MGMenuItem.button("Duplicate") { model.openTab(doc.graph, name: doc.name, presetID: doc.presetID) }
            if let url = doc.fileURL {
                MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
        }
    }
}

/// Closing and saving open workflows, shared by the Workflows panel and the menus. Questions go through the window's presenter, so
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
