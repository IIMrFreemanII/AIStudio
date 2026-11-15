import AppKit
import SwiftUI

/// The open files: a strip of tabs (with Widen and Expand), and the active file below it. In the Files pane under the
/// tree, or, expanded, in place of the chat's transcript.
struct FileEditorArea: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument
    var expanded = false
    @AppStorage("sidePanelWidths") private var paneWidths = ""

    var body: some View {
        VStack(spacing: 0) {
            strip
            if let rel = doc.openFiles.list.active, let root = doc.projectURL {
                FilePreview(url: root.appendingPathComponent(rel), title: rel, revision: doc.filesRevision,
                            onMention: { model.mentionInDraft(doc, rel) },
                            onClose: { model.closeFile(doc, rel) },
                            buffer: model.fileBuffer(doc, rel))
                    .id(rel)
                    .frame(maxHeight: .infinity)
            } else {
                Color.clear.frame(maxHeight: .infinity)
            }
        }
        .onExitCommand { if expanded { doc.editorExpanded = false } }
        .mgTestAnchor(expanded ? "file-editor-expanded" : "file-editor")
    }

    private var strip: some View {
        let list = doc.openFiles.list
        let names = Dictionary(grouping: list.tabs, by: { ($0 as NSString).lastPathComponent }).mapValues(\.count)
        return VStack(spacing: 0) {
            HStack(spacing: 4) {
                ScrollViewReader { scroller in
                ScrollView(.horizontal) {
                    HStack(spacing: 2) {
                        ForEach(list.tabs, id: \.self) { rel in
                            MGTab(title: title(rel, names), selected: rel == list.active,
                                  italic: list.isTransient(rel),
                                  strikethrough: doc.openFiles.buffers[rel]?.state == .deleted,
                                  dirty: doc.openFiles.buffers[rel]?.isDirty == true,
                                  help: rel, closeHelp: "Close this file (unsaved changes are saved)", closeLabel: "Close File",
                                  anchor: "file-tab:\(rel)",
                                  doubleClick: { model.openFile(doc, rel, pin: true) },
                                  leading: { Image(nsImage: FileIcons.icon(for: rel)).resizable().frame(width: 14, height: 14) },
                                  select: { model.openFile(doc, rel) },
                                  close: { model.closeFile(doc, rel) })
                            .id(rel)
                        }
                    }
                    .padding(.horizontal, 6)
                }
                .scrollIndicators(.never)
                // The active tab is always in view (a file opened from the tree may land past the strip's edge).
                .onChange(of: list.active, initial: true) { _, active in
                    guard let active else { return }
                    withAnimation(.easeOut(duration: 0.15)) { scroller.scrollTo(active) }
                }
                }
                Spacer(minLength: 0)
                if !expanded {
                    Button(action: widen) { Image(systemName: "arrow.left.and.right") }
                        .accessibilityLabel(doc.filesWideRestore == nil ? "Widen" : "Restore Width")
                        .mgHelp(doc.filesWideRestore == nil ? "Widen the side panel" : "Restore the side panel's width")
                        .mgTestAnchor("files-widen")
                }
                Button { doc.editorExpanded.toggle() } label: {
                    Image(systemName: expanded ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                }
                .accessibilityLabel(expanded ? "Back to Chat" : "Expand")
                .mgHelp(expanded ? "Back to the chat" : "Show the file in place of the chat")
                .mgTestAnchor("files-expand")
            }
            .buttonStyle(.mg(.icon)).controlSize(.small)
            .padding(.trailing, 8)
            .frame(height: 32)
            MGDivider()
        }
    }

    /// The file's name; with its folder when another open tab has the same name.
    private func title(_ rel: String, _ names: [String: Int]) -> String {
        let name = (rel as NSString).lastPathComponent
        guard names[name, default: 0] > 1 else { return name }
        let folder = ((rel as NSString).deletingLastPathComponent as NSString).lastPathComponent
        return folder.isEmpty ? name : "\(folder)/\(name)"
    }

    /// Widens the Files pane to nearly the window (the chat keeps 420 pt), or puts its width back.
    private func widen() {
        let width = SidePanelWidths.binding($paneWidths, .files)
        if let before = doc.filesWideRestore {
            width.wrappedValue = before
            doc.filesWideRestore = nil
        } else {
            doc.filesWideRestore = width.wrappedValue
            let window = NSApp.keyWindow?.contentView?.bounds.width ?? 1400
            let r = SidePanelWidths.range
            width.wrappedValue = min(max(min(1000, window - 420), r.lowerBound), r.upperBound)
        }
    }
}
