import AppKit
import SwiftUI

/// A file's code editor: the buffer's text view with line numbers, under a bar for what needs the user's attention
/// (a conflict, a deleted file, a read-only reason, a short notice).
struct CodeEditorView: View {
    let buffer: FileBuffer
    @State private var chipSize = CGSize(width: 90, height: 22)

    var body: some View {
        VStack(spacing: 0) {
            bar
            CodeEditorHost(buffer: buffer)
                // Top-leading and sized to the chip: an overlay filling the editor would take its clicks.
                .overlay(alignment: .topLeading) { replyChip }
        }
        .overlay {
            if let error = buffer.loadError {
                MGEmptyState(symbol: "doc.questionmark", title: "Can't show this file", message: error)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: MGEditorColor.background))
            } else if !buffer.loaded {
                MGSpinner()
            }
        }
        .task { await buffer.load() }
    }

    @ViewBuilder private var bar: some View {
        switch buffer.state {
        case .conflict:
            row("This file changed on disk and your edits overlap.") {
                Button("Keep Mine") { buffer.keepMine() }
                Button("Use Disk Version") { buffer.useDisk() }
            }
            .mgTestAnchor("editor-conflict")
        case .deleted:
            row("This file was deleted.") {
                Button("Save Again") { buffer.saveAgain() }
            }
            .mgTestAnchor("editor-deleted")
        default:
            if let message = modeMessage {
                row(message) { EmptyView() }.mgTestAnchor("editor-mode")
            } else if let note = buffer.note {
                row(note) { EmptyView() }.mgTestAnchor("editor-note")
            }
        }
    }

    /// Reply ⌘L, under the end of the selection: quotes the selected lines in the message being written. Centred on
    /// the buffer's anchor by an offset, so the overlay is only as big as the chip (alignment guides were ignored).
    @ViewBuilder private var replyChip: some View {
        if let p = buffer.replyAnchor, ChatQuoting.active {
            Button { buffer.textView.mgReplyWithSelection(nil) } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrowshape.turn.up.left")
                    Text("Reply")
                    Text("⌘L").foregroundStyle(MG.secondaryLabel)
                }
                .font(.mgFootnote.weight(.medium))
            }
            .buttonStyle(.mg(.bordered)).controlSize(.small)
            // Opaque under the bordered style's translucent fill, so the code doesn't show through.
            .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(Color(nsColor: MGEditorColor.background)))
            .fixedSize()
            .shadow(color: .black.opacity(0.18), radius: 3, y: 1)
            .mgHelp("Reply to these lines in your message")
            .mgTestAnchor("editor-reply")
            .onGeometryChange(for: CGSize.self, of: \.size) { chipSize = $0 }
            // Offset moves the chip and where it takes clicks; the overlay itself stays the chip's size.
            .offset(x: p.x - chipSize.width / 2, y: p.y - chipSize.height / 2)
        }
    }

    private var modeMessage: String? {
        switch buffer.mode {
        case .readOnly(let m), .partial(let m): m
        default: nil
        }
    }

    private func row<Actions: View>(_ message: String, @ViewBuilder actions: () -> Actions) -> some View {
        let text = Text(message).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
        let buttons = HStack(spacing: 6) { actions() }.buttonStyle(.mg(.bordered)).controlSize(.small).fixedSize()
        return VStack(spacing: 0) {
            // Side by side when the message fits beside the buttons; in a narrow pane the buttons go underneath.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    text.lineLimit(1).fixedSize()
                    Spacer(minLength: 0)
                    buttons
                }
                VStack(alignment: .leading, spacing: 6) {
                    text.fixedSize(horizontal: false, vertical: true)
                    buttons
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(MG.controlBackground)
            MGDivider()
        }
    }
}

/// Shows the buffer's scroll view. The view belongs to the buffer (made once), so it is moved in here, as the
/// terminal's is (`TerminalHost`). Unlike the terminal, one buffer can have two hosts for a moment (Expand makes the
/// chat column's before the side pane's goes), so a new host takes the view, an update takes it only when no host
/// has it (another tab was chosen), and a host that goes away lets it go. Taking it back in an update would leave it
/// in the host being removed: a blank editor.
private struct CodeEditorHost: NSViewRepresentable {
    let buffer: FileBuffer

    func makeNSView(context: Context) -> NSView {
        let box = NSView()
        place(buffer.scrollView, in: box)
        return box
    }

    func updateNSView(_ box: NSView, context: Context) {
        let view = buffer.scrollView
        guard view.superview !== box else { return }
        if view.superview != nil {
            // Held by another host. An empty box here lost it to a newer host: leave it there. A box showing another
            // buffer (a tab switch) takes it unless that other host is on screen.
            guard !box.subviews.isEmpty, view.window == nil else { return }
        }
        box.subviews.forEach { $0.removeFromSuperview() }
        place(view, in: box)
    }

    static func dismantleNSView(_ box: NSView, coordinator: ()) {
        box.subviews.forEach { $0.removeFromSuperview() }
    }

    private func place(_ view: NSView, in box: NSView) {
        view.removeFromSuperview()
        view.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(view)
        NSLayoutConstraint.activate([
            view.leadingAnchor.constraint(equalTo: box.leadingAnchor),
            view.trailingAnchor.constraint(equalTo: box.trailingAnchor),
            view.topAnchor.constraint(equalTo: box.topAnchor),
            view.bottomAnchor.constraint(equalTo: box.bottomAnchor),
        ])
    }
}
