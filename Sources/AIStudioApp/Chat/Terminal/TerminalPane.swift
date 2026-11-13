import AppKit
import SwiftUI

/// The side panel's Terminal tab: the chat's terminals as tabs (+ for another), Add to Chat and Clear, and the shown
/// terminal under them. A shell that exited leaves its text and a Restart button.
struct TerminalPane: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 4) {
                ScrollView(.horizontal) {
                    HStack(spacing: 2) {
                        ForEach(Array(doc.terminals.enumerated()), id: \.element.id) { i, t in
                            TerminalTab(session: t, index: i + 1, selected: t === doc.activeTerminal,
                                        select: {
                                            doc.activeTerminalID = t.id
                                            model.focusTerminal(doc)
                                        },
                                        close: { model.closeTerminal(doc, t.id) })
                        }
                    }
                }
                .scrollIndicators(.never)
                Button {
                    model.newTerminal(doc)
                    model.focusTerminal(doc)
                } label: { Image(systemName: "plus") }
                    .accessibilityLabel("New Terminal")
                    .buttonStyle(.mg(.icon)).controlSize(.small)
                    .mgHelp("Open another terminal")
                    .mgTestAnchor("terminal-new")
                Spacer(minLength: 4)
                if let t = doc.activeTerminal {
                    Button { model.addTerminalToChat(doc, t) } label: {
                        Label("Add to Chat", systemImage: "arrowshape.turn.up.left").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Quote the selection (or the last lines) in your message")
                    .mgTestAnchor("terminal-add")
                    Button { t.clear() } label: { Image(systemName: "eraser") }
                        .accessibilityLabel("Clear")
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .mgHelp("Clear the terminal")
                }
            }
            .padding(.horizontal, 8)
            .frame(height: 38)
            MGDivider()
            if let t = doc.activeTerminal {
                VStack(spacing: 0) {
                    TerminalHost(session: t)
                        .padding(.leading, 6).padding(.top, 4)
                        .background(MG.windowBackground)  // the terminal's own background (TerminalTheme)
                    if !t.running {
                        MGDivider()
                        HStack(spacing: 8) {
                            Text("The shell exited" + (t.exitCode.map { " (code \($0))" } ?? ""))
                                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            Spacer(minLength: 0)
                            Button("Restart") {
                                t.start()
                                model.focusTerminal(doc)
                            }
                            .buttonStyle(.mg(.bordered)).controlSize(.small)
                            .mgTestAnchor("terminal-restart")
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                    }
                }
            } else {
                VStack(spacing: 12) {
                    MGEmptyState(symbol: "terminal", title: "No terminal",
                                 message: "A shell in this chat's project folder.")
                    Button("New Terminal") {
                        model.newTerminal(doc)
                        model.focusTerminal(doc)
                    }
                    .buttonStyle(.mg(.bordered))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear { model.focusTerminal(doc) }
        .mgTestAnchor("terminal-pane")
    }
}

/// A terminal's tab: its title (the shell's, or what it set), a dot while the shell runs, × to close.
private struct TerminalTab: View {
    let session: TerminalSession
    let index: Int
    let selected: Bool
    let select: () -> Void
    let close: () -> Void
    @State private var hover = false

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(session.running ? MG.success : MG.tertiaryLabel).frame(width: 6, height: 6)
            Text(session.shortTitle).font(.mgFootnote.weight(selected ? .semibold : .regular))
                .foregroundStyle(selected ? MG.label : MG.secondaryLabel)
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: 140)
            Button(action: close) { Image(systemName: "xmark").font(.system(size: 8, weight: .bold)) }
                .buttonStyle(.plain).foregroundStyle(MG.tertiaryLabel)
                .opacity(hover || selected ? 1 : 0.001)
                .accessibilityLabel("Close Terminal")
                .mgHelp("Close this terminal (its shell ends)")
        }
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
            .fill(selected ? MG.selection : hover ? MG.hover : .clear))
        .contentShape(Rectangle())
        .onTapGesture(perform: select)
        .onHover { hover = $0 }
        .mgHelp(session.title)
        .mgTestAnchor("terminal-tab:\(index)")
    }
}

/// Shows a session's terminal view. The view belongs to the session (made once), so switching tabs moves it in here
/// instead of making a new one, and the shell keeps running.
private struct TerminalHost: NSViewRepresentable {
    let session: TerminalSession

    func makeNSView(context: Context) -> NSView {
        let box = NSView()
        place(session.view, in: box)
        return box
    }

    func updateNSView(_ box: NSView, context: Context) {
        guard session.view.superview !== box else { return }
        box.subviews.forEach { $0.removeFromSuperview() }
        place(session.view, in: box)
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
