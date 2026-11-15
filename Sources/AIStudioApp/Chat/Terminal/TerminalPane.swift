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
                            MGTab(title: t.shortTitle, selected: t === doc.activeTerminal, help: t.title,
                                  closeHelp: "Close this terminal (its shell ends)", closeLabel: "Close Terminal",
                                  anchor: "terminal-tab:\(i + 1)",
                                  leading: { Circle().fill(t.running ? MG.success : MG.tertiaryLabel).frame(width: 6, height: 6) },
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
                MGEmptyState(symbol: "terminal", title: "No terminal", message: "A shell in this chat's project folder.",
                             action: ("New Terminal", {
                                 model.newTerminal(doc)
                                 model.focusTerminal(doc)
                             }), actionAnchor: "terminal-empty-new")
            }
        }
        .onAppear { model.focusTerminal(doc) }
        .mgTestAnchor("terminal-pane")
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
