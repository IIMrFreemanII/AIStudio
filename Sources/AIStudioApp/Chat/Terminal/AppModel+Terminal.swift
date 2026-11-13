import AppKit
import LLMClient

/// A coding chat's terminals: opening and closing them, Add to Chat, and what `read_terminal` sees.
extension AppModel {
    /// Opens a terminal in the chat's project folder (its worktree for a worktree session) and shows it.
    @discardableResult
    func newTerminal(_ doc: ChatDocument) -> TerminalSession? {
        guard let folder = doc.projectURL else { return nil }
        let session = TerminalSession(folder: folder)
        session.onAddToChat = { [weak self, weak doc, weak session] in
            guard let self, let doc, let session else { return }
            self.addTerminalToChat(doc, session)
        }
        doc.terminals.append(session)
        doc.activeTerminalID = session.id
        return session
    }

    /// Closes a terminal (its shell and what it started end).
    func closeTerminal(_ doc: ChatDocument, _ id: UUID) {
        guard let i = doc.terminals.firstIndex(where: { $0.id == id }) else { return }
        doc.terminals[i].terminate()
        doc.terminals.remove(at: i)
        if doc.activeTerminalID == id { doc.activeTerminalID = doc.terminals.last?.id }
    }

    /// Ends all of a chat's terminals (the chat closes, the app quits).
    func closeTerminals(_ doc: ChatDocument) {
        doc.terminals.forEach { $0.terminate() }
        doc.terminals = []
        doc.activeTerminalID = nil
    }

    /// ⌃`: shows the terminal and puts the cursor in it; from the terminal, back to the composer.
    func toggleTerminal(_ doc: ChatDocument) {
        guard doc.isCoding else { return }
        if doc.sidePanel == .terminal, doc.flowPanelRunID == nil, let t = doc.activeTerminal,
           t.view.window?.firstResponder === t.view {
            doc.composerFocusRequest += 1
            return
        }
        showSidePanel(doc, .terminal)
        focusTerminal(doc)
    }

    /// Puts the cursor in the shown terminal (once it's in the window).
    func focusTerminal(_ doc: ChatDocument) {
        DispatchQueue.main.async { [weak doc] in
            guard let view = doc?.activeTerminal?.view else { return }
            view.window?.makeFirstResponder(view)
        }
    }

    /// The selection, or else the last lines, as a quote above the composer ("Replying to terminal output"). A quote
    /// keeps about 2,000 characters, so a long piece keeps its end (where the error usually is).
    func addTerminalToChat(_ doc: ChatDocument, _ session: TerminalSession) {
        var text = session.selection() ?? session.text(lastLines: 100)
        let room = ChatQuote.maxLength - 2
        if text.count > room { text = "…" + String(text.suffix(room)) }
        quote(doc, text, from: nil, source: "terminal output")
        doc.composerFocusRequest += 1
    }

    /// What `read_terminal` returns: a terminal's name, folder and state, then its last lines. `which` is a tab's
    /// number (from 1) or part of its title; nil: the shown one.
    func terminalReport(_ doc: ChatDocument, which: String?, lines: Int) -> String? {
        guard !doc.terminals.isEmpty else { return nil }
        let session: TerminalSession
        if let w = which?.trimmingCharacters(in: .whitespaces), !w.isEmpty {
            if let n = Int(w), doc.terminals.indices.contains(n - 1) {
                session = doc.terminals[n - 1]
            } else if let t = doc.terminals.first(where: { $0.title.localizedCaseInsensitiveContains(w) }) {
                session = t
            } else {
                let names = doc.terminals.enumerated().map { "\($0.offset + 1). \($0.element.title)" }.joined(separator: ", ")
                return "There is no terminal “\(w)”. The terminals: \(names)."
            }
        } else {
            guard let t = doc.activeTerminal else { return nil }
            session = t
        }
        let index = (doc.terminals.firstIndex { $0 === session } ?? 0) + 1
        var head = "Terminal \(index) of \(doc.terminals.count): “\(session.title)”"
        head += " · folder \(session.directory ?? session.folder.path)"
        if !session.running { head += " · the shell has exited" + (session.exitCode.map { " (code \($0))" } ?? "") }
        return head + "\n\n" + session.text(lastLines: lines)
    }
}
