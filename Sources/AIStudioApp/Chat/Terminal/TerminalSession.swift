import Agent
import AppKit
import SwiftTerm

/// One terminal of a coding chat: your login shell on a pseudo-terminal, in the chat's project folder (its worktree
/// for a worktree session). The terminal view is made once and kept, so switching tabs or rebuilding the pane never
/// restarts the shell; it ends when the tab, the chat or the app closes.
@MainActor @Observable
final class TerminalSession: Identifiable {
    let id = UUID()
    let folder: URL
    /// The shell's name, or the title it set (most shells set it to the folder or the running command).
    private(set) var title: String
    /// The folder the shell reports it's in (OSC 7), when it does.
    private(set) var directory: String?
    private(set) var running = false
    private(set) var exitCode: Int32?

    @ObservationIgnored let view: AppTerminalView
    @ObservationIgnored private var events: Events?
    /// Called for the context menu's Add to Chat.
    @ObservationIgnored var onAddToChat: (() -> Void)?

    /// The shell: yours ($SHELL) when it's there, else zsh.
    nonisolated static var shell: String {
        if let s = ProcessInfo.processInfo.environment["SHELL"], FileManager.default.isExecutableFile(atPath: s) { return s }
        return "/bin/zsh"
    }

    init(folder: URL) {
        self.folder = folder
        title = (Self.shell as NSString).lastPathComponent
        view = AppTerminalView(frame: NSRect(x: 0, y: 0, width: 600, height: 400),
                               font: Self.font,
                               options: TerminalOptions(scrollback: 10_000))
        let events = Events(session: self)
        self.events = events
        view.processDelegate = events
        view.onAddToChat = { [weak self] in self?.onAddToChat?() }
        view.optionAsMetaKey = false  // ⌥ types accented letters, as in Terminal.app
        TerminalTheme.apply(to: view)
        start()
    }

    /// JetBrains Mono, as a Nerd Font when one is installed: prompts (Starship, Powerlevel10k) and tools like eza draw
    /// icons from the Nerd Font set, which the bundled JetBrains Mono doesn't have (they'd show as "?").
    static let font: NSFont = {
        for name in ["JetBrainsMonoNF-Regular", "JetBrainsMonoNerdFont-Regular", "JetBrainsMonoNFM-Regular",
                     "JetBrainsMonoNerdFontMono-Regular", "JetBrainsMono-Regular"] {
            if let f = NSFont(name: name, size: 12) { return f }
        }
        return .monospacedSystemFont(ofSize: 12, weight: .regular)
    }()

    /// Starts (or, after it exited, restarts) the shell as a login shell in the project folder.
    func start() {
        resetModes()
        var env = ProcessInfo.processInfo.environment
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["TERM_PROGRAM"] = "AIStudio"
        if env["LANG"] == nil { env["LANG"] = "en_US.UTF-8" }
        env["PATH"] = ShellRunner.loginPath
        let shell = Self.shell
        view.startProcess(executable: shell, args: ["-l"], environment: env.map { "\($0.key)=\($0.value)" },
                          currentDirectory: folder.path)
        title = (shell as NSString).lastPathComponent
        running = true
        exitCode = nil
    }

    /// Before a (re)start: undo what a program that died mid-way may have left on (mouse reporting, which stops you
    /// selecting text; the full-screen buffer; a hidden cursor; bracketed paste; app keypad), keeping the old text.
    /// A soft reset (DECSTR) doesn't cover these.
    private func resetModes() {
        let terminal = view.getTerminal()
        if terminal.isCurrentBufferAlternate { view.feed(text: "\u{1B}[?1049l") }
        view.feed(text: "\u{1B}[?1000l\u{1B}[?1002l\u{1B}[?1003l\u{1B}[?1005l\u{1B}[?1006l\u{1B}[?1015l"
                      + "\u{1B}[?1004l\u{1B}[?2004l\u{1B}[?25h\u{1B}[?1l\u{1B}>\u{1B}[0m")
    }

    /// Ends the shell and everything it started: SIGHUP to its process group (as closing a terminal window does),
    /// SIGKILL 2 s later if it's still there.
    func terminate() {
        let pid = view.process.shellPid
        view.terminate()
        running = false
        guard pid > 0 else { return }
        kill(-pid, SIGHUP)
        kill(pid, SIGHUP)
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
            if kill(pid, 0) == 0 {
                kill(-pid, SIGKILL)
                kill(pid, SIGKILL)
            }
        }
    }

    /// The title for a tab: a shell's "user@host:/path/to/folder" as just "folder" ("~" at home).
    var shortTitle: String {
        guard let colon = title.firstIndex(of: ":"), title[..<colon].contains("@") else { return title }
        let path = title[title.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        if path.isEmpty || path == "~" { return "~" }
        return (path as NSString).lastPathComponent
    }

    /// The shell's process ID (0 before it starts).
    var pid: pid_t { view.process.shellPid }

    /// Waiting at its prompt: the shell runs, nothing it started is in front (no command running, no vim).
    var isIdle: Bool {
        guard running, pid > 0, !view.getTerminal().isCurrentBufferAlternate else { return false }
        let fd = view.process.childfd
        guard fd >= 0 else { return false }
        return tcgetpgrp(fd) == pid
    }

    /// Runs commands as if typed and Return pressed. Several lines go as one paste when the shell takes bracketed
    /// paste (zsh, bash): it gets the block as one edit, so a multi-line command (a loop, a heredoc) stays whole.
    func runCommand(_ text: String) {
        if !text.contains("\n") {
            view.send(txt: text + "\r")
        } else if view.getTerminal().bracketedPasteMode {
            view.send(txt: "\u{1B}[200~" + text + "\u{1B}[201~\r")
        } else {
            view.send(txt: text.replacingOccurrences(of: "\n", with: "\r") + "\r")
        }
    }

    /// Types text into the terminal as if from the keyboard (`\r` for Return).
    func send(_ text: String) { view.send(txt: text) }

    /// The last lines of the scrollback and screen as plain text (no colours), trailing blank lines dropped. While a
    /// full-screen program runs (vim, htop) it's that program's screen.
    func text(lastLines n: Int) -> String {
        let data = view.getTerminal().getBufferAsData()
        var lines = String(decoding: data, as: UTF8.self).components(separatedBy: "\n")
        while let l = lines.last, l.trimmingCharacters(in: .whitespaces).isEmpty { lines.removeLast() }
        return Self.withoutIcons(lines.suffix(max(1, n)).joined(separator: "\n"))
    }

    /// The text without the prompt waiting at its end, when that prompt can be told apart: the lines after the last
    /// blank line (prompts such as Starship's and Powerlevel10k's start with one), at most 3, whose first line was
    /// printed before (the same prompt, above the previous command, maybe with that command after it). Otherwise the text as it is.
    nonisolated static func withoutTrailingPrompt(_ text: String) -> String {
        var lines = text.components(separatedBy: "\n")
        while let l = lines.last, l.trimmingCharacters(in: .whitespaces).isEmpty { lines.removeLast() }
        guard let blank = lines.lastIndex(where: { $0.trimmingCharacters(in: .whitespaces).isEmpty }),
              lines.count - blank - 1 <= 3, blank + 1 < lines.count else { return text }
        let first = lines[blank + 1].trimmingCharacters(in: .whitespaces)
        guard !first.isEmpty, lines[..<blank].contains(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix(first) }) else {
            return text
        }
        return lines[..<blank].joined(separator: "\n")
    }

    /// The selected text, nil when nothing is selected.
    func selection() -> String? {
        guard let s = view.getSelection(), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return Self.withoutIcons(s)
    }

    /// Text without Nerd Font icons (prompts and tools draw them from Unicode's private use areas): outside the
    /// terminal they're unreadable boxes, to you in a quote and to the model.
    nonisolated static func withoutIcons(_ s: String) -> String {
        var out = String.UnicodeScalarView()
        for u in s.unicodeScalars where !((0xE000...0xF8FF).contains(u.value) || u.value >= 0xF0000) {
            out.append(u)
        }
        return String(out)
    }

    /// Clears the scrollback and asks the shell to redraw its prompt (Ctrl-L).
    func clear() {
        view.feed(text: "\u{1B}[3J")
        view.send(txt: "\u{0C}")
    }

    /// The terminal's events (the view keeps its delegate weakly).
    final class Events: LocalProcessTerminalViewDelegate {
        weak var session: TerminalSession?
        init(session: TerminalSession) { self.session = session }

        func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}

        func setTerminalTitle(source: LocalProcessTerminalView, title: String) {
            let t = title.trimmingCharacters(in: .whitespaces)
            DispatchQueue.main.async { [weak self] in
                if !t.isEmpty { self?.session?.title = t }
            }
        }

        func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {
            // OSC 7 sends a file:// URL.
            let path = directory.flatMap { URL(string: $0)?.path } ?? directory
            DispatchQueue.main.async { [weak self] in self?.session?.directory = path }
        }

        func processTerminated(source: TerminalView, exitCode: Int32?) {
            // SwiftTerm passes the raw wait status (130 << 8 for "exit 130"): the exit code, or 128 + the signal.
            let code = exitCode.map { s in (s & 0x7F) == 0 ? (s >> 8) & 0xFF : 128 + (s & 0x7F) }
            DispatchQueue.main.async { [weak self] in
                guard let session = self?.session else { return }
                session.running = false
                session.exitCode = code
                // The title was the shell's last (often the "exit" it ran); name the tab after the shell again.
                session.title = (TerminalSession.shell as NSString).lastPathComponent
            }
        }
    }
}

/// The terminal view, with the app's context menu and theme kept in step with light and dark mode.
final class AppTerminalView: LocalProcessTerminalView {
    var onAddToChat: (() -> Void)?

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        TerminalTheme.apply(to: self)
    }

    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = NSMenu()
        let add = NSMenuItem(title: getSelection()?.isEmpty == false ? "Add Selection to Chat" : "Add Last Lines to Chat",
                             action: #selector(addToChat), keyEquivalent: "")
        add.target = self
        menu.addItem(add)
        menu.addItem(.separator())
        for (title, action) in [("Copy", #selector(copy(_:))), ("Paste", #selector(paste(_:))), ("Select All", #selector(selectAll(_:)))] {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        return menu
    }

    @objc func addToChat() { onAddToChat?() }

    override func validateUserInterfaceItem(_ item: NSValidatedUserInterfaceItem) -> Bool {
        item.action == #selector(addToChat) ? true : super.validateUserInterfaceItem(item)
    }
}

/// The terminal's colours: the design system's background and text, and an ANSI palette tuned for each appearance.
enum TerminalTheme {
    @MainActor static func apply(to view: TerminalView) {
        let dark = view.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        var background = NSColor.black, foreground = NSColor.white, caret = NSColor.white, selection = NSColor.gray
        view.effectiveAppearance.performAsCurrentDrawingAppearance {
            // Opaque versions of the tokens: the terminal draws its own cells.
            background = (MGColor.windowBackground.usingColorSpace(.sRGB) ?? .black).withAlphaComponent(1)
            foreground = (MGColor.label.usingColorSpace(.sRGB) ?? .white)
            caret = MGColor.accent.usingColorSpace(.sRGB) ?? .systemBlue
            selection = MGColor.selection.usingColorSpace(.sRGB) ?? .selectedTextBackgroundColor
        }
        view.nativeBackgroundColor = background
        view.nativeForegroundColor = foreground
        view.caretColor = caret
        view.selectedTextBackgroundColor = selection
        view.installColors((dark ? darkPalette : lightPalette).map(color))
    }

    /// Black, red, green, yellow, blue, magenta, cyan, white, then their bright versions.
    static let darkPalette: [UInt32] = [
        0x1E1E21, 0xFF6B6B, 0x5FD38D, 0xE5C07B, 0x5AA9FF, 0xC678DD, 0x56C8D8, 0xC8C8CC,
        0x5C5C63, 0xFF8787, 0x7EE2A8, 0xF0D08A, 0x7DBBFF, 0xD69AE8, 0x7DD8E5, 0xFFFFFF,
    ]
    static let lightPalette: [UInt32] = [
        0x1D1D1F, 0xC9302C, 0x1E8E4E, 0x9A6B00, 0x0A63D6, 0x9B3FB5, 0x0F7F8F, 0xD0D0D4,
        0x6E6E73, 0xE0443F, 0x27A35C, 0xB58100, 0x2F7CF0, 0xB45AD0, 0x1593A5, 0xF5F5F7,
    ]

    static func color(_ hex: UInt32) -> SwiftTerm.Color {
        SwiftTerm.Color(red8: UInt16(hex >> 16 & 0xFF), green8: UInt16(hex >> 8 & 0xFF), blue8: UInt16(hex & 0xFF))
    }
}
