import AppKit
import CodeEditing
import Observation

/// A file open in the code editor: its text view, its state against the disk, autosave and a watcher that reconciles
/// changes made elsewhere (reload, three-way merge, conflict, deleted). Read-only buffers never write.
@MainActor @Observable
final class FileBuffer {
    enum State: Equatable { case clean, dirty, conflict(disk: String), deleted }

    let url: URL
    /// Project-relative path; nil for a file outside a project (an attachment).
    let relPath: String?
    private(set) var editable: Bool
    private(set) var mode: EditorMode = .editable
    private(set) var language: CodeLanguage?
    private(set) var state: State = .clean
    private(set) var loaded = false
    private(set) var loadError: String?
    /// A short-lived notice ("Reloaded — changed on disk."), cleared after 2.5 s.
    private(set) var note: String?
    var isDirty: Bool {
        switch state {
        case .dirty, .conflict: true
        default: false
        }
    }

    @ObservationIgnored let undo = UndoManager()
    @ObservationIgnored let textView: CodeTextView
    @ObservationIgnored let scrollView: NSScrollView
    /// The user typed (the owner pins a transient tab).
    @ObservationIgnored var onEdited: (() -> Void)?
    /// Written to disk (the owner refreshes git).
    @ObservationIgnored var onSaved: (() -> Void)?
    /// Forwarded from the text view (selection chip, ⌘L).
    @ObservationIgnored var onReply: ((NSRange) -> Void)? {
        didSet { updateReplyAnchor() }
    }
    /// Where the Reply chip goes: under the end of the selection, in the editor's coordinates (top-left origin); nil
    /// with no selection, while dragging, when it is scrolled out of view, or when replies aren't wired.
    private(set) var replyAnchor: CGPoint?
    var text: String { textView.string }
    /// Counts text changes (typing, undo), so SwiftUI can follow edits.
    private(set) var edits = 0

    @ObservationIgnored private let wantsEditable: Bool
    @ObservationIgnored private let watches: Bool
    @ObservationIgnored private var base: FileSnapshot?
    @ObservationIgnored private var loadTask: Task<Void, Never>?
    @ObservationIgnored private var autosave: Task<Void, Never>?
    @ObservationIgnored private var noteTask: Task<Void, Never>?
    @ObservationIgnored private var watcher: FileWatcher?
    @ObservationIgnored private var styler: SyntaxStyler?
    @ObservationIgnored private var ruler: LineNumberRuler?
    @ObservationIgnored private var delegate: Delegate?
    @ObservationIgnored private var observers: [NSObjectProtocol] = []
    @ObservationIgnored private var pendingReveal: ClosedRange<Int>?
    @ObservationIgnored private var revealWaitsForSize = false

    init(url: URL, relPath: String?, editable: Bool, watch: Bool) {
        self.url = url
        self.relPath = relPath
        self.wantsEditable = editable
        self.editable = editable
        self.watches = watch
        textView = CodeTextView.make()
        scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = true
        scrollView.backgroundColor = MGEditorColor.background
        scrollView.borderType = .noBorder
        let ruler = LineNumberRuler(scrollView: scrollView, textView: textView)
        scrollView.verticalRulerView = ruler
        scrollView.hasVerticalRuler = true
        scrollView.rulersVisible = true
        self.ruler = ruler
        textView.undoOwner = undo
        textView.isEditable = false
        textView.onReply = { [weak self] range in self?.onReply?(range) }
        textView.onSelectionSettled = { [weak self] in self?.updateReplyAnchor() }
        scrollView.contentView.postsBoundsChangedNotifications = true
        observers.append(NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification, object: scrollView.contentView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateReplyAnchor() }
        })
        let delegate = Delegate(self)
        self.delegate = delegate
        textView.delegate = delegate
        scrollView.postsFrameChangedNotifications = true
        observers.append(NotificationCenter.default.addObserver(
            forName: NSView.frameDidChangeNotification, object: scrollView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.sizeChanged() }
        })
    }

    // MARK: loading

    private struct Loaded: Sendable {
        var mode: EditorMode
        var text: String
        var stamp: FileStamp?
    }

    func load() async {
        if let loadTask { await loadTask.value; return }
        let task = Task { [self] in
            let result = await Self.read(url)
            guard let result else {
                loadError = "Couldn't read the file."
                loaded = true
                return
            }
            apply(result, first: true)
        }
        loadTask = task
        await task.value
    }

    private nonisolated static func read(_ url: URL) async -> Loaded? {
        await Task.detached { () -> Loaded? in
            let stamp = FileStamp.of(url)
            guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? h.close() }
            let head = (try? h.read(upToCount: EditorLimits.editableBytes + 1)) ?? Data()
            let size = stamp?.size ?? head.count
            let mode = EditorLimits.mode(head: head, size: size)
            switch mode {
            case .binary:
                return Loaded(mode: mode, text: "", stamp: stamp)
            case .partial:
                return Loaded(mode: mode, text: String(decoding: head.prefix(EditorLimits.editableBytes), as: UTF8.self), stamp: stamp)
            case .readOnly:
                var data = head
                if head.count > EditorLimits.editableBytes, let rest = try? h.readToEnd() { data.append(rest) }
                return Loaded(mode: mode, text: String(decoding: data, as: UTF8.self), stamp: stamp)
            case .editable, .plainEditable:
                return Loaded(mode: mode, text: String(decoding: head, as: UTF8.self), stamp: stamp)
            }
        }.value
    }

    /// Takes a read's result: the first load sets everything up; a read-only buffer's later reads replace the text.
    private func apply(_ r: Loaded, first: Bool) {
        mode = r.mode
        if case .binary = r.mode {
            loadError = "Not a text file."
            editable = false
            loaded = true
            return
        }
        let canEdit: Bool
        switch r.mode {
        case .editable, .plainEditable: canEdit = wantsEditable
        default: canEdit = false
        }
        editable = canEdit
        language = r.mode == .editable ? CodeLanguage.detect(path: url.path, firstLine: firstLine(of: r.text)) : nil
        let ext = url.pathExtension.lowercased()
        textView.wraps = language == .markdown || r.mode == .plainEditable || (language == nil && (ext == "txt" || ext.isEmpty))
        fitWrappedWidth()
        if first {
            textView.indentUnit = Indentation.detect(r.text).unit
            setText(r.text)
        } else {
            replaceText(r.text)
        }
        if let stamp = r.stamp { base = FileSnapshot(stamp: stamp, text: r.text) }
        textView.isEditable = canEdit
        state = .clean
        if first {
            if let language, styler == nil {
                styler = SyntaxStyler(textView: textView, scrollView: scrollView, language: language)
                styler?.textChanged(immediately: true)
            }
            if watches && canEdit { startWatcher() }
            if canEdit { observeResign() }
            loaded = true
            if let lines = pendingReveal { pendingReveal = nil; reveal(lines: lines) }
        }
    }

    private func firstLine(of text: String) -> String? {
        text.prefix(200).split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init)
    }

    // MARK: text replacement

    /// Puts the text in without an undo entry (first load).
    private func setText(_ new: String) {
        undo.disableUndoRegistration()
        let storage = textView.textStorage!
        storage.replaceCharacters(in: NSRange(location: 0, length: storage.length),
                                  with: NSAttributedString(string: new, attributes: CodeTextView.baseAttributes))
        undo.enableUndoRegistration()
        undo.removeAllActions()
        textView.typingAttributes = CodeTextView.baseAttributes
        textView.setSelectedRange(NSRange(location: 0, length: 0))
        ruler?.invalidate()
        styler?.textChanged(immediately: true)
    }

    /// Replaces the whole text without an undo entry, keeping the selection (by line and column) and the first
    /// visible line.
    private func replaceText(_ new: String) {
        let old = textView.string
        guard old != new else { return }
        let oldLines = TextLines(old), newLines = TextLines(new)
        let selection = TextLines.remap(textView.selectedRange(), from: oldLines, to: newLines)
        let clip = scrollView.contentView
        let topLine = firstVisibleLine(oldLines)
        let storage = textView.textStorage!
        undo.disableUndoRegistration()
        storage.replaceCharacters(in: NSRange(location: 0, length: storage.length),
                                  with: NSAttributedString(string: new, attributes: CodeTextView.baseAttributes))
        undo.enableUndoRegistration()
        undo.removeAllActions()
        textView.setSelectedRange(selection)
        if let lm = textView.layoutManager, lm.numberOfGlyphs > 0, clip.bounds.height > 0 {
            let start = min(newLines.range(ofLine: topLine).location, max((new as NSString).length - 1, 0))
            let rect = lm.lineFragmentRect(forGlyphAt: lm.glyphIndexForCharacter(at: start), effectiveRange: nil)
            clip.scroll(to: NSPoint(x: clip.bounds.origin.x, y: max(0, rect.minY + textView.textContainerOrigin.y - 6)))
            scrollView.reflectScrolledClipView(clip)
        }
        ruler?.invalidate()
        styler?.textChanged(immediately: true)
    }

    private func firstVisibleLine(_ lines: TextLines) -> Int {
        guard let lm = textView.layoutManager, let tc = textView.textContainer, lm.numberOfGlyphs > 0 else { return 1 }
        let top = scrollView.contentView.bounds.origin.y - textView.textContainerOrigin.y
        var point = NSPoint(x: 0, y: max(top, 0))
        point.y += 1
        let glyph = lm.glyphIndex(for: point, in: tc)
        return lines.line(at: lm.characterIndexForGlyph(at: glyph))
    }

    // MARK: reveal

    /// Selects the lines and scrolls them to the centre (after the load when it isn't done).
    func reveal(lines: ClosedRange<Int>) {
        guard loaded else { pendingReveal = lines; return }
        let range = TextLines(textView.string).range(ofLines: lines)
        textView.setSelectedRange(range)
        revealWaitsForSize = scrollView.frame.width < 1
        if revealWaitsForSize { pendingReveal = lines } else { scrollToCenter(range) }
    }

    private func scrollToCenter(_ range: NSRange) {
        guard let lm = textView.layoutManager, let tc = textView.textContainer, lm.numberOfGlyphs > 0 else { return }
        let glyphs = lm.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
        let rect = lm.boundingRect(forGlyphRange: glyphs, in: tc).offsetBy(dx: textView.textContainerOrigin.x, dy: textView.textContainerOrigin.y)
        let clip = scrollView.contentView
        let maxY = max(0, textView.frame.height - clip.bounds.height)
        let y = min(max(0, rect.midY - clip.bounds.height / 2), maxY)
        clip.scroll(to: NSPoint(x: -clip.contentInsets.left, y: y))  // the gutter floats over the left inset
        scrollView.reflectScrolledClipView(clip)
    }

    /// A wrapping text view is as wide as what shows of it. Its autoresizing only follows later changes, so a view
    /// made 600 wide and placed in a narrower pane would wrap at 600 and sit scrolled sideways. The gutter floats over
    /// the clip view's left inset (the clip's bounds start at minus its width), so that inset is left out and kept.
    private func fitWrappedWidth() {
        guard textView.wraps else { return }
        let clip = scrollView.contentView
        let width = clip.bounds.width - clip.contentInsets.left - clip.contentInsets.right
        guard width >= 1, abs(textView.frame.width - width) > 0.5 else { return }
        textView.setFrameSize(NSSize(width: width, height: textView.frame.height))
        clip.scroll(to: NSPoint(x: -clip.contentInsets.left, y: clip.bounds.origin.y))
        scrollView.reflectScrolledClipView(clip)
    }

    private func sizeChanged() {
        fitWrappedWidth()
        guard revealWaitsForSize, scrollView.frame.width >= 1, let lines = pendingReveal else { return }
        revealWaitsForSize = false
        pendingReveal = nil
        scrollToCenter(TextLines(textView.string).range(ofLines: lines))
    }

    // MARK: typing and saving

    fileprivate func userEdited() {
        edits += 1
        if state == .clean { state = .dirty }
        onEdited?()
        styler?.textChanged()
        ruler?.invalidate()
        autosave?.cancel()
        autosave = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard !Task.isCancelled else { return }
            self?.save()
        }
    }

    private func observeResign() {
        observers.append(NotificationCenter.default.addObserver(
            forName: NSApplication.willResignActiveNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.save() }
        })
    }

    /// Writes the buffer now when it has unsaved edits (not while in conflict or after a delete).
    func save() {
        guard editable, state == .dirty, let base else { return }
        if !FileSync.canSave(base: base, diskStamp: FileStamp.of(url)) { checkDisk(); return }
        let t = text
        do { try write(t) } catch {
            show("Couldn't save: \(error.localizedDescription)")
            return
        }
        guard let stamp = FileStamp.of(url) else { return }
        self.base = FileSnapshot(stamp: stamp, text: t)
        if text == t { state = .clean }
        onSaved?()
    }

    /// In place (keeps the inode, permissions and extended attributes); creates the file when it is missing.
    private func write(_ t: String) throws {
        let data = Data(t.utf8)
        if FileManager.default.fileExists(atPath: url.path) {
            let h = try FileHandle(forWritingTo: url)
            defer { try? h.close() }
            try h.truncate(atOffset: 0)
            try h.write(contentsOf: data)
            try h.synchronize()
        } else {
            try data.write(to: url)
        }
    }

    // MARK: the disk

    private func readDiskText() -> String? {
        guard let data = try? Data(contentsOf: url), data.count <= EditorLimits.editableBytes else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Compares with the disk and acts (watcher events, owners on app activation).
    func checkDisk() {
        guard loaded, editable, let base else { return }
        let stamp = FileStamp.of(url)
        if let stamp, stamp == base.stamp { return }
        if stamp == nil, state == .deleted { return }
        if let stamp, stamp.size > EditorLimits.editableBytes {
            editable = false
            textView.isEditable = false
            watcher?.stop()
            watcher = nil
            show("The file grew past 2 MB: read-only.")
            return
        }
        let diskText = stamp == nil ? nil : readDiskText()
        switch FileSync.decide(base: base, diskStamp: stamp, diskText: diskText, buffer: text,
                               dirty: isDirty || state == .deleted) {
        case .none:
            break
        case .adoptStamp(let s):
            self.base?.stamp = s
            if state == .dirty { save() }
        case .reload(let t):
            adopt(t, stamp: stamp!)
            show("Reloaded — changed on disk.")
        case .merged(let t):
            replaceText(t)
            self.base = FileSnapshot(stamp: stamp!, text: diskText ?? t)
            state = .dirty
            show("Merged changes from disk.")
            save()
        case .conflict(let d):
            state = .conflict(disk: d)
        case .deleted:
            state = .deleted
        }
    }

    /// The disk text becomes the buffer: clean.
    private func adopt(_ t: String, stamp: FileStamp) {
        replaceText(t)
        base = FileSnapshot(stamp: stamp, text: t)
        state = .clean
    }

    /// Read-only buffers: read the file again (the text is replaced, the scroll kept). Editable ones check the disk.
    func reloadFromDisk() async {
        guard loaded else { return }
        if editable { checkDisk(); return }
        guard let r = await Self.read(url) else { return }
        if case .binary = r.mode { return }
        if r.text != textView.string { apply(r, first: false) }
    }

    /// Conflict: the buffer overwrites the disk.
    func keepMine() {
        guard case .conflict(let d) = state, let base else { return }
        self.base = FileSnapshot(stamp: FileStamp.of(url) ?? base.stamp, text: readDiskText() ?? d)
        state = .dirty
        save()
    }

    /// Conflict: the disk version replaces the buffer.
    func useDisk() {
        guard case .conflict(let d) = state else { return }
        guard let stamp = FileStamp.of(url) else { state = .deleted; return }
        adopt(readDiskText() ?? d, stamp: stamp)
    }

    /// Deleted: writes the file again.
    func saveAgain() {
        guard editable, state == .deleted else { return }
        let t = text
        do { try write(t) } catch {
            show("Couldn't save: \(error.localizedDescription)")
            return
        }
        guard let stamp = FileStamp.of(url) else { return }
        base = FileSnapshot(stamp: stamp, text: t)
        state = .clean
        if watches { startWatcher() }
        onSaved?()
    }

    private func startWatcher() {
        watcher?.stop()
        watcher = FileWatcher(url: url) { [weak self] in self?.checkDisk() }
    }

    /// Saves, then stops the watcher and the timers.
    func close() {
        save()
        autosave?.cancel()
        noteTask?.cancel()
        watcher?.stop()
        watcher = nil
        styler?.stop()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
    }

    private func show(_ message: String) {
        note = message
        noteTask?.cancel()
        noteTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            guard !Task.isCancelled else { return }
            self?.note = nil
        }
    }

    // MARK: delegate

    @MainActor
    /// Puts the Reply chip under the end of the selection (or hides it).
    func updateReplyAnchor() {
        let sel = textView.selectedRange()
        // Only while the editor has focus: after a reply the message field takes it, and the chip goes.
        guard onReply != nil, sel.length > 0, NSEvent.pressedMouseButtons == 0,
              textView.window?.firstResponder === textView,
              let lm = textView.layoutManager, let tc = textView.textContainer, let box = scrollView.superview else {
            if replyAnchor != nil { replyAnchor = nil }
            return
        }
        let ns = textView.string as NSString
        // The last selected character that isn't a line break (a selection of whole lines ends after one).
        var end = min(NSMaxRange(sel), ns.length) - 1
        while end > sel.location, let c = UnicodeScalar(ns.character(at: end)), CharacterSet.newlines.contains(c) { end -= 1 }
        let glyphs = lm.glyphRange(forCharacterRange: NSRange(location: max(end, 0), length: 1), actualCharacterRange: nil)
        var rect = lm.boundingRect(forGlyphRange: glyphs, in: tc)
        rect.origin.x += textView.textContainerOrigin.x
        rect.origin.y += textView.textContainerOrigin.y
        guard textView.visibleRect.intersects(rect.insetBy(dx: -1, dy: -1)) else {
            if replyAnchor != nil { replyAnchor = nil }
            return
        }
        var p = box.convert(textView.convert(NSPoint(x: rect.maxX, y: rect.maxY), to: nil), from: nil)
        if !box.isFlipped { p.y = box.bounds.height - p.y }
        // The chip's centre: just under the line, kept inside the editor.
        p.x = min(max(p.x, 52), max(box.bounds.width - 52, 52))
        p.y = min(p.y + 16, max(box.bounds.height - 14, 14))
        if replyAnchor != p { replyAnchor = p }
    }

    private final class Delegate: NSObject, NSTextViewDelegate {
        weak var buffer: FileBuffer?
        init(_ buffer: FileBuffer) { self.buffer = buffer }

        func textDidChange(_ notification: Notification) { buffer?.userEdited() }
        func textDidEndEditing(_ notification: Notification) { buffer?.save() }
        func undoManager(for view: NSTextView) -> UndoManager? { buffer?.undo }
    }
}
