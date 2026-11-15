import AppKit
import CodeEditing
import SwiftUI

/// The code editor's text view: TextKit 1 (storage → layout manager → container → this view), plain text, a current-line
/// highlight, auto-indent on Return, Tab inserts the indent unit.
final class CodeTextView: NSTextView {
    /// Where an edit may go to the buffer's own undo manager (the window's would be shared by every tab).
    var undoOwner: UndoManager?
    var indentUnit = "    "
    /// The user asked to reply with the selection (the caret's line when nothing is selected).
    var onReply: ((NSRange) -> Void)?
    /// The file's path in the project, for the context menu's Copy Path and Lines.
    var filePath: String?
    /// The selection settled (not while dragging) or the view scrolled: the reply chip moves.
    var onSelectionSettled: (() -> Void)?

    var wraps = false {
        didSet { if oldValue != wraps { applyWrapping() } }
    }

    static var baseAttributes: [NSAttributedString.Key: Any] {
        [.font: MGType.mono.nsFont, .foregroundColor: MGEditorColor.foreground]
    }

    /// Builds the TextKit 1 stack explicitly (never touches `textLayoutManager`, so the view stays on TextKit 1).
    static func make() -> CodeTextView {
        let storage = NSTextStorage()
        let layout = NSLayoutManager()
        layout.allowsNonContiguousLayout = true
        storage.addLayoutManager(layout)
        let container = NSTextContainer(size: NSSize(width: 600, height: CGFloat.greatestFiniteMagnitude))
        layout.addTextContainer(container)
        // A real width from the start: a zero-width container would lay a long wrapped text out one character per line.
        return CodeTextView(frame: NSRect(x: 0, y: 0, width: 600, height: 100), textContainer: container)
    }

    override init(frame: NSRect, textContainer: NSTextContainer?) {
        super.init(frame: frame, textContainer: textContainer)
        configure()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    private func configure() {
        isRichText = false
        importsGraphics = false
        isAutomaticQuoteSubstitutionEnabled = false
        isAutomaticDashSubstitutionEnabled = false
        isAutomaticLinkDetectionEnabled = false
        isAutomaticDataDetectionEnabled = false
        isAutomaticTextReplacementEnabled = false
        isAutomaticTextCompletionEnabled = false
        isAutomaticSpellingCorrectionEnabled = false
        isContinuousSpellCheckingEnabled = false
        isGrammarCheckingEnabled = false
        writingToolsBehavior = .none  // no Writing Tools button beside a selection of code
        allowsUndo = true
        usesFindBar = true
        isIncrementalSearchingEnabled = true
        font = MGType.mono.nsFont
        textColor = MGEditorColor.foreground
        drawsBackground = true
        backgroundColor = MGEditorColor.background
        insertionPointColor = MGEditorColor.caret
        selectedTextAttributes = [.backgroundColor: MGEditorColor.selection]
        textContainerInset = NSSize(width: 4, height: 6)
        isVerticallyResizable = true
        minSize = NSSize(width: 0, height: 0)
        maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        applyWrapping()
    }

    private func applyWrapping() {
        guard let container = textContainer else { return }
        if wraps {
            isHorizontallyResizable = false
            autoresizingMask = [.width]
            container.widthTracksTextView = true
            container.containerSize = NSSize(width: max(bounds.width, 0), height: CGFloat.greatestFiniteMagnitude)
        } else {
            isHorizontallyResizable = true
            autoresizingMask = []
            container.widthTracksTextView = false
            container.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        }
        needsDisplay = true
    }

    override var undoManager: UndoManager? { undoOwner ?? super.undoManager }

    // MARK: editing

    override func insertNewline(_ sender: Any?) {
        guard isEditable else { return super.insertNewline(sender) }
        let ns = string as NSString
        let caret = selectedRange().location
        let line = ns.lineRange(for: NSRange(location: min(caret, ns.length), length: 0))
        var indent = ""
        var i = line.location
        while i < min(caret, ns.length) {
            let c = ns.character(at: i)
            guard c == 32 || c == 9 else { break }
            indent.append(Character(UnicodeScalar(c)!))
            i += 1
        }
        insertText("\n" + indent, replacementRange: selectedRange())
    }

    override func insertTab(_ sender: Any?) {
        guard isEditable else { return super.insertTab(sender) }
        insertText(indentUnit, replacementRange: selectedRange())
    }

    // MARK: keys

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard window?.firstResponder === self else { return super.performKeyEquivalent(with: event) }
        let mods = event.modifierFlags.intersection([.command, .shift, .option, .control])
        if mods.contains(.command), !mods.contains(.option), !mods.contains(.control) {
            switch (event.charactersIgnoringModifiers?.lowercased(), mods.contains(.shift)) {
            case ("f", false): return findAction(.showFindInterface)
            case ("g", false): return findAction(.nextMatch)
            case ("g", true): return findAction(.previousMatch)
            default: break
            }
        }
        return super.performKeyEquivalent(with: event)
    }

    private func findAction(_ action: NSTextFinder.Action) -> Bool {
        let item = NSMenuItem()
        item.tag = action.rawValue
        performFindPanelAction(item)
        return true
    }

    // MARK: reply hook

    @objc func mgReplyWithSelection(_ sender: Any?) {
        var r = selectedRange()
        if r.length == 0 {
            let ns = string as NSString
            r = ns.lineRange(for: NSRange(location: min(r.location, ns.length), length: 0))
        }
        onReply?(r)
    }

    // MARK: current line

    override func setSelectedRanges(_ ranges: [NSValue], affinity: NSSelectionAffinity, stillSelecting: Bool) {
        super.setSelectedRanges(ranges, affinity: affinity, stillSelecting: stillSelecting)
        needsDisplay = true
        if !stillSelecting { onSelectionSettled?() }
    }

    override func mouseDown(with event: NSEvent) {
        super.mouseDown(with: event)  // tracks the drag until the button is up
        onSelectionSettled?()
    }

    // The reply chip shows only while the editor has focus.
    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        if ok { DispatchQueue.main.async { [weak self] in self?.onSelectionSettled?() } }
        return ok
    }

    override func resignFirstResponder() -> Bool {
        let ok = super.resignFirstResponder()
        // A beat later, so a click on the chip itself still lands if the click moved focus.
        if ok { DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in self?.onSelectionSettled?() } }
        return ok
    }

    override func drawBackground(in rect: NSRect) {
        super.drawBackground(in: rect)
        guard isEditable, selectedRange().length == 0, let lm = layoutManager, textStorage != nil else { return }
        let caret = selectedRange().location
        let length = (string as NSString).length
        var line = NSRect.zero
        if caret >= length, !lm.extraLineFragmentRect.isEmpty {
            line = lm.extraLineFragmentRect
        } else if lm.numberOfGlyphs > 0 {
            let glyph = min(lm.glyphIndexForCharacter(at: min(caret, max(length - 1, 0))), lm.numberOfGlyphs - 1)
            line = lm.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
        } else {
            return
        }
        let origin = textContainerOrigin
        let r = NSRect(x: 0, y: line.minY + origin.y, width: bounds.width, height: line.height)
        guard r.intersects(rect) else { return }
        MGEditorColor.currentLine.setFill()
        r.fill()
    }
}

// MARK: reply from file

extension CodeTextView: MGTextReplying {
    var canReply: Bool { onReply != nil }

    /// The lines ⌘L would quote: the selection's, or the caret's line.
    var replyLines: ClosedRange<Int> { TextLines(string).lineSpan(of: selectedRange()) }

    var mgExtraItems: [MGMenuItem] {
        guard canReply else { return [] }
        let lines = replyLines
        let label = lines.count > 1 ? "Lines \(lines.lowerBound)–\(lines.upperBound)" : "Line \(lines.lowerBound)"
        var items: [MGMenuItem] = [.divider]
        if ChatQuoting.active {
            items.append(.button("Reply to \(label)", icon: "arrowshape.turn.up.left", shortcut: KeyboardShortcut("l")) { [weak self] in
                self?.mgReplyWithSelection(nil)
            })
        }
        if let filePath {
            let ref = filePath + ":" + (lines.count > 1 ? "\(lines.lowerBound)-\(lines.upperBound)" : "\(lines.lowerBound)")
            items.append(.button("Copy Path and \(label)") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(ref, forType: .string)
            })
        }
        return items.count > 1 ? items : []
    }
}
