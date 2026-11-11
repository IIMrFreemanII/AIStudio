import AppKit
import SwiftUI

// The edit menu of text fields and editors, drawn as a design-system menu in place of AppKit's: a right-click (or
// ⌃-click) on text focuses it, keeps the selection (or puts the caret at the click) and opens Undo … Select All.

@MainActor
enum MGTextMenu {
    private static var monitor: Any?

    static func install() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDown, .leftMouseDown]) { e in
            MainActor.assumeIsolated { handle(e) ? nil : e }
        }
    }

    /// The text view (or field) under a click, ignoring context-menu catchers above it.
    static func textTarget(at e: NSEvent) -> NSView? {
        guard let w = e.window, !(w is MGFloatingPanel), let frame = w.contentView?.superview else { return nil }
        MGContextCatcherView.suspended = true
        defer { MGContextCatcherView.suspended = false }
        var v = frame.hitTest(e.locationInWindow)
        while let view = v {
            if let tv = view as? NSTextView, tv.isSelectable { return tv }
            if let tf = view as? NSTextField, tf.isEditable || tf.isSelectable { return tf }
            v = view.superview
        }
        return nil
    }

    /// Opens the menu for a context click on text; false when the click isn't one.
    static func handle(_ e: NSEvent) -> Bool {
        guard MGContextCatcherView.isContextClick(e), !MGFloating.shared.isMenuOpen,
              let target = textTarget(at: e), let w = e.window else { return false }
        let tv: NSTextView
        if let t = target as? NSTextView {
            tv = t
            if w.firstResponder !== tv { w.makeFirstResponder(tv) }
            let i = tv.characterIndexForInsertion(at: tv.convert(e.locationInWindow, from: nil))
            let sel = tv.selectedRange()
            // A click inside the selection (or at its end, past the text) keeps it, as AppKit's own menu does.
            if sel.length == 0 || i < sel.location || i > NSMaxRange(sel) {
                tv.setSelectedRange(NSRange(location: i, length: 0))
            }
        } else if let tf = target as? NSTextField {
            // Already editing this field: keep its editor and selection.
            if let fe = w.firstResponder as? NSTextView, fe.isFieldEditor, fe.isDescendant(of: tf) {
                tv = fe
                let at = w.convertPoint(toScreen: e.locationInWindow)
                DispatchQueue.main.async { MGFloating.shared.openMenu(items(for: fe), at: at, in: w) }
                return true
            }
            if tf.currentEditor() == nil { w.makeFirstResponder(tf) }
            // SwiftUI's fields don't always report their editor; the window's first responder is it.
            guard let editor = (tf.currentEditor() as? NSTextView) ?? (w.firstResponder as? NSTextView) else { return false }
            tv = editor
        } else {
            return false
        }
        let at = w.convertPoint(toScreen: e.locationInWindow)
        // On the next turn: opened inside this monitor, the menu's own click-outside monitor would see this click.
        DispatchQueue.main.async { MGFloating.shared.openMenu(items(for: tv), at: at, in: w) }
        return true
    }

    static func items(for tv: NSTextView) -> [MGMenuItem] {
        let sel = tv.selectedRange().length > 0
        let editable = tv.isEditable
        let undo = tv.undoManager
        let canPaste = NSPasteboard.general.canReadObject(forClasses: [NSString.self], options: nil)
        return [
            .button(undo?.canUndo == true && !(undo?.undoActionName.isEmpty ?? true) ? "Undo \(undo!.undoActionName)" : "Undo",
                    shortcut: KeyboardShortcut("z"), enabled: editable && undo?.canUndo == true) { undo?.undo() },
            .button("Redo", shortcut: KeyboardShortcut("z", modifiers: [.command, .shift]),
                    enabled: editable && undo?.canRedo == true) { undo?.redo() },
            .divider,
            .button("Cut", shortcut: KeyboardShortcut("x"), enabled: editable && sel) { tv.cut(nil) },
            .button("Copy", shortcut: KeyboardShortcut("c"), enabled: sel) { tv.copy(nil) },
            .button("Paste", shortcut: KeyboardShortcut("v"), enabled: editable && canPaste) { tv.paste(nil) },
            .button("Delete", enabled: editable && sel) { tv.delete(nil) },
            .divider,
            .button("Select All", shortcut: KeyboardShortcut("a"), enabled: !tv.string.isEmpty) { tv.selectAll(nil) },
        ] + (sel && !editable && ChatQuoting.active ? [
            .divider,
            .button("Reply with Selection") {
                ChatQuoting.post((tv.string as NSString).substring(with: tv.selectedRange()))
            },
        ] : [])
    }
}

/// Replies to a passage of a chat: text views and code blocks post it, the chat on screen puts it in its quote bar.
enum ChatQuoting {
    static let notification = Notification.Name("AIStudio.chatQuote")
    /// A chat is on screen (its view counts itself in and out), so "Reply with Selection" is offered.
    @MainActor static var visibleChats = 0
    @MainActor static var active: Bool { visibleChats > 0 }

    @MainActor static func post(_ text: String) {
        NotificationCenter.default.post(name: notification, object: text)
    }
}
