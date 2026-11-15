import AppKit
import CodeEditing

/// Colours the text view from tree-sitter, using the layout manager's temporary attributes only (the text storage
/// is never touched, so nothing enters the undo stack). Highlights the visible characters ± two screens.
@MainActor
final class SyntaxStyler {
    private weak var textView: CodeTextView?
    private weak var scrollView: NSScrollView?
    private let highlighter: SyntaxHighlighter
    private var editCounter = 0
    /// The edit the highlighter has seen (spans are only applied when it is the latest).
    private var highlightedEdit = -1
    private var styledRange: NSRange?
    private var debounce: Task<Void, Never>?
    private var restyling: Task<Void, Never>?
    private var observer: NSObjectProtocol?

    init(textView: CodeTextView, scrollView: NSScrollView, language: CodeLanguage) {
        self.textView = textView
        self.scrollView = scrollView
        highlighter = SyntaxHighlighter(language: language)
        scrollView.contentView.postsBoundsChangedNotifications = true
        observer = NotificationCenter.default.addObserver(forName: NSView.boundsDidChangeNotification,
                                                          object: scrollView.contentView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.restyleVisible() }
        }
    }

    deinit {
        if let observer { NotificationCenter.default.removeObserver(observer) }
    }

    func stop() {
        debounce?.cancel()
        restyling?.cancel()
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
    }

    /// The text changed: highlight again after a short pause (at once for a first load or a replaced text).
    func textChanged(immediately: Bool = false) {
        editCounter += 1
        styledRange = nil
        let edit = editCounter
        debounce?.cancel()
        debounce = Task { [weak self] in
            if !immediately { try? await Task.sleep(nanoseconds: 150_000_000) }
            guard !Task.isCancelled, let self, let tv = self.textView, edit == self.editCounter else { return }
            let snapshot = tv.string
            await self.highlighter.update(text: snapshot)
            guard edit == self.editCounter else { return }
            self.highlightedEdit = edit
            self.restyleVisible()
        }
    }

    func restyleVisible() {
        guard highlightedEdit == editCounter, let tv = textView, let lm = tv.layoutManager, let tc = tv.textContainer,
              let clip = scrollView?.contentView else { return }
        let length = (tv.string as NSString).length
        guard length > 0 else { return }
        let target: NSRange
        let visibleChars: NSRange
        if clip.bounds.height < 1 {
            // Not on screen yet (no size): the start of the text.
            target = NSRange(location: 0, length: min(length, 30_000))
            visibleChars = target
        } else {
            let origin = tv.textContainerOrigin
            let visible = clip.bounds.offsetBy(dx: -origin.x, dy: -origin.y)
            visibleChars = lm.characterRange(forGlyphRange: lm.glyphRange(forBoundingRect: visible, in: tc), actualGlyphRange: nil)
            let margin = max(visible.height, 400) * 2
            let wide = visible.insetBy(dx: 0, dy: -margin)
            target = lm.characterRange(forGlyphRange: lm.glyphRange(forBoundingRect: wide, in: tc), actualGlyphRange: nil)
        }
        if let styled = styledRange, NSIntersectionRange(styled, visibleChars).length == visibleChars.length { return }
        let range = NSIntersectionRange(target, NSRange(location: 0, length: length))
        guard range.length > 0 else { return }
        let edit = editCounter
        restyling?.cancel()
        restyling = Task { [weak self, highlighter] in
            let spans = await highlighter.spans(in: range)
            guard !Task.isCancelled, let self, edit == self.editCounter, let lm = self.textView?.layoutManager else { return }
            let all = NSRange(location: 0, length: lm.textStorage?.length ?? 0)
            let clamped = NSIntersectionRange(range, all)
            lm.removeTemporaryAttribute(.foregroundColor, forCharacterRange: clamped)
            lm.removeTemporaryAttribute(.strikethroughStyle, forCharacterRange: clamped)
            for span in spans {
                let r = NSIntersectionRange(span.range, all)
                guard r.length > 0 else { continue }
                lm.addTemporaryAttribute(.foregroundColor, value: SyntaxTheme.color(span.token), forCharacterRange: r)
                if span.token == .strikethrough {
                    lm.addTemporaryAttribute(.strikethroughStyle, value: NSUnderlineStyle.single.rawValue, forCharacterRange: r)
                }
            }
            self.styledRange = range
        }
    }
}
