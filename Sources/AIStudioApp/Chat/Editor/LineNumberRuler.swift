import AppKit
import CodeEditing

/// The line-number gutter: a vertical ruler on the text view's scroll view. Wrapped lines are numbered once, at their
/// first fragment. Line starts are cached and recomputed when the text changes.
final class LineNumberRuler: NSRulerView {
    private weak var textView: NSTextView?
    private var lines = TextLines("")
    private var linesDirty = true
    private var observers: [NSObjectProtocol] = []
    private let font = MGType.monoSmall.nsFont

    init(scrollView: NSScrollView, textView: NSTextView) {
        self.textView = textView
        super.init(scrollView: scrollView, orientation: .verticalRuler)
        clientView = textView
        clipsToBounds = true
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: NSText.didChangeNotification, object: textView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.invalidate() }
        })
        observers.append(nc.addObserver(forName: NSTextView.didChangeSelectionNotification, object: textView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.needsDisplay = true }
        })
        scrollView.contentView.postsBoundsChangedNotifications = true
        observers.append(nc.addObserver(forName: NSView.boundsDidChangeNotification, object: scrollView.contentView, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.needsDisplay = true }
        })
        invalidate()
    }

    required init(coder: NSCoder) { fatalError("init(coder:) is not used") }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }

    /// The text changed: line starts are recomputed on the next draw, the width now.
    func invalidate() {
        linesDirty = true
        refreshLines()
        let digits = max(3, String(lines.count).count)
        let digitWidth = ("0" as NSString).size(withAttributes: [.font: font]).width
        let thickness = ceil(CGFloat(digits) * digitWidth) + 18
        if abs(ruleThickness - thickness) > 0.5 { ruleThickness = thickness }
        needsDisplay = true
    }

    private func refreshLines() {
        guard let tv = textView else { return }
        if linesDirty || lines.length != (tv.string as NSString).length {
            lines = TextLines(tv.string)
            linesDirty = false
        }
    }

    override var isFlipped: Bool { true }

    override func drawHashMarksAndLabels(in rect: NSRect) {
        MGEditorColor.gutterBackground.setFill()
        bounds.fill()
        MGEditorColor.gutterSeparator.setFill()
        NSRect(x: bounds.maxX - 1, y: bounds.minY, width: 1, height: bounds.height).fill()
        guard let tv = textView, let lm = tv.layoutManager, let tc = tv.textContainer,
              let clip = scrollView?.contentView else { return }
        refreshLines()
        let origin = tv.textContainerOrigin
        let visible = clip.bounds.offsetBy(dx: -origin.x, dy: -origin.y)
        let current = lines.line(at: tv.selectedRange().location)
        let right = bounds.maxX - 8

        func draw(_ n: Int, fragment: NSRect) {
            let color = n == current ? MGEditorColor.gutterCurrentLine : MGEditorColor.gutterForeground
            let s = NSAttributedString(string: String(n), attributes: [.font: font, .foregroundColor: color])
            let size = s.size()
            let y = convert(NSPoint(x: 0, y: fragment.minY + origin.y), from: tv).y + (fragment.height - size.height) / 2
            s.draw(at: NSPoint(x: right - size.width, y: y))
        }

        let glyphs = lm.glyphRange(forBoundingRect: visible, in: tc)
        if glyphs.length > 0 {
            lm.enumerateLineFragments(forGlyphRange: glyphs) { frag, _, _, range, _ in
                let index = lm.characterIndexForGlyph(at: range.location)
                let n = self.lines.line(at: index)
                if self.lines.range(ofLine: n).location == index { draw(n, fragment: frag) }
            }
        }
        let extra = lm.extraLineFragmentRect
        if !extra.isEmpty, extra.maxY >= visible.minY, extra.minY <= visible.maxY { draw(lines.count, fragment: extra) }
    }
}
