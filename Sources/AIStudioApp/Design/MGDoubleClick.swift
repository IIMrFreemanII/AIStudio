import AppKit
import SwiftUI

extension View {
    /// A click and a double-click on one view (a tab, a list row): the first click of a double-click does `single`
    /// at once, the second does `double`. SwiftUI's tap gestures drop the second click of a quick pair (a count-2
    /// tap beside a single tap never fired), so an AppKit overlay watches hit-testing for a mouse-down whose click
    /// count is 2 and acts on it there, without taking the event (taking it, as `MGContextCatcherView` does for
    /// right-clicks, made the hosting view drop it).
    @ViewBuilder func mgClick(single: @escaping () -> Void, double: (() -> Void)?) -> some View {
        let tapped = onTapGesture(perform: single)
        if let double { tapped.overlay(MGDoubleClickWatcher(action: double)) } else { tapped }
    }
}

private struct MGDoubleClickWatcher: NSViewRepresentable {
    let action: () -> Void

    func makeNSView(context: Context) -> MGDoubleClickView {
        let view = MGDoubleClickView()
        view.action = action
        return view
    }

    func updateNSView(_ view: MGDoubleClickView, context: Context) { view.action = action }
}

final class MGDoubleClickView: NSView {
    var action: (() -> Void)?
    /// When the last event acted on happened: AppKit hit-tests one event several times. (Its time, not its event
    /// number, which is 0 for posted events.)
    private var handled: TimeInterval?

    override func hitTest(_ point: NSPoint) -> NSView? {
        if let e = NSApp.currentEvent, e.type == .leftMouseDown, e.clickCount == 2, e.timestamp != handled,
           !e.modifierFlags.contains(.control), frame.contains(point) {  // `point` is in the superview's coordinates
            handled = e.timestamp
            DispatchQueue.main.async { [weak self] in self?.action?() }
        }
        return nil
    }
}
