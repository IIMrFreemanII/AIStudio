import AppKit
import SwiftUI

// The design system's Window: the system's traffic lights in the leading 78 pt and the app's own bar beside them,
// either "unified" (52 tall, the toolbar in the title row) or "title" (32 tall, the title centred).

enum MGWindowBar {
    case unified, title
    var height: CGFloat { self == .unified ? 52 : 32 }
}

/// Configures the hosting window once it exists: no system title, a transparent title bar the content runs under, and
/// the traffic lights centred in the bar. (The main and Models windows also use SwiftUI's hidden title bar style, which
/// lets the bar's own controls take clicks in the title row.)
private struct MGWindowConfigurator: NSViewRepresentable {
    let bar: MGWindowBar

    final class ConfigView: NSView {
        var bar: MGWindowBar = .unified
        private var observers: [NSObjectProtocol] = []
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            observers.forEach(NotificationCenter.default.removeObserver)
            observers.removeAll()
            configure()
            guard let w = window else { return }
            for name in [NSWindow.didResizeNotification, NSWindow.didExitFullScreenNotification,
                         NSWindow.didBecomeKeyNotification] {
                observers.append(NotificationCenter.default.addObserver(forName: name, object: w, queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated { self?.placeLights() }
                })
            }
        }
        override func layout() { super.layout(); placeLights() }

        func configure() {
            guard let w = window else { return }
            w.titleVisibility = .hidden
            w.titlebarAppearsTransparent = true
            w.styleMask.insert(.fullSizeContentView)
            w.isMovableByWindowBackground = false
            // No toolbar: its views would sit over the bar and take its clicks. The lights are moved instead.
            w.toolbar = nil
            w.titlebarSeparatorStyle = .none
            placeLights()
        }

        /// Centres the traffic lights in the bar: from 20 pt, 20 apart (12 pt lights 8 apart, the design system's Window).
        func placeLights() {
            guard let w = window, !w.styleMask.contains(.fullScreen),
                  let close = w.standardWindowButton(.closeButton), let titlebar = close.superview else { return }
            let buttons = [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton].compactMap { w.standardWindowButton($0) }
            // Grow the title bar's container to the bar's height so the lights sit inside it.
            if let container = titlebar.superview {
                var f = container.frame
                let height = bar.height
                if abs(f.height - height) > 0.5 {
                    f.size.height = height
                    f.origin.y = w.frame.height - height
                    container.frame = f
                }
            }
            for (i, b) in buttons.enumerated() {
                let y = (titlebar.frame.height - b.frame.height) / 2
                let x = 20 + CGFloat(i) * 20 - (b.frame.width - 12) / 2
                if abs(b.frame.origin.x - x) > 0.5 || abs(b.frame.origin.y - y) > 0.5 { b.setFrameOrigin(NSPoint(x: x, y: y)) }
            }
        }
    }

    func makeNSView(context: Context) -> ConfigView {
        let v = ConfigView()
        v.bar = bar
        return v
    }

    func updateNSView(_ v: ConfigView, context: Context) {
        v.bar = bar
        v.configure()
    }
}

/// Empty bar space: drags the window, and a double-click zooms or minimises it as System Settings says.
private struct MGWindowDragArea: NSViewRepresentable {
    final class DragView: NSView {
        override var mouseDownCanMoveWindow: Bool { true }
        override func mouseDown(with e: NSEvent) {
            if e.clickCount == 2 {
                switch UserDefaults.standard.string(forKey: "AppleActionOnDoubleClick") ?? "Maximize" {
                case "Minimize": window?.performMiniaturize(nil)
                case "None": break
                default: window?.performZoom(nil)
                }
            } else {
                window?.performDrag(with: e)
            }
        }
    }

    func makeNSView(context: Context) -> DragView { DragView() }
    func updateNSView(_ v: DragView, context: Context) {}
}

/// The window's own bar on the bar tint with a hairline under it. `leading` sits after the traffic lights.
struct MGTitleBar<Leading: View, Trailing: View>: View {
    let bar: MGWindowBar
    var title: String
    var subtitle: String? = nil
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        ZStack {
            MGWindowDragArea()
            if bar == .title {
                Text(title).font(.mgHeadline).lineLimit(1).padding(.horizontal, 78).allowsHitTesting(false)
            } else {
                HStack(spacing: 8) {
                    leading
                    VStack(alignment: .leading, spacing: 0) {
                        Text(title).font(.mgHeadline).lineLimit(1).truncationMode(.middle)
                        if let subtitle {
                            Text(subtitle).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                        }
                    }
                    .allowsHitTesting(false)
                    .layoutPriority(-1)
                    Spacer(minLength: 12).allowsHitTesting(false)
                    trailing
                }
                .padding(.leading, 86)
                .padding(.trailing, 12)
            }
        }
        .frame(height: bar.height)
        .frame(maxWidth: .infinity)
        .mgChrome(MG.barTint, material: .titlebar)
        .overlay(alignment: .bottom) { MGDivider() }
        .background(MGWindowConfigurator(bar: bar).frame(width: 0, height: 0))
    }
}

extension MGTitleBar where Leading == EmptyView, Trailing == EmptyView {
    /// A plain title bar (Settings, Models).
    init(title: String) {
        self.init(bar: .title, title: title, leading: { EmptyView() }, trailing: { EmptyView() })
    }
}

extension View {
    /// Puts the design system's title bar over this window's content (the content no longer sits under the lights).
    func mgWindow(_ title: String) -> some View {
        VStack(spacing: 0) {
            MGTitleBar(title: title)
            self.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .ignoresSafeArea(.container, edges: .top)
    }
}

/// A 6 pt grab strip on a column's edge that resizes it; the cursor turns into the resize arrows over it.
struct MGResizeHandle: View {
    @Binding var width: Double
    let range: ClosedRange<Double>
    /// The column is on the trailing side, so dragging left makes it wider.
    var trailingColumn = true
    @State private var start: Double?

    var body: some View {
        Color.clear
            .frame(width: 6)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in
                    if start == nil { start = width }
                    let d = Double(v.translation.width) * (trailingColumn ? -1 : 1)
                    width = min(max(start! + d, range.lowerBound), range.upperBound)
                }
                .onEnded { _ in start = nil })
    }
}

/// A horizontal divider that resizes the pane above it (a list over a preview): a hairline with a 7 pt grab area.
struct MGRowResizeHandle: View {
    @Binding var height: Double
    let range: ClosedRange<Double>
    @State private var start: Double?

    var body: some View {
        Rectangle().fill(MG.separator).frame(height: 1)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 3)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in
                    if start == nil { start = min(max(height, range.lowerBound), range.upperBound) }
                    height = min(max(start! + Double(v.translation.height), range.lowerBound), range.upperBound)
                }
                .onEnded { _ in start = nil })
    }
}
