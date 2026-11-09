import AppKit
import MLXComfyLibrary
import SwiftUI

// Floating surfaces: menus, pop-ups, tooltips and popovers live in borderless child panels, so they can reach past
// the view (and the window) the way the system's do, and are drawn entirely by the design system.

/// A borderless, non-activating panel holding one floating surface. Its content draws the glass and the shadow; the
/// panel is `margin` larger than the surface on every side so the shadow isn't clipped.
final class MGFloatingPanel: NSPanel {
    static let margin: CGFloat = 32

    init() {
        super.init(contentRect: NSRect(x: 0, y: 0, width: 10, height: 10),
                   styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        level = .popUpMenu
        isReleasedWhenClosed = false
        hidesOnDeactivate = false
        isMovable = false
        animationBehavior = .none
        collectionBehavior = [.transient, .ignoresCycle, .fullScreenAuxiliary]
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// The surface's frame on screen (the panel without its margin).
    var surfaceFrame: CGRect { frame.insetBy(dx: Self.margin, dy: Self.margin) }

    /// Shows `content` with its surface at `surface` (screen coordinates), over `parent`.
    func show<Content: View>(_ content: Content, surface: CGRect, parent: NSWindow?, interactive: Bool = true) {
        let root = content
            .padding(Self.margin)
            .frame(width: surface.width + Self.margin * 2, height: surface.height + Self.margin * 2)
            .environment(\.mgGlassBlending, .behindWindow)
            .mgRoot()
        if let host = contentView as? MGFirstMouseHostingView {
            host.rootView = AnyView(root)
        } else {
            let host = MGFirstMouseHostingView(rootView: AnyView(root))
            host.sizingOptions = []
            contentView = host
        }
        appearance = parent?.effectiveAppearance
        ignoresMouseEvents = !interactive
        setFrame(surface.insetBy(dx: -Self.margin, dy: -Self.margin), display: true)
        if let parent, parent != self.parent {
            self.parent?.removeChildWindow(self)
            parent.addChildWindow(self, ordered: .above)
        }
        orderFront(nil)
    }

    func hide() {
        parent?.removeChildWindow(self)
        orderOut(nil)
    }
}

/// Panels never become key, so their content takes the first click instead of waiting for one to activate it.
final class MGFirstMouseHostingView: NSHostingView<AnyView> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

/// Where a floating surface is anchored: a view's frame on screen, read through a tiny AppKit view behind it.
@MainActor
final class MGScreenAnchor {
    fileprivate weak var view: NSView?

    var window: NSWindow? { view?.window }

    /// The anchor view's frame in screen coordinates.
    var screenFrame: CGRect? {
        guard let view, let window = view.window else { return nil }
        return window.convertToScreen(view.convert(view.bounds, to: nil))
    }
}

private struct MGAnchorView: NSViewRepresentable {
    let anchor: MGScreenAnchor
    func makeNSView(context: Context) -> NSView {
        let v = MGPassthroughView()
        anchor.view = v
        return v
    }
    func updateNSView(_ v: NSView, context: Context) { anchor.view = v }
}

/// An AppKit view that never takes a click.
class MGPassthroughView: NSView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

extension View {
    /// Lets `anchor` report this view's frame on screen.
    func mgAnchor(_ anchor: MGScreenAnchor) -> some View {
        background(MGAnchorView(anchor: anchor).allowsHitTesting(false))
    }
}

/// The screen a point (or the key window) is on, without its menu bar and Dock.
@MainActor
func mgScreen(containing point: CGPoint) -> CGRect {
    let screen = NSScreen.screens.first { $0.frame.contains(point) } ?? NSApp.keyWindow?.screen ?? NSScreen.main
    return screen?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1440, height: 900)
}

/// Runs the floating surfaces: one menu chain, one popover and one tooltip at a time. It closes menus and popovers on
/// a click outside them, Esc, the app going to the background, or the window moving or resizing.
@MainActor
final class MGFloating {
    static let shared = MGFloating()

    private var monitors: [Any] = []
    private var observers: [NSObjectProtocol] = []

    /// The open menu chain (root first), or nil.
    private(set) var menu: MGMenuSession?
    private(set) var popover: (panel: MGFloatingPanel, onClose: () -> Void)?
    let tooltip = MGTooltipController()

    var isMenuOpen: Bool { menu != nil }
    /// Why the last menu or popover closed (for tests).
    private(set) var lastCloseReason = ""
    /// The window the open menu or popover belongs to: only its moving, resizing or closing closes them (other
    /// windows come and go, such as the input-source indicator next to a text field's caret).
    private weak var hostWindow: NSWindow?
    var isPopoverOpen: Bool { popover != nil }

    private init() {}

    // MARK: menus

    /// Opens a menu with its top-left corner at `point` (screen coordinates): a context menu.
    func openMenu(_ items: [MGMenuItem], at point: CGPoint, in window: NSWindow?) {
        closeAll()
        let session = MGMenuSession(items: items, window: window)
        let size = session.root.size(maxHeight: SurfacePlacement.maxMenuHeight(screen: mgScreen(containing: point)))
        session.present(session.root, at: SurfacePlacement.atPoint(point, size: size, screen: mgScreen(containing: point)))
        begin(session)
    }

    /// Opens a menu under `anchor` (a pop-up button), at least as wide as it.
    func openMenu(_ items: [MGMenuItem], below anchor: CGRect, in window: NSWindow?, searchable: Bool = false,
                  onClose: (() -> Void)? = nil) {
        closeAll()
        let session = MGMenuSession(items: items, window: window, searchable: searchable)
        session.onClose = onClose
        session.root.minWidth = anchor.width
        let screen = mgScreen(containing: CGPoint(x: anchor.midX, y: anchor.midY))
        let size = session.root.size(maxHeight: min(SurfacePlacement.maxMenuHeight(screen: screen), 520))
        session.present(session.root, at: SurfacePlacement.below(anchor, size: size, screen: screen))
        begin(session)
    }

    private func begin(_ session: MGMenuSession) {
        menu = session
        hostWindow = session.window
        tooltip.hide()
        installMonitors()
    }

    func closeMenu() {
        guard let session = menu else { return }
        menu = nil
        session.closeAll()
        session.onClose?()
        removeMonitorsIfIdle()
    }

    // MARK: popovers

    /// Shows `content` 4 pt below `anchor` on popover glass.
    func openPopover<Content: View>(_ content: Content, size: CGSize, below anchor: CGRect, in window: NSWindow?,
                                    onClose: @escaping () -> Void) {
        closeAll()
        let panel = MGFloatingPanel()
        let screen = mgScreen(containing: CGPoint(x: anchor.midX, y: anchor.midY))
        var frame = SurfacePlacement.below(anchor, size: size, screen: screen)
        frame.origin.x = min(max(anchor.midX - size.width / 2, screen.minX), screen.maxX - size.width)
        panel.show(MGPopoverSurface { content }, surface: frame, parent: window)
        popover = (panel, onClose)
        hostWindow = window
        installMonitors()
    }

    func closePopover() {
        guard let p = popover else { return }
        popover = nil
        p.panel.hide()
        p.onClose()
        removeMonitorsIfIdle()
    }

    func closeAll() {
        closeMenu()
        closePopover()
    }

    // MARK: events

    private func installMonitors() {
        guard monitors.isEmpty else { return }
        let mouse: NSEvent.EventTypeMask = [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        if let m = NSEvent.addLocalMonitorForEvents(matching: mouse, handler: { [weak self] e in
            MainActor.assumeIsolated { self?.mouseDown(e) ?? e }
        }) { monitors.append(m) }
        if let m = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] e in
            MainActor.assumeIsolated { self?.keyDown(e) ?? e }
        }) { monitors.append(m) }
        if let m = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel, handler: { [weak self] e in
            MainActor.assumeIsolated { self?.scroll(e) ?? e }
        }) { monitors.append(m) }
        if let m = NSEvent.addGlobalMonitorForEvents(matching: mouse, handler: { [weak self] _ in
            MainActor.assumeIsolated { self?.closeAll() }
        }) { monitors.append(m) }
        let nc = NotificationCenter.default
        for name in [NSApplication.didResignActiveNotification, NSWindow.willMoveNotification,
                     NSWindow.didResizeNotification, NSWindow.willCloseNotification] {
            observers.append(nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] note in
                MainActor.assumeIsolated {
                    // Our own panels move when they're shown; only the app's windows count.
                    if note.name != NSApplication.didResignActiveNotification {
                        guard let w = note.object as? NSWindow, let host = self?.hostWindow, w === host else { return }
                    }
                    self?.lastCloseReason = note.name.rawValue
                    self?.closeAll()
                }
            })
        }
    }

    private func removeMonitorsIfIdle() {
        guard menu == nil, popover == nil else { return }
        monitors.forEach(NSEvent.removeMonitor)
        monitors.removeAll()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers.removeAll()
    }

    /// Whether a screen point is on one of the open surfaces.
    private func onSurface(_ p: CGPoint) -> Bool {
        if let menu, menu.levels.contains(where: { $0.panel.surfaceFrame.contains(p) }) { return true }
        if let popover, popover.panel.surfaceFrame.contains(p) { return true }
        return false
    }

    private func screenPoint(_ e: NSEvent) -> CGPoint {
        e.window.map { $0.convertPoint(toScreen: e.locationInWindow) } ?? NSEvent.mouseLocation
    }

    /// A click outside the open surfaces closes them and is used up, as with the system's menus.
    private func mouseDown(_ e: NSEvent) -> NSEvent? {
        if onSurface(screenPoint(e)) { return e }
        lastCloseReason = "click outside (\(e.type.rawValue))"
        closeAll()
        return nil
    }

    private func keyDown(_ e: NSEvent) -> NSEvent? {
        if let menu {
            if e.modifierFlags.contains(.command) { closeMenu(); return e }
            menu.key(e)
            return nil
        }
        if popover != nil, e.keyCode == 53 { closePopover(); return nil }
        return e
    }

    private func scroll(_ e: NSEvent) -> NSEvent? {
        if onSurface(screenPoint(e)) { return e }
        if menu != nil { return nil }
        closePopover()
        return e
    }
}

/// The design system's Popover: popover glass, radius 10, a hairline and the popover shadow.
struct MGPopoverSurface<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .mgGlass(.popover, radius: MGRadius.lg)
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
    }
}

/// `.mgPopover`: the design system's popover, under the view it's attached to.
private struct MGPopoverModifier<Popover: View>: ViewModifier {
    @Binding var isPresented: Bool
    let size: CGSize
    @ViewBuilder var popover: () -> Popover
    @State private var anchor = MGScreenAnchor()

    func body(content: Content) -> some View {
        content
            .mgAnchor(anchor)
            .onChange(of: isPresented) { _, shown in
                if shown, let frame = anchor.screenFrame {
                    MGFloating.shared.openPopover(popover(), size: size, below: frame, in: anchor.window) {
                        isPresented = false
                    }
                } else if !shown, MGFloating.shared.isPopoverOpen {
                    MGFloating.shared.closePopover()
                }
            }
    }
}

extension View {
    func mgPopover<Content: View>(isPresented: Binding<Bool>, size: CGSize,
                                  @ViewBuilder content: @escaping () -> Content) -> some View {
        modifier(MGPopoverModifier(isPresented: isPresented, size: size, popover: content))
    }
}

/// Named anchors for the chrome smoke test (only placed in automated runs), so it can click real controls.
@MainActor
enum MGTestAnchors {
    static var anchors: [String: MGScreenAnchor] = [:]
    static func frame(_ name: String) -> CGRect? { anchors[name]?.screenFrame }
}

private struct MGTestAnchorModifier: ViewModifier {
    let name: String
    @State private var anchor = MGScreenAnchor()

    func body(content: Content) -> some View {
        if AppModel.isAutomatedRun {
            content.mgAnchor(anchor).onAppear { MGTestAnchors.anchors[name] = anchor }
        } else {
            content
        }
    }
}

extension View {
    func mgTestAnchor(_ name: String) -> some View { modifier(MGTestAnchorModifier(name: name)) }
}
