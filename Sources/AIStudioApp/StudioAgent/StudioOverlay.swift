import AppKit
import SwiftUI

/// A see-through window over the main window where the Studio agent's pointer, ring and caption are drawn. It never
/// takes a click, and moves and resizes with the main window.
final class StudioOverlayWindow: NSPanel {
    private weak var controller: StudioController?
    private var observers: [NSObjectProtocol] = []

    @MainActor
    init(controller: StudioController) {
        self.controller = controller
        super.init(contentRect: NSRect(x: 0, y: 0, width: 10, height: 10),
                   styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        ignoresMouseEvents = true
        isReleasedWhenClosed = false
        hidesOnDeactivate = false
        animationBehavior = .none
        collectionBehavior = [.transient, .ignoresCycle, .fullScreenAuxiliary]
        let host = NSHostingView(rootView: StudioOverlayView(controller: controller).mgRoot())
        host.sizingOptions = []
        contentView = host
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// Covers `main` (a child window of it, so it stays just above it and moves with it).
    @MainActor
    func cover(_ main: NSWindow) {
        if parent !== main {
            parent?.removeChildWindow(self)
            main.addChildWindow(self, ordered: .above)
            observers.forEach(NotificationCenter.default.removeObserver)
            observers = [NSWindow.didResizeNotification, NSWindow.didEndLiveResizeNotification].map { name in
                NotificationCenter.default.addObserver(forName: name, object: main, queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated { self?.setFrame(main.frame, display: true) }
                }
            }
        }
        appearance = main.effectiveAppearance
        setFrame(main.frame, display: true)
        orderFront(nil)
    }
}

/// The agent's pointer (an accent diamond tagged "Agent"), the glow ring around the control it acts on, and the caption
/// saying what it's doing.
struct StudioOverlayView: View {
    let controller: StudioController

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
            if let ring = controller.ring {
                RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                    .stroke(MG.accent, lineWidth: 2)
                    .shadow(color: MG.accent.opacity(0.9), radius: 6)
                    .shadow(color: MG.accent.opacity(0.5), radius: 14)
                    .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.accent.opacity(0.06)))
                    .frame(width: ring.width, height: ring.height)
                    .offset(x: ring.minX, y: ring.minY)
                    .opacity(controller.ringShown ? 1 : 0)
            }
            if let p = controller.pointer {
                StudioPointer(caption: controller.caption)
                    .fixedSize()
                    .offset(x: p.x - 7, y: p.y - 3)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The pointer: a diamond with an "Agent" tag, and the caption chip beside it.
struct StudioPointer: View {
    var caption: String?

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "diamond.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(MG.accent)
                .shadow(color: .black.opacity(0.35), radius: 2, y: 1)
            HStack(spacing: 5) {
                Text("Agent").font(.mgFootnote.weight(.semibold)).foregroundStyle(.white)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Capsule().fill(MG.accent))
                if let caption, !caption.isEmpty {
                    Text(caption).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(2)
                        .frame(maxWidth: 320, alignment: .leading)
                }
            }
            .padding(.horizontal, 8).padding(.vertical, 5)
            .mgGlass(.floatingPanel, radius: MGRadius.lg)
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
            .offset(y: 10)
        }
    }
}
