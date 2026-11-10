import AppKit
import AIStudioLibrary
import SwiftUI

// The design system's Tooltip in place of `.help`: a short label on tooltip glass after the pointer rests, or a hover
// card for longer text. Like the system's, a second tooltip right after the first comes up at once.

@MainActor
final class MGTooltipController {
    static let delay: TimeInterval = 0.8
    static let quickDelay: TimeInterval = 0.1

    private let panel: MGFloatingPanel = {
        let p = MGFloatingPanel()
        p.level = .floating
        return p
    }()
    private var timer: Timer?
    private var owner: UUID?
    private var hiddenAt = Date.distantPast
    private var monitor: Any?
    /// The text on screen (for tests).
    private(set) var shownText: String?

    var isShown: Bool { shownText != nil }

    /// The pointer entered a view with a tooltip.
    func enter(_ id: UUID, text: String, window: NSWindow?) {
        guard !text.isEmpty else { return }
        timer?.invalidate()
        owner = id
        let quick = isShown || Date().timeIntervalSince(hiddenAt) < 0.5
        if isShown { hide() }
        timer = Timer.scheduledTimer(withTimeInterval: quick ? Self.quickDelay : Self.delay, repeats: false) { [weak self, weak window] _ in
            MainActor.assumeIsolated {
                guard let self, self.owner == id, NSApp.isActive, !MGFloating.shared.isMenuOpen else { return }
                self.show(text, at: NSEvent.mouseLocation, window: window)
            }
        }
    }

    /// The pointer left it.
    func exit(_ id: UUID) {
        guard owner == id else { return }
        owner = nil
        timer?.invalidate()
        hide()
    }

    func show(_ text: String, at point: CGPoint, window: NSWindow?) {
        let size = Self.size(of: text)
        let frame = SurfacePlacement.tooltip(at: point, size: size, screen: mgScreen(containing: point))
        panel.show(MGTooltipView(text: text, card: Self.isCard(text)), surface: frame, parent: window, interactive: false)
        shownText = text
        if monitor == nil {
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown,
                                                                  .keyDown, .scrollWheel]) { [weak self] e in
                MainActor.assumeIsolated {
                    self?.owner = nil
                    self?.hide()
                }
                return e
            }
        }
    }

    func hide() {
        timer?.invalidate()
        if isShown { hiddenAt = Date() }
        shownText = nil
        panel.hide()
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }

    static func isCard(_ text: String) -> Bool { text.contains("\n") || text.count > 60 }

    /// One line on callout glass (8 by 4 padding), or a card at most 420 wide (8 by 6).
    static func size(of text: String) -> CGSize {
        let font = MGType.callout.nsFont
        if !isCard(text) {
            let s = (text as NSString).size(withAttributes: [.font: font])
            return CGSize(width: ceil(s.width) + 16, height: ceil(s.height) + 8)
        }
        let r = (text as NSString).boundingRect(with: CGSize(width: 420 - 16, height: 2000),
                                                options: [.usesLineFragmentOrigin, .usesFontLeading],
                                                attributes: [.font: font])
        return CGSize(width: min(ceil(r.width) + 16, 420), height: ceil(r.height) + 12)
    }
}

struct MGTooltipView: View {
    let text: String
    let card: Bool

    var body: some View {
        let radius = card ? 8 : MGRadius.md
        Text(text)
            .font(.mgCallout)
            .foregroundStyle(MG.label)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, card ? 6 : 4)
            .mgGlass(.tooltip, radius: radius)
            .mgHairline(RoundedRectangle(cornerRadius: radius, style: .continuous))
    }
}

private struct MGHelpModifier: ViewModifier {
    let text: String
    @State private var id = UUID()
    @State private var anchor = MGScreenAnchor()

    func body(content: Content) -> some View {
        content
            .mgAnchor(anchor)
            .onHover { inside in
                if inside { MGFloating.shared.tooltip.enter(id, text: text, window: anchor.window) }
                else { MGFloating.shared.tooltip.exit(id) }
            }
            .onDisappear { MGFloating.shared.tooltip.exit(id) }
            .accessibilityHint(text)
    }
}

extension View {
    /// A design-system tooltip (the `.help` of this app).
    func mgHelp(_ text: String) -> some View { modifier(MGHelpModifier(text: text)) }
}
