import Agent
import AppKit
import SwiftUI

/// A see-through window over the whole display the main window is on, where the Studio agent's actors fly: their
/// icons, the rings on the controls they act on, and their captions. It sits above the Studio panel, never takes a
/// click, and hides while another app is in front (the work goes on).
final class StudioOverlayWindow: NSPanel {
    private weak var controller: StudioController?

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
        hidesOnDeactivate = true
        animationBehavior = .none
        // Above the floating Studio panel, so icons can fly over it.
        level = NSWindow.Level(rawValue: NSWindow.Level.floating.rawValue + 1)
        collectionBehavior = [.transient, .ignoresCycle, .fullScreenAuxiliary]
        let host = NSHostingView(rootView: StudioOverlayView(controller: controller).mgRoot())
        host.sizingOptions = []
        contentView = host
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// Covers `frame` (the main window's display, without the menu bar and Dock).
    @MainActor
    func cover(_ frame: CGRect, appearance: NSAppearance) {
        self.appearance = appearance
        if self.frame != frame { setFrame(frame, display: true) }
        if !isVisible, NSApp.isActive { orderFront(nil) }
    }
}

/// Every actor on screen: its ring, its icon (flying, drifting or resting) with its label and caption, and the
/// sparkle the architect sends to a new worker. Drawn on a timeline while anything moves.
struct StudioOverlayView: View {
    let controller: StudioController

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !moving)) { tl in
            let now = tl.date
            ZStack(alignment: .topLeading) {
                Color.clear
                ForEach(controller.cursors) { c in
                    if let ring = c.ring {
                        RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                            .stroke(c.actor.color, lineWidth: 2)
                            .shadow(color: c.actor.color.opacity(0.9), radius: 6)
                            .shadow(color: c.actor.color.opacity(0.5), radius: 14)
                            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(c.actor.color.opacity(0.06)))
                            .frame(width: ring.width, height: ring.height)
                            .offset(x: ring.minX, y: ring.minY)
                            .opacity(c.ringShown ? 1 : 0)
                    }
                }
                StudioEchoesView(groups: echoGroups(at: now))
                ForEach(controller.cursors) { c in
                    let motions = controller.motions(c)
                    if let p = c.displayed(at: now, drift: motions.contains(.drift)) {
                        let size: CGFloat = c.actor.isArchitect ? 36 : 32
                        StudioPointer(cursor: c, motions: motions, pace: controller.pace, paused: controller.gate.isPaused,
                                      moving: c.isFlying(at: now) || (c.state == .idle && c.position != nil && c.homeTarget != nil))
                            .fixedSize()
                            .offset(x: p.x - size / 2 - 4, y: p.y - size / 2 - 4)
                            .transition(.scale(scale: 0.4).combined(with: .opacity))
                    }
                }
                if let h = controller.handoff, now.timeIntervalSince(h.at) < 0.5 {
                    let p = Self.handoffFlight(h).point(at: now.timeIntervalSince(h.at))
                    SparkleShape().fill(MG.accent)
                        .frame(width: 12, height: 12)
                        .shadow(color: MG.accent.opacity(0.8), radius: 4)
                        .offset(x: p.x - 6, y: p.y - 6)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// The handoff sparkle's flight (the overlay draws it for half a second).
    static func handoffFlight(_ h: (from: CGPoint, to: CGPoint, at: Date)) -> StudioFlight {
        StudioFlight(from: h.from, to: h.to, bend: 0.25, duration: 0.5)
    }

    /// Every actor's ghost echoes at `now`, and the handoff sparkle's.
    private func echoGroups(at now: Date) -> [StudioEchoGroup] {
        var groups = controller.cursors.compactMap { c -> StudioEchoGroup? in
            let e = c.echoes(at: now, drift: controller.motions(c).contains(.drift))
            return e.isEmpty ? nil : StudioEchoGroup(actor: c.actor, echoes: e)
        }
        if let h = controller.handoff, now.timeIntervalSince(h.at) < 0.5 {
            // The sparkle vanishes as it lands (the worker pops in), so its echoes stop with it.
            let leg = StudioTrailLeg(flight: Self.handoffFlight(h), start: h.at.timeIntervalSinceReferenceDate)
            groups.append(StudioEchoGroup(actor: .architect, echoes: StudioTrail.echoes([leg], now: now.timeIntervalSinceReferenceDate),
                                          sparkle: true))
        }
        return groups
    }

    /// Anything flying, trailing, drifting or handing off: the timeline runs; otherwise it rests.
    private var moving: Bool {
        let now = Date()
        if let h = controller.handoff, now.timeIntervalSince(h.at) < 0.6 { return true }
        return controller.cursors.contains { c in
            c.isTrailing(at: now)
                || (c.position != nil && (c.isFlying(at: now) || (controller.motions(c).contains(.drift) && !c.ringShown)))
        }
    }
}

/// An actor's pointer: its icon, a label in its colour ("Worker 2"), and the caption chip saying what it's doing.
struct StudioPointer: View {
    let cursor: StudioCursor
    let motions: StudioMotions
    var pace: StudioPace = .watchable
    var paused = false
    /// Flying or wandering (particles trail it).
    var moving = false

    var controllerState: StudioActorState {
        if paused, cursor.state != .idle { return .paused }
        return cursor.waiting != nil ? .waiting : cursor.state
    }

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            StudioOrb(actor: cursor.actor, state: controllerState, size: cursor.actor.isArchitect ? 36 : 32, pace: pace,
                      talking: cursor.isTalking, moving: moving, gaze: cursor.gaze, moment: cursor.moment, popIn: cursor.popIn)
                .shadow(color: .black.opacity(0.25), radius: 3, y: 1)
            if let caption = cursor.caption, !caption.isEmpty {
                HStack(spacing: 5) {
                    Text(cursor.actor.isArchitect ? "Agent" : cursor.actor.label)
                        .font(.mgFootnote.weight(.semibold)).foregroundStyle(.white)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Capsule().fill(cursor.actor.color))
                    // The label says who; the caption needn't say it again ("Worker 2 · Waiting…").
                    Text(caption.hasPrefix(cursor.actor.label + " · ") ? String(caption.dropFirst(cursor.actor.label.count + 3)) : caption)
                        .font(.mgFootnote).foregroundStyle(MG.label).lineLimit(2)
                        .frame(maxWidth: 320, alignment: .leading)
                }
                .padding(.horizontal, 8).padding(.vertical, 5)
                .mgGlass(.floatingPanel, radius: MGRadius.lg)
                .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
                .offset(y: 16)
                .transition(.opacity)
            }
        }
    }
}
