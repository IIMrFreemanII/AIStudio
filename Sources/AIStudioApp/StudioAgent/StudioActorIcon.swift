import Agent
import AppKit
import SwiftUI

/// A four-point sparkle with gently curved sides: the Studio agent's mark (its architect and workers wear it).
struct SparkleShape: Shape {
    func path(in rect: CGRect) -> Path {
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let rx = rect.width / 2, ry = rect.height / 2
        // Each side bows in towards the middle: the control points sit close to the centre.
        let k: CGFloat = 0.27, j: CGFloat = 0.13
        var p = Path()
        p.move(to: CGPoint(x: c.x, y: c.y - ry))
        p.addCurve(to: CGPoint(x: c.x + rx, y: c.y), control1: CGPoint(x: c.x + rx * j, y: c.y - ry * k),
                   control2: CGPoint(x: c.x + rx * k, y: c.y - ry * j))
        p.addCurve(to: CGPoint(x: c.x, y: c.y + ry), control1: CGPoint(x: c.x + rx * k, y: c.y + ry * j),
                   control2: CGPoint(x: c.x + rx * j, y: c.y + ry * k))
        p.addCurve(to: CGPoint(x: c.x - rx, y: c.y), control1: CGPoint(x: c.x - rx * j, y: c.y + ry * k),
                   control2: CGPoint(x: c.x - rx * k, y: c.y + ry * j))
        p.addCurve(to: CGPoint(x: c.x, y: c.y - ry), control1: CGPoint(x: c.x - rx * k, y: c.y - ry * j),
                   control2: CGPoint(x: c.x - rx * j, y: c.y - ry * k))
        p.closeSubpath()
        return p
    }
}

/// The workers' colours (teal, coral, violet), beside the design system's accent blue for the architect. They live
/// here, not in the generated MGTokens.swift, which a design-system sync would overwrite.
enum StudioColors {
    static let worker1 = Color(nsColor: MGColor.dynamic("studio-worker-1", light: (0.114, 0.620, 0.459, 1), dark: (0.176, 0.741, 0.565, 1)))
    static let worker2 = Color(nsColor: MGColor.dynamic("studio-worker-2", light: (0.847, 0.353, 0.188, 1), dark: (0.910, 0.455, 0.290, 1)))
    static let worker3 = Color(nsColor: MGColor.dynamic("studio-worker-3", light: (0.498, 0.467, 0.867, 1), dark: (0.604, 0.576, 0.918, 1)))
}

extension StudioActor {
    var color: Color {
        switch number {
        case 0: return MG.accent
        case 1: return StudioColors.worker1
        case 2: return StudioColors.worker2
        default: return StudioColors.worker3
        }
    }
}

/// An actor's icon: the architect a rounded square in the accent blue with a sparkle (and a small one at its corner),
/// a worker a circle in its colour with a sparkle and its number. It moves with what the actor is doing: it twinkles
/// while working, an amber ring pulses while it waits on you, it bursts into a check mark when done, shakes when it
/// failed, dims while paused, and pops in when it appears (`motions`, from `StudioMotion`).
struct StudioActorIcon: View {
    let actor: StudioActor
    var state: StudioActorState = .idle
    var size: CGFloat = 22
    var motions: StudioMotions = []
    /// Bumped to replay the done/failed moment.
    var moment = 0
    /// Bumped to replay the pop-in.
    var popIn = 0

    var body: some View {
        ZStack {
            if motions.contains(.pulse) {
                PulseRing(size: size)
            } else if motions.contains(.ring) {
                Circle().stroke(MG.warning, lineWidth: 2).frame(width: size + 6, height: size + 6)
            }
            badge
                .keyframeAnimator(initialValue: 0.0, trigger: shakeTrigger) { content, x in
                    content.offset(x: x)
                } keyframes: { _ in
                    KeyframeTrack {
                        LinearKeyframe(-4, duration: 0.07)
                        LinearKeyframe(4, duration: 0.07)
                        LinearKeyframe(-3, duration: 0.07)
                        LinearKeyframe(0, duration: 0.07)
                    }
                }
        }
        .frame(width: size + 8, height: size + 8)
        .opacity(motions.contains(.dim) ? 0.5 : 1)
        .keyframeAnimator(initialValue: 1.0, trigger: popIn) { content, s in
            content.scaleEffect(s)
        } keyframes: { _ in
            KeyframeTrack {
                LinearKeyframe(0.6, duration: 0.01)
                SpringKeyframe(1.08, duration: 0.2)
                SpringKeyframe(1, duration: 0.15)
            }
        }
        .accessibilityLabel(actor.label)
    }

    private var shakeTrigger: Int { motions.contains(.shake) && state == .failed ? moment : 0 }
    private var burstTrigger: Int { motions.contains(.burst) && state == .done ? moment : 0 }

    @ViewBuilder private var badge: some View {
        ZStack {
            if actor.isArchitect {
                RoundedRectangle(cornerRadius: size * 0.26, style: .continuous).fill(actor.color)
            } else {
                Circle().fill(actor.color)
            }
            mark
            if actor.isArchitect {
                Blinking(on: motions.contains(.twinkle)) {
                    SparkleShape().fill(.white).frame(width: size * 0.26, height: size * 0.26)
                }
                .offset(x: size * 0.24, y: -size * 0.24)
            } else {
                Text("\(actor.number)")
                    .font(.system(size: max(8, size * 0.42), weight: .heavy, design: .rounded))
                    .foregroundStyle(actor.color)
                    .frame(width: max(11, size * 0.56), height: max(11, size * 0.56))
                    .background(Circle().fill(.white))
                    .overlay(Circle().stroke(actor.color, lineWidth: 1))
                    .offset(x: size * 0.4, y: size * 0.4)
            }
        }
        .frame(width: size, height: size)
    }

    /// The sparkle, or the check mark / cross for a moment when the task ends.
    @ViewBuilder private var mark: some View {
        let glyph = size * (actor.isArchitect ? 0.6 : 0.58)
        Group {
            if state == .done {
                Image(systemName: "checkmark").font(.system(size: size * 0.46, weight: .heavy)).foregroundStyle(.white)
                    .transition(.scale.combined(with: .opacity))
            } else if state == .failed {
                Image(systemName: "xmark").font(.system(size: size * 0.42, weight: .heavy)).foregroundStyle(.white)
                    .transition(.opacity)
            } else {
                Twinkling(on: motions.contains(.twinkle)) {
                    SparkleShape().fill(.white).frame(width: glyph, height: glyph)
                }
                .offset(x: actor.isArchitect ? -size * 0.04 : 0, y: actor.isArchitect ? size * 0.04 : 0)
                .transition(.scale.combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.3), value: state)
        .keyframeAnimator(initialValue: 1.0, trigger: burstTrigger) { content, s in
            content.scaleEffect(s)
        } keyframes: { _ in
            KeyframeTrack {
                CubicKeyframe(1.4, duration: 0.15)
                CubicKeyframe(1, duration: 0.25)
            }
        }
    }
}

/// The sparkle breathing: 1 → 1.12 → 1 every 1.6 s.
private struct Twinkling<Content: View>: View {
    let on: Bool
    @ViewBuilder let content: Content

    var body: some View {
        if on {
            PhaseAnimator([false, true]) { big in
                content.scaleEffect(big ? 1.12 : 1)
            } animation: { _ in .easeInOut(duration: 0.8) }
        } else {
            content
        }
    }
}

/// The architect's small sparkle blinking, out of step with the big one.
private struct Blinking<Content: View>: View {
    let on: Bool
    @ViewBuilder let content: Content

    var body: some View {
        if on {
            PhaseAnimator([1.0, 0.25]) { o in
                content.opacity(o).scaleEffect(0.6 + 0.4 * o)
            } animation: { _ in .easeInOut(duration: 0.55) }
        } else {
            content
        }
    }
}

/// An amber ring pulsing out from the icon every 1.4 s (waiting on you).
private struct PulseRing: View {
    let size: CGFloat

    var body: some View {
        PhaseAnimator([false, true]) { out in
            Circle().stroke(MG.warning, lineWidth: 2)
                .frame(width: size, height: size)
                .scaleEffect(out ? 1.8 : 1)
                .opacity(out ? 0 : 0.8)
        } animation: { out in out ? .easeOut(duration: 1.4) : .linear(duration: 0.01) }
    }
}
