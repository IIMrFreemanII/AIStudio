import Agent
import SwiftUI

/// One actor's ghost echoes (or the handoff sparkle's) at a moment, ready to draw.
struct StudioEchoGroup {
    let actor: StudioActor
    let echoes: [StudioEcho]
    /// The handoff sparkle's echoes: small sparkles instead of glass.
    var sparkle = false
}

extension StudioActor {
    /// The colour of its echoes at `band` (0 front, 1 back): a worker's own tone (fixed, as the orbs are dark glass in
    /// both modes); the architect shifts violet → magenta → teal.
    func trailColor(band: Double) -> Color {
        switch number {
        case 0: return band < 1.0 / 3 ? StudioColors.violet : band < 2.0 / 3 ? StudioColors.magenta : StudioColors.teal
        case 1: return Color(red: 0.176, green: 0.741, blue: 0.565)
        case 2: return Color(red: 0.910, green: 0.455, blue: 0.290)
        default: return Color(red: 0.604, green: 0.576, blue: 0.918)
        }
    }
}

extension StudioCursor {
    /// Its echoes at `now` (the orb's drift after landing pulls them along).
    func echoes(at now: Date, drift: Bool) -> [StudioEcho] {
        guard isTrailing(at: now), let last = trail.last else { return [] }
        var offset = CGVector.zero
        if !isFlying(at: now), let p = displayed(at: now, drift: drift) {
            offset = CGVector(dx: p.x - last.flight.to.x, dy: p.y - last.flight.to.y)
        }
        return StudioTrail.echoes(trail, now: now.timeIntervalSinceReferenceDate, offset: offset)
    }
}

/// The ghost echoes behind flying orbs: glass silhouettes (dark glass, a rim and a faint glow in the trail colour, no
/// eyes), each smaller and fainter towards the back, plus small sparkles behind the handoff sparkle. One canvas for
/// all of them, drawn under the orbs.
struct StudioEchoesView: View {
    let groups: [StudioEchoGroup]
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Canvas { ctx, _ in
            let rimBoost = scheme == .light ? 1.25 : 1
            for g in groups {
                for e in g.echoes {
                    let color = g.actor.trailColor(band: e.band)
                    if g.sparkle {
                        let s = 12 * e.scale
                        let rect = CGRect(x: e.point.x - s / 2, y: e.point.y - s / 2, width: s, height: s)
                        ctx.fill(SparkleShape().path(in: rect), with: .color(color.opacity(min(1, e.opacity * 1.4))))
                        continue
                    }
                    let s = (g.actor.isArchitect ? 36 : 32) * e.scale
                    let c = e.point
                    let glowR = s * 0.6 * 1.1
                    ctx.fill(Path(ellipseIn: CGRect(x: c.x - glowR, y: c.y - glowR, width: glowR * 2, height: glowR * 2)),
                             with: .radialGradient(Gradient(colors: [color.opacity(0.8 * e.opacity), color.opacity(0)]),
                                                   center: c, startRadius: 0, endRadius: glowR))
                    let disc = CGRect(x: c.x - s / 2, y: c.y - s / 2, width: s, height: s)
                    ctx.fill(Path(ellipseIn: disc), with: .color(StudioColors.glass.opacity(e.opacity)))
                    let w = s * 0.085
                    ctx.stroke(Path(ellipseIn: disc.insetBy(dx: w / 2, dy: w / 2)),
                               with: .color(color.opacity(min(1, e.opacity * 1.5 * rimBoost))), lineWidth: w)
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}
