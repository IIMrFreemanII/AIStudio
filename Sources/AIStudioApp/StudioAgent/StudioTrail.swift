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
        let c = trailRGB(band: band)
        return Color(red: c.0, green: c.1, blue: c.2)
    }

    /// `trailColor` as red, green and blue (the 3D echoes' shader takes the numbers).
    func trailRGB(band: Double) -> (Double, Double, Double) {
        switch number {
        case 0: return band < 1.0 / 3 ? (0.50, 0.47, 0.87) : band < 2.0 / 3 ? (0.83, 0.33, 0.53) : (0.36, 0.79, 0.65)
        case 1: return (0.176, 0.741, 0.565)
        case 2: return (0.910, 0.455, 0.290)
        default: return (0.604, 0.576, 0.918)
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

/// The ghost echoes behind flying orbs: glass spheres (the 3D orb's dark glass with no eyes, its glow and rim in the
/// trail colour), each smaller and fainter towards the back, plus small sparkles behind the handoff sparkle. Drawn
/// under the orbs. Without the 3D orb (no shader library, or Reduce Transparency) they're flat glass silhouettes.
struct StudioEchoesView: View {
    let groups: [StudioEchoGroup]
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        let spheres = StudioOrbShader.available && !reduceTransparency
        ZStack(alignment: .topLeading) {
            canvas(spheres: spheres)
            if spheres {
                ForEach(Array(groups.enumerated()), id: \.offset) { _, g in
                    if !g.sparkle {
                        ForEach(Array(g.echoes.enumerated()), id: \.offset) { _, e in
                            StudioEchoSphere(size: (g.actor.isArchitect ? 36 : 32) * e.scale, color: g.actor.trailRGB(band: e.band))
                                .opacity(min(1, e.opacity * 1.3))
                                .position(e.point)
                        }
                    }
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// The glows and the handoff sparkle's echoes (and, without the 3D orb, the flat glass).
    private func canvas(spheres: Bool) -> some View {
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
                    if spheres { continue }
                    let disc = CGRect(x: c.x - s / 2, y: c.y - s / 2, width: s, height: s)
                    ctx.fill(Path(ellipseIn: disc), with: .color(StudioColors.glass.opacity(e.opacity)))
                    let w = s * 0.085
                    ctx.stroke(Path(ellipseIn: disc.insetBy(dx: w / 2, dy: w / 2)),
                               with: .color(color.opacity(min(1, e.opacity * 1.5 * rimBoost))), lineWidth: w)
                }
            }
        }
    }
}
