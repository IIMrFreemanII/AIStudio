import Agent
import AppKit
import SwiftUI

/// A four-point sparkle with gently curved sides: the architect's orbiting spark.
struct SparkleShape: Shape {
    func path(in rect: CGRect) -> Path {
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let rx = rect.width / 2, ry = rect.height / 2
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

/// The agents' colours. Workers (Mint, Ember, Plum) are teal, coral and violet; the architect is iridescent (violet,
/// magenta and teal). They live here, not in the generated MGTokens.swift, which a design-system sync would overwrite.
enum StudioColors {
    static let worker1 = Color(nsColor: MGColor.dynamic("studio-worker-1", light: (0.114, 0.620, 0.459, 1), dark: (0.176, 0.741, 0.565, 1)))
    static let worker2 = Color(nsColor: MGColor.dynamic("studio-worker-2", light: (0.847, 0.353, 0.188, 1), dark: (0.910, 0.455, 0.290, 1)))
    static let worker3 = Color(nsColor: MGColor.dynamic("studio-worker-3", light: (0.498, 0.467, 0.867, 1), dark: (0.604, 0.576, 0.918, 1)))

    // The orbs are dark glass in both modes, so their own colours are fixed.
    static let glass = Color(red: 0.071, green: 0.063, blue: 0.098)
    static let eye = Color(red: 0.93, green: 0.92, blue: 0.98)
    static let spark = Color(red: 0.66, green: 0.74, blue: 1.0)
    static let violet = Color(red: 0.50, green: 0.47, blue: 0.87)
    static let magenta = Color(red: 0.83, green: 0.33, blue: 0.53)
    static let teal = Color(red: 0.36, green: 0.79, blue: 0.65)
}

extension StudioActor {
    /// Its colour in labels, rings and rows.
    var color: Color {
        switch number {
        case 0: return MG.accent
        case 1: return StudioColors.worker1
        case 2: return StudioColors.worker2
        default: return StudioColors.worker3
        }
    }

    /// The glow inside the glass (low and right).
    var orbTint: Color { Color(red: orbTintRGB.0, green: orbTintRGB.1, blue: orbTintRGB.2) }

    /// The frosted rim: a light tint of its colour (lavender for the architect).
    var orbRim: Color { Color(red: orbRimRGB.0, green: orbRimRGB.1, blue: orbRimRGB.2) }

    /// `orbTint` as red, green and blue (the 3D orb's shader takes the numbers).
    var orbTintRGB: (Double, Double, Double) {
        switch number {
        case 0: return (0.83, 0.33, 0.53)
        case 1: return (0.11, 0.62, 0.46)
        case 2: return (0.85, 0.35, 0.19)
        default: return (0.50, 0.47, 0.87)
        }
    }

    /// `orbRim` as red, green and blue.
    var orbRimRGB: (Double, Double, Double) {
        switch number {
        case 1: return (0.62, 0.88, 0.80)
        case 2: return (0.96, 0.77, 0.70)
        default: return (0.81, 0.80, 0.96)
        }
    }
}

/// An agent as a little glass orb with eyes: a dark bubble with a frosted rim and a soft glow (iridescent for the
/// architect, which also has a tilted orbit ring with a spark riding it; one colour for each worker). Its eyes show
/// its mood (`StudioFace`): they blink and glance when idle, narrow while working, look where it's going, bounce while
/// its model writes, go wide when it waits on you, turn into happy arcs or crosses when a task ends, and close when
/// paused. Sparkles in its own colours drift around it all the time (not below 24 pt, nor with Reduce Motion).
/// Everything outside the orb (glow, ring, sparkles) draws beyond its frame, so it takes only `size` × `size` in a layout.
///
/// The orb itself is a 3D sphere (`StudioOrb.metal`): glossy dark glass lit from the upper left, the glow inside it,
/// and the eyes on its surface; it turns toward where it looks while the highlight stays put. It falls back to the
/// flat drawing when the shader library is missing or Reduce Transparency is on. Around the sphere, in depth, the
/// architect's ring passes behind and in front of it and the sparkles orbit it. In flight it tilts into the way it's
/// going; it stretches on take-off, squashes on landing, and squashes and stretches on its pop-in, its done jump and
/// while talking; the glow inside drifts slowly.
struct StudioOrb: View {
    let actor: StudioActor
    var state: StudioActorState = .idle
    var size: CGFloat = 32
    var pace: StudioPace = .watchable
    var talking = false
    var moving = false
    /// Where it looks (from its controller): up to 1 long.
    var gaze: CGVector = .zero
    /// Bumped to replay the done/failed moment.
    var moment = 0
    /// Bumped to replay the pop-in.
    var popIn = 0
    /// A still face (finished rows): its mood, no motion.
    var still = false
    /// How fast it flies, in points per second (zero when it isn't): it tilts into the way it's going.
    var velocity: CGVector = .zero

    /// Bumped when a flight starts or ends: the take-off stretch and the landing squash.
    @State private var tookOff = 0
    @State private var landed = 0

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorScheme) private var scheme

    /// Whether it draws the 3D sphere (or the flat orb).
    var isSphere: Bool { StudioOrbShader.available && !reduceTransparency }

    var face: StudioFace {
        var f = StudioFace.face(state: state, talking: talking, moving: moving, pace: pace, reduceMotion: reduceMotion || still)
        if size < 24 { f.particles = false }
        return f
    }

    private var seed: UInt64 { UInt64(actor.number + 1) &* 7919 &+ UInt64(size) }

    var body: some View {
        let f = face
        Group {
            if f.animates {
                TimelineView(.animation(minimumInterval: 1.0 / 30)) { ctx in orb(f, t: ctx.date.timeIntervalSinceReferenceDate) }
            } else if f.blinks || f.glances {
                // Only blinks and glances: redraw just when one starts or ends.
                let now = Date().timeIntervalSinceReferenceDate
                let marks = StudioFace.changes(after: now, span: 600, seed: seed, blinks: f.blinks, glances: f.glances)
                TimelineView(.explicit(marks.map { Date(timeIntervalSinceReferenceDate: $0) })) { ctx in
                    orb(f, t: ctx.date.timeIntervalSinceReferenceDate)
                }
            } else {
                orb(f, t: 0)
            }
        }
        .frame(width: size, height: size)
        // The architect's orbit ring reaches past the orb: room for it beside text.
        .padding(.horizontal, actor.isArchitect && size < 30 ? size * 0.32 : 0)
        .keyframeAnimator(initialValue: OrbBounce(), trigger: popIn) { content, v in
            content.scaleEffect(x: v.scale * (1 - 0.5 * v.stretch), y: v.scale * (1 + v.stretch))
        } keyframes: { _ in
            KeyframeTrack(\.scale) {
                LinearKeyframe(0.55, duration: 0.01)
                SpringKeyframe(1.1, duration: 0.22)
                SpringKeyframe(1, duration: 0.16)
            }
            // Stretched as it grows, a little wide as it settles.
            KeyframeTrack(\.stretch) {
                LinearKeyframe(0, duration: 0.01)
                CubicKeyframe(squashes ? 0.16 : 0, duration: 0.12)
                CubicKeyframe(squashes ? -0.07 : 0, duration: 0.14)
                SpringKeyframe(0, duration: 0.2)
            }
        }
        .keyframeAnimator(initialValue: 0.0, trigger: tookOff) { content, e in
            content.scaleEffect(x: 1 - 0.5 * e, y: 1 + e)
        } keyframes: { _ in
            KeyframeTrack {
                CubicKeyframe(0.12, duration: 0.08)
                SpringKeyframe(0, duration: 0.3)
            }
        }
        .keyframeAnimator(initialValue: 0.0, trigger: landed) { content, e in
            content.scaleEffect(x: 1 - 0.5 * e, y: 1 + e, anchor: .bottom)
        } keyframes: { _ in
            KeyframeTrack {
                CubicKeyframe(-0.16, duration: 0.07)
                CubicKeyframe(0.06, duration: 0.12)
                SpringKeyframe(0, duration: 0.25)
            }
        }
        .onChange(of: velocity != .zero) { _, flying in
            guard squashes else { return }
            if flying { tookOff &+= 1 } else { landed &+= 1 }
        }
        .keyframeAnimator(initialValue: 0.0, trigger: state == .failed && !reduceMotion ? moment : 0) { content, x in
            content.offset(x: x)
        } keyframes: { _ in
            KeyframeTrack {
                LinearKeyframe(-0.12 * size, duration: 0.07)
                LinearKeyframe(0.12 * size, duration: 0.07)
                LinearKeyframe(-0.08 * size, duration: 0.07)
                LinearKeyframe(0, duration: 0.07)
            }
        }
        .keyframeAnimator(initialValue: OrbBounce(), trigger: state == .done && !reduceMotion ? moment : 0) { content, v in
            content.scaleEffect(x: v.scale * (1 - 0.5 * v.stretch), y: v.scale * (1 + v.stretch)).offset(y: v.lift)
        } keyframes: { _ in
            KeyframeTrack(\.scale) {
                CubicKeyframe(1.25, duration: 0.15)
                CubicKeyframe(1, duration: 0.3)
            }
            // A little jump: stretched going up, squashed landing.
            KeyframeTrack(\.lift) {
                CubicKeyframe(squashes ? -0.22 * size : 0, duration: 0.2)
                CubicKeyframe(0, duration: 0.22)
            }
            KeyframeTrack(\.stretch) {
                CubicKeyframe(squashes ? 0.14 : 0, duration: 0.1)
                CubicKeyframe(0, duration: 0.12)
                CubicKeyframe(0, duration: 0.18)
                CubicKeyframe(squashes ? -0.18 : 0, duration: 0.06)
                SpringKeyframe(0, duration: 0.3)
            }
        }
        .accessibilityElement()
        .accessibilityLabel(actor.label)
    }

    /// It squashes and stretches (the 3D orb, with motion allowed).
    private var squashes: Bool { isSphere && !reduceMotion && !still }

    // MARK: layers

    private func orb(_ f: StudioFace, t: Double) -> some View {
        let s = size
        let blink = f.blinks && StudioFace.isBlinking(at: t, seed: seed)
        let look = f.glances ? StudioFace.glance(at: t, seed: seed) : gaze
        let bounce = f.bounce ? sin(t * 2 * .pi * 3) * 1.5 * s / 32 : 0
        let breathe = f.breathe ? 0.8 + 0.2 * sin(t * 1.8) : 1
        let angle = f.orbit ? t * 0.9 : -0.45
        let dim = state == .paused
        let sphere = isSphere
        let spin = f.animates ? StudioOrbPose.spin(at: t) : 0
        let flight = reduceMotion ? CGVector.zero : velocity
        let ringRoll = StudioOrbPose.pose(face: f, velocity: flight).ringRoll
        let ringAngle = f.orbit ? (t * 0.9).truncatingRemainder(dividingBy: 2 * .pi) : -0.45
        let sparkles = f.particles ? 6 : 0
        let wrapped = t.truncatingRemainder(dividingBy: 3600)
        let talk = sphere && f.bounce ? StudioOrbPose.talkStretch(at: t) : 0
        return Group {
            if sphere {
                StudioOrbSphere(actor: actor, face: f, size: s, blinking: blink, gaze: look,
                                bounce: f.bounce ? sin(t * 2 * .pi * 3) : 0, breathe: breathe, paused: dim,
                                velocity: flight, spin: spin)
                    .animation(.spring(duration: 0.35), value: look)
            } else {
                ZStack {
                    Circle().fill(StudioColors.glass)
                    Circle().fill(RadialGradient(colors: [actor.orbTint.opacity(0.75), actor.orbTint.opacity(0)],
                                                 center: UnitPoint(x: 0.72, y: 0.76), startRadius: 0, endRadius: s * 0.62))
                    if actor.isArchitect {
                        Circle().fill(RadialGradient(colors: [StudioColors.violet.opacity(0.45), .clear],
                                                     center: UnitPoint(x: 0.3, y: 0.28), startRadius: 0, endRadius: s * 0.5))
                    }
                }
            }
        }
        .frame(width: s, height: s)
        .background {
            ZStack {
                if state == .waiting { WaitingRing(size: s, pulse: !reduceMotion) }
                glow.opacity(breathe * (dim ? 0.35 : 1) * (scheme == .light ? 1.25 : 1) * (s < 24 ? 0.75 : 1))
                if sphere {
                    if actor.isArchitect || sparkles > 0 {
                        StudioOrbAround(actor: actor, size: s, front: false, ringAngle: ringAngle, ringRoll: ringRoll,
                                        particles: sparkles, time: wrapped)
                    }
                } else if actor.isArchitect {
                    orbit(angle: angle, front: false)
                }
            }
        }
        .overlay {
            ZStack {
                if !sphere {
                    rim
                    eyes(f.eyes, blink: blink).offset(x: look.dx * s * 0.08, y: look.dy * s * 0.08 + bounce)
                        .animation(.spring(duration: 0.35), value: look)
                }
                if sphere {
                    if actor.isArchitect || sparkles > 0 {
                        StudioOrbAround(actor: actor, size: s, front: true, ringAngle: ringAngle, ringRoll: ringRoll,
                                        particles: sparkles, time: wrapped)
                    }
                } else {
                    if actor.isArchitect { orbit(angle: angle, front: true) }
                    if f.particles { particles(t) }
                }
            }
        }
        .scaleEffect(x: 1 - 0.5 * talk, y: 1 + talk)
        .opacity(dim ? 0.7 : 1)
    }

    @ViewBuilder private var glow: some View {
        let s = size
        let blur = max(2, s * 0.17)
        if actor.isArchitect {
            ZStack {
                Circle().fill(StudioColors.violet).frame(width: s * 1.08, height: s * 1.08).offset(x: -s * 0.12, y: -s * 0.1).opacity(0.6)
                Circle().fill(StudioColors.magenta).frame(width: s * 0.95, height: s * 0.95).offset(x: s * 0.14, y: s * 0.15).opacity(0.5)
                Circle().fill(StudioColors.teal).frame(width: s * 0.7, height: s * 0.7).offset(x: s * 0.24, y: 0).opacity(0.42)
            }
            .blur(radius: blur)
        } else {
            Circle().fill(actor.orbTint).frame(width: s * 1.14, height: s * 1.14).opacity(0.6).blur(radius: blur)
        }
    }

    private var rim: some View {
        let s = size
        let w = max(1.2, s * 0.085)
        return ZStack {
            Circle().stroke(actor.orbRim.opacity(0.35), lineWidth: w).blur(radius: max(0.5, s * 0.03))
            Circle().strokeBorder(actor.orbRim.opacity(0.92), lineWidth: w)
            Circle().strokeBorder(Color.white.opacity(0.16), lineWidth: max(0.5, s * 0.012)).padding(w)
            Circle().trim(from: 0.56, to: 0.7)
                .stroke(Color.white.opacity(0.65), style: StrokeStyle(lineWidth: max(0.6, s * 0.028), lineCap: .round))
                .padding(w / 2)
        }
        .frame(width: s, height: s)
    }

    @ViewBuilder private func eyes(_ e: StudioEyes, blink: Bool) -> some View {
        let s = size
        let w = max(1.6, s * 0.13), h = max(3, s * 0.33)
        // Eye centres stay `gap` apart whatever each eye's width (arcs and crosses are wider than pills).
        let gap = s * (e == .happy || e == .cross ? 0.27 : 0.21)
        let eyeWidth = e == .happy ? w * 1.3 : e == .cross ? w * 1.15 : e == .closed ? w * 1.3 : w
        HStack(spacing: max(0, gap - eyeWidth)) {
            ForEach(0..<2, id: \.self) { _ in eye(e, blink: blink, w: w, h: h) }
        }
        .frame(width: gap + eyeWidth * 1.2, height: h * 1.2)
        .offset(x: -s * 0.1, y: -s * 0.06)
    }

    @ViewBuilder private func eye(_ e: StudioEyes, blink: Bool, w: CGFloat, h: CGFloat) -> some View {
        switch e {
        case .happy:
            HappyEye().stroke(StudioColors.eye, style: StrokeStyle(lineWidth: w * 0.7, lineCap: .round))
                .frame(width: w * 1.3, height: h * 0.42)
        case .cross:
            CrossEye().stroke(StudioColors.eye, style: StrokeStyle(lineWidth: w * 0.55, lineCap: .round))
                .frame(width: w * 1.15, height: w * 1.15)
        case .closed:
            Capsule().fill(StudioColors.eye).frame(width: w * 1.3, height: max(1, h * 0.14))
        default:
            let k: (CGFloat, CGFloat) = e == .focus ? (1, 0.72) : e == .wide ? (1.08, 1.15) : (1, 1)
            Capsule().fill(StudioColors.eye)
                .frame(width: w * k.0 * (blink ? 1.1 : 1), height: blink ? max(1, h * 0.12) : h * k.1)
        }
    }

    /// The architect's tilted orbit ring and its spark: the back half behind the orb, the front half over it.
    @ViewBuilder private func orbit(angle: Double, front: Bool) -> some View {
        let s = size
        let a = s * 0.81, b = s * 0.22
        let tilt = Angle.degrees(-18)
        ZStack {
            // On a light background the lavender ring and white spark would vanish: violet there.
            let light = scheme == .light
            Ellipse().trim(from: front ? 0 : 0.5, to: front ? 0.5 : 1)
                .stroke(light ? StudioColors.violet.opacity(0.7) : actor.orbRim.opacity(0.55), lineWidth: max(0.6, s * (light ? 0.028 : 0.022)))
                .frame(width: a * 2, height: b * 2)
            if (sin(angle) > 0) == front {
                SparkleShape().fill(light ? StudioColors.violet : .white)
                    .frame(width: max(4, s * 0.22), height: max(4, s * 0.22))
                    .shadow(color: StudioColors.spark.opacity(0.9), radius: max(1, s * 0.06))
                    .offset(x: a * cos(angle), y: b * sin(angle))
            }
        }
        .rotationEffect(tilt)
        .allowsHitTesting(false)
    }

    private func particles(_ t: Double) -> some View {
        let s = size
        return Canvas { ctx, box in
            let c = CGPoint(x: box.width / 2, y: box.height / 2)
            let r = max(0.8, s * 0.035)
            for (i, p) in StudioFace.particles(actor: actor.number, time: t).enumerated() {
                let at = CGPoint(x: c.x + p.offset.dx * s, y: c.y + p.offset.dy * s)
                ctx.fill(Path(ellipseIn: CGRect(x: at.x - r, y: at.y - r, width: r * 2, height: r * 2)),
                         with: .color(particleColor(i).opacity(p.opacity * 0.9)))
            }
        }
        .frame(width: s * 2.2, height: s * 2.2)
        .allowsHitTesting(false)
    }

    /// A sparkle's colour: the architect's take violet, magenta and teal in turn, a worker's its glow colour; lightened
    /// a little on a dark window so they read as light.
    private func particleColor(_ i: Int) -> Color {
        let base = actor.isArchitect ? [StudioColors.violet, StudioColors.magenta, StudioColors.teal][i % 3] : actor.orbTint
        return scheme == .dark ? base.mix(with: .white, by: 0.3) : base
    }
}

/// The 3D orb's shader (`StudioOrb.metal`, compiled into the module's resource bundle).
enum StudioOrbShader {
    static let library = ShaderLibrary.bundle(.module)
    /// The compiled library is there (the flat orb is drawn otherwise).
    static let available = Bundle.module.url(forResource: "default", withExtension: "metallib") != nil
}

/// The 3D sphere: dark glass, the glow inside it, the rim, the highlight and the eyes on its surface. Animatable in its
/// gaze, so the sphere turns smoothly toward where it looks.
private struct StudioOrbSphere: View, Animatable {
    let actor: StudioActor
    let face: StudioFace
    let size: CGFloat
    let blinking: Bool
    var gaze: CGVector
    let bounce: Double
    let breathe: Double
    let paused: Bool
    var velocity: CGVector = .zero
    var spin = 0.0
    var glass = StudioOrbGlass.standard

    @Environment(\.displayScale) private var displayScale

    var animatableData: AnimatablePair<CGFloat, CGFloat> {
        get { AnimatablePair(gaze.dx, gaze.dy) }
        set { gaze = CGVector(dx: newValue.first, dy: newValue.second) }
    }

    var body: some View {
        let pose = StudioOrbPose.pose(face: face, blinking: blinking, gaze: gaze, bounce: bounce, breathe: breathe,
                                      paused: paused, radius: size / 2, velocity: velocity)
        let tint = actor.orbTintRGB, rim = actor.orbRimRGB, g = glass
        Rectangle().fill(.white)
            .frame(width: size, height: size)
            .colorEffect(StudioOrbShader.library.studioOrbSphere(
                .boundingRect, .float(displayScale),
                .float3(tint.0, tint.1, tint.2), .float3(rim.0, rim.1, rim.2), .float(actor.isArchitect ? 1 : 0),
                .floatArray(pose.rotation),
                .float4(Double(pose.eyeShape.rawValue), pose.eyeWidth, pose.eyeHeight, pose.eyeLift),
                .float(pose.eyeGap), .float(pose.glow), .float(spin),
                .float4(g.highlightSize, g.highlightStrength, g.rim, g.frost),
                .float3(g.glowSpread, g.glowStrength, g.iridescence)))
            .allowsHitTesting(false)
    }
}

/// A happy eye: an upward arc (^).
private struct HappyEye: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: r.minX, y: r.maxY))
        p.addQuadCurve(to: CGPoint(x: r.maxX, y: r.maxY), control: CGPoint(x: r.midX, y: r.minY - r.height * 0.6))
        return p
    }
}

/// A failed eye: a small cross.
private struct CrossEye: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: r.minX, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX, y: r.maxY))
        p.move(to: CGPoint(x: r.maxX, y: r.minY))
        p.addLine(to: CGPoint(x: r.minX, y: r.maxY))
        return p
    }
}

/// An amber ring while the orb waits on you: pulsing outwards, or steady with Reduce Motion.
private struct WaitingRing: View {
    let size: CGFloat
    let pulse: Bool

    var body: some View {
        if pulse {
            PhaseAnimator([false, true]) { out in
                Circle().stroke(MG.warning, lineWidth: max(1, size * 0.06))
                    .frame(width: size, height: size)
                    .scaleEffect(out ? 1.75 : 1.05)
                    .opacity(out ? 0 : 0.85)
            } animation: { out in out ? .easeOut(duration: 1.4) : .linear(duration: 0.01) }
        } else {
            Circle().stroke(MG.warning, lineWidth: max(1, size * 0.06)).frame(width: size * 1.2, height: size * 1.2)
        }
    }
}

/// The pop-in and the done jump: size, lift (points, up is negative) and stretch (taller positive, wider negative).
private struct OrbBounce {
    var scale = 1.0
    var lift = 0.0
    var stretch = 0.0
}

/// What goes around the 3D sphere, in depth (`studioOrbAround` in StudioOrb.metal): the architect's ring and its spark,
/// and the sparkles orbiting it. Drawn behind the sphere (`front` false: the far parts, which the sphere hides) and over
/// it (`front` true), on a square 2.4 radii across around the orb's frame.
private struct StudioOrbAround: View {
    let actor: StudioActor
    let size: CGFloat
    let front: Bool
    let ringAngle: Double
    let ringRoll: Double
    let particles: Int
    /// Seconds, kept small for the shader's floats.
    let time: Double

    static let extent = 2.4

    @Environment(\.displayScale) private var displayScale
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let tint = actor.orbTintRGB, rim = actor.orbRimRGB
        let side = size * Self.extent
        Rectangle().fill(.white)
            .frame(width: side, height: side)
            .colorEffect(StudioOrbShader.library.studioOrbAround(
                .boundingRect, .float(displayScale), .float(Self.extent), .float(front ? 1 : 0),
                .float3(tint.0, tint.1, tint.2), .float3(rim.0, rim.1, rim.2), .float(actor.isArchitect ? 1 : 0),
                .float3(actor.isArchitect ? 1 : 0, ringAngle, ringRoll), .float(Double(particles)),
                .float(Double(actor.number + 1)), .float(time), .float(scheme == .light ? 1 : 0)))
            .frame(width: size, height: size)
            .allowsHitTesting(false)
    }
}
