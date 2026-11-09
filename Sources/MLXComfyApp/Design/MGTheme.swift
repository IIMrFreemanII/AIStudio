import AppKit
import CoreText
import SwiftUI

// The MetalGraphics Frosted Glass design system in SwiftUI: fonts, text styles, shadows, appearance and the root
// modifier every window's content goes through. Colours and metrics are generated into MGTokens.swift.

/// A text style from the tokens.
struct MGTextStyle {
    let size: CGFloat
    let weight: Int
    var font: Font { MGFonts.font(size: size, weight: weight) }
    var nsFont: NSFont { MGFonts.nsFont(size: size, weight: weight) }
}

/// A shadow from the tokens (CSS blur; SwiftUI's radius is about half of it).
struct MGShadowSpec {
    let x: CGFloat
    let y: CGFloat
    let blur: CGFloat
    let color: Color
}

extension View {
    func mgShadow(_ s: MGShadowSpec) -> some View {
        shadow(color: s.color, radius: s.blur / 2, x: s.x, y: s.y)
    }
}

/// JetBrains Mono, bundled (SIL OFL): registered for this process at launch. If that fails the app falls back to the
/// system monospaced font, never to a missing one.
enum MGFonts {
    private(set) static var registered = false

    static func register() {
        guard !registered else { return }
        let dir = Bundle.module.url(forResource: "Fonts", withExtension: nil)
        let files = (dir.flatMap { try? FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil) } ?? [])
            .filter { ["woff2", "woff", "ttf", "otf"].contains($0.pathExtension.lowercased()) }
        for url in files { CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil) }
        registered = NSFont(name: postScriptName(weight: 400), size: 13) != nil
    }

    static func postScriptName(weight: Int) -> String {
        switch weight {
        case ..<450: return "JetBrainsMono-Regular"
        case ..<550: return "JetBrainsMono-Medium"
        case ..<650: return "JetBrainsMono-SemiBold"
        default: return "JetBrainsMono-Bold"
        }
    }

    static func font(size: CGFloat, weight: Int) -> Font {
        registered ? .custom(postScriptName(weight: weight), fixedSize: size)
            : .system(size: size, weight: Font.Weight(mgWeight: weight), design: .monospaced)
    }

    static func nsFont(size: CGFloat, weight: Int) -> NSFont {
        NSFont(name: postScriptName(weight: weight), size: size)
            ?? .monospacedSystemFont(ofSize: size, weight: NSFont.Weight(mgWeight: weight))
    }

    /// The face the UI draws with (for logs and snapshots).
    static var description: String { registered ? "JetBrains Mono" : "system monospaced (JetBrains Mono not registered)" }
}

extension Font.Weight {
    init(mgWeight w: Int) {
        switch w {
        case ..<450: self = .regular
        case ..<550: self = .medium
        case ..<650: self = .semibold
        default: self = .bold
        }
    }

    var mgValue: Int {
        switch self {
        case .medium: return 500
        case .semibold: return 600
        case .bold, .heavy, .black: return 700
        default: return 400
        }
    }
}

extension NSFont.Weight {
    init(mgWeight w: Int) {
        switch w {
        case ..<450: self = .regular
        case ..<550: self = .medium
        case ..<650: self = .semibold
        default: self = .bold
        }
    }
}

extension Font {
    /// JetBrains Mono at a size (the `.system(size:weight:design:)` of this design system; `design` is ignored).
    static func mg(size: CGFloat, weight: Font.Weight = .regular, design: Font.Design? = nil) -> Font {
        MGFonts.font(size: size, weight: weight.mgValue)
    }

    static var mgLargeTitle: Font { MGType.largeTitle.font }
    static var mgTitle: Font { MGType.title.font }
    static var mgTitle2: Font { MGType.title2.font }
    static var mgTitle3: Font { MGType.title3.font }
    static var mgHeadline: Font { MGType.headline.font }
    static var mgBody: Font { MGType.body.font }
    static var mgCallout: Font { MGType.callout.font }
    static var mgSubheadline: Font { MGType.subheadline.font }
    static var mgFootnote: Font { MGType.footnote.font }
    static var mgMono: Font { MGType.mono.font }
}

/// Light, dark, or whatever macOS is set to (Settings › Appearance).
enum MGAppearance: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
}

/// Every window's root: the font, label colour, accent, control styles and appearance.
struct MGRoot: ViewModifier {
    @AppStorage("appearance") private var appearance = MGAppearance.system.rawValue

    func body(content: Content) -> some View {
        content
            .font(.mgBody)
            .foregroundStyle(MG.label)
            .tint(MG.accent)
            .buttonStyle(.mg(.bordered))
            .textFieldStyle(MGTextFieldStyle())
            .toggleStyle(MGSwitchStyle())
            .labeledContentStyle(MGLabeledContentStyle())
            .preferredColorScheme((MGAppearance(rawValue: appearance) ?? .system).colorScheme)
    }
}

extension View {
    func mgRoot() -> some View { modifier(MGRoot()) }

    /// A 0.5 pt hairline in the separator colour, drawn inside the shape's edge.
    func mgHairline<S: InsettableShape>(_ shape: S, color: Color = MG.separator) -> some View {
        overlay(shape.strokeBorder(color, lineWidth: 0.5))
    }
}

/// A 0.5 pt separator line (horizontal by default), or the 1 pt gap between docked panels.
struct MGDivider: View {
    var vertical = false
    var gap = false

    var body: some View {
        Rectangle()
            .fill(gap ? MG.gapTint : MG.separator)
            .frame(width: vertical ? (gap ? 1 : 0.5) : nil, height: vertical ? nil : (gap ? 1 : 0.5))
    }
}
