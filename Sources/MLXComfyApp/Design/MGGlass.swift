import AppKit
import SwiftUI

// Glass: the backdrop blurred under the role's tint (a top-to-bottom gradient), with a light rim along the top edge.
// Only for chrome and overlays; whatever text sits on is opaque. With Reduce Transparency it's the role's flat colour.

enum MGGlassRole {
    case popover, menu, tooltip, sheet, floatingPanel, dropMarker, bar

    var tint: (Color, Color) {
        switch self {
        case .popover: return (MG.glassPopoverTint, MG.glassPopoverTintBottom)
        case .menu: return (MG.glassMenuTint, MG.glassMenuTintBottom)
        case .tooltip: return (MG.glassTooltipTint, MG.glassTooltipTintBottom)
        case .sheet: return (MG.glassSheetTint, MG.glassSheetTintBottom)
        case .floatingPanel: return (MG.glassFloatingPanelTint, MG.glassFloatingPanelTintBottom)
        case .dropMarker: return (MG.glassDropMarkerTint, MG.glassDropMarkerTintBottom)
        case .bar: return (MG.glassBarTint, MG.glassBarTintBottom)
        }
    }

    var rim: (Color, Color) {
        switch self {
        case .popover: return (MG.glassPopoverRim, MG.glassPopoverRimBottom)
        case .menu: return (MG.glassMenuRim, MG.glassMenuRimBottom)
        case .tooltip: return (MG.glassTooltipRim, MG.glassTooltipRimBottom)
        case .sheet: return (MG.glassSheetRim, MG.glassSheetRimBottom)
        case .floatingPanel: return (MG.glassFloatingPanelRim, MG.glassFloatingPanelRimBottom)
        case .dropMarker: return (MG.glassDropMarkerRim, MG.glassDropMarkerRimBottom)
        case .bar: return (MG.glassBarRim, MG.glassBarRimBottom)
        }
    }

    var fallback: Color {
        switch self {
        case .popover: return MG.glassPopoverFallback
        case .menu: return MG.glassMenuFallback
        case .tooltip: return MG.glassTooltipFallback
        case .sheet: return MG.glassSheetFallback
        case .floatingPanel: return MG.glassFloatingPanelFallback
        case .dropMarker: return MG.glassDropMarkerFallback
        case .bar: return MG.glassBarFallback
        }
    }

    var shadow: MGShadowSpec? {
        switch self {
        case .popover, .menu, .tooltip: return MGShadow.popover
        case .sheet: return MGShadow.sheet
        case .floatingPanel: return MGShadow.float
        case .dropMarker, .bar: return nil
        }
    }

    var material: NSVisualEffectView.Material {
        switch self {
        case .popover: return .popover
        case .menu: return .menu
        case .tooltip: return .toolTip
        case .sheet: return .sheet
        case .floatingPanel, .dropMarker: return .hudWindow
        case .bar: return .headerView
        }
    }
}

/// The platform blur under a glass tint.
struct MGVisualEffect: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blending: NSVisualEffectView.BlendingMode = .withinWindow

    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.state = .active
        v.material = material
        v.blendingMode = blending
        return v
    }

    func updateNSView(_ v: NSVisualEffectView, context: Context) {
        v.material = material
        v.blendingMode = blending
    }
}

/// The glass background for a role in a shape.
struct MGGlassBackground<S: InsettableShape>: View {
    let role: MGGlassRole
    let shape: S
    var blending: NSVisualEffectView.BlendingMode = .withinWindow
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        ZStack {
            if reduceTransparency {
                shape.fill(role.fallback)
            } else {
                MGVisualEffect(material: role.material, blending: blending).clipShape(shape)
                shape.fill(LinearGradient(colors: [role.tint.0, role.tint.1], startPoint: .top, endPoint: .bottom))
            }
            shape.strokeBorder(LinearGradient(colors: [role.rim.0, role.rim.1], startPoint: .top, endPoint: .bottom),
                               lineWidth: 0.5)
        }
        .compositingGroup()
        .modifier(OptionalShadow(spec: role.shadow))
    }
}

private struct OptionalShadow: ViewModifier {
    let spec: MGShadowSpec?
    func body(content: Content) -> some View {
        if let spec { content.mgShadow(spec) } else { content }
    }
}

extension View {
    /// Puts this view on glass of `role` in `shape`.
    func mgGlass<S: InsettableShape>(_ role: MGGlassRole, in shape: S) -> some View {
        background(MGGlassBackground(role: role, shape: shape))
    }

    /// Glass in a rounded rectangle (radius 10 for cards and popovers, 12 for sheets).
    func mgGlass(_ role: MGGlassRole, radius: CGFloat = MGRadius.lg) -> some View {
        mgGlass(role, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
    }

    /// A bar over the desktop blur (toolbars, tab bars) or a sidebar: the platform material under the token tint.
    func mgChrome(_ tint: Color, material: NSVisualEffectView.Material = .sidebar) -> some View {
        background {
            ZStack {
                MGVisualEffect(material: material, blending: .behindWindow)
                tint
            }
            .ignoresSafeArea()
        }
    }
}
