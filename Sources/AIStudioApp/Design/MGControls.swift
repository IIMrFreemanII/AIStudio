import AppKit
import SwiftUI

// SwiftUI versions of the design system's controls, matched to its metrics: Button, TextField, Toggle, LabeledContent,
// segmented Picker and ListRow.

// MARK: buttons

enum MGButtonKind {
    /// An accent label (and a hover fill).
    case borderless
    /// A label, nothing else.
    case plain
    /// A fill.
    case bordered
    /// The accent.
    case prominent
    /// A glyph (or short label) in the label colour with a hover fill: toolbars, rails, close buttons.
    case icon
}

/// Buttons: 22 tall (18 mini, 20 small, 28 large), radius 6, fills and accent from the tokens; disabled dims to 0.4.
struct MGButtonStyle: ButtonStyle {
    var kind: MGButtonKind = .bordered
    var destructive = false

    func makeBody(configuration: Configuration) -> some View {
        MGButtonBody(configuration: configuration, kind: kind,
                     destructive: destructive || configuration.role == .destructive)
    }
}

extension ButtonStyle where Self == MGButtonStyle {
    static func mg(_ kind: MGButtonKind, destructive: Bool = false) -> MGButtonStyle {
        MGButtonStyle(kind: kind, destructive: destructive)
    }
}

private struct MGButtonBody: View {
    let configuration: ButtonStyle.Configuration
    let kind: MGButtonKind
    let destructive: Bool
    @Environment(\.isEnabled) private var enabled
    @Environment(\.controlSize) private var size
    @State private var hover = false

    var height: CGFloat {
        switch size {
        case .mini: return 18
        case .small: return 20
        case .large, .extraLarge: return 28
        default: return 22
        }
    }

    var fill: Color {
        switch kind {
        case .bordered: return configuration.isPressed ? MG.fillPressed : hover ? MG.fillHover : MG.fill
        case .prominent: return destructive ? MG.destructive : MG.accent
        case .borderless, .icon: return configuration.isPressed ? MG.pressed : hover ? MG.hover : .clear
        case .plain: return .clear
        }
    }

    var foreground: Color {
        switch kind {
        case .prominent: return MG.accentForeground
        case .borderless: return destructive ? MG.destructive : MG.accent
        case .bordered, .icon: return destructive ? MG.destructive : MG.label
        case .plain: return destructive ? MG.destructive : MG.label
        }
    }

    var body: some View {
        if kind == .plain {
            configuration.label
                .foregroundStyle(foreground)
                .opacity(enabled ? (configuration.isPressed ? 0.6 : 1) : 0.4)
                .contentShape(Rectangle())
        } else {
            configuration.label
                .font(size == .small || size == .mini ? .mgCallout : .mgBody)
                .lineLimit(1)
                .foregroundStyle(foreground)
                .padding(.horizontal, kind == .icon ? 4 : size == .small || size == .mini ? 8 : 10)
                .frame(minWidth: kind == .icon ? height : nil, minHeight: height)
                .background {
                    RoundedRectangle(cornerRadius: kind == .icon ? MGRadius.sm : MGRadius.md, style: .continuous)
                        .fill(fill)
                        .brightness(kind == .prominent ? (configuration.isPressed ? -0.08 : hover ? 0.05 : 0) : 0)
                }
                .contentShape(RoundedRectangle(cornerRadius: MGRadius.md))
                .opacity(enabled ? 1 : 0.4)
                .onHover { hover = $0 && enabled }
                .animation(.easeOut(duration: 0.12), value: hover)
        }
    }
}

// MARK: text fields

/// A field on the control background with a 0.5 pt border; focused, a 3.5 pt focus ring.
struct MGTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        MGFieldChrome { configuration.textFieldStyle(.plain) }
    }
}

/// The chrome of a field (also for search fields and other controls that need it).
struct MGFieldChrome<Content: View>: View {
    var icon: String? = nil
    @ViewBuilder var content: Content
    @FocusState private var focused: Bool
    @Environment(\.isEnabled) private var enabled

    var body: some View {
        HStack(spacing: 5) {
            if let icon { Image(systemName: icon).font(.system(size: 11)).foregroundStyle(MG.secondaryLabel) }
            content.focused($focused)
        }
        .font(.mgBody)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .frame(minHeight: 22)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .overlay {
            if focused {
                RoundedRectangle(cornerRadius: MGRadius.md + 1.75, style: .continuous)
                    .stroke(MG.focusRing, lineWidth: 3.5)
                    .padding(-1.75)
                    .allowsHitTesting(false)
            }
        }
        .opacity(enabled ? 1 : 0.5)
        .animation(.easeOut(duration: 0.12), value: focused)
    }
}

/// A search field: the field chrome with a magnifier and a clear button.
struct MGSearchField: View {
    let prompt: String
    @Binding var text: String

    var body: some View {
        MGFieldChrome(icon: "magnifyingglass") {
            HStack(spacing: 4) {
                TextField(prompt, text: $text).textFieldStyle(.plain)
                if !text.isEmpty {
                    Button { text = "" } label: {
                        Image(systemName: "xmark.circle.fill").font(.system(size: 11)).foregroundStyle(MG.tertiaryLabel)
                    }
                    .buttonStyle(.plain)
                    .mgHelp("Clear")
                }
            }
        }
    }
}

// MARK: toggles

/// A switch: the accent track when on, the fill when off, a white knob with a control shadow. The label goes before it.
struct MGSwitchStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        MGSwitch(configuration: configuration)
    }
}

private struct MGSwitch: View {
    let configuration: ToggleStyle.Configuration
    @Environment(\.isEnabled) private var enabled
    @Environment(\.controlSize) private var size

    var body: some View {
        let small = size == .small || size == .mini
        let w: CGFloat = small ? 26 : 32, h: CGFloat = small ? 15 : 18
        HStack(spacing: 8) {
            configuration.label
            Spacer(minLength: 0)
            Capsule()
                .fill(configuration.isOn ? MG.accent : MG.fill)
                .overlay(Capsule().strokeBorder(MG.separator, lineWidth: configuration.isOn ? 0 : 0.5))
                .overlay(alignment: configuration.isOn ? .trailing : .leading) {
                    Circle().fill(MG.controlKnob)
                        .mgShadow(MGShadow.control)
                        .padding(1.5)
                }
                .frame(width: w, height: h)
                .animation(.spring(response: 0.25, dampingFraction: 0.8), value: configuration.isOn)
                .contentShape(Capsule())
                .onTapGesture { if enabled { configuration.isOn.toggle() } }
                .accessibilityElement()
                .accessibilityAddTraits(.isButton)
                .accessibilityValue(configuration.isOn ? "on" : "off")
                .accessibilityAction { configuration.isOn.toggle() }
        }
        .opacity(enabled ? 1 : 0.4)
    }
}

/// A checkbox in the design system's colours (for compact places such as node cards).
struct MGCheckboxStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: MGRadius.xs, style: .continuous)
                .fill(configuration.isOn ? MG.accent : MG.controlBackground)
                .mgHairline(RoundedRectangle(cornerRadius: MGRadius.xs, style: .continuous),
                            color: configuration.isOn ? .clear : MG.border)
                .overlay {
                    if configuration.isOn {
                        Image(systemName: "checkmark").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.accentForeground)
                    }
                }
                .frame(width: 13, height: 13)
            configuration.label
        }
        .contentShape(Rectangle())
        .onTapGesture { configuration.isOn.toggle() }
    }
}

// MARK: labeled content

/// Label on the leading side in the label colour, value trailing in the secondary label colour.
struct MGLabeledContentStyle: LabeledContentStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            configuration.label.foregroundStyle(MG.label)
            Spacer(minLength: 8)
            configuration.content.foregroundStyle(MG.secondaryLabel).multilineTextAlignment(.trailing)
        }
    }
}

// MARK: segmented picker

/// A segmented control: a fill track with the selected segment raised.
struct MGSegmentedPicker<Value: Hashable>: View {
    @Binding var selection: Value
    let options: [(value: Value, title: String)]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options.indices, id: \.self) { i in
                let selected = options[i].value == selection
                Text(options[i].title)
                    .font(.mgCallout)
                    .lineLimit(1)
                    .foregroundStyle(selected ? MG.label : MG.secondaryLabel)
                    .padding(.horizontal, 10)
                    .frame(maxWidth: .infinity, minHeight: 18)
                    .background {
                        if selected {
                            RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                                .fill(MG.segmentSelected)
                                .mgShadow(MGShadow.control)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { selection = options[i].value }
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .animation(.easeOut(duration: 0.18), value: selection)
    }
}

// MARK: list rows

/// A list row: 24 tall (38 with a subtitle), the highlight inset 8 and the content 8 inside it. Hover uses `hover`;
/// selected uses `selection` (or the accent when prominent).
struct MGRow: ViewModifier {
    var selected = false
    var prominent = false
    var height: CGFloat = 24
    var indent: CGFloat = 0
    @State private var hover = false

    func body(content: Content) -> some View {
        content
            .padding(.leading, 8 + indent)
            .padding(.trailing, 8)
            .frame(minHeight: height)
            .frame(maxWidth: .infinity, alignment: .leading)
            .foregroundStyle(selected && prominent ? MG.accentForeground : MG.label)
            .fontWeight(selected && !prominent ? .medium : nil)
            .background {
                RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                    .fill(selected ? (prominent ? MG.accent : MG.selection) : hover ? MG.hover : .clear)
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 8)
            .onHover { hover = $0 }
    }
}

extension View {
    func mgRow(selected: Bool = false, prominent: Bool = false, height: CGFloat = 24, indent: CGFloat = 0) -> some View {
        modifier(MGRow(selected: selected, prominent: prominent, height: height, indent: indent))
    }
}

/// A small section title over a list (callout, secondary label).
struct MGListHeader: View {
    let title: String
    var body: some View {
        Text(title)
            .font(.mgCallout)
            .foregroundStyle(MG.secondaryLabel)
            .padding(.horizontal, 16)
            .padding(.top, 10)
            .padding(.bottom, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A coloured tile with a letter or glyph, as the design system's KindBadge.
struct MGKindBadge: View {
    let color: Color
    var letter: String? = nil
    var symbol: String? = nil
    var size: CGFloat = 16

    var body: some View {
        RoundedRectangle(cornerRadius: MGRadius.xs + 1, style: .continuous)
            .fill(color)
            .overlay {
                if let symbol {
                    Image(systemName: symbol).font(.system(size: size * 0.55, weight: .semibold))
                } else if let letter {
                    Text(letter).font(.mg(size: size * 0.62, weight: .bold))
                }
            }
            .foregroundStyle(MG.accentForeground)
            .frame(width: size, height: size)
    }
}

// MARK: lists

extension View {
    /// A List drawn as the design system's list: no system background, selection or separators (rows use `mgRow`).
    func mgList() -> some View {
        listStyle(.plain)
            .scrollContentBackground(.hidden)
            .environment(\.defaultMinListRowHeight, 24)
            .mgAppKitScrollIndicators()
    }

    /// A row inside an `mgList()`: no system chrome or insets.
    func mgListRow() -> some View {
        listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
            .listRowInsets(EdgeInsets(top: 1, leading: 0, bottom: 1, trailing: 0))
    }
}

/// A multi-line text editor in field chrome.
struct MGTextEditor: View {
    @Binding var text: String
    @FocusState private var focused: Bool

    var body: some View {
        TextEditor(text: $text)
            .font(.mgBody)
            .scrollContentBackground(.hidden)
            .mgAppKitScrollIndicators()
            .focused($focused)
            .padding(4)
            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
            .overlay {
                if focused {
                    RoundedRectangle(cornerRadius: MGRadius.md + 1.75, style: .continuous)
                        .stroke(MG.focusRing, lineWidth: 3.5).padding(-1.75).allowsHitTesting(false)
                }
            }
    }
}

/// An empty state: a glyph, a headline and a line of secondary text, centred.
struct MGEmptyState: View {
    let symbol: String
    let title: String
    var message: String? = nil

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: symbol).font(.system(size: 34, weight: .light)).foregroundStyle(MG.tertiaryLabel)
            Text(title).font(.mgTitle3)
            if let message {
                Text(message).font(.mgCallout).foregroundStyle(MG.secondaryLabel).multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: slider, stepper, progress

/// The design system's Slider: a 4 pt track, the accent up to an 18 pt knob. Arrow keys step it.
struct MGSlider<V: BinaryFloatingPoint>: View where V.Stride: BinaryFloatingPoint {
    @Binding var value: V
    let range: ClosedRange<V>
    var step: V? = nil
    @Environment(\.isEnabled) private var enabled
    @State private var dragging = false

    init(value: Binding<V>, in range: ClosedRange<V>, step: V? = nil) {
        _value = value
        self.range = range
        self.step = step
    }

    var fraction: CGFloat {
        let span = range.upperBound - range.lowerBound
        return span > 0 ? CGFloat((value - range.lowerBound) / span) : 0
    }

    func set(_ f: CGFloat) {
        let f = min(max(f, 0), 1)
        var v = range.lowerBound + V(f) * (range.upperBound - range.lowerBound)
        if let step, step > 0 { v = (v / step).rounded() * step }
        value = min(max(v, range.lowerBound), range.upperBound)
    }

    func nudge(_ dir: V) {
        let s = step ?? (range.upperBound - range.lowerBound) / 20
        value = min(max(value + dir * s, range.lowerBound), range.upperBound)
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x = fraction * max(w - 18, 0)
            ZStack(alignment: .leading) {
                Capsule().fill(MG.fill).frame(height: 4)
                Capsule().fill(MG.accent).frame(width: x + 9, height: 4)
                Circle().fill(MG.controlKnob)
                    .overlay(Circle().strokeBorder(MG.shadow.opacity(0.8), lineWidth: 0.5))
                    .mgShadow(MGShadow.control)
                    .frame(width: 18, height: 18)
                    .scaleEffect(dragging ? 1.06 : 1)
                    .offset(x: x)
            }
            .frame(height: 20)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { g in
                    guard enabled else { return }
                    dragging = true
                    set((g.location.x - 9) / max(w - 18, 1))
                }
                .onEnded { _ in dragging = false })
        }
        .frame(minWidth: 60, idealWidth: 180)
        .frame(height: 20)
        .opacity(enabled ? 1 : 0.4)
        .animation(.easeOut(duration: 0.1), value: dragging)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.leftArrow) { nudge(-1); return .handled }
        .onKeyPress(.rightArrow) { nudge(1); return .handled }
        .accessibilityElement()
        .accessibilityValue(String(format: "%.0f%%", Double(fraction) * 100))
        .accessibilityAdjustableAction { dir in nudge(dir == .increment ? 1 : -1) }
    }
}

/// The design system's Stepper: − and + keys on the fill; a key at the end of the range dims to 0.3.
struct MGStepper: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    var step = 1
    @Environment(\.isEnabled) private var enabled

    init(value: Binding<Int>, in range: ClosedRange<Int>, step: Int = 1) {
        _value = value
        self.range = range
        self.step = step
    }

    var body: some View {
        HStack(spacing: 1) {
            key("minus", enabled: value > range.lowerBound, corners: .leading) {
                value = max(range.lowerBound, value - step)
            }
            key("plus", enabled: value < range.upperBound, corners: .trailing) {
                value = min(range.upperBound, value + step)
            }
        }
        .padding(1)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .opacity(enabled ? 1 : 0.4)
        .accessibilityElement()
        .accessibilityValue("\(value)")
        .accessibilityAdjustableAction { dir in
            value = dir == .increment ? min(range.upperBound, value + step) : max(range.lowerBound, value - step)
        }
    }

    func key(_ symbol: String, enabled on: Bool, corners: HorizontalEdge, action: @escaping () -> Void) -> some View {
        let shape = UnevenRoundedRectangle(topLeadingRadius: corners == .leading ? 5 : 0,
                                           bottomLeadingRadius: corners == .leading ? 5 : 0,
                                           bottomTrailingRadius: corners == .trailing ? 5 : 0,
                                           topTrailingRadius: corners == .trailing ? 5 : 0, style: .continuous)
        return Button(action: action) {
            Image(systemName: symbol).font(.system(size: 9, weight: .semibold))
                .frame(width: 24, height: 22)
                .background(shape.fill(MG.controlButton))
                .contentShape(shape)
        }
        .buttonStyle(.mg(.plain))
        .opacity(on ? 1 : 0.3)
        .disabled(!on)
    }
}

/// The design system's ProgressView: a 6 pt capsule filled with the accent; without a value, a segment slides along it.
struct MGProgress: View {
    var value: Double? = nil
    var total: Double = 1
    var tint: Color = MG.accent
    var height: CGFloat = 6
    @State private var phase = false

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(MG.fill)
                if let value {
                    Capsule().fill(tint)
                        .frame(width: max(height, w * CGFloat(min(max(value / max(total, 1e-9), 0), 1))))
                        .animation(.easeOut(duration: 0.2), value: value)
                } else {
                    Capsule().fill(tint).frame(width: w * 0.3)
                        .offset(x: phase ? w * 0.7 : 0)
                        .onAppear {
                            withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { phase = true }
                        }
                }
            }
            .clipShape(Capsule())
        }
        .frame(minWidth: 40, idealWidth: 180)
        .frame(height: height)
        .accessibilityElement()
        .accessibilityValue(value.map { String(format: "%.0f%%", $0 / max(total, 1e-9) * 100) } ?? "in progress")
    }
}

/// A spinner: a three-quarter accent arc turning (mini 10, small 14, regular 20).
struct MGSpinner: View {
    var tint: Color = MG.accent
    @Environment(\.controlSize) private var size
    @State private var spin = false

    var diameter: CGFloat {
        switch size {
        case .mini: return 10
        case .small: return 14
        case .large, .extraLarge: return 28
        default: return 20
        }
    }

    var body: some View {
        Circle()
            .trim(from: 0, to: 0.75)
            .stroke(tint, style: StrokeStyle(lineWidth: max(1.5, diameter / 8), lineCap: .round))
            .frame(width: diameter, height: diameter)
            .rotationEffect(.degrees(spin ? 360 : 0))
            .onAppear { withAnimation(.linear(duration: 0.9).repeatForever(autoreverses: false)) { spin = true } }
            .accessibilityLabel("In progress")
    }
}
