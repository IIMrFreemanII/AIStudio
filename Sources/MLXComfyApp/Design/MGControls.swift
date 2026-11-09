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
                    .help("Clear")
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
