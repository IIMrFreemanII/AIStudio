import SwiftUI

/// One tab of a strip of tabs (terminals, open files): a leading view, a title, and × to close (shown on hover and on
/// the selected tab; a dirty tab shows a dot in its place until hovered).
struct MGTab<Leading: View>: View {
    let title: String
    let selected: Bool
    var italic = false
    var strikethrough = false
    var dirty = false
    var help: String?
    var closeHelp = "Close this tab"
    var closeLabel = "Close Tab"
    var anchor: String?
    var doubleClick: (() -> Void)?
    let leading: Leading
    let select: () -> Void
    let close: () -> Void
    @State private var hover = false

    init(title: String, selected: Bool, italic: Bool = false, strikethrough: Bool = false, dirty: Bool = false,
         help: String? = nil, closeHelp: String = "Close this tab", closeLabel: String = "Close Tab", anchor: String? = nil,
         doubleClick: (() -> Void)? = nil, @ViewBuilder leading: () -> Leading,
         select: @escaping () -> Void, close: @escaping () -> Void) {
        self.title = title
        self.selected = selected
        self.italic = italic
        self.strikethrough = strikethrough
        self.dirty = dirty
        self.help = help
        self.closeHelp = closeHelp
        self.closeLabel = closeLabel
        self.anchor = anchor
        self.doubleClick = doubleClick
        self.leading = leading()
        self.select = select
        self.close = close
    }

    var body: some View {
        HStack(spacing: 5) {
            leading
            Text(title).font(.mgFootnote.weight(selected ? .semibold : .regular))
                .strikethrough(strikethrough)
                .foregroundStyle(selected ? MG.label : MG.secondaryLabel)
                .lineLimit(1).truncationMode(.middle)
                // Only upright JetBrains Mono faces ship, so `.italic()` would draw nothing different: slant it.
                .transformEffect(italic ? CGAffineTransform(a: 1, b: 0, c: -0.2, d: 1, tx: 1.5, ty: 0) : .identity)
                .frame(maxWidth: 140)
            ZStack {
                if dirty && !hover {
                    Circle().fill(MG.secondaryLabel).frame(width: 7, height: 7)
                } else {
                    Button(action: close) { Image(systemName: "xmark").font(.system(size: 8, weight: .bold)) }
                        .buttonStyle(.plain).foregroundStyle(MG.tertiaryLabel)
                        .opacity(hover || selected ? 1 : 0.001)
                        .accessibilityLabel(closeLabel)
                        .mgHelp(closeHelp)
                }
            }
            .frame(width: 12)
        }
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
            .fill(selected ? MG.selection : hover ? MG.hover : .clear))
        .contentShape(Rectangle())
        .mgClick(single: select, double: doubleClick)
        .onHover { hover = $0 }
        .mgHelp(help ?? title)
        .modifier(OptionalAnchor(name: anchor))
    }
}

private struct OptionalAnchor: ViewModifier {
    let name: String?
    @ViewBuilder func body(content: Content) -> some View {
        if let name { content.mgTestAnchor(name) } else { content }
    }
}
