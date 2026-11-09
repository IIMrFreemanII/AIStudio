import SwiftUI

// Form, Section and FormRow: settings on the grouped background, sections as cards (radius 10) with 0.5 pt hairlines
// between rows, headers in callout and footers in subheadline.

/// A scrolling column of sections on the grouped background (at most `maxWidth` wide, centred).
struct MGForm<Content: View>: View {
    var maxWidth: CGFloat = 600
    var padding: CGFloat = MGSpace.lg
    @ViewBuilder var content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MGSpace.xl) { content }
                .padding(padding)
                .frame(maxWidth: maxWidth)
                .frame(maxWidth: .infinity)
        }
        .scrollContentBackground(.hidden)
        .background(MG.groupedBackground)
    }
}

/// A card of rows: each direct child is a row, separated by hairlines.
struct MGSection<Content: View, Header: View, Footer: View>: View {
    @ViewBuilder var content: Content
    @ViewBuilder var header: Header
    @ViewBuilder var footer: Footer

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
                .font(.mgCallout)
                .foregroundStyle(MG.secondaryLabel)
                .padding(.horizontal, 10)
            VStack(spacing: 0) {
                Group(subviews: content) { rows in
                    ForEach(rows) { row in
                        if row.id != rows.first?.id { MGDivider().padding(.leading, 12) }
                        row
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .frame(minHeight: 32)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
            footer
                .font(.mgSubheadline)
                .foregroundStyle(MG.secondaryLabel)
                .padding(.horizontal, 10)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

extension MGSection where Header == EmptyView, Footer == EmptyView {
    init(@ViewBuilder content: () -> Content) {
        self.content = content()
        self.header = EmptyView()
        self.footer = EmptyView()
    }
}

extension MGSection where Header == Text, Footer == EmptyView {
    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.content = content()
        self.header = Text(title)
        self.footer = EmptyView()
    }
}

extension MGSection where Footer == EmptyView {
    init(@ViewBuilder content: () -> Content, @ViewBuilder header: () -> Header) {
        self.content = content()
        self.header = header()
        self.footer = EmptyView()
    }
}

extension MGSection where Header == EmptyView {
    init(@ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) {
        self.content = content()
        self.header = EmptyView()
        self.footer = footer()
    }
}

extension MGSection where Header == Text {
    init(_ title: String, @ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) {
        self.content = content()
        self.header = Text(title)
        self.footer = footer()
    }
}

/// A row with a label on the leading side and its control trailing.
struct MGFormRow<Control: View>: View {
    let label: String
    @ViewBuilder var control: Control

    init(_ label: String, @ViewBuilder control: () -> Control) {
        self.label = label
        self.control = control()
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label).foregroundStyle(MG.label)
            Spacer(minLength: 8)
            control
        }
    }
}
