import AppKit
import CharacterCore
import SwiftUI

/// One option category as a row of chips: tap to pick, tap again to drop. A pick-one category swaps its choice; a full
/// one drops its oldest pick. Look options show their bundled example photo as a tile, colors a swatch, and jobs and
/// habits an icon; every option's hint shows under the row while it's hovered (and in its tooltip).
struct TraitChips: View {
    let category: CharacterOptions.Category
    @Binding var c: AppModel.Companion
    /// The option under the pointer: its hint shows under the row.
    @State private var hovered: String?

    static let tileSize = CGSize(width: 76, height: 112)

    private var picked: [String] { c.traits[category.id] ?? [] }
    private var options: [CharacterOptions.Option] { CharacterOptions.options(category, gender: c.gender) }

    /// "Label: hint" for the hovered option, else the latest pick.
    private var hintLine: String? {
        guard let id = hovered ?? picked.last, let o = category.options.first(where: { $0.id == id }) else { return nil }
        let hint = CharacterOptions.hint(category: category.id, option: o.id)
        return hint.isEmpty ? o.label : "\(o.label): \(hint)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(category.title).foregroundStyle(MG.label)
                if category.max > 1 {
                    Text("up to \(category.max)").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                }
            }
            MGFlowLayout(spacing: CharacterOptions.photoCategories.contains(category.id) ? 8 : 6) {
                ForEach(options) { o in item(o, on: picked.contains(o.id)) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            // Always there (a space when empty), so hovering doesn't move the rows below.
            Text(hintLine ?? " ").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder private func item(_ o: CharacterOptions.Option, on: Bool) -> some View {
        Group {
            if CharacterOptions.photoCategories.contains(category.id),
               let image = OptionImages.image(category: category.id, option: o.id, gender: c.gender) {
                tile(o, image: image, on: on)
            } else {
                chip(o, on: on)
            }
        }
        .onHover { inside in
            if inside { hovered = o.id } else if hovered == o.id { hovered = nil }
        }
        .accessibilityAddTraits(on ? .isSelected : [])
        .accessibilityHint(CharacterOptions.hint(category: category.id, option: o.id))
        .mgHelp(helpText(o, on: on))
    }

    private func helpText(_ o: CharacterOptions.Option, on: Bool) -> String {
        let hint = CharacterOptions.hint(category: category.id, option: o.id)
        return (on ? "Drop “\(o.label)”" : "Pick “\(o.label)”") + (hint.isEmpty ? "" : ": \(hint)")
    }

    private func toggle(_ o: CharacterOptions.Option) {
        c.traits = CharacterOptions.toggle(c.traits, category: category.id, option: o.id)
    }

    private func tile(_ o: CharacterOptions.Option, image: NSImage, on: Bool) -> some View {
        let shape = RoundedRectangle(cornerRadius: 8, style: .continuous)
        return Button { toggle(o) } label: {
            VStack(spacing: 4) {
                Image(nsImage: image).resizable().interpolation(.high).aspectRatio(contentMode: .fill)
                    .frame(width: Self.tileSize.width, height: Self.tileSize.height)
                    .clipShape(shape)
                    .overlay(shape.strokeBorder(on ? MG.accent : MG.border, lineWidth: on ? 2.5 : 1))
                    .overlay(alignment: .topTrailing) {
                        if on {
                            Image(systemName: "checkmark.circle.fill").font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(MG.accentForeground, MG.accent).padding(4)
                        }
                    }
                Text(o.label).font(.mgFootnote).lineLimit(1).truncationMode(.tail)
                    .foregroundStyle(on ? MG.accent : MG.label)
                    .frame(width: Self.tileSize.width)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func chip(_ o: CharacterOptions.Option, on: Bool) -> some View {
        Button { toggle(o) } label: {
            HStack(spacing: 5) {
                switch CharacterOptions.visual(category: category.id, option: o.id) {
                case .swatch(let hex):
                    Circle().fill(Self.color(hex)).frame(width: 12, height: 12)
                        .overlay(Circle().strokeBorder(Color.black.opacity(0.18), lineWidth: 0.5))
                case .symbol(let name):
                    Image(systemName: name).font(.system(size: 11))
                case .photo, .none:
                    EmptyView()
                }
                Text(o.label).font(.mgFootnote).lineLimit(1)
            }
            .foregroundStyle(on ? MG.accentForeground : MG.label)
            .padding(.horizontal, 10).frame(height: 26)
            .background(Capsule().fill(on ? MG.accent : MG.fill))
            .mgHairline(Capsule(), color: on ? MG.accent : MG.border)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    /// "#RRGGBB" as a color (gray when it doesn't parse).
    static func color(_ hex: String) -> Color {
        guard hex.count == 7, hex.hasPrefix("#"), let v = UInt32(hex.dropFirst(), radix: 16) else { return .gray }
        return Color(red: Double((v >> 16) & 0xFF) / 255, green: Double((v >> 8) & 0xFF) / 255,
                     blue: Double(v & 0xFF) / 255)
    }
}

/// Every category of one option group, then (when given) a free-text field for what the chips don't cover.
struct TraitGroupView: View {
    let group: CharacterOptions.Group
    @Binding var c: AppModel.Companion
    var detailsTitle: String? = nil
    var details: Binding<String>? = nil

    var body: some View {
        // A category with nothing for this gender (Breasts for a man) isn't shown.
        ForEach(CharacterOptions.categories(group).filter { !CharacterOptions.options($0, gender: c.gender).isEmpty }) {
            TraitChips(category: $0, c: $c)
        }
        if let details, let detailsTitle {
            VStack(alignment: .leading, spacing: 6) {
                Text(detailsTitle).foregroundStyle(MG.label)
                MGTextEditor(text: details).frame(height: 52)
            }
            .padding(.vertical, 4)
        }
    }
}
