import CharacterCore
import SwiftUI

/// One option category as a row of chips: tap to pick, tap again to drop. A pick-one category swaps its choice; a full
/// one drops its oldest pick.
struct TraitChips: View {
    let category: CharacterOptions.Category
    @Binding var c: AppModel.Companion

    private var picked: [String] { c.traits[category.id] ?? [] }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(category.title).foregroundStyle(MG.label)
                if category.max > 1 {
                    Text("up to \(category.max)").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                }
            }
            MGFlowLayout(spacing: 6) {
                ForEach(CharacterOptions.options(category, gender: c.gender)) { o in
                    chip(o, on: picked.contains(o.id))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
    }

    private func chip(_ o: CharacterOptions.Option, on: Bool) -> some View {
        Button {
            c.traits = CharacterOptions.toggle(c.traits, category: category.id, option: o.id)
        } label: {
            Text(o.label).font(.mgFootnote).lineLimit(1)
                .foregroundStyle(on ? MG.accentForeground : MG.label)
                .padding(.horizontal, 10).frame(height: 26)
                .background(Capsule().fill(on ? MG.accent : MG.fill))
                .mgHairline(Capsule(), color: on ? MG.accent : MG.border)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(on ? .isSelected : [])
        .mgHelp(on ? "Drop “\(o.label)”" : "Pick “\(o.label)”")
    }
}

/// Every category of one option group, then (when given) a free-text field for what the chips don't cover.
struct TraitGroupView: View {
    let group: CharacterOptions.Group
    @Binding var c: AppModel.Companion
    var detailsTitle: String? = nil
    var details: Binding<String>? = nil

    var body: some View {
        ForEach(CharacterOptions.categories(group)) { TraitChips(category: $0, c: $c) }
        if let details, let detailsTitle {
            VStack(alignment: .leading, spacing: 6) {
                Text(detailsTitle).foregroundStyle(MG.label)
                MGTextEditor(text: details).frame(height: 52)
            }
            .padding(.vertical, 4)
        }
    }
}
