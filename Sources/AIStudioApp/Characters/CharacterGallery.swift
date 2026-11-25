import CharacterCore
import SwiftUI

/// File › New Character Chat…: the user's characters, and the built-in templates to start from.
struct CharacterGallery: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Characters").font(.mgTitle2)
                Spacer()
                Button {
                    open(AppModel.Companion(name: "", age: 25, gender: "woman"), isNew: true)
                } label: {
                    Label("New Character", systemImage: "plus")
                }
                .buttonStyle(.mg(.bordered))
                .mgHelp("Write a character from scratch")
            }
            .padding(18)
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Your characters").font(.mgHeadline)
                        if model.characters.isEmpty {
                            MGEmptyState(symbol: "heart", title: "No characters yet",
                                         message: "Start from a template below, or write your own.")
                                .frame(height: 170)
                        } else {
                            grid(model.characters, mine: true)
                        }
                    }
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Start from a template").font(.mgHeadline)
                        grid(CharacterTemplates.all, mine: false)
                    }
                }
                .padding(18)
            }
            .mgScrollIndicators()
            .background(MG.groupedBackground)
            MGDivider()
            HStack {
                Spacer()
                Button("Close") { dismiss() }
                    .buttonStyle(.mg(.bordered))
                    .keyboardShortcut(.cancelAction)
                    .mgHelp("Close the gallery")
            }
            .padding(14)
        }
        .frame(width: 720, height: 600)
    }

    private func grid(_ items: [AppModel.Companion], mine: Bool) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 340), spacing: 12)], alignment: .leading, spacing: 12) {
            ForEach(items) { c in CharacterCard(character: c, mine: mine, chat: { chat(c) }, edit: { edit(c) }, use: { use(c) }) }
        }
    }

    private func open(_ c: AppModel.Companion, isNew: Bool) {
        model.editingCharacter = CharacterEdit(character: c, isNew: isNew)
    }
    private func edit(_ c: AppModel.Companion) { open(c, isNew: false) }
    private func use(_ t: AppModel.Companion) { open(CharacterTemplates.make(t), isNew: true) }
    private func chat(_ c: AppModel.Companion) {
        dismiss()
        model.chat(withCharacter: c.id)
    }
}

struct CharacterCard: View {
    let character: AppModel.Companion
    let mine: Bool
    var chat: () -> Void = {}
    var edit: () -> Void = {}
    var use: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                CharacterAvatar(name: character.name, size: 40)
                VStack(alignment: .leading, spacing: 2) {
                    Text(character.name.isEmpty ? "Unnamed" : character.name).font(.mgCallout.weight(.semibold)).lineLimit(1)
                    Text("\(character.age) · \(character.gender)").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        .lineLimit(1)
                    Text(character.personality).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            HStack(spacing: 8) {
                if mine {
                    Button("Chat", action: chat).buttonStyle(.mg(.prominent)).controlSize(.small)
                        .mgHelp("Open your chat with \(character.name)")
                    Button("Edit", action: edit).buttonStyle(.mg(.bordered)).controlSize(.small)
                        .mgHelp("Change \(character.name)'s persona and settings")
                } else {
                    Button("Use", action: use).buttonStyle(.mg(.bordered)).controlSize(.small)
                        .mgHelp("Make your own character from this template")
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator, lineWidth: 0.5))
    }
}
