import CharacterCore
import SwiftUI

/// File › New Character Chat…: the user's characters, and the built-in templates to start from.
struct CharacterGallery: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    /// Choosing who's in a new group chat (nil: not choosing), in the order they were ticked.
    @State private var group: [UUID]?

    init(group: [UUID]? = nil) { _group = State(initialValue: group) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Characters").font(.mgTitle2)
                Spacer()
                if group == nil {
                    Button {
                        group = []
                    } label: {
                        Label("New Group Chat", systemImage: "person.3")
                    }
                    .buttonStyle(.mg(.bordered))
                    .disabled(model.characters.count < 2)
                    .mgHelp(model.characters.count < 2 ? "Make at least two characters for a group chat"
                            : "Chat with two to four of your characters at once")
                }
                Button {
                    create(AppModel.Companion(name: "", age: 25, gender: "woman"))
                } label: {
                    Label("New Character", systemImage: "plus")
                }
                .buttonStyle(.mg(.bordered))
                .mgHelp("Make a character step by step: looks, personality, habits, photo and voice")
            }
            .padding(18)
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(group == nil ? "Your characters" : "Who's in the group chat? Pick two to four.").font(.mgHeadline)
                        if model.characters.isEmpty {
                            MGEmptyState(symbol: "heart", title: "No characters yet",
                                         message: "Start from a template below, or write your own.")
                                .frame(height: 170)
                        } else {
                            grid(model.characters, mine: true)
                        }
                    }
                    if group == nil {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Start from a template").font(.mgHeadline)
                            grid(CharacterTemplates.all, mine: false)
                        }
                    }
                }
                .padding(18)
            }
            .mgScrollIndicators()
            .background(MG.groupedBackground)
            MGDivider()
            HStack {
                Spacer()
                if let picked = group {
                    Button("Cancel") { group = nil }
                        .buttonStyle(.mg(.bordered))
                        .mgHelp("Back to your characters")
                    Button(picked.count < 2 ? "Start Group Chat" : "Start Group Chat (\(picked.count))") { startGroup(picked) }
                        .buttonStyle(.mg(.prominent))
                        .disabled(picked.count < 2)
                        .mgHelp("Start a chat with the characters you ticked; they answer in turn")
                }
                Button("Close") { dismiss() }
                    .buttonStyle(.mg(.bordered))
                    .keyboardShortcut(.cancelAction)
                    .mgHelp("Close the gallery")
            }
            .padding(14)
        }
        .frame(width: 720, height: 600)
        .onAppear { model.sweepStaleDrafts() }
    }

    private func grid(_ items: [AppModel.Companion], mine: Bool) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 340), spacing: 12)], alignment: .leading, spacing: 12) {
            ForEach(items) { c in
                if mine, let picked = group {
                    CharacterPickCard(character: c, photo: model.portraitURL(c), picked: picked.contains(c.id),
                                      full: picked.count >= 4) {
                        if picked.contains(c.id) { group?.removeAll { $0 == c.id } } else { group?.append(c.id) }
                    }
                } else {
                    CharacterCard(character: c, mine: mine, photo: mine ? model.portraitURL(c) : nil,
                                  chat: { chat(c) }, edit: { edit(c) }, use: { use(c) })
                }
            }
        }
    }

    private func open(_ c: AppModel.Companion, isNew: Bool) {
        model.editingCharacter = CharacterEdit(character: c, isNew: isNew)
    }
    private func create(_ c: AppModel.Companion) {
        model.creatingCharacter = CharacterEdit(character: c, isNew: true)
    }
    private func edit(_ c: AppModel.Companion) { open(c, isNew: false) }
    private func use(_ t: AppModel.Companion) { create(CharacterTemplates.make(t)) }
    private func startGroup(_ ids: [UUID]) {
        dismiss()
        model.newCharacterChat(ids)
    }
    private func chat(_ c: AppModel.Companion) {
        dismiss()
        model.chat(withCharacter: c.id)
    }
}

struct CharacterCard: View {
    let character: AppModel.Companion
    let mine: Bool
    var photo: URL? = nil
    var chat: () -> Void = {}
    var edit: () -> Void = {}
    var use: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                CharacterAvatar(name: character.name, size: 56, photo: photo)
                VStack(alignment: .leading, spacing: 2) {
                    Text(character.name.isEmpty ? "Unnamed" : character.name).font(.mgCallout.weight(.semibold)).lineLimit(1)
                    Text("\(character.age) · \(character.gender)").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        .lineLimit(1)
                    Text(character.fullPersonality).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
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
                        .mgHelp("Make your own character from this template, step by step")
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator, lineWidth: 0.5))
    }
}

/// A character card while choosing a group chat's cast: tick it in or out.
struct CharacterPickCard: View {
    let character: AppModel.Companion
    var photo: URL? = nil
    let picked: Bool
    /// Four are ticked already: only ticked ones can change.
    let full: Bool
    let toggle: () -> Void

    var body: some View {
        Button(action: toggle) {
            HStack(spacing: 10) {
                Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 18)).foregroundStyle(picked ? MG.accent : MG.tertiaryLabel)
                CharacterAvatar(name: character.name, size: 32, photo: photo)
                VStack(alignment: .leading, spacing: 2) {
                    Text(character.name.isEmpty ? "Unnamed" : character.name).font(.mgCallout.weight(.semibold)).lineLimit(1)
                    Text(character.fullPersonality).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
            .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous)
                .strokeBorder(picked ? MG.accent : MG.separator, lineWidth: picked ? 1.5 : 0.5))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(full && !picked)
        .mgHelp(picked ? "Leave \(character.name) out" : full ? "A group chat has at most four characters" : "Add \(character.name)")
        .accessibilityAddTraits(picked ? .isSelected : [])
    }
}
