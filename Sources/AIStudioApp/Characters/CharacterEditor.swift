import CharacterCore
import SwiftUI

/// What the editor sheet edits: a copy of a character, new (unsaved) or existing.
struct CharacterEdit: Identifiable {
    let id = UUID()
    var character: CharacterCore.Character
    var isNew: Bool
}

/// A character by id (the memory sheet's item).
struct CharacterRef: Identifiable {
    let id: UUID
}

/// The persona, voice and chat settings of one character.
struct CharacterEditor: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let edit: CharacterEdit
    /// The sheet's height (taller only in snapshots, to see the whole form).
    var height: CGFloat = 680
    @State private var c: AppModel.Companion
    @State private var saveProblems: [String] = []

    init(edit: CharacterEdit) {
        self.edit = edit
        _c = State(initialValue: edit.character)
    }

    private var problems: [String] {
        var p = AgeGuard.problems(c)
        if c.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { p.insert("Add a name.", at: 0) }
        return p
    }

    var body: some View {
        let shown = saveProblems.isEmpty ? problems : saveProblems
        VStack(spacing: 0) {
            HStack {
                Text(edit.isNew ? "New Character" : "Edit \(edit.character.name)").font(.mgTitle3)
                Spacer()
            }
            .padding(.horizontal, 18).padding(.vertical, 14)
            MGDivider()
            MGForm {
                MGSection("Persona") {
                    MGFormRow("Name") { TextField("Name", text: $c.name).frame(maxWidth: 300) }
                    MGFormRow("Age") {
                        HStack(spacing: 10) {
                            Text("\(c.age)").monospacedDigit().foregroundStyle(MG.secondaryLabel)
                            MGStepper(value: $c.age, in: 18...99).mgHelp("Characters are adults: 18 or older")
                        }
                    }
                    MGFormRow("Gender") { TextField("woman, man, non-binary…", text: $c.gender).frame(maxWidth: 300) }
                    TraitGroupView(group: .basics, c: $c)
                    field("Backstory", $c.backstory)
                    field("Greeting", $c.greeting)
                } footer: {
                    Text("Greeting: their first message in a new chat. Leave empty to start yourself.")
                }
                MGSection("Look") {
                    TraitGroupView(group: .look, c: $c, detailsTitle: "More look details", details: $c.look)
                }
                MGSection("Personality") {
                    TraitGroupView(group: .personality, c: $c, detailsTitle: "More about them", details: $c.personality)
                    field("Speaking style", $c.speakingStyle)
                }
                MGSection("Habits and interests") {
                    TraitGroupView(group: .habits, c: $c, detailsTitle: "Things they love", details: $c.likes)
                }
                CharacterLookSection(c: $c, title: "Profile photo")
                CharacterPhotosSection(c: $c)
                CharacterVoiceSection(c: $c)
                MGSection("Chat") {
                    MGFormRow("Temperature") {
                        HStack(spacing: 10) {
                            MGSlider(value: $c.temperature, in: 0.3...1.5).frame(width: 200)
                                .mgHelp("Higher is more varied and surprising, lower is steadier")
                            Text(String(format: "%.2f", c.temperature)).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                                .frame(width: 40, alignment: .trailing)
                        }
                    }
                    MGFormRow("Top-p") {
                        HStack(spacing: 10) {
                            MGSlider(value: $c.topP, in: 0.5...1.0).frame(width: 200)
                                .mgHelp("How much of the likely words the model picks from")
                            Text(String(format: "%.2f", c.topP)).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                                .frame(width: 40, alignment: .trailing)
                        }
                    }
                    MGFormRow("Media on their own per day") {
                        HStack(spacing: 10) {
                            Text("\(c.initiative)").monospacedDigit().foregroundStyle(MG.secondaryLabel)
                            MGStepper(value: $c.initiative, in: 0...10)
                                .mgHelp("How many photos, videos or voice messages they send unasked each day")
                        }
                    }
                    MGFormRow("Texts you first") {
                        MGMenuButton(kind: .bordered) {
                            for h in CheckIn.choices {
                                MGMenuItem.button(CheckIn.label(h), checked: c.checkInHours == h) { c.checkInHours = h }
                            }
                        } label: {
                            HStack(spacing: 6) {
                                Text(CheckIn.label(c.checkInHours))
                                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold))
                                    .foregroundStyle(MG.secondaryLabel)
                            }
                        }
                        .mgHelp("When you open your chat after this long away, they send a message first (it uses the chat's model)")
                    }
                    CharacterMediaKindsRow(c: $c)
                } footer: {
                    Text("Videos and video messages always wait for your OK before they're made.")
                }
            }
            MGDivider()
            VStack(alignment: .leading, spacing: 10) {
                if !shown.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(shown, id: \.self) { Text($0).font(.mgFootnote).foregroundStyle(MG.destructive) }
                    }
                }
                HStack(spacing: 8) {
                    if !edit.isNew {
                        Button("Delete", role: .destructive) { confirmDelete() }
                            .buttonStyle(.mg(.bordered, destructive: true))
                            .mgHelp("Delete this character; their chats stay")
                    }
                    Spacer()
                    Button("Cancel", role: .cancel) { cancel() }
                        .buttonStyle(.mg(.bordered)).keyboardShortcut(.cancelAction)
                        .mgHelp("Close without saving")
                    Button("Save") { save(chat: false) }
                        .buttonStyle(.mg(.bordered)).disabled(!problems.isEmpty)
                        .mgHelp("Save the character")
                    Button("Save & Chat") { save(chat: true) }
                        .buttonStyle(.mg(.prominent)).disabled(!problems.isEmpty)
                        .mgHelp("Save, then open a chat with them")
                }
            }
            .padding(14)
        }
        .frame(width: 560, height: height)
        .onChange(of: c) { saveProblems = [] }
    }

    private func field(_ title: String, _ text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).foregroundStyle(MG.label)
            MGTextEditor(text: text).frame(height: 62)
        }
        .padding(.vertical, 4)
    }

    /// A new character that was never saved takes the portraits and voices made for it along.
    private func cancel() {
        if edit.isNew { model.discardDraft(c.id) }
        dismiss()
    }

    private func save(chat: Bool) {
        var saved = c
        saved.name = saved.name.trimmingCharacters(in: .whitespacesAndNewlines)
        let result = model.saveCharacter(saved)
        guard result.isEmpty else { saveProblems = result; return }
        dismiss()
        if chat {
            model.showCharacterGallery = false
            model.chat(withCharacter: saved.id)
        }
    }

    private func confirmDelete() {
        let id = c.id
        model.presenter.alert("Delete “\(edit.character.name)”?", message: "The character and what they remember move to the Trash. Their chats stay.", actions: [
            .destructive("Delete") {
                model.deleteCharacter(id)
                dismiss()
            },
            .cancel(),
        ])
    }
}
