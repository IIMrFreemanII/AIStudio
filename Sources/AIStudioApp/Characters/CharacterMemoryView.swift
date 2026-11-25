import CharacterCore
import SwiftUI

/// What a character remembers about the user: relationship, facts and the story so far. Everything can be corrected.
struct CharacterMemoryView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let characterID: UUID
    @State private var memory = CharacterMemory()
    @State private var character: AppModel.Companion?
    @State private var loaded = false
    @State private var newFact = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                if let character { CharacterAvatar(name: character.name, size: 28) }
                Text("\(character?.name ?? "Character")'s Memory").font(.mgTitle3)
                Spacer()
            }
            .padding(.horizontal, 18).padding(.vertical, 14)
            MGDivider()
            MGForm {
                MGSection("Relationship") {
                    MGSegmentedPicker(selection: $memory.relationship.stage,
                                      options: CharacterMemory.Stage.allCases.map { ($0, $0.label) })
                        .mgHelp("How close you two are")
                    MGFormRow("Affection") {
                        HStack(spacing: 10) {
                            MGSlider(value: affection, in: 0...100, step: 1).frame(width: 220)
                                .mgHelp("How much they like you, 0 to 100")
                            Text("\(memory.relationship.affection)").monospacedDigit().foregroundStyle(MG.secondaryLabel)
                                .frame(width: 30, alignment: .trailing)
                        }
                    }
                    MGFormRow("Mood") { TextField("Mood", text: $memory.relationship.mood).frame(maxWidth: 260) }
                }
                MGSection("What they know about you") {
                    if memory.facts.isEmpty {
                        Text("Nothing yet. They pick things up as you chat.").foregroundStyle(MG.secondaryLabel)
                    }
                    ForEach(memory.facts) { f in
                        HStack(spacing: 8) {
                            Text(f.text).frame(maxWidth: .infinity, alignment: .leading)
                            Button { memory.facts.removeAll { $0.id == f.id } } label: { Image(systemName: "trash") }
                                .accessibilityLabel("Forget This")
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Make them forget this")
                        }
                    }
                    HStack(spacing: 8) {
                        TextField("Add something they should know", text: $newFact).onSubmit(addFact)
                        Button("Add", action: addFact).buttonStyle(.mg(.bordered)).controlSize(.small)
                            .disabled(newFact.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            .mgHelp("Add this to what they know about you")
                    }
                }
                MGSection("Album") {
                    CharacterAlbumGrid(characterID: characterID, items: memory.album)
                } footer: {
                    Text("Click a photo to open it.")
                }
                MGSection("Your story so far") {
                    MGTextEditor(text: $memory.summary).frame(height: 110).padding(.vertical, 4)
                } footer: {
                    Text(String(format: "Spent on this character so far (memory updates, portraits and photos): $%.2f", memory.spentUSD))
                }
            }
            MGDivider()
            HStack(spacing: 8) {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .buttonStyle(.mg(.bordered)).keyboardShortcut(.cancelAction)
                    .mgHelp("Close without saving")
                Button("Save") { save() }
                    .buttonStyle(.mg(.prominent))
                    .mgHelp("Save these changes to their memory")
            }
            .padding(14)
        }
        .frame(width: 560, height: 680)
        .task {
            guard !loaded else { return }
            loaded = true
            memory = model.characterStore.memory(characterID)
            character = model.character(characterID)
        }
    }

    private var affection: Binding<Double> {
        Binding(get: { Double(memory.relationship.affection) },
                set: { memory.relationship.affection = max(0, min(100, Int($0.rounded()))) })
    }

    private func addFact() {
        let t = newFact.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        memory.facts.append(CharacterMemory.Fact(text: t))
        newFact = ""
    }

    private func save() {
        // The memory pass may have written since this opened: keep its cost and counter.
        var m = memory
        let fresh = model.characterStore.memory(characterID)
        m.spentUSD = fresh.spentUSD
        m.turnsSinceUpdate = fresh.turnsSinceUpdate
        m.album = fresh.album
        try? model.characterStore.saveMemory(m, for: characterID)
        dismiss()
    }
}
