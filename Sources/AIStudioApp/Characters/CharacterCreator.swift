import CharacterCore
import SwiftUI

/// New Character: a character made step by step from option chips: who they are, their look, personality and
/// habits, then their profile photo (drawn from the chosen look) and their voice (designed from the chosen voice).
/// Photos and voice samples are made before the character is saved; Cancel removes them again.
struct CharacterCreator: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let edit: CharacterEdit
    @State private var c: AppModel.Companion
    @State private var step: Int
    @State private var saved = false
    @State private var saveProblems: [String] = []

    static let steps = ["Basics", "Look", "Personality", "Habits", "Photo", "Voice"]
    private static let genders = ["woman", "man", "non-binary"]

    init(edit: CharacterEdit, step: Int = 0) {
        self.edit = edit
        _c = State(initialValue: edit.character)
        _step = State(initialValue: min(max(step, 0), Self.steps.count - 1))
    }

    private var problems: [String] {
        var p = AgeGuard.problems(c)
        if c.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { p.insert("Add a name.", at: 0) }
        return p
    }

    private var isLast: Bool { step == Self.steps.count - 1 }

    var body: some View {
        let shown = saveProblems.isEmpty ? problems : saveProblems
        VStack(spacing: 0) {
            header
            MGDivider()
            MGForm { page }
            MGDivider()
            VStack(alignment: .leading, spacing: 10) {
                if !shown.isEmpty, step == 0 || isLast || !saveProblems.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(shown, id: \.self) { Text($0).font(.mgFootnote).foregroundStyle(MG.destructive) }
                    }
                }
                HStack(spacing: 8) {
                    Button("Cancel", role: .cancel) { dismiss() }
                        .buttonStyle(.mg(.bordered)).keyboardShortcut(.cancelAction)
                        .mgHelp("Close without making the character (photos and voices made here are removed)")
                    Spacer()
                    if step > 0 {
                        Button("Back") { step -= 1 }
                            .buttonStyle(.mg(.bordered))
                            .mgHelp("The previous step")
                    }
                    if isLast {
                        Button("Create") { save(chat: false) }
                            .buttonStyle(.mg(.bordered)).disabled(!problems.isEmpty)
                            .mgHelp("Save the character")
                        Button("Create & Chat") { save(chat: true) }
                            .buttonStyle(.mg(.prominent)).disabled(!problems.isEmpty)
                            .mgHelp("Save the character, then open a chat with them")
                    } else {
                        Button(skippable ? "Skip" : "Next") { step += 1 }
                            .buttonStyle(.mg(skippable ? .bordered : .prominent))
                            .disabled(!problems.isEmpty)
                            .keyboardShortcut(.defaultAction)
                            .mgHelp(problems.isEmpty ? "The next step" : "Give them a name first")
                    }
                }
            }
            .padding(14)
        }
        .frame(width: 640, height: 660)
        .onChange(of: c) { saveProblems = [] }
        .onDisappear { if !saved { model.discardDraft(c.id) } }
    }

    /// The photo step with no photo chosen yet: its button says Skip.
    private var skippable: Bool { step == 4 && c.selectedPortrait == nil }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                CharacterAvatar(c, model: model, size: 36)
                VStack(alignment: .leading, spacing: 1) {
                    Text(c.name.isEmpty ? "New Character" : c.name).font(.mgTitle3).lineLimit(1)
                    Text("Step \(step + 1) of \(Self.steps.count): \(Self.steps[step])")
                        .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                }
                Spacer()
            }
            HStack(spacing: 6) {
                ForEach(Self.steps.indices, id: \.self) { i in
                    Button { if problems.isEmpty || i == 0 { step = i } } label: {
                        VStack(spacing: 4) {
                            Capsule().fill(i <= step ? MG.accent : MG.fill).frame(height: 4)
                            Text(Self.steps[i]).font(.mgFootnote)
                                .foregroundStyle(i == step ? MG.label : MG.secondaryLabel)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(i == step ? .isSelected : [])
                    .mgHelp("Go to \(Self.steps[i])")
                }
            }
        }
        .padding(.horizontal, 18).padding(.vertical, 14)
    }

    @ViewBuilder private var page: some View {
        switch step {
        case 0:
            MGSection("Who they are") {
                MGFormRow("Name") { TextField("Name", text: $c.name).frame(maxWidth: 300) }
                MGFormRow("Age") {
                    HStack(spacing: 10) {
                        Text("\(c.age)").monospacedDigit().foregroundStyle(MG.secondaryLabel)
                        MGStepper(value: $c.age, in: 18...99).mgHelp("Characters are adults: 18 or older")
                    }
                }
                MGFormRow("Gender") {
                    MGSegmentedPicker(selection: gender, options: Self.genders.map { ($0, $0.capitalized) }
                        + (Self.genders.contains(c.gender) ? [] : [(c.gender, c.gender.isEmpty ? "Other" : c.gender.capitalized)]))
                        .frame(maxWidth: 340)
                        .mgHelp("Some look options depend on it")
                }
                TraitGroupView(group: .basics, c: $c, detailsTitle: "Backstory", details: $c.backstory)
            } footer: {
                Text("Every step is optional except the name; tap an option again to drop it.")
            }
        case 1:
            MGSection("Look") {
                TraitGroupView(group: .look, c: $c, detailsTitle: "More details", details: $c.look)
            } footer: {
                Text("Their profile photo and every photo they send are drawn from this.")
            }
        case 2:
            MGSection("Personality") {
                TraitGroupView(group: .personality, c: $c, detailsTitle: "More about them", details: $c.personality)
            }
        case 3:
            MGSection("Habits and interests") {
                TraitGroupView(group: .habits, c: $c, detailsTitle: "Things they love", details: $c.likes)
            }
        case 4:
            MGSection("Drawn as") {
                MGFormRow("Style") {
                    MGSegmentedPicker(selection: $c.artModel.style,
                                      options: ArtModel.Style.allCases.map { ($0, $0 == .anime ? "Anime" : "Photoreal") })
                        .frame(width: 200)
                        .mgHelp("Photoreal looks like a camera picture; anime like a drawing")
                }
                MGFormRow("Image model") {
                    MGSegmentedPicker(selection: family,
                                      options: ArtModel.Family.allCases.map { ($0, $0 == .chroma ? "Chroma" : "SDXL") })
                        .frame(width: 200)
                        .mgHelp("SDXL is quick and keeps the face closely; Chroma follows descriptions more closely and costs about 10× more")
                }
                if !c.fullLook.isEmpty {
                    Text("From their look: \(c.fullLook)").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            CharacterLookSection(c: $c, title: "Profile photo", thumbSize: 132)
        default:
            CharacterVoiceSection(c: $c)
        }
    }

    private var gender: Binding<String> {
        Binding(get: { c.gender }, set: { c.gender = $0 })
    }

    /// Switching families drops the fine-tune and LoRAs (they only fit the model they were made for).
    private var family: Binding<ArtModel.Family> {
        Binding(get: { c.artModel.family }, set: { f in
            guard f != c.artModel.family else { return }
            c.artModel.family = f
            c.artModel.variant = nil
            c.loras = []
        })
    }

    private func save(chat: Bool) {
        var new = c
        new.name = new.name.trimmingCharacters(in: .whitespacesAndNewlines)
        let result = model.saveCharacter(new)
        guard result.isEmpty else { saveProblems = result; return }
        saved = true
        dismiss()
        if chat {
            model.showCharacterGallery = false
            model.chat(withCharacter: new.id)
        }
    }
}
