import AppKit
import CharacterCore
import MovieCore
import NodesStandard
import SwiftUI

/// Builds a prompt from choices: the model to write for, the subject (a person with the character creator's look
/// options, or anything), what happens and where, then chips for medium, shot, camera, light, time, mood, colour,
/// composition and (video) camera moves and sound. The right side shows the prompt and negative as that model reads
/// best, live, and a preview draws it (PromptPreviewPanel). `onUse` (from a node) writes them into the node; the window
/// copies them.
struct PromptBuilderView: View {
    @Environment(AppModel.self) private var app
    @Binding var draft: PromptBuilderDraft
    /// The model is fixed by the node it writes into.
    var lockedModel: Bool = false
    /// In a node's sheet: the library family of the node's LoRAs (nil: it takes none).
    var nodeLoRAFamily: String? = nil
    var useTitle = "Use in Node"
    var onUse: ((WrittenPrompt) -> Void)? = nil

    @State private var copied: String?
    @AppStorage(AppModel.adultOptionsKey) private var adult = false
    /// The look options take room (photo tiles): folded until opened; open at first when something is picked.
    @State private var lookOpen = false

    private var written: WrittenPrompt? { draft.written }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    subjectSection
                    if loraFamily != nil { lorasSection }
                    if draft.isPerson { lookSection }
                    sceneSection
                    ForEach(PromptCatalog.groups(for: draft.model, adult: adult && draft.isPerson).filter { $0.field != .place }) { g in
                        ChoiceChips(group: g, picks: $draft.picks, model: draft.model)
                    }
                    extrasSection
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minWidth: 460)
            .onAppear {
                lookOpen = !draft.traits.isEmpty
                draft.keepLoRAs(fitting: loraFamily, library: library)
            }
            .onChange(of: draft.model) { _, _ in draft.keepLoRAs(fitting: loraFamily, library: library) }
            Divider()
            ScrollView { output }.frame(width: 400)
        }
        .background(MG.windowBackground)
    }

    // MARK: input

    private var subjectSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Model and subject")
            PromptBuilderCharacterPicker(draft: $draft, lockedModel: lockedModel)
            if !lockedModel {
                MGFormRow("Write for") {
                    MGPicker(selection: $draft.model, options: PromptModel.allCases.map { ($0, $0.title) }, maxWidth: 240)
                }
            } else {
                Text("Writing for \(draft.model.title)").font(.mgCallout).foregroundStyle(MG.secondaryLabel)
            }
            Toggle("A person", isOn: $draft.isPerson)
            if draft.isPerson {
                HStack(spacing: 12) {
                    MGPicker(selection: $draft.gender, options: [("woman", "Woman"), ("man", "Man"), ("person", "Person")],
                             maxWidth: 120)
                    Stepper(value: $draft.age, in: AgeGuard.minimumAge...90) { Text("\(draft.age) years old") }
                }
                TextField("Who they are (optional): a lighthouse keeper, a jazz singer…", text: $draft.subject)
                if draft.model == .minimaxImage {
                    Toggle("The reference picture (<Picture 1>) is their face", isOn: $draft.faceReference)
                }
            } else {
                TextField("The subject: a red fox, a vintage motorbike, a misty harbour…", text: $draft.subject)
            }
        }
    }

    /// The character creator's look options, as chips with their example photos.
    private var lookSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { lookOpen.toggle() } label: {
                HStack(spacing: 6) {
                    Image(systemName: lookOpen ? "chevron.down" : "chevron.right").font(.system(size: 11))
                    sectionTitle("Look")
                    if !draft.traits.isEmpty {
                        Text("\(draft.traits.values.reduce(0) { $0 + $1.count }) picked").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if lookOpen { lookChips }
        }
    }

    private var lookChips: some View {
        VStack(alignment: .leading, spacing: 6) {
            let person = Binding<AppModel.Companion>(
                get: {
                    var c = AppModel.Companion(name: "Prompt", age: draft.age, gender: draft.gender, look: "")
                    c.traits = draft.traits
                    return c
                },
                set: { draft.traits = $0.traits })
            TraitGroupView(group: .look, c: person)
        }
    }

    private var sceneSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("What happens and where")
            TextField(draft.model.isVideo ? "What happens: she turns, laughs and walks out of frame…"
                      : "What they're doing: reading by the window, mid-laugh…", text: $draft.action, axis: .vertical)
                .lineLimit(1...3)
            TextField("Place (or pick one below)", text: $draft.place)
            if let places = PromptCatalog.group("place") {
                // A place chip fills the field, so it can be edited further.
                MGFlowLayout(spacing: 6) {
                    ForEach(places.choices) { c in
                        ChoiceChip(label: c.label, on: draft.place == c.words) {
                            draft.place = draft.place == c.words ? "" : c.words
                        }
                    }
                }
            }
        }
    }

    private var extrasSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Extras")
            if draft.model.drawsText {
                TextField("Exact text in the picture (a sign, a title)", text: $draft.text)
            }
            if draft.model.dialect == .tags || draft.model.dialect == .sentences {
                TextField(draft.loras.isEmpty ? "LoRA trigger words, comma-separated"
                          : "More trigger words, comma-separated (the picked LoRAs' are added)", text: $draft.triggerWords)
            }
            TextField(draft.model.takesNegative ? "Keep out (negative prompt), comma-separated"
                      : "Keep out (\(draft.model.title) has no negative prompt: these become words)", text: $draft.avoid)
            Button("Clear All") {
                let model = draft.model
                draft = PromptBuilderDraft()
                draft.model = model
            }
            .controlSize(.small)
        }
    }

    // MARK: LoRAs

    private var library: LinkLibrary? { app.store?.library }
    private var loraFamily: String? { lockedModel ? nodeLoRAFamily : PromptBuilderDraft.loraFamily(draft.model) }
    private var availableLoRAs: [LibraryEntry] { loraFamily.map { library?.loras(for: $0) ?? [] } ?? [] }
    private var loraFamilyName: String { LoRASources.familyName(loraFamily) }

    /// LoRAs from the Models library that fit the model: picked with a strength, their trigger words join the prompt.
    private var lorasSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("LoRAs")
            ForEach(Array(draft.loras.enumerated()), id: \.offset) { i, _ in loraRow(i) }
            HStack(spacing: 10) {
                Button("Add LoRA") { addLoRA() }
                    .buttonStyle(.mg(.bordered)).controlSize(.small)
                    .disabled(availableLoRAs.isEmpty || draft.loras.count >= PromptPreview.maxLoRAs)
                    .mgHelp(availableLoRAs.isEmpty ? "Add a \(loraFamilyName) LoRA in the Models library first (⇧⌘M)"
                            : "Use a \(loraFamilyName) LoRA from the Models library (up to \(PromptPreview.maxLoRAs)): its "
                            + "trigger words join the prompt, and the preview and Use in Node draw with it")
                    .mgTestAnchor("prompt-builder-add-lora")
                if availableLoRAs.isEmpty {
                    Text("No \(loraFamilyName) LoRAs in the Models library yet").font(.mgFootnote)
                        .foregroundStyle(MG.secondaryLabel)
                }
                Spacer(minLength: 0)
            }
        }
    }

    /// The next LoRA not picked yet (or the first one).
    private func addLoRA() {
        let used = Set(draft.loras.map(\.id))
        guard let e = availableLoRAs.first(where: { !used.contains($0.id) }) ?? availableLoRAs.first else { return }
        draft.loras.append(PromptBuilderLoRA(id: e.id, triggerWords: e.triggerWords))
    }

    private func loraOptions(current: String) -> [(value: String, title: String)] {
        var o = availableLoRAs.map { (value: $0.id, title: LinkLibrary.loraTitle($0, for: loraFamily ?? "")) }
        if !o.contains(where: { $0.value == current }) { o.append((current, "Missing: \(current)")) }
        return o
    }

    @ViewBuilder private func loraRow(_ i: Int) -> some View {
        if draft.loras.indices.contains(i) {
            let l = draft.loras[i]
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    MGPicker(selection: Binding(get: { draft.loras.indices.contains(i) ? draft.loras[i].id : "" },
                                                set: { id in
                                                    guard draft.loras.indices.contains(i) else { return }
                                                    draft.loras[i].id = id
                                                    draft.loras[i].triggerWords = library?.item(id)?.triggerWords ?? []
                                                }),
                             options: loraOptions(current: l.id), placeholder: "Pick a LoRA", maxWidth: 220)
                        .mgHelp("A \(loraFamilyName) LoRA from the Models library")
                    MGSlider(value: Binding(get: { draft.loras.indices.contains(i) ? draft.loras[i].strength : 0 },
                                            set: { if draft.loras.indices.contains(i) { draft.loras[i].strength = $0 } }),
                             in: -1...2, step: 0.05).frame(width: 100)
                        .mgHelp("How strongly it shapes the picture, −1 to 2. Around 0.6–0.8 keeps the prompt in charge")
                    Text(String(format: "%.2f", l.strength)).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                        .frame(width: 38, alignment: .trailing)
                    Button { if draft.loras.indices.contains(i) { draft.loras.remove(at: i) } } label: { Image(systemName: "trash") }
                        .accessibilityLabel("Remove LoRA")
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .mgHelp("Remove this LoRA")
                }
                if !l.triggerWords.isEmpty {
                    Text("Adds: " + l.triggerWords.joined(separator: ", ")).font(.mgFootnote)
                        .foregroundStyle(MG.tertiaryLabel).lineLimit(2)
                }
            }
        }
    }

    private func sectionTitle(_ s: String) -> some View {
        Text(s).font(.mgHeadline).foregroundStyle(MG.label)
    }

    // MARK: output

    private var output: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(draft.model.title).font(.mgHeadline)
            Text(Self.howItReads(draft.model)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .fixedSize(horizontal: false, vertical: true)
            if let problem = draft.problem {
                Text(problem).font(.mgCallout).foregroundStyle(MG.destructive)
            }
            box("Prompt", written?.prompt ?? "")
            if draft.model.takesNegative { box("Negative", written?.negative ?? "") }
            if draft.model.dialect == .tags, let p = written?.prompt, !p.isEmpty {
                let tokens = Self.estimatedTokens(p)
                Text("About \(tokens) tokens" + (tokens > 75 ? ": past CLIP's first 75 the server splits it into chunks, which weigh less" : ""))
                    .font(.mgFootnote).foregroundStyle(tokens > 75 ? MG.warning : MG.tertiaryLabel)
            }
            HStack {
                Button(copied == "prompt" ? "Copied" : "Copy Prompt") { copy(written?.prompt ?? "", as: "prompt") }
                    .disabled((written?.prompt ?? "").isEmpty).fixedSize()
                if draft.model.takesNegative {
                    Button(copied == "negative" ? "Copied" : "Copy Negative") { copy(written?.negative ?? "", as: "negative") }
                        .disabled((written?.negative ?? "").isEmpty).fixedSize()
                }
                Spacer(minLength: 0)
            }
            if let onUse, let w = written {
                HStack {
                    Spacer()
                    Button(useTitle) { onUse(w) }.buttonStyle(.mg(.prominent)).disabled(w.prompt.isEmpty).fixedSize()
                }
            }
            Divider().padding(.vertical, 4)
            PromptPreviewPanel(draft: $draft, allowsModelChange: !lockedModel)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func box(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
            ScrollView {
                Text(text.isEmpty ? " " : text).font(.mgCallout).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(8)
            }
            .frame(minHeight: title == "Prompt" ? 180 : 70, maxHeight: title == "Prompt" ? 320 : 120)
            .background(RoundedRectangle(cornerRadius: 6).fill(MG.fill))
        }
    }

    private func copy(_ s: String, as what: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        copied = what
    }

    /// A rough CLIP token count: words and punctuation, plus a share for long words split into pieces.
    static func estimatedTokens(_ s: String) -> Int {
        var n = 0
        for word in s.split(whereSeparator: { $0.isWhitespace }) {
            let letters = word.filter { $0.isLetter || $0.isNumber }.count
            n += max(1, Int((Double(letters) / 6).rounded(.up))) + word.filter { ",.:()".contains($0) }.count
        }
        return n
    }

    /// One line on how the model reads a prompt (docs/prompting.md).
    static func howItReads(_ m: PromptModel) -> String {
        switch m.dialect {
        case .tags:
            return "Reads a short comma list: the first words weigh most. Picks are weighted (…:1.2)"
                + (m.takesWeights ? "" : " where the model supports it") + "; exclusions go in the negative."
                + (m.qualityTags.isEmpty ? "" : " Adds this family's quality tags.")
        case .sentences:
            return "Reads plain sentences describing what is visible." + (m.takesNegative
                ? " Exclusions go in the negative." : " It has no negative prompt: exclusions are written as sentences.")
        case .narrative:
            return "Reads one detailed paragraph: purpose and framing first, then the subject, then light and colour. "
                + "No negative prompt and no quality words; exclusions are closing sentences, and it is told to draw no text unless you ask for some."
        case .motion:
            return "Reads one flowing paragraph: the shot and camera move, then the action as it unfolds, the look, place and light"
                + (m.hasSound ? ", and what is heard." : ".") + (m.takesNegative ? " Exclusions go in the negative." : "")
        }
    }
}

/// One catalog group as a row of chips.
struct ChoiceChips: View {
    let group: PromptCatalog.Group
    @Binding var picks: [String: [String]]
    /// The words a chip adds differ by model (`Choice.words(for:)`).
    var model: PromptModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(group.title).foregroundStyle(MG.label)
                if group.max > 1 { Text("up to \(group.max)").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel) }
            }
            MGFlowLayout(spacing: 6) {
                ForEach(group.choices) { c in
                    ChoiceChip(label: c.label, on: (picks[group.id] ?? []).contains(c.id)) {
                        picks = PromptCatalog.toggle(picks, group: group.id, choice: c.id)
                    }
                    .mgHelp(c.words(for: model))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct ChoiceChip: View {
    let label: String
    let on: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label).font(.mgFootnote).lineLimit(1)
                .foregroundStyle(on ? MG.accentForeground : MG.label)
                .padding(.horizontal, 10).frame(height: 26)
                .background(Capsule().fill(on ? MG.accent : MG.fill))
                .mgHairline(Capsule(), color: on ? MG.accent : MG.border)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}

/// The Prompt Builder window (Generate ▸ Prompt Builder…): pick the model, copy the result. The draft is kept between
/// launches.
struct PromptBuilderWindow: View {
    @AppStorage(PromptBuilderDraft.storageKey) private var stored = ""
    @State private var draft = PromptBuilderDraft()
    @State private var loaded = false

    var body: some View {
        PromptBuilderView(draft: $draft)
            .frame(minWidth: 880, minHeight: 600)
            .onAppear {
                guard !loaded else { return }
                loaded = true
                if let d = try? JSONDecoder().decode(PromptBuilderDraft.self, from: Data(stored.utf8)) { draft = d }
            }
            .onChange(of: draft) { _, d in
                if let data = try? JSONEncoder().encode(d) { stored = String(decoding: data, as: UTF8.self) }
            }
    }
}

/// The builder as a sheet for one node: the model comes from the node, and Use in Node writes the prompt (and the
/// negative, when the node has one) as one undo step.
struct PromptBuilderSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    @AppStorage(PromptBuilderDraft.storageKey) private var stored = ""
    let target: PromptBuilderTarget
    @State private var draft = PromptBuilderDraft()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Prompt Builder").font(.mgHeadline)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            PromptBuilderView(draft: $draft, lockedModel: true, nodeLoRAFamily: target.loraFamily) { w in
                model.mutate("Prompt Builder", undoManager: model.activeUndoManager) { g in
                    try? g.setValue(.string(w.prompt), node: target.id, input: "prompt")
                    if target.takesNegative { try? g.setValue(.string(w.negative), node: target.id, input: "negative") }
                    // The picked LoRAs, wired into the node (ahead of any it already has).
                    if target.loraFamily != nil { PromptBuilderDraft.wireLoRAs(draft.loras, into: target.id, graph: &g) }
                }
                dismiss()
            }
        }
        .frame(width: 960, height: 760)
        .onAppear {
            if let d = try? JSONDecoder().decode(PromptBuilderDraft.self, from: Data(stored.utf8)) { draft = d }
            draft.model = target.model
        }
        .onChange(of: draft) { _, d in
            if let data = try? JSONEncoder().encode(d) { stored = String(decoding: data, as: UTF8.self) }
        }
    }
}
