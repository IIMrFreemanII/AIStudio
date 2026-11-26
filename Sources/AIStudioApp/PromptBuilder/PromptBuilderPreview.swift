import AppKit
import CharacterCore
import MovieCore
import NodesRemote
import SwiftUI

/// "Start from Character": a saved character's look, image model and trigger words fill the draft, and the preview
/// draws with their face, LoRAs and fine-tune.
struct PromptBuilderCharacterPicker: View {
    @Environment(AppModel.self) private var model
    @Binding var draft: PromptBuilderDraft
    var lockedModel = false

    var body: some View {
        MGFormRow("Start from") {
            MGPicker(selection: Binding<UUID?>(
                get: { draft.characterID },
                set: { id in
                    guard let id, let c = model.character(id) else {
                        draft.detachCharacter()
                        return
                    }
                    let context = model.artContext(c)
                    let keep = draft.model
                    draft.apply(character: c, model: CharacterArt.promptModel(c, context: context),
                                triggerWords: context.triggerWords)
                    if lockedModel { draft.model = keep }
                }),
                options: [(nil, "No character")] + model.characters.map { (Optional($0.id), $0.name) }
                    + missing,
                maxWidth: 240)
        }
        .mgHelp("Fills the look from a saved character; the preview then uses their face. Changes here don't change the character.")
    }

    private var missing: [(value: UUID?, title: String)] {
        guard let id = draft.characterID, !model.characters.contains(where: { $0.id == id }) else { return [] }
        return [(Optional(id), draft.characterName.isEmpty ? "Missing character" : "\(draft.characterName) (deleted)")]
    }
}

/// The preview: draws the written prompt on the first GPU server and keeps the newest pictures, each with the
/// choices that made it, to compare two side by side.
struct PromptPreviewPanel: View {
    @Environment(AppModel.self) private var model
    @Binding var draft: PromptBuilderDraft

    @State private var shots: [PromptPreviewShot] = []
    @State private var shown: UUID?
    @State private var comparing = false
    @State private var running: Task<Void, Never>?
    @State private var status = ""
    @State private var error: String?
    @State private var spent = 0.0

    private var canDraw: Bool { PromptPreview.nodeType(draft.model) != nil }
    private var current: PromptPreviewShot? { shots.first { $0.id == shown } ?? shots.first }
    /// The one before `current` in the strip (older).
    private var previous: PromptPreviewShot? {
        guard let c = current, let i = shots.firstIndex(of: c), i + 1 < shots.count else { return nil }
        return shots[i + 1]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Preview").font(.mgHeadline)
            if canDraw { controls } else {
                Text("Preview draws stills: pick an image model.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            if let error { Text(error).font(.mgCallout).foregroundStyle(MG.destructive).fixedSize(horizontal: false, vertical: true) }
            if comparing, let a = previous, let b = current { compare(a, b) } else if let s = current { picture(s) }
            if !shots.isEmpty { strip }
        }
        .onAppear { shots = model.promptPreviews() }
        .onDisappear { running?.cancel() }
    }

    // MARK: controls

    @ViewBuilder private var controls: some View {
        if let c = model.previewCharacter(draft) {
            Text("Draws \(c.name) with their face, LoRAs and fine-tune.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
        } else if draft.characterID != nil {
            Text("\(draft.characterName)'s face is only used on their own image model.").font(.mgFootnote)
                .foregroundStyle(MG.secondaryLabel)
        }
        HStack(spacing: 8) {
            MGPicker(selection: $draft.aspect, options: PromptPreview.Aspect.allCases.map { ($0, $0.title) }, maxWidth: 110)
            Toggle("Keep seed", isOn: $draft.keepSeed).toggleStyle(.checkbox).fixedSize()
                .mgHelp("The same seed every time, so a changed option shows only its own effect")
            Button { draft.seed = Int.random(in: 0..<Int(Int32.max)) } label: { Image(systemName: "dice") }
                .buttonStyle(.mg(.icon)).mgHelp("New seed (\(draft.seed))")
            Spacer(minLength: 0)
        }
        if PromptPreview.takesFineTune(draft.model), let family = libraryFamily {
            MGPicker(selection: fineTune, options: fineTuneOptions(family), maxWidth: 260)
        }
        HStack(spacing: 8) {
            if running == nil {
                Button("Preview") { start() }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut("r", modifiers: .command)
                    .disabled(draft.problem != nil || (draft.written?.prompt ?? "").isEmpty)
                    .fixedSize()
            } else {
                Button("Stop") { running?.cancel() }.fixedSize()
                ProgressView().controlSize(.small)
            }
            Text(running == nil ? priceLine : status).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
            Spacer(minLength: 0)
        }
    }

    private var priceLine: String {
        var s = ""
        if let e = model.promptPreviewEstimate(draft) {
            s = "≈ $" + String(format: e.usd < 0.01 ? "%.3f" : "%.2f", e.usd) + (e.warm ? " · warm" : " · loads first")
        } else if model.connections.gpuServers.isEmpty {
            s = "Add a GPU server in Servers first"
        }
        if spent > 0 { s += (s.isEmpty ? "" : " · ") + "spent $" + String(format: "%.3f", spent) }
        return s
    }

    private var libraryFamily: String? {
        PromptPreview.nodeType(draft.model).flatMap { RemoteNodes.libraryFamilies[$0] }
    }

    /// Picking an SDXL fine-tune also sets the prompt model to its Civitai base (Pony, Illustrious quality tags).
    private var fineTune: Binding<String> {
        Binding(get: { draft.variant }, set: { v in
            draft.variant = v
            if [.sdxl, .sdxlPony, .sdxlIllustrious].contains(draft.model) {
                draft.model = PromptModel.sdxl(baseModel: v.isEmpty ? nil : model.store?.library.item(v)?.baseModel)
            }
        })
    }

    private func fineTuneOptions(_ family: String) -> [(value: String, title: String)] {
        var o: [(value: String, title: String)] = [("", RemoteNodes.baseTitle(family: family))]
        o += (model.store?.library.models(for: family) ?? []).map { ($0.id, $0.title) }
        if !draft.variant.isEmpty, !o.contains(where: { $0.value == draft.variant }) {
            o.append((draft.variant, "Missing: \(draft.variant)"))
        }
        return o
    }

    private func start() {
        error = nil
        if !draft.keepSeed { draft.seed = Int.random(in: 0..<Int(Int32.max)) }
        let d = draft
        status = "Starting…"
        running = Task {
            defer { running = nil }
            do {
                let shot = try await model.previewPrompt(d) { status = $0 }
                spent += shot.costUSD ?? 0
                shots = model.promptPreviews()
                shown = shot.id
            } catch is CancellationError {
            } catch {
                self.error = "\(error)"
            }
        }
    }

    // MARK: pictures

    private func picture(_ s: PromptPreviewShot) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            image(s).frame(maxWidth: .infinity, maxHeight: 420)
            HStack(spacing: 8) {
                Text("\(s.model.title) · seed \(s.seed)" + (s.costUSD.map { " · $" + String(format: "%.3f", $0) } ?? ""))
                    .font(.mgFootnote).foregroundStyle(MG.tertiaryLabel).lineLimit(1)
                Spacer(minLength: 0)
                if previous != nil { Button("Compare") { comparing = true }.controlSize(.small).fixedSize() }
                Button("Use These Choices") { restore(s) }.controlSize(.small).fixedSize()
                    .mgHelp("Sets the builder back to the choices this picture was made with")
            }
        }
    }

    private func compare(_ a: PromptPreviewShot, _ b: PromptPreviewShot) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 6) {
                VStack(spacing: 2) { image(a); Text("Before").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel) }
                VStack(spacing: 2) { image(b); Text("After").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel) }
            }
            .frame(maxHeight: 300)
            let changes = b.draft.changes(from: a.draft)
            Text(changes.isEmpty ? "Same choices" : changes.joined(separator: " · ")).font(.mgFootnote)
                .foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Done") { comparing = false }.controlSize(.small).fixedSize()
            }
        }
    }

    private func image(_ s: PromptPreviewShot) -> some View {
        Group {
            if let ns = NSImage(contentsOf: model.promptPreviewURL(s)) {
                Image(nsImage: ns).resizable().aspectRatio(contentMode: .fit)
            } else {
                RoundedRectangle(cornerRadius: 6).fill(MG.fill).aspectRatio(0.7, contentMode: .fit)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .contextMenu { menu(s) }
        .onTapGesture(count: 2) { NSWorkspace.shared.open(model.promptPreviewURL(s)) }
        .mgHelp(s.prompt)
    }

    private var strip: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 6) {
                ForEach(shots) { s in
                    Button { shown = s.id; comparing = false } label: {
                        thumb(s).frame(width: 52, height: 52)
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(s.id == current?.id ? MG.accent : .clear, lineWidth: 2))
                    }
                    .buttonStyle(.plain)
                    .contextMenu { menu(s) }
                }
            }
            .padding(.vertical, 2)
        }
    }

    private func thumb(_ s: PromptPreviewShot) -> some View {
        Group {
            if let ns = NSImage(contentsOf: model.promptPreviewURL(s)) {
                Image(nsImage: ns).resizable().aspectRatio(contentMode: .fill)
            } else {
                MG.fill
            }
        }
        .frame(width: 52, height: 52).clipShape(RoundedRectangle(cornerRadius: 5))
    }

    @ViewBuilder private func menu(_ s: PromptPreviewShot) -> some View {
        Button("Open") { NSWorkspace.shared.open(model.promptPreviewURL(s)) }
        Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([model.promptPreviewURL(s)]) }
        Button("Copy Image") {
            if let ns = NSImage(contentsOf: model.promptPreviewURL(s)) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.writeObjects([ns])
            }
        }
        Button("Copy Prompt") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(s.prompt, forType: .string)
        }
        Button("Use These Choices") { restore(s) }
        Divider()
        Button("Delete") {
            model.deletePromptPreview(s)
            shots = model.promptPreviews()
            if shown == s.id { shown = nil }
        }
    }

    /// Back to that picture's choices (the model stays when the builder writes into a node).
    private func restore(_ s: PromptPreviewShot) {
        let keep = draft.model
        draft = s.draft
        if !allowsModelChange { draft.model = keep }
    }

    /// Set by the builder: false in a node's sheet (the node fixes the model).
    var allowsModelChange = true
}
