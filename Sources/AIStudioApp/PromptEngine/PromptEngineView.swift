import AppKit
import CharacterCore
import GraphRuntime
import ImageIO
import LLMClient
import MovieCore
import SwiftUI

/// The Prompt Builder's Engine pane: type an idea, and the prompt engine picks a model, writes the prompt by its
/// rules, drafts it (each draft behind a cost sheet), checks the draft against a checklist and fixes what's wrong.
/// The left column is the request; the right one the run as it goes, and the result with Use / Copy / Open.
struct PromptEngineView: View {
    @Environment(AppModel.self) private var app
    /// The model the result must use (a node's), nil: the engine picks.
    var fixedModel: PromptModel? = nil
    var useTitle = "Use in Node"
    var onUse: ((EngineTry) -> Void)? = nil

    @AppStorage("promptEngine.idea") private var idea = ""
    @State private var characterID: UUID?
    @State private var llmChoice = ""
    @State private var session: PromptEngineSession?
    @State private var copied = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView { request.padding(20) }.frame(width: 340)
            Divider()
            ScrollView { run.padding(20).frame(maxWidth: .infinity, alignment: .topLeading) }
                .frame(minWidth: 460)
        }
        .background(MG.windowBackground)
        .mgSheet("prompt-engine-gate", item: Binding(get: { session?.gate }, set: { session?.gate = $0 })) { gate in
            CostGateSheet(gate: gate)
        }
        .task { await loadModels() }
        .onDisappear { session?.stop() }
    }

    // MARK: the request

    private var visionModels: [(value: String, title: String)] {
        app.llm.listings.sorted { $0.key < $1.key }.flatMap { server, listing in
            listing.models.filter { $0.installed == true && $0.vision == true }
                .map { ("\(server)/\($0.id)", $0.name.isEmpty ? $0.id : $0.name) }
        }
    }

    private var chosenKey: LLMKey? {
        guard let slash = llmChoice.firstIndex(of: "/") else { return nil }
        return LLMKey(serverID: String(llmChoice[..<slash]), modelID: String(llmChoice[llmChoice.index(after: slash)...]))
    }

    private var running: Bool { session?.running == true }

    private var request: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompt Engine").font(.mgHeadline)
            Text("Describe the picture. The engine picks a model, writes the prompt the way that model reads it, drafts it, checks the draft against a checklist and fixes what's wrong. Each draft shows its cost first.")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 4) {
                Text("The idea").font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                TextEditor(text: $idea)
                    .font(.mgBody)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(height: 120)
                    .background(RoundedRectangle(cornerRadius: 6).fill(MG.controlBackground))
                    .mgHairline(RoundedRectangle(cornerRadius: 6), color: MG.border)
                    .disabled(running)
                    .mgTestAnchor("prompt-engine-idea")
            }
            row("Model") {
                if let fixedModel {
                    Text(fixedModel.title).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                } else {
                    Text("The engine picks").font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                }
            }
            row("Character") {
                MGPicker(selection: $characterID,
                         options: [(nil, "None")] + app.characters.map { (Optional($0.id), $0.name) }, maxWidth: 200)
            }
            .mgHelp("Drafts draw this character with their face reference and LoRAs, on their own image models.")
            .disabled(running)
            row("Thinks with") {
                if visionModels.isEmpty {
                    Text("No vision model installed").font(.mgCallout).foregroundStyle(MG.warning)
                } else {
                    MGPicker(selection: $llmChoice, options: visionModels, maxWidth: 200)
                }
            }
            .mgHelp("The language model that writes the prompts and looks at the drafts. Only models that can see pictures are listed.")
            .disabled(running)
            HStack {
                Spacer()
                if running {
                    Button("Stop") { session?.stop() }.buttonStyle(.mg(.bordered)).fixedSize()
                } else {
                    Button("Start") { start() }
                        .buttonStyle(.mg(.prominent))
                        .disabled(idea.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || chosenKey == nil)
                        .keyboardShortcut(.return, modifiers: .command)
                        .fixedSize()
                        .mgTestAnchor("prompt-engine-start")
                }
            }
            if let p = AgeGuard.youthTerm(in: idea) {
                Text("The idea describes someone under 18 (“\(p)”): the engine will refuse it.")
                    .font(.mgFootnote).foregroundStyle(MG.destructive)
            }
        }
    }

    /// A label column of one width, so the rows' values line up.
    private func row<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        HStack(spacing: 10) {
            Text(label).font(.mgBody).foregroundStyle(MG.label).frame(width: 96, alignment: .leading)
            content()
            Spacer(minLength: 0)
        }
    }

    private func loadModels() async {
        for s in app.connections.gpuServers { await app.llm.refresh(s.id) }
        if llmChoice.isEmpty || !visionModels.contains(where: { $0.value == llmChoice }) {
            if let k = app.defaultPromptEngineLLM { llmChoice = "\(k.serverID)/\(k.modelID)" }
        }
    }

    private func start() {
        guard let key = chosenKey else { return }
        app.setPromptEngineLLM(key)
        session?.stop()
        let s = PromptEngineSession(app: app, idea: idea.trimmingCharacters(in: .whitespacesAndNewlines),
                                    fixedModel: fixedModel, characterID: characterID, llm: key)
        s.showsOwnGate = true
        session = s
        copied = false
        s.start()
    }

    // MARK: the run

    @ViewBuilder
    private var run: some View {
        if let s = session {
            PromptEngineRunView(session: s, record: s.record, useTitle: useTitle, onUse: onUse, copied: $copied)
        } else {
            MGEmptyState(symbol: "wand.and.stars", title: "Nothing run yet",
                         message: "Type an idea and press Start. You'll see the checklist, each draft with what the engine thought of it, and what it changed for the next one.")
                .frame(minHeight: 420)
        }
    }
}

/// One run: status, checklist, the tries with their verdicts, and the result.
struct PromptEngineRunView: View {
    @Environment(AppModel.self) private var app
    let session: PromptEngineSession
    let record: PromptEngineRecord
    var useTitle: String
    var onUse: ((EngineTry) -> Void)?
    @Binding var copied: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                if session.running { ProgressView().controlSize(.small) }
                Text(record.status.isEmpty ? "Waiting" : record.status).font(.mgCallout).lineLimit(2)
                Spacer()
                Text("Spent \(CostEstimate.format(record.gpuUSD + record.llmUSD))")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .mgHelp("GPU \(CostEstimate.format(record.gpuUSD)) for drafts, \(CostEstimate.format(record.llmUSD)) for the language model.")
            }
            if let e = session.error { Text(e).font(.mgCallout).foregroundStyle(MG.destructive) }
            if !record.checklist.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Checklist").font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                    MGFlowLayout(spacing: 6) {
                        ForEach(record.checklist, id: \.self) { item in
                            Text(item).font(.mgFootnote).padding(.horizontal, 10).frame(height: 24)
                                .background(Capsule().fill(MG.fill))
                        }
                    }
                }
            }
            if let r = record.result { result(r) }
            ForEach(record.tries.reversed()) { t in
                PromptEngineTryCard(attempt: t, verdict: record.verdict(t.n), best: record.result?.best?.n == t.n)
            }
        }
    }

    @ViewBuilder
    private func result(_ r: EngineResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let best = r.best {
                Text("Result: try \(best.n) on \(best.model.title)").font(.mgHeadline)
                if !r.notes.isEmpty {
                    Text(r.notes).font(.mgCallout).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
                }
                HStack {
                    Button(copied ? "Copied" : "Copy Prompt") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(best.prompt, forType: .string)
                        copied = true
                    }
                    .fixedSize()
                    Button("Open as Workflow") { app.openEngineTry(best) }
                        .fixedSize()
                        .mgHelp("A new workflow tab with this model, prompt, LoRAs and seed.")
                    Spacer(minLength: 0)
                    if let onUse {
                        Button(useTitle) { onUse(best) }.buttonStyle(.mg(.prominent)).fixedSize()
                    }
                }
            } else {
                Text("No result").font(.mgHeadline)
                Text(r.notes.isEmpty ? "The run ended without a draft." : r.notes).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(MG.card))
        .mgHairline(RoundedRectangle(cornerRadius: 8), color: MG.border)
    }
}

/// One draft: its picture, settings, and what the engine thought of it.
struct PromptEngineTryCard: View {
    let attempt: EngineTry
    let verdict: EngineVerdict?
    let best: Bool
    @State private var image: CGImage?
    @State private var showPrompt = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Group {
                if let image {
                    Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fit)
                } else {
                    RoundedRectangle(cornerRadius: 6).fill(MG.fill)
                }
            }
            .frame(width: 120, height: 170)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .onTapGesture { NSWorkspace.shared.open(URL(fileURLWithPath: attempt.file)) }
            .mgHelp("Open the picture")
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text("Try \(attempt.n)").font(.mgHeadline)
                    badge
                    if best { Text("Chosen").font(.mgFootnote).foregroundStyle(MG.accent) }
                }
                Text("\(attempt.model.title) · seed \(attempt.seed)"
                     + (attempt.loras.isEmpty ? "" : " · \(attempt.loras.count) LoRA\(attempt.loras.count == 1 ? "" : "s")")
                     + (attempt.costUSD.map { " · \(CostEstimate.format($0))" } ?? ""))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                if let v = verdict {
                    ForEach(v.issues, id: \.self) { i in
                        Label(i, systemImage: "xmark.circle").font(.mgFootnote).foregroundStyle(MG.label)
                    }
                    if !v.change.isEmpty {
                        Text("Next: \(v.change)").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Button(showPrompt ? "Hide Prompt" : "Show Prompt") { showPrompt.toggle() }
                    .buttonStyle(.plain).font(.mgFootnote).foregroundStyle(MG.accent).fixedSize()
                if showPrompt {
                    Text(attempt.prompt).font(.mgFootnote).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    if !attempt.negative.isEmpty {
                        Text("Negative: " + attempt.negative).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(MG.card))
        .mgHairline(RoundedRectangle(cornerRadius: 8), color: best ? MG.accent : MG.border)
        .task(id: attempt.file) { image = await Self.thumbnail(attempt.file) }
    }

    @ViewBuilder
    private var badge: some View {
        if let v = verdict {
            Text(v.pass ? "Passed" : "Needs work").font(.mgFootnote)
                .foregroundStyle(v.pass ? MG.success : MG.warning)
        } else {
            Text("Checking…").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
        }
    }

    private static func thumbnail(_ path: String) async -> CGImage? {
        await Task.detached(priority: .utility) {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
            return CGImageSourceCreateThumbnailAtIndex(src, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 480,
            ] as CFDictionary)
        }.value
    }
}

extension AppModel {
    /// A try as a new workflow tab: its model's Cloud node with the prompt, negative, seed, size, fine-tune and LoRAs.
    func openEngineTry(_ t: EngineTry) {
        guard let template = PromptPreview.template(t.model, negative: !t.negative.isEmpty, fast: t.fast, loras: t.loras.count)
        else { return }
        var values = PromptPreview.values(WrittenPrompt(prompt: t.prompt, negative: t.negative), model: t.model, seed: t.seed,
                                          aspect: t.aspect, server: connections.gpuServers.first?.id ?? "",
                                          fineTune: t.variant.isEmpty ? nil : t.variant)
        values.merge(PromptPreview.loraValues(t.loras.map { (id: $0.id, strength: $0.strength) })) { $1 }
        guard let graph = try? template.bind(values, optional: Set(values.keys), remove: []) else { return }
        openTab(graph, name: "Prompt Engine try \(t.n)")
    }
}
