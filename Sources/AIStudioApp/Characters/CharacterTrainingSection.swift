import CharacterCore
import MovieCore
import NodesStandard
import SwiftUI

/// The editor's "Character LoRA" section: training pictures (drawn with the character's face on their model), the
/// ones to train on (15–30), and Train LoRA, whose result joins their LoRAs. Only for saved SDXL and MiniMax characters.
struct CharacterTrainingSection: View {
    @Environment(AppModel.self) private var model
    @Binding var c: AppModel.Companion

    private var set: TrainingSet {
        _ = model.trainingRevision  // redraw when pictures or state change
        return model.trainingSet(c.id)
    }
    private var running: Bool { model.trainingTasks[c.id] != nil }
    private var status: String? { model.trainingStatus[c.id] }
    private var supported: Bool { CharacterTraining.task(c.artModel.family) != nil }
    private var saved: Bool { model.character(c.id) != nil }

    var body: some View {
        MGSection("Character LoRA") {
            if !saved {
                Text("Save the character first.").foregroundStyle(MG.secondaryLabel)
            } else if !supported {
                Text("LoRA training is for SDXL and MiniMax characters.").foregroundStyle(MG.secondaryLabel)
            } else {
                content
            }
        } footer: {
            Text("A LoRA trained on 15–30 pictures of them keeps their face, hair and body the same in every photo; "
                 + "clothes, places and poses stay free.")
        }
        .onChange(of: set.loraID) { _, id in
            // A training that finished while the editor is open: its LoRA joins this draft too.
            guard let id, !c.loras.contains(where: { $0.lora == id }), c.loras.count < CharacterArt.maxLoRAs else { return }
            c.loras.append(CharacterLoRA(lora: id, strength: 0.9))
        }
    }

    @ViewBuilder private var content: some View {
        let s = set
        if !s.pictures.isEmpty { grid(s) }
        HStack(spacing: 8) {
            Button(s.pictures.isEmpty ? "Make Training Pictures" : "Make 10 More") { makePictures(s.pictures.isEmpty ? CharacterTraining.pictureCount : 10) }
                .buttonStyle(.mg(s.pictures.isEmpty ? .prominent : .bordered))
                .disabled(running || c.portraits.isEmpty)
                .mgHelp(c.portraits.isEmpty ? "Make or pick a profile photo first: the pictures use its face"
                        : "Draw varied pictures of them (angles, framings, outfits, places, light) with their face")
                .mgTestAnchor("training-make")
            if !s.pictures.isEmpty {
                Button("Train LoRA") { train() }
                    .buttonStyle(.mg(.prominent))
                    .disabled(running || !s.canTrain)
                    .mgHelp(s.canTrain ? "Train a LoRA on the picked pictures on your GPU server"
                            : "Pick \(CharacterTraining.minPicked)–\(CharacterTraining.maxPicked) pictures to train on")
                    .mgTestAnchor("training-train")
            }
            if running {
                MGSpinner().controlSize(.small)
                Button("Cancel") { model.trainingTasks[c.id]?.cancel() }
                    .buttonStyle(.mg(.bordered)).controlSize(.small)
            }
            Spacer(minLength: 0)
        }
        VStack(alignment: .leading, spacing: 2) {
            if let status { Text(status).font(.mgFootnote).foregroundStyle(MG.secondaryLabel) }
            Text(summary(s)).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
            if let error = s.error { Text(error).font(.mgFootnote).foregroundStyle(MG.destructive) }
        }
    }

    private func grid(_ s: TrainingSet) -> some View {
        let folder = model.characterStore.trainingFolder(c.id)
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 84, maximum: 96), spacing: 8)], alignment: .leading,
                         spacing: 8) {
            ForEach(s.pictures) { p in
                TrainingThumb(url: folder.appendingPathComponent(p.file), picked: p.picked, revision: model.trainingRevision,
                              toggle: { model.setTrainingPicked(c.id, file: p.file, picked: !p.picked) },
                              redraw: { redraw(p.file) })
                    .disabled(running)
            }
        }
    }

    private func summary(_ s: TrainingSet) -> String {
        var parts: [String] = []
        if !s.pictures.isEmpty { parts.append("\(s.picked.count) of \(s.pictures.count) picked") }
        let e = model.trainingEstimate(c, pictures: s.pictures.isEmpty ? CharacterTraining.pictureCount : 10,
                                       picked: max(s.picked.count, CharacterTraining.minPicked))
        if let d = e.draw { parts.append("pictures ≈ " + CostEstimate.format(d)) }
        if let t = e.train { parts.append("training ≈ " + CostEstimate.format(t)) }
        if s.spentUSD > 0 { parts.append("spent " + CostEstimate.format(s.spentUSD)) }
        if let id = s.loraID, let item = model.store?.library.item(id) {
            parts.append("trained: \(item.title)" + (s.trigger.map { ", trigger \($0)" } ?? ""))
        }
        return parts.joined(separator: " · ")
    }

    private func makePictures(_ count: Int) {
        let who = c
        model.runTraining(c.id) { progress in try await model.makeTrainingPictures(who, count: count, progress: progress) }
    }

    private func redraw(_ file: String) {
        let who = c
        model.runTraining(c.id) { progress in try await model.redrawTrainingPicture(who, file: file, progress: progress) }
    }

    private func train() {
        let who = c
        model.runTraining(c.id) { progress in _ = try await model.trainLoRA(who, progress: progress) }
    }
}

private struct TrainingThumb: View {
    let url: URL
    let picked: Bool
    let revision: Int
    let toggle: () -> Void
    let redraw: () -> Void
    @State private var hover = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
        MGFileThumb(url: url)
            .id("\(url.path)#\(revision)")
            .frame(width: 84, height: 84)
            .background(MG.fill)
            .clipShape(shape)
            .opacity(picked ? 1 : 0.4)
            .overlay { if picked { shape.strokeBorder(MG.accent, lineWidth: 2) } }
            .overlay(alignment: .topTrailing) {
                if hover {
                    Button(action: redraw) { Image(systemName: "arrow.clockwise") }
                        .accessibilityLabel("Redraw Picture")
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .background(Circle().fill(.regularMaterial))
                        .padding(4)
                        .mgHelp("Draw this picture again")
                }
            }
            .contentShape(shape)
            .onTapGesture(perform: toggle)
            .onHover { hover = $0 }
            .mgHelp(picked ? "Trained on: click to leave it out" : "Left out: click to train on it")
            .accessibilityAddTraits(.isButton)
            .accessibilityLabel(picked ? "Training picture, picked" : "Training picture, left out")
    }
}
