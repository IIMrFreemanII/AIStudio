import LLMClient
import MovieCore
import SwiftUI

// MARK: idea

struct MovieIdeaPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let hasStory = !doc.movie.scenes.isEmpty
        MoviePage(title: "Idea", subtitle: "What the film is about, how it looks and how long it runs. The model develops the story from it.") {
            if hasStory {
                Button("Go to Story") { doc.stage = .story }.buttonStyle(.mg(.bordered))
            }
            Button(hasStory ? "Rewrite Story…" : "Write Story…") { model.requestStory(doc) }
                .buttonStyle(.mg(.prominent))
                .disabled(doc.isBusy || doc.movie.idea.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || doc.movie.llm == nil)
                .mgHelp(hasStory ? "Writes a new story bible and outline: replaces the cast, the places, the scenes and the shot list"
                                 : "The characters, places and an outline of every scene (two requests; you see the cost first)")
                .mgTestAnchor("movie-write-story")
        } content: {
            VStack(alignment: .leading, spacing: 18) {
                MovieField(label: "Title") {
                    TextField("Untitled Movie", text: $doc.movie.title).frame(maxWidth: 420)
                }
                MovieField(label: "Idea", hint: "a logline, a paragraph, or a full treatment") {
                    TextField("A lighthouse keeper's daughter keeps the lamp burning through the storm of the century…",
                              text: $doc.movie.idea, axis: .vertical)
                        .lineLimit(4...12)
                        .mgTestAnchor("movie-idea")
                }
                HStack(alignment: .top, spacing: 28) {
                    MovieField(label: "Style") {
                        MGPicker(selection: $doc.movie.style, options: MovieStyle.all.map { ($0.id, $0.name) })
                            .frame(width: 200, alignment: .leading)
                    }
                    MovieField(label: "Shape") {
                        MGSegmentedPicker(selection: $doc.movie.aspect,
                                          options: MovieAspect.allCases.map { ($0, $0.title) })
                            .frame(width: 200)
                    }
                    MovieField(label: "Length", hint: MovieFormat.time(Double(doc.movie.targetSeconds))) {
                        MGSlider(value: lengthBinding, in: 30...Double(MovieProject.maxTargetSeconds), step: 15)
                            .frame(width: 200)
                            .mgTestAnchor("movie-length")
                    }
                }
                MovieBudgetField(doc: doc)
                Text(MovieStyle.named(doc.movie.style).brief.capitalizedFirst)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                MovieField(label: "Writes with", hint: "a chat model on your GPU server") {
                    MovieLLMPicker(selection: $doc.movie.llm)
                }
            }
        }
    }
}

extension MovieIdeaPage {
    /// The target length: 15 s steps up to 5 minutes, then whole minutes up to 20.
    var lengthBinding: Binding<Double> {
        Binding(get: { Double(doc.movie.targetSeconds) }, set: { (v: Double) in
            let step: Double = v <= 300 ? 15 : 60
            doc.movie.targetSeconds = Int((v / step).rounded() * step)
        })
    }
}

/// The movie's budget, and what the rest of it will roughly cost with its engines.
struct MovieBudgetField: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let estimate = FilmEstimate(movie: doc.movie, timings: GPUTimings.load(model.gpuTimingsURL))
        MovieField(label: "Budget", hint: "cost sheets warn before a run would pass it") {
            HStack(spacing: 10) {
                Text("$").foregroundStyle(MG.secondaryLabel)
                TextField("none", value: $doc.movie.budget, format: .number.precision(.fractionLength(0...2)))
                    .frame(width: 90)
                    .mgTestAnchor("movie-budget")
                Text("≈ \(CostEstimate.format(estimate.usd)) to finish with these engines · \(CostEstimate.format(doc.movie.costSpent)) spent")
                    .font(.mgFootnote)
                    .foregroundStyle(doc.movie.wouldExceedBudget(estimate.usd) ? MG.warning : MG.secondaryLabel)
                    .mgHelp(estimate.parts.map { "\($0.label): \(CostEstimate.format($0.usd))" }.joined(separator: "\n")
                            + "\n\nFrom the learned GPU timings; each run's cost sheet has the real figure.")
            }
        }
    }
}

/// The chat models on the GPU servers, for writing the script and shot list.
struct MovieLLMPicker: View {
    @Environment(AppModel.self) private var model
    @Binding var selection: MovieLLM?

    var body: some View {
        let options: [(value: MovieLLM?, title: String)] = model.llm.models.map {
            (MovieLLM(serverID: $0.server.id, modelID: $0.model.id), "\($0.model.name) · \($0.model.gpuLabel) · \($0.server.name)")
        }
        if options.isEmpty {
            Text("No chat models: add one in Servers › your GPU server › Models.")
                .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                .task { await model.llm.refreshAll() }
        } else {
            MGPicker(selection: $selection, options: options, placeholder: "Choose a model…", maxWidth: 460)
                .onChange(of: selection) { _, new in model.rememberMovieLLM(new) }
                // The models list arrives after launch: a movie made before it gets the usual model then.
                .onAppear { if selection == nil { selection = model.defaultMovieLLM ?? options.first?.value } }
        }
    }
}

// MARK: script

struct MovieScriptPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.isApproved(.script)
        let written = doc.movie.scenes.filter(\.isWritten).count
        let total = doc.movie.scenes.count
        MoviePage(title: "Script",
                  subtitle: written < total ? "\(written) of \(total) scenes written. Each is written from the outline, with the scenes before it."
                      : "\(total) scenes · \(doc.movie.scenes.reduce(0) { $0 + $1.dialogue.count }) lines. Edit anything; the shot lists are written from what's here.") {
            Toggle("Critique pass", isOn: $doc.movie.critique)
                .toggleStyle(MGCheckboxStyle()).font(.mgFootnote)
                .mgHelp("After the scenes, an editor reads each sequence and revises its dialogue, pacing and continuity (one more request per sequence)")
            Button(written == 0 ? "Write Script…" : "Rewrite All…") { model.requestScript(doc) }
                .buttonStyle(.mg(written == 0 ? .prominent : .bordered)).disabled(doc.isBusy || total == 0)
                .mgHelp("Writes every scene in order, one request each (you see the cost first)")
                .mgTestAnchor("movie-write-script")
            Button(approved ? "Approved" : "Approve Script") {
                doc.approve(.script)
                doc.stage = .shots
                if doc.scopeShots.isEmpty {
                    // A long film's shot lists come a sequence at a time: the one shown (or the first).
                    let q = doc.sequence ?? (doc.movie.sequences.count > 1 ? doc.movie.sequences.first?.id : nil)
                    model.requestShots(doc, scenes: q.map { doc.movie.scenes(inSequence: $0).map(\.id) })
                }
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || written < total || total == 0)
            .mgHelp(written < total ? "Every scene needs writing first"
                    : "Locks this version in and writes the shot list from it (you see the cost first)")
            .mgTestAnchor("movie-approve-script")
        } content: {
            LazyVStack(alignment: .leading, spacing: 22) {
                MovieField(label: "Logline") {
                    TextField("One sentence", text: $doc.movie.logline, axis: .vertical).lineLimit(1...3)
                }
                ForEach($doc.movie.scenes) { $scene in
                    if doc.movie.sequences.count > 1, doc.movie.sequences.first(where: { $0.id == scene.sequence }).map({ q in
                        doc.movie.scenes.first { $0.sequence == q.id }?.id == scene.id
                    }) == true, let i = doc.movie.sequences.firstIndex(where: { $0.id == scene.sequence }) {
                        sequenceHeader(i)
                    }
                    SceneEditor(scene: $scene, cast: doc.movie.characters, sequences: doc.movie.sequences,
                                busy: doc.isBusy, rewrite: { model.requestScript(doc, scenes: [scene.id]) }) {
                        doc.movie.scenes.removeAll { $0.id == scene.id }
                    }
                }
                Button {
                    let n = (doc.movie.scenes.count + 1)
                    var id = "s\(n)"
                    while doc.movie.scenes.contains(where: { $0.id == id }) { id += "b" }
                    doc.movie.scenes.append(MovieScene(id: id, heading: "INT. NEW SCENE – DAY",
                                                       sequence: doc.movie.scenes.last?.sequence ?? MovieSequence.wholeFilm.id))
                } label: { Label("Add Scene", systemImage: "plus") }
                    .buttonStyle(.mg(.borderless))
            }
        }
        .onChange(of: doc.movie.scenes) { _, _ in if doc.isApproved(.script) { doc.reopen(.script) } }
    }

    /// A sequence's title over its first scene (editable), Rewrite Sequence, and the editor's notes on it.
    private func sequenceHeader(_ i: Int) -> some View {
        let q = doc.movie.sequences[i]
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Text("SEQUENCE \(i + 1) · ACT \(q.act)")
                    .font(.mg(size: 10.5, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                TextField("Sequence title", text: $doc.movie.sequences[i].title).font(.mgHeadline)
                Button("Rewrite Sequence…") {
                    model.requestScript(doc, scenes: doc.movie.scenes(inSequence: q.id).map(\.id))
                }
                .buttonStyle(.mg(.borderless)).controlSize(.small).disabled(doc.isBusy)
                .mgHelp("Writes this sequence's scenes again, then its critique (you see the cost first)")
            }
            if let notes = doc.movie.critiqueNotes[q.id], !notes.isEmpty {
                DisclosureGroup("Editor's notes") {
                    Text(notes).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.mgFootnote)
                .mgTestAnchor("movie-critique-notes")
            }
        }
        .padding(.top, 6)
    }
}

private struct SceneEditor: View {
    @Binding var scene: MovieScene
    let cast: [MovieCharacter]
    var sequences: [MovieSequence] = []
    var busy = false
    var rewrite: () -> Void = {}
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(scene.id.uppercased()).font(.mgMono).foregroundStyle(MG.tertiaryLabel)
                TextField("INT. PLACE – TIME", text: $scene.heading).font(.mgHeadline)
                if sequences.count > 1 {
                    MGPicker(selection: $scene.sequence,
                             options: sequences.enumerated().map { i, q in (q.id, "Sequence \(i + 1)") }, maxWidth: 130)
                        .frame(width: 130)
                        .mgHelp("The sequence this scene is made in")
                }
                Button(action: rewrite) { Image(systemName: "arrow.clockwise") }.buttonStyle(.mg(.icon)).disabled(busy)
                    .mgHelp(scene.isWritten ? "Write this scene again (you see the cost first)" : "Write this scene (you see the cost first)")
                Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.mg(.icon)).mgHelp("Remove this scene")
            }
            if !scene.beat.isEmpty {
                Text(scene.beat).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel).mgHelp("The scene's beat, from the outline (Story)")
            }
            if !scene.isWritten {
                Text("Not written yet").font(.mgFootnote).foregroundStyle(MG.warning)
            }
            TextField("Summary", text: $scene.summary).foregroundStyle(MG.secondaryLabel)
            TextField("Action", text: $scene.action, axis: .vertical).lineLimit(2...10)
            ForEach($scene.dialogue) { $line in
                HStack(alignment: .top, spacing: 8) {
                    MGPicker(selection: $line.character, options: cast.map { ($0.id, $0.name) }, maxWidth: 140)
                        .frame(width: 140)
                    TextField("Line", text: $line.text, axis: .vertical).lineLimit(1...4)
                    TextField("how", text: $line.delivery).frame(width: 130).foregroundStyle(MG.secondaryLabel)
                    Button { scene.dialogue.removeAll { $0.id == line.id } } label: { Image(systemName: "minus.circle") }
                        .buttonStyle(.mg(.icon)).mgHelp("Remove this line")
                }
            }
            HStack(spacing: 14) {
                Button {
                    scene.dialogue.append(DialogueLine(character: cast.first?.id ?? "", text: ""))
                } label: { Label("Add Line", systemImage: "text.bubble") }
                    .buttonStyle(.mg(.borderless)).disabled(cast.isEmpty)
                Spacer()
            }
            HStack(spacing: 6) {
                Image(systemName: "music.note").foregroundStyle(MG.tertiaryLabel).mgHelp("The score for this scene")
                TextField("Music mood: genre, tempo, instruments", text: $scene.mood).font(.mgFootnote)
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
    }
}

// MARK: shot list

struct MovieShotsPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.isApproved(.shots)
        let shots = doc.scopeShots
        let runtime = shots.reduce(0) { $0 + $1.duration }
        let index = doc.movie.shotIndex
        let scenes = doc.scopeScenes
        let budgets = doc.sequence == nil ? [:] : MoviePlanner.sceneSeconds(for: doc.movie)
        let aim = doc.sequence == nil ? Double(doc.movie.targetSeconds) : scenes.reduce(0) { $0 + (budgets[$1.id] ?? 0) }
        MoviePage(title: "Shot List" + (doc.scopeTitle.map { " · " + $0 } ?? ""),
                  subtitle: shots.isEmpty ? "Each shot becomes one clip: a keyframe still, animated for its duration."
                  : "\(shots.count) shots · \(MovieFormat.time(runtime)) of \(MovieFormat.time(aim)) planned") {
            Button(shots.isEmpty ? "Write Shot List…" : "Rewrite All…") {
                model.requestShots(doc, scenes: doc.sequence == nil ? nil : scenes.map(\.id))
            }
                .buttonStyle(.mg(shots.isEmpty ? .prominent : .bordered))
                .disabled(doc.isBusy || scenes.isEmpty)
                .mgTestAnchor("movie-write-shots")
            if !shots.isEmpty {
                Button(approved ? "Approved" : "Approve Shot List") {
                    doc.approve(.shots)
                }
                .buttonStyle(.mg(.prominent))
                .disabled(approved || doc.isBusy)
                .mgHelp("Locks the shots in; the cast and keyframes are made from them next")
                .mgTestAnchor("movie-approve-shots")
            }
        } content: {
            LazyVStack(alignment: .leading, spacing: 26) {
                if shots.isEmpty && !doc.isBusy {
                    MGEmptyState(symbol: "rectangle.split.3x3", title: "No shots yet",
                                 message: "Write the shot list: the model breaks each scene into shots, one request per scene.")
                        .frame(maxWidth: .infinity, minHeight: 240)
                }
                ForEach(scenes) { scene in
                    let sceneShots = shots.filter { $0.scene == scene.id }
                    let ids = sceneShots.map(\.id)
                    if !ids.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(scene.heading).font(.mgHeadline)
                                Text("\(ids.count) shots · \(MovieFormat.time(sceneShots.reduce(0) { $0 + $1.duration }))")
                                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                                Spacer()
                                Button("Rewrite Scene…") { model.requestShots(doc, scenes: [scene.id]) }
                                    .buttonStyle(.mg(.borderless)).controlSize(.small).disabled(doc.isBusy)
                                    .mgHelp("Breaks this scene down again (one request)")
                            }
                            if let note = doc.shotNotes[scene.id] {
                                Label(note.prefix(1).uppercased() + note.dropFirst(), systemImage: "exclamationmark.triangle")
                                    .font(.mgFootnote).foregroundStyle(MG.warning)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .mgHelp("Still so after asking the model again: fix it here, or rewrite the scene")
                                    .mgTestAnchor("movie-shot-note")
                            }
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 420), spacing: 12, alignment: .top)],
                                      alignment: .leading, spacing: 12) {
                                ForEach(ids, id: \.self) { id in
                                    if let binding = shotBinding(id, index: index) {
                                        ShotCard(shot: binding, number: (index[id] ?? 0) + 1, cast: doc.movie.characters,
                                                 talkingCap: MoviePlanner.talkingCap(for: doc.movie),
                                                 keyframe: binding.wrappedValue.selectedKeyframeFile.map {
                                                     model.movieStore.url(doc.id, relative: $0)
                                                 },
                                                 move: { move(id, by: $0) }, remove: { remove(id) })
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // The plan, not its takes: a keyframe made later isn't a shot list edit.
        .onChange(of: shots.map(\.planPart)) { _, _ in
            if doc.isApproved(.shots), !doc.isBusy { doc.reopen(.shots) }
        }
    }

    /// A shot by id; `index` is where it was when the page was drawn (checked, since shots move).
    private func shotBinding(_ id: UUID, index: [UUID: Int]) -> Binding<Shot>? {
        guard index[id] != nil else { return nil }
        func at() -> Int? {
            if let i = index[id], doc.movie.shots.indices.contains(i), doc.movie.shots[i].id == id { return i }
            return doc.movie.shots.firstIndex { $0.id == id }
        }
        return Binding(get: { at().map { doc.movie.shots[$0] } ?? Shot(scene: "") },
                       set: { new in if let i = at() { doc.movie.shots[i] = new } })
    }

    /// Moves a shot within its scene.
    private func move(_ id: UUID, by offset: Int) {
        guard let i = doc.movie.shots.firstIndex(where: { $0.id == id }) else { return }
        let j = i + offset
        guard doc.movie.shots.indices.contains(j), doc.movie.shots[j].scene == doc.movie.shots[i].scene else { return }
        doc.movie.shots.swapAt(i, j)
    }

    private func remove(_ id: UUID) { doc.movie.shots.removeAll { $0.id == id } }
}

private struct ShotCard: View {
    @Binding var shot: Shot
    let number: Int
    let cast: [MovieCharacter]
    /// How long a lip-synced talking shot can be with the movie's engines.
    var talkingCap: Double = Shot.durationRange.upperBound
    /// The chosen keyframe's file, when there is one.
    var keyframe: URL? = nil
    let move: (Int) -> Void
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("\(number)").font(.mg(size: 12, weight: .bold)).monospacedDigit()
                    .frame(minWidth: 22, minHeight: 22)
                    .background(Circle().fill(MG.accent.opacity(0.15)))
                    .foregroundStyle(MG.accent)
                TextField("framing", text: $shot.framing).frame(width: 96).font(.mgFootnote)
                if let c = shot.coverage {
                    Text(c.title).font(.mg(size: 9.5, weight: .medium)).foregroundStyle(MG.tertiaryLabel)
                        .mgHelp("Its part in the scene's coverage")
                }
                TextField("camera", text: $shot.camera).font(.mgFootnote)
                Spacer(minLength: 2)
                Button { move(-1) } label: { Image(systemName: "chevron.left") }.buttonStyle(.mg(.icon)).mgHelp("Earlier")
                Button { move(1) } label: { Image(systemName: "chevron.right") }.buttonStyle(.mg(.icon)).mgHelp("Later")
                Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.mg(.icon)).mgHelp("Remove this shot")
            }
            // The chosen keyframe, once the Keyframes stage made one.
            RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(MG.fill)
                .aspectRatio(16 / 9, contentMode: .fit)
                .overlay {
                    if let url = keyframe { MovieStill(url: url, maxPixels: 840) } else {
                    VStack(spacing: 4) {
                        Image(systemName: "photo").font(.mg(size: 18))
                        Text(MovieFormat.seconds(shot.duration)).font(.mgFootnote)
                    }
                    .foregroundStyle(MG.tertiaryLabel)
                    }
                }
            MovieField(label: "Keyframe") {
                TextField("The first frame: who, where, light, composition", text: $shot.keyframePrompt, axis: .vertical)
                    .lineLimit(2...6).font(.mgCallout)
            }
            MovieField(label: "Motion") {
                TextField("What moves", text: $shot.motionPrompt, axis: .vertical).lineLimit(1...4).font(.mgCallout)
            }
            if !shot.dialogue.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(shot.dialogue) { line in
                        (Text((cast.first { $0.id == line.character }?.name ?? line.character).uppercased() + "  ")
                            .font(.mg(size: 10.5, weight: .semibold)).foregroundColor(MG.secondaryLabel)
                         + Text(line.text).font(.mgCallout))
                    }
                }
            }
            TextField("Sound effects", text: $shot.sfx).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            HStack(spacing: 12) {
                HStack(spacing: 4) {
                    Text("Length").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    MGSlider(value: $shot.duration,
                             in: Shot.durationRange.lowerBound...(shot.talking ? max(talkingCap, Shot.durationRange.upperBound)
                                                                  : Shot.durationRange.upperBound),
                             step: 0.5).frame(width: 90)
                    Text(MovieFormat.seconds(shot.duration)).font(.mgFootnote).monospacedDigit().frame(width: 38, alignment: .leading)
                }
                Spacer(minLength: 0)
                MGPicker(selection: $shot.transition, options: ShotTransition.allCases.map { ($0, $0.title) }, maxWidth: 110)
                    .frame(width: 110)
                    .mgHelp("How this shot starts")
            }
            HStack(spacing: 16) {
                Toggle("Lip-sync", isOn: $shot.talking).disabled(shot.dialogue.isEmpty)
                    .mgHelp(shot.dialogue.isEmpty ? "No lines in this shot" : "The speaker is on screen: their lips move with the voice")
                Toggle("Continues previous", isOn: $shot.continuesPrevious)
                    .mgHelp("Starts from the previous shot's last frame (same moment, no cut in time)")
            }
            .toggleStyle(MGCheckboxStyle())
            .font(.mgFootnote)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .mgTestAnchor("movie-shot-card")
        .mgScrollTarget("movie-shot:\(shot.id.uuidString)")
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
