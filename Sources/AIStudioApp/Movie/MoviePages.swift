import LLMClient
import MovieCore
import SwiftUI

// MARK: idea

struct MovieIdeaPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let hasScript = !doc.movie.scenes.isEmpty
        MoviePage(title: "Idea", subtitle: "What the film is about, how it looks and how long it runs. The model writes a script from it.") {
            if hasScript {
                Button("Go to Script") { doc.stage = .script }.buttonStyle(.mg(.bordered))
            }
            Button(hasScript ? "Rewrite Script…" : "Write Script…") { model.requestScript(doc) }
                .buttonStyle(.mg(.prominent))
                .disabled(doc.isBusy || doc.movie.idea.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || doc.movie.llm == nil)
                .mgHelp(hasScript ? "Writes a new script: replaces the cast, the scenes and the shot list"
                                  : "Shows the cost first; nothing runs until you confirm")
                .mgTestAnchor("movie-write-script")
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
                        MGSlider(value: Binding(get: { Double(doc.movie.targetSeconds) },
                                                set: { doc.movie.targetSeconds = Int($0) }),
                                 in: 30...300, step: 15)
                            .frame(width: 200)
                    }
                }
                Text(MovieStyle.named(doc.movie.style).brief.capitalizedFirst)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                MovieField(label: "Writes with", hint: "a chat model on your GPU server") {
                    MovieLLMPicker(selection: $doc.movie.llm)
                }
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
        let approved = doc.movie.approved.contains(.script)
        MoviePage(title: "Script", subtitle: doc.movie.logline.isEmpty ? "Edit anything; the shot list is written from what's here." : doc.movie.logline) {
            Button("Rewrite…") { model.requestScript(doc) }.buttonStyle(.mg(.bordered)).disabled(doc.isBusy)
                .mgHelp("Writes a new script from the idea: replaces the cast, the scenes and the shot list")
            Button(approved ? "Approved" : "Approve Script") {
                doc.movie.approve(.script)
                doc.stage = .shots
                if doc.movie.shots.isEmpty { model.requestShots(doc) }
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || doc.movie.scenes.isEmpty)
            .mgHelp("Locks this version in and writes the shot list from it (you see the cost first)")
            .mgTestAnchor("movie-approve-script")
        } content: {
            VStack(alignment: .leading, spacing: 22) {
                MovieField(label: "Logline") {
                    TextField("One sentence", text: $doc.movie.logline, axis: .vertical).lineLimit(1...3)
                }
                castSection
                ForEach($doc.movie.scenes) { $scene in
                    SceneEditor(scene: $scene, cast: doc.movie.characters) {
                        doc.movie.scenes.removeAll { $0.id == scene.id }
                    }
                }
                Button {
                    let n = (doc.movie.scenes.count + 1)
                    var id = "s\(n)"
                    while doc.movie.scenes.contains(where: { $0.id == id }) { id += "b" }
                    doc.movie.scenes.append(MovieScene(id: id, heading: "INT. NEW SCENE – DAY"))
                } label: { Label("Add Scene", systemImage: "plus") }
                    .buttonStyle(.mg(.borderless))
            }
        }
        .onChange(of: doc.movie.scenes) { _, _ in if doc.movie.approved.contains(.script) { doc.movie.reopen(.script) } }
        .onChange(of: doc.movie.characters) { _, _ in if doc.movie.approved.contains(.script) { doc.movie.reopen(.script) } }
    }

    private var castSection: some View {
        MovieField(label: "Cast", hint: "look and voice are what keep them the same in every shot") {
            VStack(spacing: 8) {
                ForEach($doc.movie.characters) { $c in
                    HStack(alignment: .top, spacing: 10) {
                        TextField("Name", text: $c.name).frame(width: 140)
                        VStack(alignment: .leading, spacing: 6) {
                            TextField("Look: age, face, hair, build, clothes", text: $c.look, axis: .vertical).lineLimit(1...4)
                            TextField("Voice: gender, age, pitch, texture, accent", text: $c.voice, axis: .vertical).lineLimit(1...3)
                        }
                        Button { doc.movie.characters.removeAll { $0.id == c.id } } label: { Image(systemName: "trash") }
                            .buttonStyle(.mg(.icon)).mgHelp("Remove \(c.name)")
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
                }
                HStack {
                    Button {
                        var id = "character-\(doc.movie.characters.count + 1)"
                        while doc.movie.characters.contains(where: { $0.id == id }) { id += "b" }
                        doc.movie.characters.append(MovieCharacter(id: id, name: "New Character", look: "", voice: ""))
                    } label: { Label("Add Character", systemImage: "plus") }
                        .buttonStyle(.mg(.borderless))
                    Spacer()
                }
            }
        }
    }
}

private struct SceneEditor: View {
    @Binding var scene: MovieScene
    let cast: [MovieCharacter]
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(scene.id.uppercased()).font(.mgMono).foregroundStyle(MG.tertiaryLabel)
                TextField("INT. PLACE – TIME", text: $scene.heading).font(.mgHeadline)
                Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.mg(.icon)).mgHelp("Remove this scene")
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
        let approved = doc.movie.approved.contains(.shots)
        let runtime = doc.movie.runtime
        MoviePage(title: "Shot List",
                  subtitle: doc.movie.shots.isEmpty ? "Each shot becomes one clip: a keyframe still, animated for its duration."
                  : "\(doc.movie.shots.count) shots · \(MovieFormat.time(runtime)) of \(MovieFormat.time(Double(doc.movie.targetSeconds))) planned") {
            Button(doc.movie.shots.isEmpty ? "Write Shot List…" : "Rewrite All…") { model.requestShots(doc) }
                .buttonStyle(.mg(doc.movie.shots.isEmpty ? .prominent : .bordered))
                .disabled(doc.isBusy || doc.movie.scenes.isEmpty)
                .mgTestAnchor("movie-write-shots")
            if !doc.movie.shots.isEmpty {
                Button(approved ? "Approved" : "Approve Shot List") {
                    doc.movie.approve(.shots)
                }
                .buttonStyle(.mg(.prominent))
                .disabled(approved || doc.isBusy)
                .mgHelp("Locks the shots in; the cast and keyframes are made from them next")
                .mgTestAnchor("movie-approve-shots")
            }
        } content: {
            VStack(alignment: .leading, spacing: 26) {
                if doc.movie.shots.isEmpty && !doc.isBusy {
                    MGEmptyState(symbol: "rectangle.split.3x3", title: "No shots yet",
                                 message: "Write the shot list: the model breaks each scene into shots, one request per scene.")
                        .frame(maxWidth: .infinity, minHeight: 240)
                }
                ForEach(doc.movie.scenes) { scene in
                    let ids = doc.movie.shots(in: scene.id).map(\.id)
                    if !ids.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(scene.heading).font(.mgHeadline)
                                Text("\(ids.count) shots · \(MovieFormat.time(doc.movie.shots(in: scene.id).reduce(0) { $0 + $1.duration }))")
                                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                                Spacer()
                                Button("Rewrite Scene…") { model.requestShots(doc, scenes: [scene.id]) }
                                    .buttonStyle(.mg(.borderless)).controlSize(.small).disabled(doc.isBusy)
                                    .mgHelp("Breaks this scene down again (one request)")
                            }
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 420), spacing: 12, alignment: .top)],
                                      alignment: .leading, spacing: 12) {
                                ForEach(ids, id: \.self) { id in
                                    if let binding = shotBinding(id) {
                                        ShotCard(shot: binding, number: shotNumber(id), cast: doc.movie.characters,
                                                 move: { move(id, by: $0) }, remove: { remove(id) })
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        .onChange(of: doc.movie.shots) { _, _ in
            if doc.movie.approved.contains(.shots), !doc.isBusy { doc.movie.reopen(.shots) }
        }
    }

    private func shotBinding(_ id: UUID) -> Binding<Shot>? {
        guard doc.movie.shots.contains(where: { $0.id == id }) else { return nil }
        return Binding(get: { doc.movie.shots.first { $0.id == id } ?? Shot(scene: "") },
                       set: { new in
                           if let i = doc.movie.shots.firstIndex(where: { $0.id == id }) { doc.movie.shots[i] = new }
                       })
    }

    private func shotNumber(_ id: UUID) -> Int { (doc.movie.shots.firstIndex { $0.id == id } ?? 0) + 1 }

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
                TextField("camera", text: $shot.camera).font(.mgFootnote)
                Spacer(minLength: 2)
                Button { move(-1) } label: { Image(systemName: "chevron.left") }.buttonStyle(.mg(.icon)).mgHelp("Earlier")
                Button { move(1) } label: { Image(systemName: "chevron.right") }.buttonStyle(.mg(.icon)).mgHelp("Later")
                Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.mg(.icon)).mgHelp("Remove this shot")
            }
            // The keyframe goes here once it's made (next update).
            RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(MG.fill)
                .aspectRatio(16 / 9, contentMode: .fit)
                .overlay {
                    VStack(spacing: 4) {
                        Image(systemName: "photo").font(.mg(size: 18))
                        Text(MovieFormat.seconds(shot.duration)).font(.mgFootnote)
                    }
                    .foregroundStyle(MG.tertiaryLabel)
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
                    MGSlider(value: $shot.duration, in: Shot.durationRange, step: 0.5).frame(width: 90)
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
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
