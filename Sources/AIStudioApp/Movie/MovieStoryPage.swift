import MovieCore
import SwiftUI

/// Story: the bible (premise, tone, the score's theme, the characters with what drives them, the places) and the
/// outline (acts → sequences → scenes, each with its beat and seconds). Approved, the script is written from it a
/// scene at a time.
struct MovieStoryPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    /// What the script is written from: a change takes back the approval.
    private var storyKey: [String] {
        let m = doc.movie
        return [m.premise, m.tone, m.theme, m.logline] + m.characters.flatMap(\.storyPart)
            + m.locations.flatMap { [$0.id, $0.name, $0.look] }
            + m.sequences.flatMap { [$0.id, $0.title, "\($0.act)", $0.summary] }
            + m.scenes.flatMap { [$0.id, $0.heading, $0.location, $0.beat, $0.summary, $0.sequence,
                                  $0.characters.joined(separator: ","), "\($0.seconds ?? 0)"] }
    }

    var body: some View {
        let approved = doc.isApproved(.story)
        let seconds = MoviePlanner.sceneSeconds(for: doc.movie)
        let planned = doc.movie.scenes.reduce(0) { $0 + (seconds[$1.id] ?? 0) }
        MoviePage(title: "Story",
                  subtitle: doc.movie.scenes.isEmpty ? "The characters, the places and the outline the script is written from."
                      : "\(doc.movie.characters.count) characters · \(doc.movie.locations.count) places · "
                      + "\(doc.movie.sequences.count) sequences · \(doc.movie.scenes.count) scenes · "
                      + "\(MovieFormat.time(planned)) of \(MovieFormat.time(Double(doc.movie.targetSeconds)))") {
            Button("Rewrite Story…") { model.requestStory(doc) }
                .buttonStyle(.mg(.bordered)).disabled(doc.isBusy || doc.movie.llm == nil)
                .mgHelp("Writes a new bible and outline from the idea: replaces the cast, places, scenes and shot list")
            Button(approved ? "Approved" : "Approve Story") {
                doc.approve(.story)
                doc.stage = .script
                if !doc.movie.scenes.contains(where: \.isWritten) { model.requestScript(doc) }
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || doc.movie.scenes.isEmpty)
            .mgHelp("Locks the story in and writes the script from it, a scene at a time (you see the cost first)")
            .mgTestAnchor("movie-approve-story")
        } content: {
            LazyVStack(alignment: .leading, spacing: 22) {
                MovieField(label: "Premise", hint: "the whole story: setup, escalation, crisis, climax, resolution") {
                    TextField("What happens", text: $doc.movie.premise, axis: .vertical).lineLimit(3...12)
                        .mgTestAnchor("movie-premise")
                }
                HStack(alignment: .top, spacing: 20) {
                    MovieField(label: "Tone") {
                        TextField("How it feels", text: $doc.movie.tone, axis: .vertical).lineLimit(1...3)
                    }
                    MovieField(label: "Main theme", hint: "for the score") {
                        TextField("Instruments, melody, how it grows", text: $doc.movie.theme, axis: .vertical).lineLimit(1...3)
                    }
                }
                charactersSection
                locationsSection
                outlineSection(seconds: seconds)
            }
        }
        .onChange(of: storyKey) { _, _ in
            if doc.isApproved(.story), !doc.isBusy { doc.reopen(.story) }
        }
    }

    // MARK: characters

    private var charactersSection: some View {
        MovieField(label: "Characters", hint: "what they want and what stops them drives every scene; look and voice keep them the same") {
            VStack(spacing: 8) {
                ForEach($doc.movie.characters) { $c in
                    CharacterBibleCard(character: $c) { doc.movie.characters.removeAll { $0.id == c.id } }
                }
                HStack(spacing: 14) {
                    Button {
                        add(MovieCharacter(id: "", name: "New Character", look: "", voice: ""))
                    } label: { Label("Add Character", systemImage: "plus") }
                    Button {
                        add(MovieCharacter(id: "narrator", name: "Narrator", look: "", voice: "", role: "narrator",
                                           isNarrator: true))
                    } label: { Label("Add Narrator", systemImage: "text.bubble") }
                        .disabled(doc.movie.characters.contains(where: \.isNarrator))
                    Spacer()
                }
                .buttonStyle(.mg(.borderless))
            }
        }
    }

    private func add(_ c: MovieCharacter) {
        var c = c
        var id = c.id.isEmpty ? "character-\(doc.movie.characters.count + 1)" : c.id
        while doc.movie.characters.contains(where: { $0.id == id }) { id += "b" }
        c.id = id
        doc.movie.characters.append(c)
    }

    // MARK: places

    private var locationsSection: some View {
        MovieField(label: "Places", hint: "a fixed look keeps a place the same in every scene there") {
            VStack(spacing: 8) {
                ForEach($doc.movie.locations) { $l in
                    HStack(alignment: .top, spacing: 10) {
                        TextField("Name", text: $l.name).frame(width: 220).font(.mgHeadline)
                        TextField("Look: architecture, materials, light, colours, weather", text: $l.look, axis: .vertical)
                            .lineLimit(1...4).font(.mgCallout)
                        Button { doc.movie.locations.removeAll { $0.id == l.id } } label: { Image(systemName: "trash") }
                            .buttonStyle(.mg(.icon)).mgHelp("Remove \(l.name)")
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
                }
                HStack {
                    Button {
                        var id = "place-\(doc.movie.locations.count + 1)"
                        while doc.movie.locations.contains(where: { $0.id == id }) { id += "b" }
                        doc.movie.locations.append(MovieLocation(id: id, name: "New Place"))
                    } label: { Label("Add Place", systemImage: "plus") }
                        .buttonStyle(.mg(.borderless))
                    Spacer()
                }
            }
        }
    }

    // MARK: outline

    private func outlineSection(seconds: [String: Double]) -> some View {
        MovieField(label: "Outline", hint: "acts, sequences and scenes; each scene's beat and length") {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(Array(doc.movie.sequences.indices), id: \.self) { i in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 10) {
                            Text("ACT \(doc.movie.sequences[i].act) · SEQUENCE \(i + 1)")
                                .font(.mg(size: 10.5, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                            TextField("Sequence title", text: $doc.movie.sequences[i].title).font(.mgHeadline)
                            let total = doc.movie.scenes(inSequence: doc.movie.sequences[i].id)
                                .reduce(0) { $0 + (seconds[$1.id] ?? 0) }
                            Text(MovieFormat.time(total)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                        }
                        TextField("What this sequence does", text: $doc.movie.sequences[i].summary, axis: .vertical)
                            .lineLimit(1...3).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        ForEach($doc.movie.scenes) { $scene in
                            if scene.sequence == doc.movie.sequences[i].id {
                                OutlineSceneRow(scene: $scene, seconds: seconds[scene.id] ?? 0,
                                                cast: doc.movie.characters, places: doc.movie.locations)
                            }
                        }
                    }
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
                    .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
                }
            }
            .mgTestAnchor("movie-outline")
        }
    }
}

private struct CharacterBibleCard: View {
    @Binding var character: MovieCharacter
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                TextField("Name", text: $character.name).frame(width: 160).font(.mgHeadline)
                TextField("Role", text: $character.role).frame(width: 140).font(.mgFootnote)
                if character.isNarrator {
                    Text("narrator · heard only").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                }
                Spacer()
                Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.mg(.icon)).mgHelp("Remove \(character.name)")
            }
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 6) {
                row("Wants", "What they want", $character.goal)
                row("Flaw", "What stands in their way from inside", $character.flaw)
                row("Arc", "How they change", $character.arc)
                row("Relations", "Who they are to the others", $character.relationships)
                if !character.isNarrator { row("Look", "Age, face, hair, build, clothes", $character.look) }
                row("Voice", "Gender, age, pitch, texture, accent, way of speaking", $character.voice)
                row("Motif", "Their musical motif", $character.motif)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .mgTestAnchor("movie-character-bible")
    }

    private func row(_ label: String, _ prompt: String, _ text: Binding<String>) -> some View {
        GridRow {
            Text(label).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).gridColumnAlignment(.trailing)
            TextField(prompt, text: text, axis: .vertical).lineLimit(1...3).font(.mgCallout)
        }
    }
}

private struct OutlineSceneRow: View {
    @Binding var scene: MovieScene
    let seconds: Double
    let cast: [MovieCharacter]
    let places: [MovieLocation]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(scene.id.uppercased()).font(.mgMono).foregroundStyle(MG.tertiaryLabel)
                TextField("INT. PLACE – TIME", text: $scene.heading).font(.mgSubheadline)
                if !places.isEmpty {
                    MGPicker(selection: $scene.location,
                             options: [("", "No place")] + places.map { ($0.id, $0.name) }, maxWidth: 150)
                        .frame(width: 150)
                        .mgHelp("Where it happens: its look goes into every keyframe")
                }
                Stepper(value: Binding(get: { scene.seconds ?? seconds }, set: { scene.seconds = max(4, $0) }),
                        in: 4...600, step: 5) {
                    Text(MovieFormat.time(scene.seconds ?? seconds)).font(.mgFootnote).monospacedDigit()
                }
                .mgHelp("How long the scene runs on screen")
            }
            TextField("Beat: what they want / what opposes them / how it turns", text: $scene.beat, axis: .vertical)
                .lineLimit(1...3).font(.mgCallout)
            if !scene.characters.isEmpty {
                Text(scene.characters.compactMap { id in cast.first { $0.id == id }?.name }.joined(separator: ", "))
                    .font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
            }
        }
        .padding(.vertical, 4)
    }
}

extension MovieCharacter {
    /// What the story is written from (a sheet or voice take isn't a story edit).
    var storyPart: [String] {
        [id, name, look, voice, role, goal, flaw, arc, relationships, motif, isNarrator ? "n" : ""]
    }
}
