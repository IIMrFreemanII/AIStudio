import AVFoundation
import Foundation
import MovieCore
import NodesRemote

/// The Voices and Sound stages: each character's voice (designed once from their description, then cloned for every
/// line, so it stays the same), the shots' lines, their sound effects and each scene's music, rendered through the
/// movie's templates on a GPU server behind the cost gate. Every take is measured when it's made (length and
/// loudness), so the edit can lay it out and level it without reading the files again.
extension AppModel {
    // MARK: voices

    /// The characters who speak in a shot.
    func speakingCharacters(_ movie: MovieProject) -> [MovieCharacter] {
        let ids = Set(movie.shots.flatMap { $0.dialogue.map(\.character) })
        return movie.characters.filter { ids.contains($0.id) }
    }

    /// Voices › Make Voices: one voice sample per speaking character (`characters` nil: those without one yet, else
    /// all of them).
    func requestVoiceSamples(_ doc: MovieDocument, characters ids: [String]? = nil) {
        let speaking = speakingCharacters(doc.movie)
        let missing = speaking.filter { $0.voiceSamples.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? speaking.map(\.id) : missing)
        guard !ids.isEmpty else { return }
        let seconds = Double(MovieVoices.sampleText.count) / 15
        gateImages(doc, title: ids.count == 1 ? "Make a Voice" : "Make Voices", work: [
            ImageWork(label: ids.count == 1 ? "1 voice sample" : "\(ids.count) voice samples", task: "qwen3-tts",
                      jobs: ids.count, megapixels: MovieVoices.work(seconds: seconds)),
        ], extraNotes: ["Each voice is designed from the character's voice description, saying one sample sentence. "
                            + "Their lines are then spoken in that voice."]) { [weak self] in
            self?.makeVoiceSamples(doc, characters: ids)
        }
    }

    private func makeVoiceSamples(_ doc: MovieDocument, characters ids: [String]) {
        runImageJob(doc, title: "Making voices", count: ids.count, task: { _ in "qwen3-tts" }) {
            [weak self] doc, i, server in
            guard let self, let c = doc.movie.character(ids[i]) else { return }
            doc.job?.detail = "\(c.name) (\(i + 1) of \(ids.count))"
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let template = self.movieTemplate(MovieVoices.TemplateName.sample, for: doc) { MovieVoices.sampleTemplate() }
            let take = try await self.renderSound(doc, template: template,
                                                  render: MovieVoices.sampleRender(c, seed: seed, server: server),
                                                  to: "audio/voice-\(c.id)", seed: seed,
                                                  title: "\(doc.movie.title): \(c.name)'s voice")
            guard let j = doc.movie.characters.firstIndex(where: { $0.id == c.id }) else { return }
            doc.movie.characters[j].voiceSamples.append(take)
            if doc.movie.characters[j].selectedVoiceSample == nil { doc.movie.characters[j].selectedVoiceSample = take.id }
        }
    }

    /// A line to voice: which shot, which line.
    struct LineRef: Hashable {
        var shot: UUID
        var line: UUID
    }

    /// Voices › Speak Lines: every line of the shots (nil: all shots) that has no take yet; when they all have one,
    /// a new take of each.
    func requestLines(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let shots = doc.movie.shots.filter { ids?.contains($0.id) ?? true }
        let all = shots.flatMap { s in s.dialogue.filter { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { (ref: LineRef(shot: s.id, line: $0.id), line: $0) } }
        let missing = all.filter { $0.line.takes.isEmpty }
        let items = missing.isEmpty ? all : missing
        guard !items.isEmpty else { return }
        let lacking = Set(items.compactMap { item -> String? in
            guard let c = doc.movie.character(item.line.character) else { return item.line.character }
            return c.selectedVoiceSampleFile == nil ? c.name : nil
        })
        guard lacking.isEmpty else {
            doc.error = "Make and choose a voice for \(lacking.sorted().joined(separator: ", ")) first."
            return
        }
        let seconds = items.map { Double($0.line.text.count) / 15 }.reduce(0, +) / Double(items.count)
        gateImages(doc, title: items.count == 1 ? "Speak a Line" : "Speak Lines", work: [
            ImageWork(label: items.count == 1 ? "1 line" : "\(items.count) lines", task: "qwen3-tts", jobs: items.count,
                      megapixels: MovieVoices.work(seconds: max(1, seconds))),
        ], extraNotes: ["Lines are spoken in each character's chosen voice. Shots too short for their lines get longer "
                            + "(up to \(Int(MovieVoices.maxTalking)) s for a lip-synced shot, 7.5 s otherwise)."]) { [weak self] in
            self?.makeLines(doc, items.map(\.ref))
        }
    }

    private func makeLines(_ doc: MovieDocument, _ refs: [LineRef]) {
        runImageJob(doc, title: "Speaking lines", count: refs.count, task: { _ in "qwen3-tts" }) {
            [weak self] doc, i, server in
            guard let self, let j = doc.movie.shots.firstIndex(where: { $0.id == refs[i].shot }),
                  let k = doc.movie.shots[j].dialogue.firstIndex(where: { $0.id == refs[i].line }) else { return }
            let line = doc.movie.shots[j].dialogue[k]
            guard let c = doc.movie.character(line.character), let sample = c.selectedVoiceSampleFile else { return }
            doc.job?.detail = "Shot \(j + 1), \(c.name) (\(i + 1) of \(refs.count))"
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let template = self.movieTemplate(MovieVoices.TemplateName.line, for: doc) { MovieVoices.lineTemplate() }
            let render = MovieVoices.lineRender(line, sample: self.movieStore.url(doc.id, relative: sample), seed: seed,
                                                server: server)
            let take = try await self.renderSound(doc, template: template, render: render,
                                                  to: "audio/line-\(line.id.uuidString.prefix(8).lowercased())",
                                                  seed: seed, title: "\(doc.movie.title): shot \(j + 1) line")
            guard let j = doc.movie.shots.firstIndex(where: { $0.id == refs[i].shot }),
                  let k = doc.movie.shots[j].dialogue.firstIndex(where: { $0.id == refs[i].line }) else { return }
            doc.movie.shots[j].dialogue[k].takes.append(take)
            if doc.movie.shots[j].dialogue[k].selectedTake == nil { doc.movie.shots[j].dialogue[k].selectedTake = take.id }
            self.fitShotsToLines(doc)
        }
    }

    /// Makes shots long enough for their chosen lines; says which still don't fit.
    func fitShotsToLines(_ doc: MovieDocument) {
        let unfit = MovieVoices.fitDurations(&doc.movie)
        if !unfit.isEmpty {
            let numbers = unfit.compactMap { id in doc.movie.shots.firstIndex { $0.id == id }.map { $0 + 1 } }
            doc.error = "The lines of shot \(numbers.map(String.init).joined(separator: ", ")) are longer than the "
                + "shot can be: shorten them, or split the shot on the Shot List page."
        }
    }

    func selectVoiceSample(_ doc: MovieDocument, character id: String, take: UUID) {
        guard let j = doc.movie.characters.firstIndex(where: { $0.id == id }),
              doc.movie.characters[j].selectedVoiceSample != take else { return }
        doc.movie.characters[j].selectedVoiceSample = take
        doc.movie.reopen(.voices)  // lines spoken in the old voice no longer match
        saveMovie(doc)
    }

    func selectLine(_ doc: MovieDocument, shot: UUID, line: UUID, take: UUID) {
        guard let j = doc.movie.shots.firstIndex(where: { $0.id == shot }),
              let k = doc.movie.shots[j].dialogue.firstIndex(where: { $0.id == line }),
              doc.movie.shots[j].dialogue[k].selectedTake != take else { return }
        doc.movie.shots[j].dialogue[k].selectedTake = take
        fitShotsToLines(doc)
        doc.movie.reopen(.voices)
        saveMovie(doc)
    }

    // MARK: sound

    /// Sound › Make Sound Effects: one take per shot with an effects prompt (`shots` nil: those without one yet,
    /// else all of them).
    func requestEffects(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let withPrompt = doc.movie.shots.filter { MovieSound.effectsPrompt($0) != nil }
        let missing = withPrompt.filter { $0.sounds.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? withPrompt.map(\.id) : missing)
        let shots = withPrompt.filter { ids.contains($0.id) }
        guard !shots.isEmpty else { return }
        let seconds = shots.map(\.duration).reduce(0, +) / Double(shots.count)
        gateImages(doc, title: shots.count == 1 ? "Make a Sound Effect" : "Make Sound Effects", work: [
            ImageWork(label: shots.count == 1 ? "1 sound effect" : "\(shots.count) sound effects", task: "stable-audio-3",
                      jobs: shots.count, megapixels: MovieVoices.work(seconds: seconds)),
        ], extraNotes: ["Each shot's effects are as long as the shot, from its sound prompt."]) { [weak self] in
            self?.makeEffects(doc, shots: shots.map(\.id))
        }
    }

    private func makeEffects(_ doc: MovieDocument, shots ids: [UUID]) {
        runImageJob(doc, title: "Making sound effects", count: ids.count, task: { _ in "stable-audio-3" }) {
            [weak self] doc, i, server in
            guard let self, let j = doc.movie.shots.firstIndex(where: { $0.id == ids[i] }) else { return }
            let shot = doc.movie.shots[j]
            let seed = Int.random(in: 0..<Int(UInt32.max))
            guard let render = MovieSound.effectsRender(shot, seed: seed, server: server) else { return }
            doc.job?.detail = "Shot \(j + 1) (\(i + 1) of \(ids.count))"
            let template = self.movieTemplate(MovieSound.TemplateName.effects, for: doc) { MovieSound.effectsTemplate() }
            let take = try await self.renderSound(doc, template: template, render: render,
                                                  to: "audio/fx-\(shot.id.uuidString.prefix(8).lowercased())",
                                                  seed: seed, title: "\(doc.movie.title): shot \(j + 1) sound")
            guard let j = doc.movie.shots.firstIndex(where: { $0.id == ids[i] }) else { return }
            doc.movie.shots[j].sounds.append(take)
            if doc.movie.shots[j].selectedSound == nil { doc.movie.shots[j].selectedSound = take.id }
        }
    }

    /// Sound › Make Music: one piece per scene (`scenes` nil: those without one yet, else all of them), as long as
    /// the scene.
    func requestMusic(_ doc: MovieDocument, scenes ids: [String]? = nil) {
        doc.movie.music = MovieSound.cues(for: doc.movie)
        let missing = doc.movie.music.filter { $0.takes.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? doc.movie.music.map(\.id) : missing)
        let cues = doc.movie.music.filter { ids.contains($0.id) }
        guard !cues.isEmpty else { return }
        let seconds = cues.map { MovieSound.sceneSeconds($0.id, in: doc.movie) }.reduce(0, +) / Double(cues.count)
        gateImages(doc, title: cues.count == 1 ? "Make Music" : "Make Music for \(cues.count) Scenes", work: [
            ImageWork(label: cues.count == 1 ? "1 piece of music" : "\(cues.count) pieces of music", task: "ace-step-xl",
                      jobs: cues.count, megapixels: MovieVoices.work(seconds: max(10, seconds + 2))),
        ], extraNotes: ["One instrumental piece per scene, a little longer than the scene; it dips under the dialogue."]) { [weak self] in
            self?.makeMusic(doc, scenes: cues.map(\.id))
        }
    }

    private func makeMusic(_ doc: MovieDocument, scenes ids: [String]) {
        runImageJob(doc, title: "Making music", count: ids.count, task: { _ in "ace-step-xl" }) {
            [weak self] doc, i, server in
            guard let self, let cue = doc.movie.music.first(where: { $0.id == ids[i] }) else { return }
            let heading = doc.movie.scenes.first { $0.id == cue.id }?.heading ?? cue.id
            doc.job?.detail = "\(heading) (\(i + 1) of \(ids.count))"
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let render = MovieSound.musicRender(cue, seconds: MovieSound.sceneSeconds(cue.id, in: doc.movie), seed: seed,
                                                server: server)
            let template = self.movieTemplate(MovieSound.TemplateName.music, for: doc) { MovieSound.musicTemplate() }
            let take = try await self.renderSound(doc, template: template, render: render, to: "audio/music-\(cue.id)",
                                                  seed: seed, title: "\(doc.movie.title): music, \(heading)")
            guard let j = doc.movie.music.firstIndex(where: { $0.id == cue.id }) else { return }
            doc.movie.music[j].takes.append(take)
            if doc.movie.music[j].selected == nil { doc.movie.music[j].selected = take.id }
        }
    }

    func selectSound(_ doc: MovieDocument, shot id: UUID, take: UUID) {
        guard let j = doc.movie.shots.firstIndex(where: { $0.id == id }), doc.movie.shots[j].selectedSound != take
        else { return }
        doc.movie.shots[j].selectedSound = take
        doc.movie.reopen(.sound)
        saveMovie(doc)
    }

    func selectMusic(_ doc: MovieDocument, scene id: String, take: UUID) {
        guard let j = doc.movie.music.firstIndex(where: { $0.id == id }), doc.movie.music[j].selected != take
        else { return }
        doc.movie.music[j].selected = take
        doc.movie.reopen(.sound)
        saveMovie(doc)
    }

    /// Takes a sound take out of the movie (a voice, a line, an effect or a piece of music) and its file to the
    /// Trash; the choice moves to the newest take left.
    func deleteSoundTake(_ doc: MovieDocument, take: UUID) {
        var file: String?
        var stage: MovieStage?
        for j in doc.movie.characters.indices where doc.movie.characters[j].voiceSamples.contains(where: { $0.id == take }) {
            file = doc.movie.characters[j].voiceSamples.first { $0.id == take }?.file
            doc.movie.characters[j].voiceSamples.removeAll { $0.id == take }
            if doc.movie.characters[j].selectedVoiceSample == take {
                doc.movie.characters[j].selectedVoiceSample = doc.movie.characters[j].voiceSamples.last?.id
                stage = .voices
            }
        }
        for j in doc.movie.shots.indices {
            for k in doc.movie.shots[j].dialogue.indices
            where doc.movie.shots[j].dialogue[k].takes.contains(where: { $0.id == take }) {
                file = doc.movie.shots[j].dialogue[k].takes.first { $0.id == take }?.file
                doc.movie.shots[j].dialogue[k].takes.removeAll { $0.id == take }
                if doc.movie.shots[j].dialogue[k].selectedTake == take {
                    doc.movie.shots[j].dialogue[k].selectedTake = doc.movie.shots[j].dialogue[k].takes.last?.id
                    stage = .voices
                }
            }
            if doc.movie.shots[j].sounds.contains(where: { $0.id == take }) {
                file = doc.movie.shots[j].sounds.first { $0.id == take }?.file
                doc.movie.shots[j].sounds.removeAll { $0.id == take }
                if doc.movie.shots[j].selectedSound == take {
                    doc.movie.shots[j].selectedSound = doc.movie.shots[j].sounds.last?.id
                    stage = .sound
                }
            }
        }
        for j in doc.movie.music.indices where doc.movie.music[j].takes.contains(where: { $0.id == take }) {
            file = doc.movie.music[j].takes.first { $0.id == take }?.file
            doc.movie.music[j].takes.removeAll { $0.id == take }
            if doc.movie.music[j].selected == take {
                doc.movie.music[j].selected = doc.movie.music[j].takes.last?.id
                stage = .sound
            }
        }
        if let stage { doc.movie.reopen(stage) }
        if let file {
            let url = movieStore.url(doc.id, relative: file)
            if Self.isAutomatedRun { try? FileManager.default.removeItem(at: url) }
            else { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        }
        saveMovie(doc)
    }

    // MARK: running

    /// Renders one sound through a template, copies it into the movie's folder (`base` plus a short id and the
    /// file's own extension), measures its length and loudness, and records its cost and time.
    private func renderSound(_ doc: MovieDocument, template: MovieTemplate, render: MovieArt.Render, to base: String,
                             seed: Int, title: String) async throws -> Take {
        let (outputs, jobs) = try await runMovieGraph(doc, template: template, render: render, title: title)
        guard let made = outputs.first(where: { $0.kind == .audio })?.url else { throw MovieSoundError.noAudio }
        let ext = made.pathExtension.isEmpty ? "wav" : made.pathExtension
        let file = "\(base)-\(UUID().uuidString.prefix(6).lowercased()).\(ext)"
        let url = movieStore.url(doc.id, relative: file)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: made, to: url)
        let measured = try? MovieLoudness.measure(url)
        let cost = recordJobs(jobs, doc: doc, work: measured.map { MovieVoices.work(seconds: $0.seconds) })
        return Take(file: file, seed: seed, costUSD: cost, seconds: measured?.seconds, gainDB: measured?.gainDB)
    }
}

extension MovieCharacter {
    var selectedVoiceSampleFile: String? { voiceSamples.first { $0.id == selectedVoiceSample }?.file }
}

extension MovieProject {
    /// Lines that will play silent: no chosen take.
    var silentLines: Int {
        shots.reduce(0) { n, s in
            n + s.dialogue.filter { $0.selectedTake == nil && !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }.count
        }
    }

    var lineCount: Int {
        shots.reduce(0) { $0 + $1.dialogue.filter { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }.count }
    }
}

enum MovieSoundError: Error, CustomStringConvertible {
    case noAudio
    var description: String { "the template gave no sound (it needs a Preview Audio or Save Audio at the end)" }
}
