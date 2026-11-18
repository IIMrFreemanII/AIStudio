import AppKit
import Foundation
import MovieCore
import NodesRemote

/// The Shots and Edit stages: each shot's clip (Wan 2.2 image-to-video on a GPU server, from its keyframe or, for a
/// continuing shot, from the clip before's last frame), behind the cost gate; then the edit's composition and the
/// MP4 export, which run on this Mac.
extension AppModel {
    // MARK: clips

    /// Shots › Make Clips: one take per shot (`shots` nil: those without a clip from the movie's engine yet, else all),
    /// in film order; a
    /// continuing shot starts from its predecessor's new clip, the others render at the same time.
    func requestClips(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let scoped = doc.scopeShots
        let missing = scoped.filter { !$0.hasClip(by: doc.movie.videoEngine) }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? scoped.map(\.id) : missing)
        let shots = doc.movie.shots.filter { ids.contains($0.id) }
        guard !shots.isEmpty else { return }
        let lacking = shots.filter { $0.selectedKeyframeFile == nil }.compactMap { number(of: $0, in: doc.movie) }
        guard lacking.isEmpty else {
            doc.error = "Choose a keyframe for shot \(lacking.map(String.init).joined(separator: ", ")) first (Keyframes)."
            return
        }
        if doc.movie.videoEngine == .minimaxH3 {
            requestH3Clips(doc, shots: shots)
            return
        }
        let size = doc.movie.aspect.videoSize
        let engine = doc.movie.lipSyncEngine
        let talking = shots.filter { MovieShots.usesLipSync($0, in: doc.movie) }
        // InfiniteTalk dubs each talking shot's ordinary clip, so those are made too.
        let plain = shots.filter { !MovieShots.usesLipSync($0, in: doc.movie) || engine == .infiniteTalk }
        let work = plain.map { MovieShots.work(width: size.width, height: size.height,
                                               frames: MovieShots.frames(for: $0.duration)) }
        let talkWork = talking.map { MovieShots.talkingWork(width: size.width, height: size.height, seconds: $0.duration) }
        var notes = plain.isEmpty ? ["Clips are \(size.width)×\(size.height) at 16 fps."]
            : ["Clips are \(size.width)×\(size.height) at 16 fps, Wan 2.2's fast 4-step mode."]
        let continuing = shots.filter(\.continuesPrevious).count
        if continuing > 0 {
            notes.append("\(continuing == 1 ? "1 shot continues" : "\(continuing) shots continue") the one before: it "
                + "starts from that clip's last frame and ends on its own keyframe.")
        }
        if !talking.isEmpty {
            let how: String = switch engine {
            case .wanS2V:
                "Talking shots are animated from their lines by Wan 2.2 S2V (lips in sync, 40 steps): several minutes "
                    + "of GPU time each."
            case .longcat:
                "Talking shots are animated from their lines by LongCat Avatar (lips in sync, 8 steps). With two "
                    + "speakers, each keeps the side of the frame the shot list gave them (else the first to speak is "
                    + "taken to be on the left)."
            case .infiniteTalk:
                "Talking shots are made like the others, then InfiniteTalk re-syncs the speaker to their lines (lips, "
                    + "head and body; the camera move stays)."
            }
            notes.append(how + " Turn lip-sync off on this page to animate them like the others.")
        }
        if engine == .infiniteTalk {
            let long = talking.filter { $0.duration > Shot.durationRange.upperBound + 1e-9 }
                .compactMap { number(of: $0, in: doc.movie) }
            if !long.isEmpty {
                notes.append("Shot \(long.map(String.init).joined(separator: ", ")) is longer than a clip can be "
                    + "(7.5 s): past the clip's end its last frame holds while the lines finish.")
            }
        }
        let unvoiced = shots.filter { $0.talking && !MovieShots.usesLipSync($0, in: doc.movie) }.count
        if unvoiced > 0, doc.movie.lipSync {
            notes.append("\(unvoiced == 1 ? "1 talking shot has" : "\(unvoiced) talking shots have") no spoken lines yet "
                + "(Voices), so it's animated without lip-sync.")
        }
        func avg(_ w: [Double]) -> Double { w.isEmpty ? 0 : w.reduce(0, +) / Double(w.count) }
        gateImages(doc, title: shots.count == 1 ? "Make a Clip" : "Make Clips", work: [
            ImageWork(label: plain.count == 1 ? "1 clip" : "\(plain.count) clips", task: "wan22-i2v", jobs: plain.count,
                      megapixels: avg(work)),
            ImageWork(label: engine == .infiniteTalk
                        ? (talking.count == 1 ? "1 lip-sync" : "\(talking.count) lip-syncs")
                        : (talking.count == 1 ? "1 talking clip" : "\(talking.count) talking clips"),
                      task: engine.task, jobs: talking.count, megapixels: avg(talkWork)),
        ], extraNotes: notes) { [weak self] in
            self?.makeClips(doc, shots: shots.map(\.id))
        }
    }

    /// Make Clips with MiniMax H3: talking shots from references (keyframe, the speakers' sheets and voices), the
    /// rest from their start picture; each clip with its own sound.
    private func requestH3Clips(_ doc: MovieDocument, shots asked: [Shot]) {
        // Room for the lines first (H3 speaks them inside the clip), so the sheet prices the real lengths.
        let unfit = MovieH3.fitDurations(&doc.movie).compactMap { id in doc.movie.shots.first { $0.id == id } }
            .compactMap { number(of: $0, in: doc.movie) }
        let shots = asked.compactMap { s in doc.movie.shots.first { $0.id == s.id } }
        let talking = shots.filter { MovieH3.usesReferences($0, in: doc.movie) }
        let plain = shots.filter { !MovieH3.usesReferences($0, in: doc.movie) }
        let size = doc.movie.videoSize
        func avg(_ w: [Double]) -> Double { w.isEmpty ? 0 : w.reduce(0, +) / Double(w.count) }
        var notes = ["Clips are \(size.width)×\(size.height) at 24 fps with their own sound (lines, effects, "
                     + "ambience), by MiniMax H3's fast 8-step mode; each is at least 5 s, and the edit plays the "
                     + "shot's length.",
                     "MiniMax H3's licence excludes the US, EU, UK and Korea without MiniMax's own licence; products "
                     + "made with it must show \"MiniMax H3\"."]
        if !talking.isEmpty {
            notes.append("Talking shots are made from their keyframe, the speakers' character sheets and their voices "
                         + "(Voices), so each character speaks in their own voice.")
        }
        let voiceless = shots.filter { !MovieH3.speakers($0).isEmpty && !MovieH3.usesReferences($0, in: doc.movie) }.count
        if voiceless > 0 {
            notes.append("\(voiceless == 1 ? "1 shot's speakers have" : "\(voiceless) shots' speakers have") no voice yet: "
                         + "H3 makes one up from the voice description.")
        }
        if !unfit.isEmpty {
            notes.append("The lines of shot \(unfit.map(String.init).joined(separator: ", ")) are longer than a clip can "
                         + "be (15 s): shorten them, or split the shot.")
        }
        gateImages(doc, title: shots.count == 1 ? "Make a Clip" : "Make Clips", work: [
            ImageWork(label: plain.count == 1 ? "1 clip" : "\(plain.count) clips", task: "h3-video", jobs: plain.count,
                      megapixels: avg(plain.map { MovieH3.work($0, in: doc.movie) })),
            ImageWork(label: talking.count == 1 ? "1 talking clip" : "\(talking.count) talking clips",
                      task: "h3-references", jobs: talking.count,
                      megapixels: avg(talking.map { MovieH3.work($0, in: doc.movie) })),
        ], extraNotes: notes) { [weak self] in
            self?.makeClips(doc, shots: shots.map(\.id))
        }
    }

    private func number(of shot: Shot, in movie: MovieProject) -> Int? {
        movie.shots.firstIndex { $0.id == shot.id }.map { $0 + 1 }
    }

    private func makeClips(_ doc: MovieDocument, shots ids: [UUID]) {
        // A continuing shot starts from the clip before's last frame, so it waits for that clip; the rest run at once.
        let engine = doc.movie.lipSyncEngine
        let h3 = doc.movie.videoEngine == .minimaxH3
        let tasks = ids.map { id -> String in
            guard let shot = doc.movie.shots.first(where: { $0.id == id }) else { return "wan22-i2v" }
            if h3 { return MovieH3.usesReferences(shot, in: doc.movie) ? "h3-references" : "h3-video" }
            return MovieShots.usesLipSync(shot, in: doc.movie) ? engine.task : "wan22-i2v"
        }
        runImageJob(doc, title: "Making clips", count: ids.count, chains: MovieShots.chains(ids, in: doc.movie),
                    task: { tasks[$0] }) { [weak self] doc, i, server in
            guard let self, let shot = doc.movie.shots.first(where: { $0.id == ids[i] }),
                  let ends = MovieShots.ends(for: shot, in: doc.movie) else { return }
            let number = (doc.movie.shots.firstIndex { $0.id == shot.id } ?? 0) + 1
            doc.job?.detail = "Shot \(number) (\(i + 1) of \(ids.count))"
            let folder = self.movieStore.folder(doc.id)
            let start: URL
            switch ends.start {
            case .keyframe(let file):
                start = folder.appendingPathComponent(file)
            case .lastFrame(let clip):
                start = folder.appendingPathComponent(MovieShots.lastFrameFile(of: clip))
                if !FileManager.default.fileExists(atPath: start.path) {
                    let frame = try await MovieComposer.frame(of: folder.appendingPathComponent(clip), at: .last)
                    try MovieComposer.writePNG(frame, to: start)
                }
            }
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let lipSync = !h3 && MovieShots.usesLipSync(shot, in: doc.movie)
            let render: MovieArt.Render
            let template: MovieTemplate
            if h3, tasks[i] == "h3-references", let key = shot.selectedKeyframeFile {
                template = self.movieTemplate(MovieH3.TemplateName.references, for: doc) { MovieH3.referencesTemplate() }
                render = MovieH3.referencesRender(shot, movie: doc.movie, folder: folder,
                                                  keyframe: folder.appendingPathComponent(key), seed: seed, server: server)
            } else if h3 {
                template = self.movieTemplate(MovieH3.TemplateName.video, for: doc) { MovieH3.videoTemplate() }
                render = MovieH3.videoRender(shot, movie: doc.movie, folder: folder, startImage: start, ends: ends,
                                             seed: seed, server: server)
            } else if lipSync {
                // The shot's lines as one recording as long as the shot: what the lips follow.
                let voice = folder.appendingPathComponent(MovieShots.voiceTrackFile(for: shot))
                let incoming = MovieVoices.incoming(doc.movie)[shot.id] ?? 0
                try MovieVoices.voiceTrack(shot, folder: folder, duration: shot.duration, incoming: incoming, to: voice)
                let name = MovieShots.TemplateName.talking(engine)
                template = self.movieTemplate(name, for: doc) { MovieShots.talkingTemplate(for: engine) }
                switch engine {
                case .wanS2V:
                    render = MovieShots.talkingRender(shot, movie: doc.movie, startImage: start, voice: voice,
                                                      seed: seed, server: server)
                case .longcat:
                    // Two speakers: one track each (left first, by the scene's blocking), on the shot's timeline.
                    let speakers = MovieShots.leftToRight(MovieVoices.speakers(shot),
                                                          scene: doc.movie.scenes.first { $0.id == shot.scene })
                    var voices = [voice]
                    if speakers.count >= 2 {
                        voices = try speakers.prefix(2).enumerated().map { k, character in
                            let url = folder.appendingPathComponent(MovieShots.voiceTrackFile(for: shot, speaker: k))
                            try MovieVoices.voiceTrack(shot, folder: folder, duration: shot.duration,
                                                       incoming: incoming, speaker: character, to: url)
                            return url
                        }
                    }
                    render = MovieShots.longCatRender(shot, movie: doc.movie, startImage: start, voices: voices,
                                                      seed: seed, server: server)
                case .infiniteTalk:
                    render = MovieShots.dubRender(shot, movie: doc.movie, folder: folder, startImage: start,
                                                  ends: ends, voice: voice, seed: seed, server: server)
                }
            } else {
                render = MovieShots.render(shot, movie: doc.movie, folder: folder, startImage: start, ends: ends,
                                           seed: seed, server: server)
                template = self.movieTemplate(MovieShots.TemplateName.shot, for: doc) { MovieShots.template() }
            }
            let (outputs, jobs) = try await self.runMovieGraph(doc, template: template, render: render,
                                                               title: "\(doc.movie.title): shot \(number)")
            guard let made = outputs.first(where: { $0.kind == .video })?.url else { throw MovieVideoError.noClip }
            let base = "shots/shot-\(shot.id.uuidString.prefix(8).lowercased())-\(UUID().uuidString.prefix(6).lowercased())"
            let file = base + ".mp4", poster = base + ".png"
            let url = folder.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: made, to: url)
            if let still = try? await MovieComposer.frame(of: url, at: .middle) {
                try? MovieComposer.writePNG(still, to: folder.appendingPathComponent(poster))
            }
            let size = doc.movie.aspect.videoSize
            let clipWork = h3 ? MovieH3.work(shot, in: doc.movie)
                : MovieShots.work(width: size.width, height: size.height, frames: MovieShots.frames(for: shot.duration))
            let talkWork = MovieShots.talkingWork(width: size.width, height: size.height, seconds: shot.duration)
            // A dub graph runs the ordinary clip and the lip-sync: each job learns its own task's timing.
            var cost: Double?
            for job in jobs {
                let c = self.recordJobs([job], doc: doc, work: lipSync && job.task != "wan22-i2v" ? talkWork : clipWork)
                if let c { cost = (cost ?? 0) + c }
            }
            guard let j = doc.movie.shots.firstIndex(where: { $0.id == shot.id }) else { return }
            var take = Take(file: file, poster: poster, seed: seed, costUSD: cost, engine: doc.movie.videoEngine.rawValue)
            if h3 && !shot.dialogue.isEmpty { take.voice = try? await MovieSpeech.spans(in: url) }
            doc.movie.shots[j].videos.append(take)
            if doc.movie.shots[j].selectedVideo == nil { doc.movie.shots[j].selectedVideo = take.id }
        }
    }

    func selectClip(_ doc: MovieDocument, shot id: UUID, take: UUID) {
        guard let j = doc.movie.shots.firstIndex(where: { $0.id == id }), doc.movie.shots[j].selectedVideo != take
        else { return }
        doc.movie.shots[j].selectedVideo = take
        doc.movie.shots[j].trimStart = 0
        doc.movie.shots[j].trimEnd = 0
        doc.reopen(.video, shot: id)
        saveMovie(doc)
    }

    // MARK: edit

    /// The edit's timeline, with each chosen clip's real length.
    /// The edit's timeline: the whole film, or one sequence (`sequence`) on its own.
    func movieTimeline(_ doc: MovieDocument, sequence: String? = nil) async -> MovieTimeline {
        let folder = movieStore.folder(doc.id)
        await findClipVoices(doc, in: folder)
        let movie = sequence.map { doc.movie.excerpt(sequence: $0) } ?? doc.movie
        let files = movie.shots.compactMap(\.selectedVideoFile)
        let durations = await MovieComposer.durations(of: files, in: folder)
        return MovieTimeline(movie: movie, durations: durations)
    }

    /// Finds where the chosen H3 clips of talking shots speak, for takes made before that was recorded, and saves.
    private func findClipVoices(_ doc: MovieDocument, in folder: URL) async {
        var found = false
        for shot in doc.movie.shots where !shot.dialogue.isEmpty {
            guard let take = shot.selectedVideoTake, take.voice == nil,
                  take.engine == VideoEngine.minimaxH3.rawValue,
                  let spans = try? await MovieSpeech.spans(in: folder.appendingPathComponent(take.file)),
                  let j = doc.movie.shots.firstIndex(where: { $0.id == shot.id }),
                  let k = doc.movie.shots[j].videos.firstIndex(where: { $0.id == take.id }) else { continue }
            doc.movie.shots[j].videos[k].voice = spans
            found = true
        }
        if found { saveMovie(doc) }
    }

    /// The export's file name: "<title>.mp4", or "<title> – <sequence>.mp4" for one sequence.
    func exportName(_ doc: MovieDocument, sequence: String? = nil) -> String {
        var name = doc.movie.title
        if let sequence, let i = doc.movie.sequences.firstIndex(where: { $0.id == sequence }) {
            let q = doc.movie.sequences[i]
            name += " – " + (q.title.trimmingCharacters(in: .whitespaces).isEmpty ? "Sequence \(i + 1)" : q.title)
        }
        return name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-") + ".mp4"
    }

    /// Edit › Export: the film (or one sequence of it, `sequence`) as an MP4 at `url` (nil: `renders/<name>.mp4` in
    /// the movie's folder), with a chapter per sequence and an .srt beside it. Runs as the movie's job; returns where
    /// it wrote.
    @discardableResult
    func exportMovie(_ doc: MovieDocument, to url: URL? = nil, codec: MovieComposer.Codec = .h264,
                     sequence: String? = nil) async -> URL? {
        guard doc.job == nil else { return nil }
        doc.error = nil
        let url = url ?? movieStore.folder(doc.id, .renders).appendingPathComponent(exportName(doc, sequence: sequence))
        doc.job = MovieJob(title: "Exporting \(url.lastPathComponent)", detail: codec.title, progress: 0)
        defer { doc.job = nil }
        do {
            let timeline = await movieTimeline(doc, sequence: sequence)
            let composition = try await MovieComposer.build(timeline, folder: movieStore.folder(doc.id),
                                                            size: doc.movie.videoSize)
            try await MovieComposer.export(composition, to: url, codec: codec,
                                           burnSubtitles: doc.movie.mix.burnSubtitles) { p in
                Task { @MainActor in doc.job?.progress = p }
            }
            // The lines as subtitles beside the film (players pick up a same-named .srt).
            let srt = url.deletingPathExtension().appendingPathExtension("srt")
            if timeline.subtitles.isEmpty { try? FileManager.default.removeItem(at: srt) }
            else { try MovieComposer.writeSRT(timeline.subtitles, to: srt, speakers: doc.movie.mix.speakerNames) }
            doc.lastExport = url
            return url
        } catch {
            doc.error = "\(error)"
            return nil
        }
    }

    /// Asks where to save, then exports the film (or one sequence).
    func exportMovieAs(_ doc: MovieDocument, codec: MovieComposer.Codec, sequence: String? = nil) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = exportName(doc, sequence: sequence)
        panel.directoryURL = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { @MainActor in
            if let out = await exportMovie(doc, to: url, codec: codec, sequence: sequence) {
                NSWorkspace.shared.activateFileViewerSelecting([out])
            }
        }
    }

    /// Moves a shot to another place in the film (the edit's drag and drop).
    func moveShot(_ doc: MovieDocument, _ id: UUID, before target: UUID?) {
        guard let from = doc.movie.shots.firstIndex(where: { $0.id == id }) else { return }
        let shot = doc.movie.shots.remove(at: from)
        let to = target.flatMap { t in doc.movie.shots.firstIndex { $0.id == t } } ?? doc.movie.shots.endIndex
        doc.movie.shots.insert(shot, at: to)
        doc.movie.reopen(.edit)
        saveMovie(doc)
    }
}

enum MovieVideoError: Error, CustomStringConvertible {
    case noClip
    var description: String { "the template gave no clip (it needs a Preview Video or Save Video at the end)" }
}
