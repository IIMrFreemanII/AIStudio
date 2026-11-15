import AppKit
import Foundation
import MovieCore
import NodesRemote

/// The Shots and Edit stages: each shot's clip (Wan 2.2 image-to-video on a GPU server, from its keyframe or, for a
/// continuing shot, from the clip before's last frame), behind the cost gate; then the edit's composition and the
/// MP4 export, which run on this Mac.
extension AppModel {
    // MARK: clips

    /// Shots › Make Clips: one take per shot (`shots` nil: those without a clip yet, else all), in film order so a
    /// continuing shot starts from its predecessor's new clip.
    func requestClips(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let missing = doc.movie.shots.filter { $0.videos.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? doc.movie.shots.map(\.id) : missing)
        let shots = doc.movie.shots.filter { ids.contains($0.id) }
        guard !shots.isEmpty else { return }
        let lacking = shots.filter { $0.selectedKeyframeFile == nil }.compactMap { number(of: $0, in: doc.movie) }
        guard lacking.isEmpty else {
            doc.error = "Choose a keyframe for shot \(lacking.map(String.init).joined(separator: ", ")) first (Keyframes)."
            return
        }
        let size = doc.movie.aspect.videoSize
        let work = shots.map { MovieShots.work(width: size.width, height: size.height,
                                               frames: MovieShots.frames(for: $0.duration)) }
        var notes = ["Clips are \(size.width)×\(size.height) at 16 fps, Wan 2.2's fast 4-step mode."]
        let continuing = shots.filter(\.continuesPrevious).count
        if continuing > 0 {
            notes.append("\(continuing == 1 ? "1 shot continues" : "\(continuing) shots continue") the one before: it "
                + "starts from that clip's last frame and ends on its own keyframe.")
        }
        let talking = shots.filter(\.talking).count
        if talking > 0 {
            notes.append("Talking shots are animated without lip-sync for now: voices and lip-sync come in a later update.")
        }
        gateImages(doc, title: shots.count == 1 ? "Make a Clip" : "Make Clips", work: [
            ImageWork(label: shots.count == 1 ? "1 clip" : "\(shots.count) clips", task: "wan22-i2v", jobs: shots.count,
                      megapixels: work.reduce(0, +) / Double(work.count)),
        ], extraNotes: notes) { [weak self] in
            self?.makeClips(doc, shots: shots.map(\.id))
        }
    }

    private func number(of shot: Shot, in movie: MovieProject) -> Int? {
        movie.shots.firstIndex { $0.id == shot.id }.map { $0 + 1 }
    }

    private func makeClips(_ doc: MovieDocument, shots ids: [UUID]) {
        runImageJob(doc, title: "Making clips", count: ids.count) { [weak self] doc, i, server in
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
            let render = MovieShots.render(shot, movie: doc.movie, folder: folder, startImage: start, ends: ends,
                                           seed: seed, server: server)
            let template = self.movieTemplate(MovieShots.TemplateName.shot, for: doc) { MovieShots.template() }
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
            let cost = self.recordJobs(jobs, doc: doc, work: MovieShots.work(
                width: size.width, height: size.height, frames: MovieShots.frames(for: shot.duration)))
            guard let j = doc.movie.shots.firstIndex(where: { $0.id == shot.id }) else { return }
            let take = Take(file: file, poster: poster, seed: seed, costUSD: cost)
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
        doc.movie.reopen(.video)
        saveMovie(doc)
    }

    // MARK: edit

    /// The edit's timeline, with each chosen clip's real length.
    func movieTimeline(_ doc: MovieDocument) async -> MovieTimeline {
        let files = doc.movie.shots.compactMap(\.selectedVideoFile)
        let durations = await MovieComposer.durations(of: files, in: movieStore.folder(doc.id))
        return MovieTimeline(movie: doc.movie, durations: durations)
    }

    /// Edit › Export: the film as an MP4 at `url` (nil: `renders/<title>.mp4` in the movie's folder). Runs as the
    /// movie's job; returns where it wrote.
    @discardableResult
    func exportMovie(_ doc: MovieDocument, to url: URL? = nil, codec: MovieComposer.Codec = .h264) async -> URL? {
        guard doc.job == nil else { return nil }
        doc.error = nil
        let name = doc.movie.title.replacingOccurrences(of: "/", with: "-")
        let url = url ?? movieStore.folder(doc.id, .renders).appendingPathComponent(name + ".mp4")
        doc.job = MovieJob(title: "Exporting \(url.lastPathComponent)", detail: codec.title, progress: 0)
        defer { doc.job = nil }
        do {
            let timeline = await movieTimeline(doc)
            let composition = try await MovieComposer.build(timeline, folder: movieStore.folder(doc.id),
                                                            size: doc.movie.aspect.videoSize)
            try await MovieComposer.export(composition, to: url, codec: codec) { p in
                Task { @MainActor in doc.job?.progress = p }
            }
            doc.lastExport = url
            return url
        } catch {
            doc.error = "\(error)"
            return nil
        }
    }

    /// Asks where to save, then exports.
    func exportMovieAs(_ doc: MovieDocument, codec: MovieComposer.Codec) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = doc.movie.title.replacingOccurrences(of: "/", with: "-") + ".mp4"
        panel.directoryURL = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { @MainActor in
            if let out = await exportMovie(doc, to: url, codec: codec) {
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
