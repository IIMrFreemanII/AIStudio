import Foundation
import MovieCore

/// A movie's takes (clips, keyframes, sheets, sounds) in the full-window viewer: ← → between a row's takes, Esc
/// closes. Nothing plays underneath while it's open.
extension AppModel {
    /// Opens `files` (relative to the movie's folder; missing ones left out) at `index`.
    func openMovieViewer(_ doc: MovieDocument, files: [String], titles: [String], index: Int) {
        let folder = movieStore.folder(doc.id)
        let items = zip(files, titles).enumerated().compactMap { i, item -> (Int, URL, String)? in
            let url = folder.appendingPathComponent(item.0)
            return FileManager.default.fileExists(atPath: url.path) ? (i, url, item.1) : nil
        }
        guard !items.isEmpty else { return }
        MovieSoundPlayer.shared.stop()
        let at = items.firstIndex { $0.0 >= index } ?? items.count - 1
        chatViewer = ChatViewerItem(urls: items.map(\.1), index: at, titles: items.map(\.2))
    }

    /// A row of takes, titled "`name` · Take N", at `take`.
    func openMovieViewer(_ doc: MovieDocument, takes: [Take], name: String, take: UUID?) {
        let items = MovieViewerItems.takes(takes, name: name)
        openMovieViewer(doc, files: items.files, titles: items.titles,
                        index: MovieViewerItems.index(of: take, in: takes))
    }

    /// A shot's clips (`take` nil: the chosen one).
    func viewShotClips(_ doc: MovieDocument, shot id: UUID, take: UUID? = nil) {
        guard let (shot, n) = shotNumbered(doc, id) else { return }
        openMovieViewer(doc, takes: shot.videos, name: "Shot \(n)", take: take ?? shot.selectedVideo)
    }

    /// A shot's keyframes (`take` nil: the chosen one).
    func viewKeyframes(_ doc: MovieDocument, shot id: UUID, take: UUID? = nil) {
        guard let (shot, n) = shotNumbered(doc, id) else { return }
        openMovieViewer(doc, takes: shot.keyframes, name: "Shot \(n) keyframe", take: take ?? shot.selectedKeyframe)
    }

    /// A character's reference sheets (`take` nil: the chosen one).
    func viewSheets(_ doc: MovieDocument, character id: String, take: UUID? = nil) {
        guard let c = doc.movie.characters.first(where: { $0.id == id }) else { return }
        openMovieViewer(doc, takes: c.sheets, name: "\(c.name) sheet", take: take ?? c.selectedSheet)
    }

    private func shotNumbered(_ doc: MovieDocument, _ id: UUID) -> (Shot, Int)? {
        guard let i = doc.movie.shots.firstIndex(where: { $0.id == id }) else { return nil }
        return (doc.movie.shots[i], i + 1)
    }
}
