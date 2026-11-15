import Foundation
import MovieCore
import Observation

/// An open movie: the project as edited, what it's doing now (writing the script, breaking down a scene…) and the
/// last error. Saved by the session timer when it changed.
@MainActor @Observable
final class MovieDocument: Identifiable {
    let id: UUID
    var movie: MovieProject
    /// What was last written to disk.
    @ObservationIgnored var saved: MovieProject?

    /// The stage on screen (any unlocked one; defaults to the first not approved yet).
    var stage: MovieStage

    /// The job running for this movie, if any: one at a time.
    var job: MovieJob?
    var error: String?

    /// A stage's work waiting for the user's OK on its cost.
    var pendingGate: CostGate?

    init(movie: MovieProject, saved: MovieProject? = nil) {
        id = movie.id
        self.movie = movie
        self.saved = saved
        stage = movie.currentStage == .edit && !movie.approved.contains(.edit) ? .edit : movie.currentStage
        if !stage.isAvailable { stage = MovieStage.allCases.last(where: { $0.isAvailable && movie.isUnlocked($0) }) ?? .idea }
    }

    /// Changed since it was written (the modification date set on save doesn't count).
    var isDirty: Bool {
        guard var s = saved else { return true }
        s.modified = movie.modified
        return s != movie
    }
    var isBusy: Bool { job != nil }
}

/// Work running for a movie: what it is, how far it got, and its task (for Stop).
struct MovieJob {
    var title: String
    var detail: String = ""
    /// 0…1 when known.
    var progress: Double?
    var task: Task<Void, Never>?
}

/// A cost confirmation on screen: the estimate, and what to run when the user approves.
struct CostGate: Identifiable {
    let id = UUID()
    var estimate: CostEstimate
    var run: @MainActor () -> Void
}
