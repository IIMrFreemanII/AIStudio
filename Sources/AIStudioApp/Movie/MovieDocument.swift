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
    /// How many renders the next job may send at once, as chosen on its cost sheet (nil: as many as the server runs).
    var parallelLimit: Int?

    /// Where the film was last exported (this session).
    var lastExport: URL?

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
///
/// Several renders of a job can run at once, each in its own lane (`MovieJob.$lane.withValue(n)`): `detail` reads
/// and writes the current lane's line, and the job bar shows every lane's (`lines`).
struct MovieJob {
    var title: String
    /// 0…1 when known.
    var progress: Double?
    var task: Task<Void, Never>?
    private var lanes: [Int: String] = [:]

    @TaskLocal static var lane = 0

    init(title: String, detail: String = "", progress: Double? = nil) {
        self.title = title
        self.progress = progress
        if !detail.isEmpty { lanes[0] = detail }
    }

    /// What the current lane's render is doing ("Shot 3 (2 of 10) · step 2/4").
    var detail: String {
        get { lanes[Self.lane] ?? "" }
        set { lanes[Self.lane] = newValue.isEmpty ? nil : newValue }
    }

    /// Every running render's line, in lane order.
    var lines: [String] { lanes.sorted { $0.key < $1.key }.map(\.value) }

    mutating func endLane(_ lane: Int) { lanes[lane] = nil }
}

/// A cost confirmation on screen: the estimate, and what to run when the user approves.
struct CostGate: Identifiable {
    let id = UUID()
    var estimate: CostEstimate
    var run: @MainActor () -> Void
    /// Told the chosen plan's `parallel` before `run`, when the estimate offers plans.
    var choose: @MainActor (Int) -> Void = { _ in }
}
