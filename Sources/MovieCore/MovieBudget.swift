import Foundation

/// What making the rest of a movie will roughly cost with its engines: every stage's GPU work not done yet (or, before
/// the shot list, as many shots as the target length suggests), from the learned timings. One cold start and an idle
/// tail per task and sequence, since a long film is made a sequence at a time. An estimate for the budget, not a quote:
/// each run's cost sheet has the real figure.
public struct FilmEstimate: Sendable, Equatable {
    public struct Part: Sendable, Equatable, Identifiable {
        public var id: String { label }
        public var label: String
        public var usd: Double
    }

    public var parts: [Part]
    public var usd: Double { parts.reduce(0) { $0 + $1.usd } }

    /// The GPU each task runs on by default (the server's `GPU_*` settings) and its price per second, with the H3
    /// classes' large memory requests added.
    public static let taskPrice: [String: Double] = {
        let h100 = MovieCost.gpuPrices["H100"]!, l40s = MovieCost.gpuPrices["L40S"]!
        let h3 = 6.20 / 3600
        return [
            "qwen-image": h100, "qwen-image-edit": h100, "wan22-i2v": h100, "wan22-s2v": h100,
            "longcat-avatar": h100, "infinitetalk": h100, "qwen3-tts": l40s, "breeze-tts": l40s,
            "stable-audio-3": l40s, "ace-step-xl": l40s, "h3-video": h3, "h3-references": h3,
        ]
    }()

    /// A sequence's idle tail per task (the server keeps a GPU up a while after its last job).
    public static let idleSeconds = 120.0
    /// Words spoken per second, for line lengths before they're voiced.
    public static let wordsPerSecond = 2.5

    public init(parts: [Part]) { self.parts = parts }

    public init(movie: MovieProject, timings: GPUTimings) {
        let sequences = Double(max(1, movie.sequences.count))
        var parts: [Part] = []
        func add(_ label: String, task: String, jobs: [Double], runs: Double = sequences) {
            guard !jobs.isEmpty, let price = Self.taskPrice[task] else { return }
            let work = jobs.reduce(0) { $0 + timings.seconds(task: task, megapixels: $1) }
            let start = min(Double(jobs.count), runs) * (timings.loadSeconds(task: task) + Self.idleSeconds)
            parts.append(Part(label: label, usd: (work + start) * price))
        }
        func words(_ s: String) -> Double { Double(s.split(whereSeparator: \.isWhitespace).count) }

        // The shots: as planned, or about one per 4.5 s of the target with a line in every other one.
        let planned = !movie.shots.isEmpty
        var shots = movie.shots
        if !planned {
            let n = max(1, Int((Double(movie.targetSeconds) / 4.5).rounded()))
            shots = (0..<n).map { i in
                Shot(scene: "", duration: 4.5,
                     dialogue: i % 2 == 0 ? [DialogueLine(character: "", text: "a line of about eight words said here")] : [],
                     talking: i % 2 == 0)
            }
        }

        // Cast: a sheet for each character without one.
        let cast = movie.characters.filter { $0.selectedSheetFile == nil }.count
        let castCount = movie.characters.isEmpty && !planned ? 3 : cast
        add("Character sheets", task: "qwen-image", jobs: Array(repeating: 1.5, count: castCount), runs: 1)

        // Keyframes.
        let size = movie.aspect.keyframeSize
        let kmp = Double(size.width * size.height) / 1e6
        add("Keyframes", task: "qwen-image-edit",
            jobs: shots.filter { $0.selectedKeyframeFile == nil }.map { _ in kmp })

        // Clips.
        let engine = movie.videoEngine
        let needClip = shots.filter { !$0.hasClip(by: engine) }
        if engine == .minimaxH3 {
            let jobs = needClip.map { MovieH3.work($0, in: movie) }
            add("Clips (MiniMax H3)", task: "h3-video", jobs: jobs)
        } else {
            let v = movie.aspect.videoSize
            let talking = needClip.filter { $0.talking && movie.lipSync && !$0.dialogue.isEmpty }
            let silent = needClip.filter { !($0.talking && movie.lipSync && !$0.dialogue.isEmpty) }
            add("Clips (Wan 2.2)", task: "wan22-i2v", jobs: silent.map {
                MovieShots.work(width: v.width, height: v.height, frames: Int((min($0.duration, 7.5) * 16).rounded(.up)) + 1)
            })
            add("Lip-synced clips (\(movie.lipSyncEngine.title))", task: movie.lipSyncEngine.task, jobs: talking.map {
                MovieShots.talkingWork(width: v.width, height: v.height, seconds: $0.duration)
            })
            // Voices: samples, then every line.
            let speakers = Set(shots.flatMap { $0.dialogue.map(\.character) })
            let samples = planned
                ? movie.characters.filter { speakers.contains($0.id) && $0.selectedVoiceSample == nil }.count
                : max(2, movie.characters.count)
            let task = movie.voiceEngine.task
            add("Voices", task: task, jobs: Array(repeating: 6, count: samples), runs: 1)
            add("Lines", task: task, jobs: shots.flatMap { s in
                s.dialogue.filter { $0.selectedTake == nil }.map { max(1, words($0.text) / Self.wordsPerSecond) }
            })
            // Effects: one per shot, as long as the shot.
            add("Effects", task: "stable-audio-3", jobs: shots.filter { $0.selectedSound == nil }.map(\.duration))
        }

        // Music: one cue per scene (or per minute of the target before the script), as long as the scene.
        if planned {
            var seconds: [String: Double] = [:]
            for s in shots { seconds[s.scene, default: 0] += s.duration }
            let missing = seconds.filter { scene, _ in
                movie.music.first { $0.id == scene }?.selectedTake == nil
            }
            add("Music", task: "ace-step-xl", jobs: missing.values.map { $0 + 2 })
        } else {
            let n = max(1, movie.targetSeconds / 60)
            add("Music", task: "ace-step-xl",
                jobs: Array(repeating: Double(movie.targetSeconds) / Double(n) + 2, count: n))
        }
        self.parts = parts
    }
}

extension MovieProject {
    /// Whether spending `usd` more would take the movie past its budget.
    public func wouldExceedBudget(_ usd: Double) -> Bool {
        guard let budget else { return false }
        return costSpent + usd > budget + 0.005
    }

    /// Said on a cost sheet whose run would pass the budget; nil when it wouldn't (or there's none).
    public func budgetWarning(for usd: Double) -> String? {
        guard let budget, wouldExceedBudget(usd) else { return nil }
        return "This run would take the movie past its \(CostEstimate.format(budget)) budget: "
            + "\(CostEstimate.format(costSpent)) spent + \(CostEstimate.format(usd)) = \(CostEstimate.format(costSpent + usd))."
    }
}
