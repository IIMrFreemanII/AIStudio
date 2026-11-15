import Foundation

/// What the full-window viewer shows for a row of takes: their files in order, each with a title ("Shot 9 · Take 2").
public enum MovieViewerItems {
    public static func takes(_ takes: [Take], name: String) -> (files: [String], titles: [String]) {
        (takes.map(\.file), takes.indices.map { "\(name) · Take \($0 + 1)" })
    }

    /// Where `take` is in `takes` (the first when it isn't there).
    public static func index(of take: UUID?, in takes: [Take]) -> Int {
        take.flatMap { id in takes.firstIndex { $0.id == id } } ?? 0
    }
}

/// Playing a take from a point on its bar.
public enum MoviePlayback {
    /// Seconds into a take of `duration` for a click at `fraction` of its bar, clamped to the take.
    public static func time(at fraction: Double, of duration: Double) -> Double {
        guard duration > 0, fraction.isFinite else { return 0 }
        return min(max(fraction, 0), 1) * duration
    }

    /// Where playing (again) starts: from `position`, or from the top when it's at (or past) the end.
    public static func resume(at position: Double, of duration: Double) -> Double {
        position >= duration - 0.01 || position < 0 ? 0 : position
    }

    /// "0:01.2".
    public static func clock(_ seconds: Double) -> String {
        let s = seconds.isFinite ? max(seconds, 0) : 0
        return String(format: "%d:%04.1f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
    }
}
