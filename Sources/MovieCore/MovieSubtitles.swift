import Foundation

/// Subtitles as SubRip text.
public enum MovieSubtitles {
    public static func srt(_ subtitles: [MovieTimeline.Subtitle]) -> String {
        subtitles.sorted { $0.start < $1.start }.enumerated().map { i, s in
            "\(i + 1)\n\(stamp(s.start)) --> \(stamp(s.end))\n\(s.shown)\n\n"
        }.joined()
    }

    static func stamp(_ t: Double) -> String {
        let ms = Int((max(0, t) * 1000).rounded())
        return String(format: "%02d:%02d:%02d,%03d", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
    }
}
