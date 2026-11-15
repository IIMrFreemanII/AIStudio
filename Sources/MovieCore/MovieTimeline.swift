import Foundation

/// The edit: the chosen clips laid end to end, with the transitions between them. Pure timing (seconds); the
/// composer turns it into an `AVComposition`, and the timeline view draws it.
///
/// Transitions say how a shot starts: a **cut**; a **crossfade** (it overlaps the end of the shot before and fades
/// in over it); **fade in** (from black); **fade out** (a dip to black: the shot before fades out, this one fades in).
public struct MovieTimeline: Sendable, Equatable {
    public struct Clip: Sendable, Equatable, Identifiable {
        public var id: UUID { shot }
        public var shot: UUID
        /// The clip file (relative to the movie's folder).
        public var file: String
        /// The poster to draw it with (relative), when there is one.
        public var poster: String?
        /// The file's own length.
        public var sourceDuration: Double
        /// Where it starts on the timeline.
        public var start: Double
        /// From where in the file it plays.
        public var trimStart: Double
        /// How long it plays.
        public var duration: Double
        /// Fading in at its start (a crossfade over the clip before, or from black), and out to black at its end.
        public var fadeIn: Double
        public var fadeOut: Double
        public var transition: ShotTransition
        /// Which of the two video tracks (clips alternate, so a crossfade's two clips are on different ones).
        public var track: Int

        public var end: Double { start + duration }

        /// How opaque it is at a time on the timeline (0 outside it).
        public func opacity(at t: Double) -> Double {
            let local = t - start
            guard local >= 0, local <= duration else { return 0 }
            var o = 1.0
            if fadeIn > 0 { o = min(o, local / fadeIn) }
            if fadeOut > 0 { o = min(o, (duration - local) / fadeOut) }
            return max(0, min(1, o))
        }
    }

    /// The length of a crossfade or a fade.
    public static let fade = 0.5
    /// The shortest a clip can be trimmed to.
    public static let minimumClip = 0.5

    public var clips: [Clip]
    /// Shots left out: no chosen clip yet.
    public var missing: [UUID]

    public var duration: Double { clips.map(\.end).max() ?? 0 }

    /// Lays out the shots' chosen clips in film order. `durations` gives each clip file's length (a file it doesn't
    /// list counts as the shot's planned duration).
    public init(movie: MovieProject, durations: [String: Double]) {
        var clips: [Clip] = []
        var missing: [UUID] = []
        var t = 0.0
        for shot in movie.shots {
            guard let take = shot.selectedVideoTake else {
                missing.append(shot.id)
                continue
            }
            let source = durations[take.file] ?? shot.duration
            let trimStart = max(0, min(shot.trimStart, source - Self.minimumClip))
            let trimEnd = max(0, min(shot.trimEnd, source - trimStart - Self.minimumClip))
            let length = max(Self.minimumClip, source - trimStart - trimEnd)
            var clip = Clip(shot: shot.id, file: take.file, poster: take.poster, sourceDuration: source, start: t,
                            trimStart: trimStart, duration: length, fadeIn: 0, fadeOut: 0, transition: shot.transition,
                            track: clips.count % 2)
            switch shot.transition {
            case .cut:
                break
            case .crossfade:
                if let prev = clips.last {
                    let overlap = min(Self.fade, prev.duration / 2, length / 2)
                    clip.start = prev.end - overlap
                    clip.fadeIn = overlap
                }
            case .fadeFromBlack:
                clip.fadeIn = min(Self.fade, length / 2)
            case .fadeToBlack:
                if !clips.isEmpty {
                    clips[clips.count - 1].fadeOut = min(Self.fade, clips[clips.count - 1].duration / 2)
                }
                clip.fadeIn = min(Self.fade, length / 2)
            }
            clips.append(clip)
            t = clip.end
        }
        self.clips = clips
        self.missing = missing
    }

    /// A stretch of the timeline where the same clips show and every opacity changes linearly: what one video
    /// composition instruction covers. `layers` are top first (a clip fading in is above the one it fades over).
    public struct Segment: Sendable, Equatable {
        public struct Layer: Sendable, Equatable {
            public var clip: Int
            public var opacityFrom: Double
            public var opacityTo: Double
        }
        public var start: Double
        public var end: Double
        public var layers: [Layer]
    }

    /// The whole timeline cut at every clip's start, end and fade boundary.
    public var segments: [Segment] {
        var cuts = Set<Double>([0])
        for c in clips {
            cuts.formUnion([c.start, c.end, c.start + c.fadeIn, c.end - c.fadeOut])
        }
        let times = cuts.filter { $0 >= 0 && $0 <= duration }.sorted()
        var out: [Segment] = []
        for (a, b) in zip(times, times.dropFirst()) where b - a > 1e-6 {
            let mid = (a + b) / 2
            let active = clips.indices.filter { clips[$0].start <= mid && mid < clips[$0].end }
                .sorted { clips[$0].start > clips[$1].start }
            out.append(Segment(start: a, end: b, layers: active.map {
                Segment.Layer(clip: $0, opacityFrom: clips[$0].opacity(at: a), opacityTo: clips[$0].opacity(at: b))
            }))
        }
        return out
    }

    /// The clip on top at a time (for the playhead's shot).
    public func clip(at t: Double) -> Clip? {
        clips.last { $0.start <= t && t < $0.end }
    }
}
