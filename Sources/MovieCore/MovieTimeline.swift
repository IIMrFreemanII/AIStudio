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

    // MARK: audio

    public enum SoundKind: String, Sendable, CaseIterable { case dialogue, effects, music }

    /// A piece of sound on the timeline: a line, a shot's effects, or a scene's score.
    public struct Sound: Sendable, Equatable, Identifiable {
        /// "line-<line uuid>", "fx-<shot uuid>", "music-<scene id>".
        public var id: String
        public var kind: SoundKind
        /// Relative to the movie's folder.
        public var file: String
        public var label: String
        /// On the timeline.
        public var start: Double
        /// Into the file.
        public var offset: Double
        public var duration: Double
        /// Linear: the mix's gain for its kind × the take's loudness correction.
        public var gain: Double
        public var fadeIn: Double
        public var fadeOut: Double
        /// The lane within its kind; sounds on one lane never overlap.
        public var track: Int

        public var end: Double { start + duration }
    }

    public struct Subtitle: Sendable, Equatable {
        public var start: Double
        public var end: Double
        public var speaker: String
        public var text: String
    }

    public struct VolumePoint: Sendable, Equatable {
        public var time: Double
        public var gain: Double
    }

    public var sounds: [Sound] = []
    public var subtitles: [Subtitle] = []
    /// The music's gain while someone speaks.
    public var duck: Double = 0.3

    /// How long the music takes to dip and to come back.
    public static let duckRamp = 0.25

    /// The sound's gain over time as piecewise-linear points (absolute timeline times): its gain × its fades, and for
    /// music × the dip under dialogue (lines closer than half a second count as one).
    public func volume(for sound: Sound) -> [VolumePoint] {
        var spans: [(Double, Double)] = []
        if sound.kind == .music {
            for d in sounds.filter({ $0.kind == .dialogue }).sorted(by: { $0.start < $1.start }) {
                if let last = spans.last, d.start - last.1 < 0.5 {
                    spans[spans.count - 1].1 = max(last.1, d.end)
                } else {
                    spans.append((d.start, d.end))
                }
            }
        }
        let ramp = Self.duckRamp
        func duckFactor(_ t: Double) -> Double {
            var f = 1.0
            for (a, b) in spans {
                if t >= a && t <= b { f = min(f, duck) }
                else if t > a - ramp && t < a { f = min(f, 1 + (duck - 1) * (t - (a - ramp)) / ramp) }
                else if t > b && t < b + ramp { f = min(f, duck + (1 - duck) * (t - b) / ramp) }
            }
            return f
        }
        var times: [Double] = [sound.start, sound.end, sound.start + sound.fadeIn, sound.end - sound.fadeOut]
        for (a, b) in spans { times += [a - ramp, a, b, b + ramp] }
        times = times.filter { $0 >= sound.start - 1e-9 && $0 <= sound.end + 1e-9 }
            .map { max(sound.start, min(sound.end, $0)) }.sorted()
        var unique: [Double] = []
        for t in times where unique.last.map({ t - $0 >= 1e-6 }) ?? true { unique.append(t) }
        if let last = unique.last, abs(last - sound.end) >= 1e-6 { unique.append(sound.end) }
        return unique.map { t in
            let local = t - sound.start
            var fade = 1.0
            if sound.fadeIn > 0 { fade = min(fade, local / sound.fadeIn) }
            if sound.fadeOut > 0 { fade = min(fade, (sound.duration - local) / sound.fadeOut) }
            return VolumePoint(time: t, gain: sound.gain * max(0, min(1, fade)) * duckFactor(t))
        }
    }

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
        duck = movie.mix.duck
        buildSound(movie: movie, durations: durations)
    }

    private static func linear(_ db: Double?) -> Double { pow(10, (db ?? 0) / 20) }

    private mutating func buildSound(movie: MovieProject, durations: [String: Double]) {
        func length(_ take: Take) -> Double? { take.seconds ?? durations[take.file] }
        var out: [Sound] = []
        var subs: [Subtitle] = []
        for c in clips {
            guard let shot = movie.shots.first(where: { $0.id == c.shot }) else { continue }
            for p in MovieVoices.layout(shot) {
                guard let line = shot.dialogue.first(where: { $0.id == p.line }),
                      let take = line.takes.first(where: { $0.id == line.selectedTake }) else { continue }
                let t0 = c.start - c.trimStart + p.start
                let lo = max(t0, c.start), hi = min(t0 + p.seconds, c.end)
                guard hi - lo >= 0.05 else { continue }
                let name = movie.character(line.character)?.name ?? line.character
                out.append(Sound(id: "line-\(line.id.uuidString)", kind: .dialogue, file: take.file, label: name, start: lo,
                                 offset: lo - t0, duration: hi - lo, gain: movie.mix.dialogue * Self.linear(take.gainDB),
                                 fadeIn: 0.02, fadeOut: 0.05, track: 0))
                subs.append(Subtitle(start: lo, end: hi, speaker: name,
                                     text: line.text.trimmingCharacters(in: .whitespacesAndNewlines)))
            }
            if let take = shot.sounds.first(where: { $0.id == shot.selectedSound }), let len = length(take) {
                let d = min(c.duration, len - c.trimStart)
                if d > 0.05 {
                    let n = (movie.shots.firstIndex { $0.id == shot.id } ?? 0) + 1
                    out.append(Sound(id: "fx-\(shot.id.uuidString)", kind: .effects, file: take.file, label: "Shot \(n)",
                                     start: c.start, offset: c.trimStart, duration: d,
                                     gain: movie.mix.effects * Self.linear(take.gainDB), fadeIn: max(c.fadeIn, 0.05),
                                     fadeOut: max(c.fadeOut, 0.15), track: 0))
                }
            }
        }
        for cue in movie.music {
            guard let take = cue.selectedTake, let len = length(take) else { continue }
            let sceneClips = clips.filter { c in movie.shots.first { $0.id == c.shot }?.scene == cue.id }
            guard let first = sceneClips.map(\.start).min(), let last = sceneClips.map(\.end).max() else { continue }
            let d = min(last - first, len)
            guard d > 0 else { continue }
            let heading = movie.scenes.first { $0.id == cue.id }?.heading ?? cue.id
            out.append(Sound(id: "music-\(cue.id)", kind: .music, file: take.file, label: heading, start: first, offset: 0,
                             duration: d, gain: movie.mix.music * Self.linear(take.gainDB),
                             fadeIn: min(first < 0.01 ? 0.5 : 1.0, d / 3), fadeOut: min(1.5, d / 3), track: 0))
        }
        // Lanes, per kind: the lowest one free at the sound's start.
        for kind in SoundKind.allCases {
            var ends: [Double] = []
            for i in out.indices.filter({ out[$0].kind == kind }).sorted(by: { out[$0].start < out[$1].start }) {
                if let lane = ends.firstIndex(where: { $0 <= out[i].start + 1e-6 }) {
                    out[i].track = lane
                    ends[lane] = out[i].end
                } else {
                    out[i].track = ends.count
                    ends.append(out[i].end)
                }
            }
        }
        sounds = out
        subtitles = subs.sorted { $0.start < $1.start }
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
