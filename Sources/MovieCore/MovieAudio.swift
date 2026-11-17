import AVFoundation
import Foundation
import GraphRuntime

public enum MovieAudioError: Error, CustomStringConvertible {
    case noLines
    case unreadable(String)

    public var description: String {
        switch self {
        case .noLines: "No line of this shot has a voice yet"
        case .unreadable(let f): "Couldn't read the audio file \(f)"
        }
    }
}

/// Voices: a character's designed voice sample, each line cloned from it, and the shot's lines laid out in time.
public enum MovieVoices {
    /// The sentence every character's voice is designed saying.
    public static let sampleText = "I remember the way the light fell across the water that morning, quiet and gold, before anyone else was awake."

    public enum TemplateName {
        public static let sample = "voice-sample"
        public static let line = "voice-line"
    }

    /// Seconds of quiet before the first line, between lines, and after the last.
    public static let leadIn = 0.3
    public static let gap = 0.25
    public static let tail = 0.4

    /// The longest a lip-synced shot may be (S2V, LongCat); other shots are capped by `Shot.durationRange`, and so are
    /// InfiniteTalk's, since it dubs the shot's ordinary clip (at most 121 frames).
    public static let maxTalking = 12.0

    /// Text to Speech (voice design, no reference) into a Preview Audio.
    public static func sampleTemplate(name: String = TemplateName.sample) -> MovieTemplate {
        let voice = GraphNode(id: "voice", type: "RemoteTextToSpeech", inputs: [
            "text": .value(.string(sampleText)), "language": .value(.string("Auto")), "voice": .value(.string("designed")),
            "description": .value(.string("")), "seed": .value(.int(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 40, y: 60))
        let out = GraphNode(id: "out", type: "PreviewAudio", inputs: ["audio": .link(node: "voice", output: "audio")],
                            position: GraphPoint(x: 400, y: 80))
        let slots = Dictionary(uniqueKeysWithValues: ["text", "description", "voice", "language", "seed", "server"]
            .map { ($0, MovieTemplate.Slot(node: "voice", input: $0)) })
        return MovieTemplate(name: name, graph: Graph(nodes: [voice, out]), slots: slots)
    }

    /// Load Audio (the character's sample) into Text to Speech as its reference (voice cloning) into a Preview Audio.
    public static func lineTemplate(name: String = TemplateName.line) -> MovieTemplate {
        let reference = GraphNode(id: "reference", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 60))
        let voice = GraphNode(id: "voice", type: "RemoteTextToSpeech", inputs: [
            "reference": .link(node: "reference", output: "audio"),
            "text": .value(.string("")), "reference_text": .value(.string(sampleText)),
            "language": .value(.string("Auto")), "seed": .value(.int(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewAudio", inputs: ["audio": .link(node: "voice", output: "audio")],
                            position: GraphPoint(x: 700, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: ["text", "reference_text", "language", "seed", "server"]
            .map { ($0, MovieTemplate.Slot(node: "voice", input: $0)) })
        slots["reference"] = .init(node: "reference", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [reference, voice, out]), slots: slots)
    }

    /// A character's voice sample: the sample sentence, spoken in a voice designed from their description.
    public static func sampleRender(_ c: MovieCharacter, seed: Int, server: String) -> MovieArt.Render {
        var description = c.voice.trimmingCharacters(in: .whitespacesAndNewlines)
        if description.isEmpty { description = "a clear, natural speaking voice" }
        return MovieArt.Render(values: [
            "text": .string(sampleText), "description": .string(description), "voice": .string("designed"),
            "language": .string("Auto"), "seed": .int(seed), "server": .string(server),
        ], unusedPictures: [])
    }

    /// A line spoken by cloning the character's voice sample (`sample`, absolute), so the voice is the same from line
    /// to line. Cloning can't apply the line's `delivery` ("quietly, scared"); it isn't used here.
    public static func lineRender(_ line: DialogueLine, sample: URL, seed: Int, server: String) -> MovieArt.Render {
        MovieArt.Render(values: [
            "text": .string(line.text.trimmingCharacters(in: .whitespacesAndNewlines)),
            "reference": .string(sample.path), "reference_text": .string(sampleText),
            "language": .string("Auto"), "seed": .int(seed), "server": .string(server),
        ], unusedPictures: [])
    }

    // MARK: layout

    public struct Placed: Equatable, Sendable {
        public var line: UUID
        /// Seconds from the shot's start.
        public var start: Double
        public var seconds: Double
    }

    /// The characters who speak voiced lines in the shot, in the order they first speak.
    public static func speakers(_ shot: Shot) -> [String] {
        var out: [String] = []
        for p in layout(shot) {
            if let c = shot.dialogue.first(where: { $0.id == p.line })?.character, !out.contains(c) { out.append(c) }
        }
        return out
    }

    /// The shot's voiced lines in order: the first after `leadIn`, each next after the one before and a `gap`. Lines
    /// without a chosen take of known length are left out.
    public static func layout(_ shot: Shot) -> [Placed] {
        var out: [Placed] = []
        var t = leadIn
        for line in shot.dialogue {
            guard let take = line.takes.first(where: { $0.id == line.selectedTake }), let s = take.seconds, s > 0
            else { continue }
            out.append(Placed(line: line.id, start: t, seconds: s))
            t += s + gap
        }
        return out
    }

    /// How long the shot must be to say its lines; nil when none is voiced.
    public static func neededDuration(_ shot: Shot) -> Double? {
        layout(shot).last.map { $0.start + $0.seconds + tail }
    }

    public static func maxDuration(_ shot: Shot, lipSync: Bool, engine: LipSyncEngine = .wanS2V) -> Double {
        lipSync && shot.talking && engine != .infiniteTalk ? maxTalking : Shot.durationRange.upperBound
    }

    /// Stretches shots that are too short for their lines (rounded up to 0.1 s, never shortened, never past the
    /// longest the shot can be). Returns the shots whose lines still don't fit.
    @discardableResult
    public static func fitDurations(_ movie: inout MovieProject) -> [UUID] {
        var unfit: [UUID] = []
        let lipSync = movie.lipSync
        for i in movie.shots.indices {
            let shot = movie.shots[i]
            guard let needed = neededDuration(shot) else { continue }
            let cap = maxDuration(shot, lipSync: lipSync, engine: movie.lipSyncEngine)
            if needed > cap + 1e-9 { unfit.append(shot.id) }
            if needed > shot.duration {
                movie.shots[i].duration = max(shot.duration, (min(needed, cap) * 10 - 1e-9).rounded(.up) / 10)
            }
        }
        return unfit
    }

    /// The GPU "size" of an audio job for the learned timings: seconds of audio.
    public static func work(seconds: Double) -> Double { seconds }

    // MARK: voice track

    /// The sample rate Wan's audio encoder reads.
    public static let trackRate = 16_000.0

    /// Writes the shot's lines as one mono 16-bit 16 kHz WAV, `duration` seconds long: quiet except the lines at their
    /// layout times (each scaled by its take's `gainDB`, summed, clipped); a line past `duration` is cut. With
    /// `speaker` (a character's id), only that character's lines: one track per voice, still on the shot's timeline.
    public static func voiceTrack(_ shot: Shot, folder: URL, duration: Double, speaker: String? = nil,
                                  to url: URL) throws {
        let placed = layout(shot).filter { p in
            speaker == nil || shot.dialogue.first { $0.id == p.line }?.character == speaker
        }
        guard !placed.isEmpty else { throw MovieAudioError.noLines }
        let total = max(1, Int((duration * trackRate).rounded()))
        var mix = [Float](repeating: 0, count: total)
        for p in placed {
            guard let line = shot.dialogue.first(where: { $0.id == p.line }),
                  let take = line.takes.first(where: { $0.id == line.selectedTake }) else { continue }
            let samples = try mono16k(folder.appendingPathComponent(take.file))
            let gain = Float(pow(10, (take.gainDB ?? 0) / 20))
            let at = Int((p.start * trackRate).rounded())
            guard at < total else { continue }
            for (k, v) in samples.enumerated() where at + k < total { mix[at + k] += v * gain }
        }
        for i in mix.indices { mix[i] = max(-1, min(1, mix[i])) }

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: url)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: trackRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
        ]
        let out = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: out.processingFormat, frameCapacity: AVAudioFrameCount(total))
        else { throw MovieAudioError.unreadable(url.lastPathComponent) }
        buffer.frameLength = AVAudioFrameCount(total)
        mix.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: total) }
        try out.write(from: buffer)
    }

    /// A whole audio file as 16 kHz mono floats.
    static func mono16k(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw MovieAudioError.unreadable(url.lastPathComponent) }
        let src = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let input = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames),
              let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: trackRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: src, to: dst)
        else { throw MovieAudioError.unreadable(url.lastPathComponent) }
        try file.read(into: input)
        let capacity = AVAudioFrameCount(Double(frames) * trackRate / src.sampleRate) + 4096
        guard let output = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: capacity)
        else { throw MovieAudioError.unreadable(url.lastPathComponent) }
        nonisolated(unsafe) var fed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if fed {
                inputStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error || error != nil { throw MovieAudioError.unreadable(url.lastPathComponent) }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }
}

/// Sound effects and music: templates, prompts and the scores' cues.
public enum MovieSound {
    public enum TemplateName {
        public static let effects = "sound-effects"
        public static let music = "music"
    }

    /// Text to Sound into a Preview Audio.
    public static func effectsTemplate(name: String = TemplateName.effects) -> MovieTemplate {
        let sound = GraphNode(id: "sound", type: "RemoteTextToSound", inputs: [
            "prompt": .value(.string("")), "seconds": .value(.double(5)), "seed": .value(.int(0)),
            "steps": .value(.int(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 40, y: 60))
        let out = GraphNode(id: "out", type: "PreviewAudio", inputs: ["audio": .link(node: "sound", output: "audio")],
                            position: GraphPoint(x: 400, y: 80))
        let slots = Dictionary(uniqueKeysWithValues: ["prompt", "seconds", "seed", "server"]
            .map { ($0, MovieTemplate.Slot(node: "sound", input: $0)) })
        return MovieTemplate(name: name, graph: Graph(nodes: [sound, out]), slots: slots)
    }

    /// Text to Music into a Preview Audio.
    public static func musicTemplate(name: String = TemplateName.music) -> MovieTemplate {
        let music = GraphNode(id: "music", type: "RemoteTextToMusic", inputs: [
            "tags": .value(.string("")), "lyrics": .value(.string("[Instrumental]")), "seconds": .value(.double(30)),
            "seed": .value(.int(0)), "steps": .value(.int(0)), "planner": .value(.bool(true)), "bpm": .value(.int(0)),
            "key": .value(.string("auto")), "time_signature": .value(.string("auto")),
            "language": .value(.string("auto")), "server": .value(.string("")),
        ], position: GraphPoint(x: 40, y: 60))
        let out = GraphNode(id: "out", type: "PreviewAudio", inputs: ["audio": .link(node: "music", output: "audio")],
                            position: GraphPoint(x: 400, y: 80))
        let slots = Dictionary(uniqueKeysWithValues: ["tags", "lyrics", "seconds", "seed", "server"]
            .map { ($0, MovieTemplate.Slot(node: "music", input: $0)) })
        return MovieTemplate(name: name, graph: Graph(nodes: [music, out]), slots: slots)
    }

    /// The shot's sound-effects prompt, nil when it has no `sfx`.
    public static func effectsPrompt(_ shot: Shot) -> String? {
        let sfx = shot.sfx.trimmingCharacters(in: .whitespacesAndNewlines)
        return sfx.isEmpty ? nil : "\(sfx), sound effects only, no music, no speech"
    }

    public static func effectsRender(_ shot: Shot, seed: Int, server: String) -> MovieArt.Render? {
        guard let prompt = effectsPrompt(shot) else { return nil }
        return MovieArt.Render(values: [
            "prompt": .string(prompt), "seconds": .double(max(1, (shot.duration * 2).rounded(.up) / 2)),
            "seed": .int(seed), "server": .string(server),
        ], unusedPictures: [])
    }

    public static func musicPrompt(for scene: MovieScene, style: MovieStyle) -> String {
        let mood = scene.mood.trimmingCharacters(in: .whitespacesAndNewlines)
        return ((mood.isEmpty ? [] : [mood]) + [style.musicTags, "instrumental"]).joined(separator: ", ")
    }

    /// How long a scene's shots run together.
    public static func sceneSeconds(_ scene: String, in movie: MovieProject) -> Double {
        movie.shots.filter { $0.scene == scene }.reduce(0) { $0 + $1.duration }
    }

    /// One cue per scene that has shots; cues already in the movie are kept as they are.
    public static func cues(for movie: MovieProject) -> [MusicCue] {
        let style = MovieStyle.named(movie.style)
        return movie.scenes.compactMap { scene in
            guard movie.shots.contains(where: { $0.scene == scene.id }) else { return nil }
            return movie.music.first { $0.id == scene.id } ?? MusicCue(id: scene.id, prompt: musicPrompt(for: scene, style: style))
        }
    }

    public static func musicRender(_ cue: MusicCue, seconds: Double, seed: Int, server: String) -> MovieArt.Render {
        MovieArt.Render(values: [
            "tags": .string(cue.prompt), "lyrics": .string("[Instrumental]"),
            "seconds": .double(max(10, (seconds + 2).rounded(.up))), "seed": .int(seed), "server": .string(server),
        ], unusedPictures: [])
    }
}

/// Loudness: the level of a recording, so every line and cue starts from the same one.
public enum MovieLoudness {
    public static let target = -20.0

    /// The file's length and the gain (dB, ±20) that brings its level to `target`. The level is the RMS in dBFS of
    /// 50 ms blocks louder than −50 dBFS (a silence gate): not a full LUFS meter. A silent file gets 0 dB.
    public static func measure(_ url: URL) throws -> (seconds: Double, gainDB: Double) {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw MovieAudioError.unreadable(url.lastPathComponent) }
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        let rate = format.sampleRate
        let seconds = Double(file.length) / rate
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return (seconds, 0) }
        try file.read(into: buffer)
        let n = Int(buffer.frameLength), channels = Int(format.channelCount)
        guard let data = buffer.floatChannelData, n > 0, channels > 0 else { return (seconds, 0) }
        var mono = [Float](repeating: 0, count: n)
        for c in 0..<channels { for i in 0..<n { mono[i] += data[c][i] / Float(channels) } }

        let block = max(1, Int(rate * 0.05))
        var sum = 0.0, count = 0
        var i = 0
        while i < n {
            let end = min(n, i + block)
            var s = 0.0
            for k in i..<end { s += Double(mono[k]) * Double(mono[k]) }
            let meanSquare = s / Double(end - i)
            if 10 * log10(max(meanSquare, 1e-20)) > -50 {
                sum += meanSquare
                count += 1
            }
            i = end
        }
        guard count > 0 else { return (seconds, 0) }
        let level = 10 * log10(sum / Double(count))
        return (seconds, max(-20, min(20, target - level)))
    }
}
