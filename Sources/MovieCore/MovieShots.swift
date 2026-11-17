import Foundation
import GraphRuntime

/// The Shots stage: each shot's clip, animated from its keyframe by Wan 2.2 image-to-video. A shot that continues
/// the one before starts from that clip's last frame and ends on its own keyframe, so the moment carries on without a
/// cut and still lands where the shot was planned.
public enum MovieShots {
    /// Wan 2.2 makes 16 frames a second, 4k+1 frames a clip, at most 121 (7.5 s).
    public static let fps = 16.0
    public static let maxFrames = 121

    /// The clip length for a shot's planned duration: the nearest 4k+1 frames, 5…121.
    public static func frames(for duration: Double) -> Int {
        let k = Int((duration * fps / 4).rounded())
        return min(maxFrames, max(5, k * 4 + 1))
    }

    /// What the clip shows moving: the shot's motion, its camera move and the style's motion words.
    public static func motionPrompt(for shot: Shot, in movie: MovieProject) -> String {
        let style = MovieStyle.named(movie.style)
        var parts: [String] = []
        var motion = shot.motionPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if motion.isEmpty { motion = shot.keyframePrompt.trimmingCharacters(in: .whitespacesAndNewlines) }
        if !motion.isEmpty { parts.append(motion.hasSuffix(".") ? motion : motion + ".") }
        let camera = shot.camera.trimmingCharacters(in: .whitespacesAndNewlines)
        if !camera.isEmpty { parts.append("Camera: \(camera).") }
        parts.append(style.motionSuffix)
        return parts.joined(separator: " ")
    }

    /// Where a clip starts and ends.
    public struct Ends: Equatable, Sendable {
        public enum Start: Equatable, Sendable {
            /// The shot's chosen keyframe (relative file).
            case keyframe(String)
            /// The last frame of this clip (relative file): the shot before, which this one continues.
            case lastFrame(of: String)
        }
        public var start: Start
        /// The keyframe to end on (a continuing shot's own), or nil.
        public var end: String?
    }

    /// A continuing shot (in the same scene, after a shot with a chosen clip) starts from that clip's last frame and
    /// ends on its keyframe; any other starts from its keyframe. Nil: no keyframe chosen yet.
    public static func ends(for shot: Shot, in movie: MovieProject) -> Ends? {
        guard let key = shot.selectedKeyframeFile else { return nil }
        if shot.continuesPrevious, let prev = movie.shot(before: shot), prev.scene == shot.scene,
           let clip = prev.selectedVideoFile {
            return Ends(start: .lastFrame(of: clip), end: key)
        }
        return Ends(start: .keyframe(key), end: nil)
    }

    /// Which of `ids` must render in order, as runs of their indices (each in film order): a shot that continues the
    /// shot just before it (in its scene) starts from that shot's new keyframe or clip when both are in the batch, so it joins its
    /// run; every other shot is a run of its own and can render at the same time as the rest.
    public static func chains(_ ids: [UUID], in movie: MovieProject) -> [[Int]] {
        let position = Dictionary(movie.shots.enumerated().map { ($1.id, $0) }, uniquingKeysWith: { a, _ in a })
        var chains: [[Int]] = []
        var chainOf: [UUID: Int] = [:]
        for (i, id) in ids.enumerated().sorted(by: { (position[$0.1] ?? .max) < (position[$1.1] ?? .max) }) {
            if let shot = movie.shots.first(where: { $0.id == id }), shot.continuesPrevious,
               let prev = movie.shot(before: shot), prev.scene == shot.scene, let c = chainOf[prev.id] {
                chains[c].append(i)
                chainOf[id] = c
            } else {
                chainOf[id] = chains.count
                chains.append([i])
            }
        }
        return chains
    }

    // MARK: template

    public enum TemplateName {
        public static let shot = "shot-i2v"
        public static let talking = "shot-s2v"
        public static let talkingLongCat = "shot-longcat"
        public static let talkingDub = "shot-infinitetalk"

        /// The talking-shot template of a lip-sync model.
        public static func talking(_ engine: LipSyncEngine) -> String {
            switch engine {
            case .wanS2V: talking
            case .longcat: talkingLongCat
            case .infiniteTalk: talkingDub
            }
        }
    }

    /// The slots of the shot template: the clip's settings on the Wan node, and the paths of its two Load Images.
    public static let videoSlots = ["prompt", "negative", "seed", "width", "height", "length", "server"]

    /// Load Image (start) and Load Image (end) into Wan 2.2 Image to Video (Cloud GPU) into a Preview Video.
    public static func template(name: String = TemplateName.shot) -> MovieTemplate {
        let start = GraphNode(id: "start", type: "LoadImage", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 60))
        let end = GraphNode(id: "end", type: "LoadImage", inputs: ["path": .value(.string(""))],
                            position: GraphPoint(x: 40, y: 220))
        let wan = GraphNode(id: "video", type: "RemoteWanImageToVideo", inputs: [
            "image": .link(node: "start", output: "image"), "last_image": .link(node: "end", output: "image"),
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(832)), "height": .value(.int(480)), "length": .value(.int(81)),
            "fast": .value(.bool(true)), "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "video", output: "video")],
                            position: GraphPoint(x: 720, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: videoSlots.map { ($0, MovieTemplate.Slot(node: "video", input: $0)) })
        slots["start"] = .init(node: "start", input: "path")
        slots["end"] = .init(node: "end", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [start, end, wan, out]), slots: slots)
    }

    /// A clip's values: `startImage` is the picture to start from (the keyframe, or the previous clip's last frame
    /// written out as a file), absolute; the end picture is left out (its Load Image removed) unless the shot has one.
    public static func render(_ shot: Shot, movie: MovieProject, folder: URL, startImage: URL, ends: Ends, seed: Int,
                              server: String) -> MovieArt.Render {
        let size = movie.aspect.videoSize
        var values: [String: JSONValue] = [
            "prompt": .string(motionPrompt(for: shot, in: movie)),
            "negative": .string(MovieStyle.named(movie.style).negative), "seed": .int(seed),
            "width": .int(size.width), "height": .int(size.height), "length": .int(frames(for: shot.duration)),
            "server": .string(server), "start": .string(startImage.path),
        ]
        if let end = ends.end { values["end"] = .string(folder.appendingPathComponent(end).path) }
        return MovieArt.Render(values: values, unusedPictures: ends.end == nil ? ["end"] : [])
    }

    /// Load Image (start) and Load Audio (voice) into Wan 2.2 Speech to Video (Cloud GPU) into a Preview Video.
    public static func talkingTemplate(name: String = TemplateName.talking) -> MovieTemplate {
        let start = GraphNode(id: "start", type: "LoadImage", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 60))
        let voice = GraphNode(id: "voice", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 220))
        let wan = GraphNode(id: "video", type: "RemoteWanSpeechToVideo", inputs: [
            "image": .link(node: "start", output: "image"), "audio": .link(node: "voice", output: "audio"),
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(832)), "height": .value(.int(480)), "steps": .value(.int(0)),
            "guidance": .value(.double(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "video", output: "video")],
                            position: GraphPoint(x: 720, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: ["prompt", "negative", "seed", "width", "height", "server"]
            .map { ($0, MovieTemplate.Slot(node: "video", input: $0)) })
        slots["start"] = .init(node: "start", input: "path")
        slots["voice"] = .init(node: "voice", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [start, voice, wan, out]), slots: slots)
    }

    /// A talking shot's values: Wan animates `startImage` to the voice track (`MovieVoices.voiceTrack`), absolute paths.
    public static func talkingRender(_ shot: Shot, movie: MovieProject, startImage: URL, voice: URL, seed: Int,
                                     server: String) -> MovieArt.Render {
        let size = movie.aspect.videoSize
        return MovieArt.Render(values: [
            "prompt": .string(talkingPrompt(for: shot, in: movie)),
            "negative": .string(MovieStyle.named(movie.style).negative), "seed": .int(seed),
            "width": .int(size.width), "height": .int(size.height), "server": .string(server),
            "start": .string(startImage.path), "voice": .string(voice.path),
        ], unusedPictures: [])
    }

    /// The talking-shot template of the movie's lip-sync model (built in; a movie can keep an edited copy).
    public static func talkingTemplate(for engine: LipSyncEngine) -> MovieTemplate {
        switch engine {
        case .wanS2V: talkingTemplate()
        case .longcat: longCatTemplate()
        case .infiniteTalk: dubTemplate()
        }
    }

    /// The prompt of a talking shot: its motion, and that the person speaks.
    static func talkingPrompt(for shot: Shot, in movie: MovieProject) -> String {
        motionPrompt(for: shot, in: movie) + " The person speaks, their lips moving with the words."
    }

    /// Load Image (start) and Load Audio (voice, and voice2 for a second speaker) into LongCat Avatar Talking Video
    /// (Cloud GPU) into a Preview Video.
    public static func longCatTemplate(name: String = TemplateName.talkingLongCat) -> MovieTemplate {
        let start = GraphNode(id: "start", type: "LoadImage", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 60))
        let voice = GraphNode(id: "voice", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 220))
        let voice2 = GraphNode(id: "voice2", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                               position: GraphPoint(x: 40, y: 380))
        let avatar = GraphNode(id: "video", type: "RemoteLongCatAvatar", inputs: [
            "image": .link(node: "start", output: "image"), "audio": .link(node: "voice", output: "audio"),
            "audio2": .link(node: "voice2", output: "audio"), "prompt": .value(.string("")), "seed": .value(.int(0)),
            "resolution": .value(.string("480p")), "steps": .value(.int(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "video", output: "video")],
                            position: GraphPoint(x: 720, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: ["prompt", "seed", "server"]
            .map { ($0, MovieTemplate.Slot(node: "video", input: $0)) })
        slots["start"] = .init(node: "start", input: "path")
        slots["voice"] = .init(node: "voice", input: "path")
        slots["voice2"] = .init(node: "voice2", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [start, voice, voice2, avatar, out]), slots: slots)
    }

    /// A LongCat talking shot's values: `voices` are one track per speaker in the order they first speak (the first
    /// is taken to be the left-hand person, the second the right-hand one); with one, the second Load Audio goes.
    public static func longCatRender(_ shot: Shot, movie: MovieProject, startImage: URL, voices: [URL], seed: Int,
                                     server: String) -> MovieArt.Render {
        var values: [String: JSONValue] = [
            "prompt": .string(talkingPrompt(for: shot, in: movie)), "seed": .int(seed), "server": .string(server),
            "start": .string(startImage.path), "voice": .string(voices[0].path),
        ]
        if voices.count > 1 { values["voice2"] = .string(voices[1].path) }
        return MovieArt.Render(values: values, unusedPictures: voices.count > 1 ? [] : ["voice2"])
    }

    /// The shot template (Wan 2.2 image to video, start and end pictures) with InfiniteTalk Lip-Sync (Cloud GPU)
    /// after it: the clip is made as usual, then the speaker re-synced to the voice track.
    public static func dubTemplate(name: String = TemplateName.talkingDub) -> MovieTemplate {
        var base = template(name: name)
        let voice = GraphNode(id: "voice", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 340, y: 380))
        let talk = GraphNode(id: "talk", type: "RemoteInfiniteTalk", inputs: [
            "video": .link(node: "video", output: "video"), "audio": .link(node: "voice", output: "audio"),
            "prompt": .value(.string("")), "seed": .value(.int(0)), "resolution": .value(.string("480p")),
            "steps": .value(.int(0)), "audio_guidance": .value(.double(0)), "server": .value(.string("")),
        ], position: GraphPoint(x: 720, y: 220))
        var nodes = base.graph.nodes.filter { $0.id != "out" }
        nodes += [voice, talk, GraphNode(id: "out", type: "PreviewVideo",
                                         inputs: ["video": .link(node: "talk", output: "video")],
                                         position: GraphPoint(x: 1080, y: 240))]
        base.graph = Graph(nodes: nodes)
        base.slots["voice"] = .init(node: "voice", input: "path")
        base.slots["talk_prompt"] = .init(node: "talk", input: "prompt")
        base.slots["talk_seed"] = .init(node: "talk", input: "seed")
        base.slots["talk_server"] = .init(node: "talk", input: "server")
        return base
    }

    /// An InfiniteTalk shot's values: the ordinary clip's (`render`) plus the voice track and the dub's settings.
    public static func dubRender(_ shot: Shot, movie: MovieProject, folder: URL, startImage: URL, ends: Ends,
                                 voice: URL, seed: Int, server: String) -> MovieArt.Render {
        var r = render(shot, movie: movie, folder: folder, startImage: startImage, ends: ends, seed: seed,
                       server: server)
        r.values["voice"] = .string(voice.path)
        r.values["talk_prompt"] = .string(talkingPrompt(for: shot, in: movie))
        r.values["talk_seed"] = .int(seed)
        r.values["talk_server"] = .string(server)
        return r
    }

    /// Whether the shot is animated from its voice: lip sync is on, it's a talking shot, and has voiced lines.
    public static func usesLipSync(_ shot: Shot, in movie: MovieProject) -> Bool {
        movie.lipSync && shot.talking && !MovieVoices.layout(shot).isEmpty
    }

    /// Where a shot's voice track is written (relative to the movie's folder); `speaker` k: the k-th speaker's own.
    public static func voiceTrackFile(for shot: Shot, speaker: Int? = nil) -> String {
        "audio/shot-\(shot.id.uuidString.lowercased().prefix(8))-voice\(speaker.map { "-\($0 + 1)" } ?? "").wav"
    }

    /// GPU work of a talking clip: like `work`, for `seconds` of video at 16 fps.
    public static func talkingWork(width: Int, height: Int, seconds: Double) -> Double {
        work(width: width, height: height, frames: Int((seconds * fps).rounded(.up)))
    }

    /// Where a clip's last frame is written, to start the shot that continues it.
    public static func lastFrameFile(of clip: String) -> String {
        (clip as NSString).deletingPathExtension + "-last.png"
    }

    /// GPU work of a clip for the learned timings: megapixels × frames, since time grows with both.
    public static func work(width: Int, height: Int, frames: Int) -> Double {
        Double(width * height * frames) / 1e6
    }
}

extension Shot {
    /// Whether the shot has a clip made by `engine`, so a switch of engine offers to make the missing ones.
    public func hasClip(by engine: VideoEngine) -> Bool {
        videos.contains { ($0.engine.flatMap(VideoEngine.init(rawValue:)) ?? .wan22) == engine }
    }

    public var selectedVideoFile: String? {
        videos.first { $0.id == selectedVideo }?.file
    }

    public var selectedVideoTake: Take? {
        videos.first { $0.id == selectedVideo }
    }
}

extension MovieProject {
    /// Every shot has a chosen clip.
    public var videosAreReady: Bool { !shots.isEmpty && shots.allSatisfy { $0.selectedVideoFile != nil } }
}
