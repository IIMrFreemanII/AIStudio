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

    // MARK: template

    public enum TemplateName {
        public static let shot = "shot-i2v"
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
