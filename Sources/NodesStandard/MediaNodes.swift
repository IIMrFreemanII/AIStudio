import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore

/// `<prefix>_00001.<ext>` numbering in `dir`, continuing from the highest existing number for that extension.
func nextMediaFilename(in dir: URL, prefix: String, ext: String) -> URL {
    let existing = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    let used = existing.compactMap { name -> Int? in
        guard name.hasPrefix(prefix + "_"), name.hasSuffix("." + ext) else { return nil }
        return Int(name.dropFirst(prefix.count + 1).dropLast(ext.count + 1))
    }
    return dir.appendingPathComponent(String(format: "%@_%05d.%@", prefix, (used.max() ?? 0) + 1, ext))
}

func mediaMetadata(_ context: ExecutionContext) -> [String: String] {
    [StandardNodes.graphMetadataKey: context.graph.jsonString(), "aistudio.node": context.nodeID]
}

/// Temporary folder for previews (cleared at launch by the app).
public let previewDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("AIStudio-previews")

// MARK: test signals

/// A moving colour pattern, for checking video plumbing without a model.
public struct TestPatternNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "TestPattern", displayName: "Test Pattern", category: "video",
        inputs: [
            InputSpec("width", .int, default: .int(512), widget: .number(min: 16, max: 2048, step: 16)),
            InputSpec("height", .int, default: .int(288), widget: .number(min: 16, max: 2048, step: 16)),
            InputSpec("seconds", .float, default: .double(2), widget: .number(min: 0.1, max: 60, step: 0.5)),
            InputSpec("fps", .float, default: .double(24), widget: .number(min: 1, max: 60, step: 1)),
        ],
        outputs: [OutputSpec("video", .video)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let fps = try inputs.float("fps")
        let frames = max(1, Int((try inputs.float("seconds") * fps).rounded()))
        return ["video": .object(MediaOps.testPattern(width: try inputs.int("width"), height: try inputs.int("height"),
                                                      frames: frames, fps: fps))]
    }
}

/// A sine tone, for checking audio plumbing without a model.
public struct TestToneNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "TestTone", displayName: "Test Tone", category: "audio",
        inputs: [
            InputSpec("frequency", .float, default: .double(440), widget: .number(min: 20, max: 20000, step: 10)),
            InputSpec("seconds", .float, default: .double(2), widget: .number(min: 0.1, max: 600, step: 0.5)),
            InputSpec("sample_rate", .int, default: .int(48000), widget: .number(min: 8000, max: 96000, step: 100)),
            InputSpec("channels", .int, default: .int(2), widget: .number(min: 1, max: 2, step: 1)),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["audio": .object(MediaOps.tone(frequency: try inputs.float("frequency"), seconds: try inputs.float("seconds"),
                                        sampleRate: try inputs.int("sample_rate"), channels: try inputs.int("channels")))]
    }
}

// MARK: load

public struct LoadVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoadVideo", displayName: "Load Video", category: "video",
        inputs: [
            InputSpec("path", .string, default: .string(""), widget: .filePath),
            // 0 = every frame.
            InputSpec("max_frames", .int, default: .int(0), widget: .number(min: 0, max: 10000, step: 1), zeroLabel: "all"),
        ],
        outputs: [OutputSpec("video", .video), OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let n = try inputs.int("max_frames")
        let clip = try await MediaIO.readVideo(try StandardNodes.inputFile(try inputs.string("path")), maxFrames: n > 0 ? n : nil)
        let silence = AudioClip(samples: MLXArray.zeros([2, max(1, Int(clip.duration * 48000))]), sampleRate: 48000)
        return ["video": .object(clip), "audio": .object(clip.audio ?? silence)]
    }
}

public struct LoadAudioNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoadAudio", displayName: "Load Audio", category: "audio",
        inputs: [InputSpec("path", .string, default: .string(""), widget: .filePath)],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["audio": .object(try await MediaIO.readAudio(try StandardNodes.inputFile(try inputs.string("path"))))]
    }
}

// MARK: combine

/// `b` mixed into `a`, starting `offset` seconds in (resampled and channel-matched to `a`).
public struct MixAudioNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "MixAudio", displayName: "Mix Audio", category: "audio",
        inputs: [
            InputSpec("a", .audio),
            InputSpec("b", .audio),
            InputSpec("gain", .float, default: .double(1), widget: .number(min: 0, max: 4, step: 0.05)),
            InputSpec("offset", .float, default: .double(0), widget: .number(min: 0, max: 600, step: 0.1)),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let a: AudioClip = try inputs.object("a"), b: AudioClip = try inputs.object("b")
        let m = MediaOps.mix(a, b, gain: Float(try inputs.float("gain")), offset: try inputs.float("offset"))
        eval(m.samples)
        return ["audio": .object(m)]
    }
}

/// Puts `audio` on a video: replacing its soundtrack, or mixed into it. The track is fitted to the video length.
public struct AddAudioToVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "AddAudioToVideo", displayName: "Add Audio to Video", category: "video",
        inputs: [
            InputSpec("video", .video),
            InputSpec("audio", .audio),
            InputSpec("offset", .float, default: .double(0), widget: .number(min: 0, max: 600, step: 0.1)),
            InputSpec("gain", .float, default: .double(1), widget: .number(min: 0, max: 4, step: 0.05)),
            InputSpec("mode", .string, default: .string("replace"), widget: .choice(["replace", "mix"])),
        ],
        outputs: [OutputSpec("video", .video)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        var video: VideoClip = try inputs.object("video")
        let audio: AudioClip = try inputs.object("audio")
        let gain = Float(try inputs.float("gain")), offset = try inputs.float("offset")
        var track = MediaOps.fit(audio, seconds: video.duration, offset: offset, gain: gain)
        if try inputs.string("mode") == "mix", let existing = video.audio {
            track = MediaOps.fit(MediaOps.mix(existing, track), seconds: video.duration)
        }
        eval(track.samples)
        video.audio = track
        return ["video": .object(video)]
    }
}

// MARK: save & preview

public struct SaveVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "SaveVideo", displayName: "Save Video", category: "video",
        inputs: [
            InputSpec("video", .video),
            // Overrides the video's own soundtrack when connected.
            InputSpec("audio", .audio, optional: true),
            InputSpec("filename_prefix", .string, default: .string("aistudio")),
            InputSpec("directory", .string, default: .string("outputs"), widget: .filePath),
            InputSpec("format", .string, default: .string("mp4"), widget: .choice(["mp4", "mov"])),
            InputSpec("codec", .string, default: .string("h264"), widget: .choice(MediaIO.VideoCodec.allCases.map(\.rawValue))),
            InputSpec("quality", .float, default: .double(0.7), widget: .number(min: 0, max: 1, step: 0.05)),
        ],
        outputs: [], isOutput: true, alwaysExecute: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let video: VideoClip = try inputs.object("video")
        let dir = try StandardNodes.outputDirectory(try inputs.string("directory"))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = nextMediaFilename(in: dir, prefix: try inputs.string("filename_prefix"), ext: try inputs.string("format"))
        let audio = inputs.optionalObject("audio", as: AudioClip.self)
        if let source = video.sourceURL, audio == nil {
            // An unchanged download (a remote node's result): keep its encoding, add the graph.
            try await MediaIO.remuxVideo(source, to: url, metadata: mediaMetadata(context))
        } else {
            try await MediaIO.writeVideo(video, audio: audio, to: url,
                                         codec: MediaIO.VideoCodec(rawValue: try inputs.string("codec")) ?? .h264,
                                         quality: try inputs.float("quality"), metadata: mediaMetadata(context))
        }
        return ["video_url": .object(url), "video": .object(video)]
    }
}

public struct SaveAudioNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "SaveAudio", displayName: "Save Audio", category: "audio",
        inputs: [
            InputSpec("audio", .audio),
            InputSpec("filename_prefix", .string, default: .string("aistudio")),
            InputSpec("directory", .string, default: .string("outputs"), widget: .filePath),
            InputSpec("format", .string, default: .string("wav"), widget: .choice(MediaIO.AudioFormat.allCases.map(\.rawValue))),
        ],
        outputs: [], isOutput: true, alwaysExecute: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let audio: AudioClip = try inputs.object("audio")
        let dir = try StandardNodes.outputDirectory(try inputs.string("directory"))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let format = MediaIO.AudioFormat(rawValue: try inputs.string("format")) ?? .wav
        let url = nextMediaFilename(in: dir, prefix: try inputs.string("filename_prefix"), ext: format.rawValue)
        try await MediaIO.writeAudio(audio, to: url, format: format, metadata: mediaMetadata(context))
        return ["audio_url": .object(url), "audio": .object(audio)]
    }
}

/// Shows a video in the app without saving it (written to a temporary file for playback).
public struct PreviewVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "PreviewVideo", displayName: "Preview Video", category: "video",
        inputs: [InputSpec("video", .video)], outputs: [], isOutput: true, alwaysExecute: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let video: VideoClip = try inputs.object("video")
        let folder = try StandardNodes.previewFolder()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent(UUID().uuidString + ".mp4")
        try await MediaIO.writeVideo(video, to: url, metadata: mediaMetadata(context))
        return ["video_url": .object(url), "preview": .bool(true)]
    }
}

public struct PreviewAudioNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "PreviewAudio", displayName: "Preview Audio", category: "audio",
        inputs: [InputSpec("audio", .audio)], outputs: [], isOutput: true, alwaysExecute: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let audio: AudioClip = try inputs.object("audio")
        let folder = try StandardNodes.previewFolder()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent(UUID().uuidString + ".wav")
        try await MediaIO.writeAudio(audio, to: url, format: .wav, metadata: mediaMetadata(context))
        return ["audio_url": .object(url), "preview": .bool(true)]
    }
}
