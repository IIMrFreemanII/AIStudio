import AVFoundation
import QuartzCore
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Turns a `MovieTimeline` into an `AVComposition` (two alternating video tracks, so crossfades can overlap, with
/// opacity ramps for the fades) and a video composition at 24 fps, which both the edit's player and the MP4 export
/// use. Works from the clip files; nothing is decoded into memory.
public enum MovieComposer {
    public enum Codec: String, CaseIterable, Sendable, Identifiable {
        case h264, hevc
        public var id: String { rawValue }
        public var title: String { self == .h264 ? "H.264" : "HEVC" }
    }

    public enum ComposerError: Error, CustomStringConvertible {
        case empty
        case noVideo(String)
        case exportFailed(String)
        case noFrame(String)
        case noAudio(String)

        public var description: String {
            switch self {
            case .empty: "No shot has a chosen clip yet"
            case .noVideo(let f): "\(f) has no video track"
            case .exportFailed(let why): "The export failed: \(why)"
            case .noFrame(let f): "Couldn't read a frame of \(f)"
            case .noAudio(let f): "\(f) has no audio track"
            }
        }
    }

    /// The film's frame rate: Wan's 16 fps clips play at their own speed, each frame shown 1–2 times.
    public static let fps: Int32 = 24
    /// A timescale every clip time lands on exactly (16 and 24 fps frames, half-second fades).
    static let timescale: CMTimeScale = 9600

    static func time(_ seconds: Double) -> CMTime {
        CMTime(value: CMTimeValue((seconds * Double(timescale)).rounded()), timescale: timescale)
    }

    /// The composition and how to draw it; give both to an `AVPlayerItem` or the export.
    public struct Composition: @unchecked Sendable {
        public let asset: AVMutableComposition
        public let video: AVMutableVideoComposition
        /// The sound's volumes (nil when the film has no sound).
        public let audioMix: AVMutableAudioMix?
        public let subtitles: [MovieTimeline.Subtitle]
        public var duration: Double { asset.duration.seconds }
    }

    /// Each clip file's length, as the timeline needs them.
    public static func durations(of files: [String], in folder: URL) async -> [String: Double] {
        var out: [String: Double] = [:]
        for file in Set(files) {
            if let d = await duration(of: folder.appendingPathComponent(file)) { out[file] = d }
        }
        return out
    }

    /// A clip file's video length (the video track's, not counting any longer soundtrack).
    public static func duration(of url: URL) async -> Double? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let range = try? await track.load(.timeRange) else { return nil }
        return range.duration.seconds
    }

    public static func build(_ timeline: MovieTimeline, folder: URL, size: (width: Int, height: Int)) async throws
        -> Composition
    {
        guard !timeline.clips.isEmpty else { throw ComposerError.empty }
        let comp = AVMutableComposition()
        let render = CGSize(width: size.width, height: size.height)
        var tracks: [AVMutableCompositionTrack] = []
        for _ in 0..<2 {
            guard let t = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
            else { throw ComposerError.exportFailed("couldn't add a video track") }
            tracks.append(t)
        }
        var transforms: [CGAffineTransform] = []
        for clip in timeline.clips {
            let asset = AVURLAsset(url: folder.appendingPathComponent(clip.file))
            guard let source = try await asset.loadTracks(withMediaType: .video).first else {
                throw ComposerError.noVideo(clip.file)
            }
            let (natural, preferred, range) = try await source.load(.naturalSize, .preferredTransform, .timeRange)
            let track = tracks[clip.track]
            let at = time(clip.start)
            let trackEnd = track.timeRange.end
            if trackEnd < at {
                track.insertEmptyTimeRange(CMTimeRange(start: trackEnd, end: at))
            }
            let wanted = CMTimeRange(start: range.start + time(clip.trimStart), duration: time(clip.duration))
            try track.insertTimeRange(wanted.intersection(range), of: source, at: at)
            transforms.append(fill(natural, preferred, into: render))
        }

        let video = AVMutableVideoComposition()
        video.renderSize = render
        video.frameDuration = CMTime(value: 1, timescale: fps)
        video.instructions = timeline.segments.map { seg in
            let instruction = AVMutableVideoCompositionInstruction()
            instruction.timeRange = CMTimeRange(start: time(seg.start), end: time(seg.end))
            instruction.backgroundColor = CGColor(gray: 0, alpha: 1)
            instruction.layerInstructions = seg.layers.map { layer in
                let clip = timeline.clips[layer.clip]
                let li = AVMutableVideoCompositionLayerInstruction(assetTrack: tracks[clip.track])
                li.setTransform(transforms[layer.clip], at: instruction.timeRange.start)
                if layer.opacityFrom < 1 || layer.opacityTo < 1 {
                    li.setOpacityRamp(fromStartOpacity: Float(layer.opacityFrom), toEndOpacity: Float(layer.opacityTo),
                                      timeRange: instruction.timeRange)
                } else {
                    li.setOpacity(1, at: instruction.timeRange.start)
                }
                return li
            }
            return instruction
        }

        // Sound: one composition track per kind and lane, with volume ramps from the timeline's volume points.
        var audioTracks: [String: AVMutableCompositionTrack] = [:]
        var params: [String: AVMutableAudioMixInputParameters] = [:]
        var order: [String] = []
        let limit = time(timeline.duration)
        for sound in timeline.sounds.sorted(by: { $0.start < $1.start }) {
            let asset = AVURLAsset(url: folder.appendingPathComponent(sound.file))
            guard let source = try await asset.loadTracks(withMediaType: .audio).first else {
                throw ComposerError.noAudio(sound.file)
            }
            let range = try await source.load(.timeRange)
            let key = "\(sound.kind.rawValue)-\(sound.track)"
            if audioTracks[key] == nil {
                guard let t = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
                else { throw ComposerError.exportFailed("couldn't add an audio track") }
                audioTracks[key] = t
                params[key] = AVMutableAudioMixInputParameters(track: t)
                order.append(key)
            }
            let track = audioTracks[key]!
            let at = time(sound.start)
            guard at < limit else { continue }
            if track.timeRange.end < at { track.insertEmptyTimeRange(CMTimeRange(start: track.timeRange.end, end: at)) }
            let wanted = CMTimeRange(start: range.start + time(sound.offset), duration: min(time(sound.duration), limit - at))
            let part = wanted.intersection(range)
            guard part.duration > .zero else { continue }
            try track.insertTimeRange(part, of: source, at: at)
            let points = timeline.volume(for: sound)
            guard let first = points.first else { continue }
            params[key]!.setVolume(Float(first.gain), at: time(first.time))
            for (a, b) in zip(points, points.dropFirst()) {
                let from = time(a.time), to = min(time(b.time), limit)
                guard to > from else { continue }
                params[key]!.setVolumeRamp(fromStartVolume: Float(a.gain), toEndVolume: Float(b.gain),
                                           timeRange: CMTimeRange(start: from, end: to))
            }
        }
        var mix: AVMutableAudioMix?
        if !order.isEmpty {
            mix = AVMutableAudioMix()
            mix!.inputParameters = order.compactMap { params[$0] }
        }
        return Composition(asset: comp, video: video, audioMix: mix, subtitles: timeline.subtitles)
    }

    /// Scales a source to cover the render size and centres it (a clip of another shape is cropped, not stretched).
    static func fill(_ natural: CGSize, _ preferred: CGAffineTransform, into size: CGSize) -> CGAffineTransform {
        let rect = CGRect(origin: .zero, size: natural).applying(preferred)
        let shown = CGSize(width: abs(rect.width), height: abs(rect.height))
        guard shown.width > 0, shown.height > 0 else { return .identity }
        let s = max(size.width / shown.width, size.height / shown.height)
        return preferred
            .concatenating(CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .concatenating(CGAffineTransform(scaleX: s, y: s))
            .concatenating(CGAffineTransform(translationX: (size.width - shown.width * s) / 2,
                                             y: (size.height - shown.height * s) / 2))
    }

    /// Writes the film as an MP4. `progress` hears 0…1 while it runs.
    public static func export(_ composition: Composition, to url: URL, codec: Codec = .h264,
                              burnSubtitles: Bool = false, progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        let preset = codec == .hevc ? AVAssetExportPresetHEVCHighestQuality : AVAssetExportPresetHighestQuality
        guard let session = AVAssetExportSession(asset: composition.asset, presetName: preset) else {
            throw ComposerError.exportFailed("no export session for \(codec.title)")
        }
        session.videoComposition = composition.video
        session.audioMix = composition.audioMix
        if burnSubtitles, !composition.subtitles.isEmpty {
            session.videoComposition = burned(composition.video, subtitles: composition.subtitles)
        }
        session.shouldOptimizeForNetworkUse = true
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: url)
        let watcher = Task {
            for await state in session.states(updateInterval: 0.25) {
                if case .exporting(let p) = state { progress(p.fractionCompleted) }
            }
        }
        defer { watcher.cancel() }
        do {
            try await session.export(to: url, as: .mp4)
        } catch {
            throw ComposerError.exportFailed(error.localizedDescription)
        }
        progress(1)
    }

    /// A copy of the video composition with the subtitles drawn over the picture. Only for the export: an `AVPlayer`
    /// refuses a composition with an animation tool.
    static func burned(_ video: AVMutableVideoComposition, subtitles: [MovieTimeline.Subtitle])
        -> AVMutableVideoComposition
    {
        let copy = video.mutableCopy() as! AVMutableVideoComposition
        let size = copy.renderSize
        let parent = CALayer()
        parent.frame = CGRect(origin: .zero, size: size)
        let videoLayer = CALayer()
        videoLayer.frame = parent.frame
        parent.addSublayer(videoLayer)
        let fontSize = size.height * 0.055
        for s in subtitles {
            let layer = CATextLayer()
            layer.string = s.text
            layer.fontSize = fontSize
            layer.foregroundColor = CGColor(gray: 1, alpha: 1)
            layer.alignmentMode = .center
            layer.isWrapped = true
            layer.contentsScale = 2
            layer.shadowColor = CGColor(gray: 0, alpha: 1)
            layer.shadowOpacity = 0.9
            layer.shadowRadius = 3
            layer.shadowOffset = .zero
            let width = size.width * 0.9
            let height = fontSize * 3.4
            layer.frame = CGRect(x: (size.width - width) / 2, y: size.height * 0.06, width: width, height: height)
            layer.opacity = 0
            let show = CABasicAnimation(keyPath: "opacity")
            show.fromValue = 1
            show.toValue = 1
            show.beginTime = AVCoreAnimationBeginTimeAtZero + s.start
            show.duration = max(0.01, s.end - s.start)
            show.isRemovedOnCompletion = false
            show.fillMode = .removed
            layer.add(show, forKey: "show")
            parent.addSublayer(layer)
        }
        copy.animationTool = AVVideoCompositionCoreAnimationTool(postProcessingAsVideoLayer: videoLayer, in: parent)
        return copy
    }

    public static func writeSRT(_ subtitles: [MovieTimeline.Subtitle], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try MovieSubtitles.srt(subtitles).write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: frames

    public enum FramePosition: Sendable {
        case first, middle, last
    }

    /// One frame of a clip, exactly (no nearby keyframe instead).
    public static func frame(of url: URL, at position: FramePosition) async throws -> CGImage {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ComposerError.noVideo(url.lastPathComponent)
        }
        let (range, rate) = try await track.load(.timeRange, .nominalFrameRate)
        let frame = 1 / Double(rate > 0 ? rate : 16)
        let t: Double = switch position {
        case .first: range.start.seconds
        case .middle: range.start.seconds + range.duration.seconds / 2
        case .last: range.end.seconds - frame / 2
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: frame, preferredTimescale: timescale)
        generator.requestedTimeToleranceAfter = .zero
        do {
            return try await generator.image(at: CMTime(seconds: max(0, t), preferredTimescale: timescale)).image
        } catch {
            throw ComposerError.noFrame(url.lastPathComponent)
        }
    }

    public static func writePNG(_ image: CGImage, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
