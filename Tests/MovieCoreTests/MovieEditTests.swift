import AVFoundation
import CoreGraphics
import Foundation
import GraphRuntime
@testable import MovieCore
import Testing

// MARK: shots

@Test func videoAndEditStagesFollowKeyframesWhileVoicesAndSoundWait() {
    var m = MovieProject()
    for s in [MovieStage.idea, .script, .shots, .cast, .keyframes] { m.approve(s) }
    #expect(m.currentStage == .video && m.isUnlocked(.video) && !m.isUnlocked(.edit))
    m.approve(.video)
    #expect(m.currentStage == .edit && m.isUnlocked(.edit))
    #expect(MovieStage.video.previous == .keyframes && !MovieStage.voices.isAvailable)
}

@Test func oldShotsDecodeWithoutTrimsAndTakesWithoutPosters() throws {
    var shot = Shot(scene: "s1", videos: [Take(file: "shots/a.mp4")])
    shot.trimStart = 1
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(shot)) as! [String: Any]
    json["trimStart"] = nil
    json["trimEnd"] = nil
    var take = (json["videos"] as! [[String: Any]])[0]
    take["poster"] = nil
    json["videos"] = [take]
    let back = try JSONDecoder().decode(Shot.self, from: JSONSerialization.data(withJSONObject: json))
    #expect(back.trimStart == 0 && back.trimEnd == 0 && back.videos.first?.poster == nil && back.id == shot.id)
}

@Test func clipLengthsAreWansFourKPlusOneFrames() {
    #expect(MovieShots.frames(for: 5) == 81 && MovieShots.frames(for: 4) == 65 && MovieShots.frames(for: 3) == 49)
    #expect(MovieShots.frames(for: 7.5) == 121 && MovieShots.frames(for: 12) == 121 && MovieShots.frames(for: 0.1) == 5)
}

private func clipMovie() -> MovieProject {
    var m = MovieProject(style: "noir")
    let k = (0..<3).map { Take(file: "keyframes/\($0).png") }
    let v = Take(file: "shots/one.mp4")
    m.scenes = [MovieScene(id: "s1", heading: "INT. ROOM"), MovieScene(id: "s2", heading: "EXT. STREET")]
    m.shots = [
        Shot(scene: "s1", duration: 4, camera: "slow push in", keyframePrompt: "a desk", motionPrompt: "smoke curls",
             keyframes: [k[0]], selectedKeyframe: k[0].id, videos: [v], selectedVideo: v.id),
        Shot(scene: "s1", duration: 3, motionPrompt: "", continuesPrevious: true, keyframes: [k[1]],
             selectedKeyframe: k[1].id),
        Shot(scene: "s2", continuesPrevious: true, keyframes: [k[2]], selectedKeyframe: k[2].id),
    ]
    return m
}

@Test func aContinuingShotStartsOnTheClipBeforeAndEndsOnItsKeyframe() throws {
    let m = clipMovie()
    #expect(MovieShots.ends(for: m.shots[0], in: m) == .init(start: .keyframe("keyframes/0.png"), end: nil))
    #expect(MovieShots.ends(for: m.shots[1], in: m) == .init(start: .lastFrame(of: "shots/one.mp4"), end: "keyframes/1.png"))
    // Across a scene change it starts afresh.
    #expect(MovieShots.ends(for: m.shots[2], in: m) == .init(start: .keyframe("keyframes/2.png"), end: nil))
    #expect(MovieShots.lastFrameFile(of: "shots/one.mp4") == "shots/one-last.png")

    let folder = URL(fileURLWithPath: "/m")
    let ends = try #require(MovieShots.ends(for: m.shots[1], in: m))
    let r = MovieShots.render(m.shots[1], movie: m, folder: folder, startImage: URL(fileURLWithPath: "/m/shots/one-last.png"),
                              ends: ends, seed: 9, server: "gpu")
    #expect(r.unusedPictures.isEmpty && r.values["end"] == .string("/m/keyframes/1.png"))
    #expect(r.values["length"] == .int(49) && r.values["width"] == .int(832) && r.values["height"] == .int(480))
    let g = try MovieShots.template().bind(r.values, optional: Set(r.values.keys), remove: r.unusedPictures)
    #expect(g["video"]?.inputs["last_image"] == .link(node: "end", output: "image"))
    #expect(g["start"]?.inputs["path"] == .value(.string("/m/shots/one-last.png")))

    // A shot without an end drops the end picture's node and link.
    let first = try #require(MovieShots.ends(for: m.shots[0], in: m))
    let r0 = MovieShots.render(m.shots[0], movie: m, folder: folder, startImage: folder.appendingPathComponent("keyframes/0.png"),
                               ends: first, seed: 1, server: "")
    let g0 = try MovieShots.template().bind(r0.values, optional: Set(r0.values.keys), remove: r0.unusedPictures)
    #expect(g0["end"] == nil && g0["video"]?.inputs["last_image"] == nil && r0.values["length"] == .int(65))
    let prompt = MovieShots.motionPrompt(for: m.shots[0], in: m)
    #expect(prompt == "smoke curls. Camera: slow push in. moody noir camera motion")
    #expect(!m.videosAreReady)
}

// MARK: timeline

private func editMovie(_ transitions: [ShotTransition], trims: [(Double, Double)] = []) -> MovieProject {
    var m = MovieProject()
    m.shots = transitions.enumerated().map { i, t in
        let v = Take(file: "shots/\(i).mp4", poster: "shots/\(i).png")
        let trim = i < trims.count ? trims[i] : (0, 0)
        return Shot(scene: "s1", transition: t, videos: [v], selectedVideo: v.id, trimStart: trim.0, trimEnd: trim.1)
    }
    return m
}

@Test func timelinePlacesCutsCrossfadesAndFades() {
    let m = editMovie([.fadeFromBlack, .cut, .crossfade, .fadeToBlack])
    let d = Dictionary(uniqueKeysWithValues: (0..<4).map { ("shots/\($0).mp4", 4.0) })
    let t = MovieTimeline(movie: m, durations: d)
    #expect(t.clips.map(\.start) == [0, 4, 7.5, 11.5])
    #expect(t.clips.map(\.fadeIn) == [0.5, 0, 0.5, 0.5] && t.clips.map(\.fadeOut) == [0, 0, 0.5, 0])
    #expect(t.clips.map(\.track) == [0, 1, 0, 1] && t.duration == 15.5)
    #expect(t.clips[0].opacity(at: 0.25) == 0.5 && t.clips[2].opacity(at: 11.25) == 0.5)
    // The segments cover the film without gaps; in the crossfade the incoming clip is on top, fading in.
    let segs = t.segments
    #expect(segs.first?.start == 0 && segs.last?.end == 15.5)
    #expect(zip(segs, segs.dropFirst()).allSatisfy { $0.end == $1.start })
    let cross = try! #require(segs.first { $0.start == 7.5 })
    #expect(cross.end == 8 && cross.layers.map(\.clip) == [2, 1])
    #expect(cross.layers[0].opacityFrom == 0 && cross.layers[0].opacityTo == 1 && cross.layers[1].opacityTo == 1)
    #expect(t.clip(at: 7.75)?.shot == m.shots[2].id)
}

@Test func timelineTrimsAreClampedAndMissingClipsListed() {
    var m = editMovie([.cut, .cut], trims: [(1, 0.5), (3.8, 2)])
    m.shots.append(Shot(scene: "s1"))
    let t = MovieTimeline(movie: m, durations: ["shots/0.mp4": 4, "shots/1.mp4": 4])
    #expect(t.clips[0].trimStart == 1 && t.clips[0].duration == 2.5)
    // A trim past the clip leaves the shortest clip.
    #expect(t.clips[1].trimStart == 3.5 && t.clips[1].duration == MovieTimeline.minimumClip)
    #expect(t.missing == [m.shots[2].id] && t.duration == 3)
}

// MARK: composer

/// A solid-colour clip at 16 fps, as Wan's are.
private func writeClip(_ url: URL, frames: Int, rgb: (UInt8, UInt8, UInt8), size: (Int, Int) = (64, 48)) async throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: size.0, AVVideoHeightKey: size.1,
    ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: size.0, kCVPixelBufferHeightKey as String: size.1,
    ])
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)
    for i in 0..<frames {
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, size.0, size.1, kCVPixelFormatType_32BGRA, nil, &buffer)
        let pb = try #require(buffer)
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let row = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<size.1 {
            for x in 0..<size.0 {
                let p = base + y * row + x * 4
                p[0] = rgb.2; p[1] = rgb.1; p[2] = rgb.0; p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 16))
    }
    input.markAsFinished()
    await writer.finishWriting()
    #expect(writer.status == .completed)
}

private func centre(_ image: CGImage) -> (Double, Double, Double) {
    var px = [UInt8](repeating: 0, count: 4)
    let ctx = CGContext(data: &px, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(image, in: CGRect(x: -image.width / 2, y: -image.height / 2, width: image.width, height: image.height))
    return (Double(px[0]) / 255, Double(px[1]) / 255, Double(px[2]) / 255)
}

@Test func composerExportsACrossfadedFilmAt24FPS() async throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("movie-edit-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder.appendingPathComponent("shots"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let colours: [(UInt8, UInt8, UInt8)] = [(230, 20, 20), (20, 230, 20), (20, 20, 230)]
    for (i, c) in colours.enumerated() {
        try await writeClip(folder.appendingPathComponent("shots/\(i).mp4"), frames: 33, rgb: c)  // 2.0625 s
    }
    let m = editMovie([.cut, .crossfade, .fadeToBlack])
    let durations = await MovieComposer.durations(of: (0..<3).map { "shots/\($0).mp4" }, in: folder)
    #expect(durations.values.allSatisfy { abs($0 - 2.0625) < 0.001 })
    let timeline = MovieTimeline(movie: m, durations: durations)
    #expect(abs(timeline.duration - (3 * 2.0625 - 0.5)) < 0.01, "\(timeline.duration)")
    let comp = try await MovieComposer.build(timeline, folder: folder, size: (128, 96))
    #expect(abs(comp.duration - timeline.duration) < 0.01)

    let out = folder.appendingPathComponent("renders/film.mp4")
    try await MovieComposer.export(comp, to: out, codec: .h264)
    let asset = AVURLAsset(url: out)
    let tracks = try await asset.loadTracks(withMediaType: .video)
    #expect(tracks.count == 1)
    let track = try #require(tracks.first)
    let (size, fps, range) = try await track.load(.naturalSize, .nominalFrameRate, .timeRange)
    #expect(size == CGSize(width: 128, height: 96) && abs(fps - 24) < 0.5)
    #expect(abs(range.duration.seconds - timeline.duration) < 0.1)

    // Red, then green (scaled up from 64×48), a mix in the crossfade, black in the dip, blue at the end.
    let gen = AVAssetImageGenerator(asset: asset)
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    func colour(at t: Double) async throws -> (Double, Double, Double) {
        centre(try await gen.image(at: CMTime(seconds: t, preferredTimescale: 600)).image)
    }
    let red = try await colour(at: 1), green = try await colour(at: 2.7)
    #expect(red.0 > 0.7 && red.1 < 0.3 && green.1 > 0.7 && green.0 < 0.3)
    let mix = try await colour(at: 1.8125)  // halfway through the crossfade (1.5625…2.0625)
    #expect(mix.0 > 0.25 && mix.1 > 0.25, "crossfade: \(mix)")
    let dip = try await colour(at: 3.625 - 0.02)  // the end of clip 2, faded to black
    #expect(dip.0 + dip.1 + dip.2 < 0.3, "dip: \(dip)")
    let blue = try await colour(at: 5.2)
    #expect(blue.2 > 0.7 && blue.0 < 0.3)

    // Frames of a clip: the last one, exactly.
    let last = try await MovieComposer.frame(of: folder.appendingPathComponent("shots/2.mp4"), at: .last)
    #expect(centre(last).2 > 0.7 && last.width == 64)
}
