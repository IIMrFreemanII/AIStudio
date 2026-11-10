import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import MLX

/// Reading and writing video and audio files, with the app's metadata (the graph) embedded.
///
/// Metadata is a JSON object of strings. MP4/MOV/M4A carry it in the iTunes comment (`©cmt`, which AVFoundation
/// keeps in MPEG-4 files, unlike custom QuickTime keys) behind a marker prefix; WAV files carry it in a custom
/// `mlxc` RIFF chunk that players ignore.
public enum MediaIO {
    public enum VideoCodec: String, CaseIterable, Sendable { case h264, hevc }
    public enum AudioFormat: String, CaseIterable, Sendable { case wav, m4a }

    public enum MediaError: Error, CustomStringConvertible {
        case writer(String)
        case reader(String)
        case noTrack(URL)

        public var description: String {
            switch self {
            case .writer(let m): return "could not write media: \(m)"
            case .reader(let m): return "could not read media: \(m)"
            case .noTrack(let u): return "\(u.lastPathComponent) has no usable track"
            }
        }
    }

    static let commentPrefix = "mlxcomfy:"
    public static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]
    public static let audioExtensions: Set<String> = ["wav", "m4a", "aac", "mp3", "aif", "aiff", "caf"]

    // MARK: metadata

    static func encodeMetadata(_ m: [String: String]) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: m, options: [.sortedKeys])) ?? Data("{}".utf8)
        return commentPrefix + (String(data: data, encoding: .utf8) ?? "{}")
    }

    static func decodeMetadata(_ s: String) -> [String: String] {
        guard s.hasPrefix(commentPrefix),
              let obj = try? JSONSerialization.jsonObject(with: Data(s.dropFirst(commentPrefix.count).utf8))
        else { return [:] }
        return obj as? [String: String] ?? [:]
    }

    static func metadataItems(_ m: [String: String]) -> [AVMetadataItem] {
        guard !m.isEmpty else { return [] }
        let item = AVMutableMetadataItem()
        item.keySpace = .iTunes
        item.key = AVMetadataKey.iTunesMetadataKeyUserComment.rawValue as NSString
        item.value = encodeMetadata(m) as NSString
        return [item]
    }

    /// The app's metadata in a media file (empty if none).
    public static func readMetadata(_ url: URL) async -> [String: String] {
        if url.pathExtension.lowercased() == "wav" { return readWAVChunk(url) }
        guard let items = try? await AVURLAsset(url: url).load(.metadata) else { return [:] }
        for item in items {
            if let s = try? await item.load(.stringValue), s.hasPrefix(commentPrefix) { return decodeMetadata(s) }
        }
        return [:]
    }

    // MARK: writing video

    /// Encodes `clip` (and `audio`, or the clip's own soundtrack) as MP4/MOV, trimming audio to the video length.
    /// Copies an encoded video into an MP4/MOV container without decoding it, with the app's metadata added (the
    /// samples are passed through unchanged).
    public static func remuxVideo(_ source: URL, to url: URL, metadata: [String: String] = [:]) async throws {
        try? FileManager.default.removeItem(at: url)
        let asset = AVURLAsset(url: source)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
            throw MediaError.reader("can't copy \(source.lastPathComponent)")
        }
        session.metadata = metadataItems(metadata)
        session.shouldOptimizeForNetworkUse = true
        try await session.export(to: url, as: url.pathExtension.lowercased() == "mov" ? .mov : .mp4)
    }

    public static func writeVideo(_ clip: VideoClip, audio: AudioClip? = nil, to url: URL, codec: VideoCodec = .h264,
                                  quality: Double = 0.7, metadata: [String: String] = [:]) async throws {
        try? FileManager.default.removeItem(at: url)
        let type: AVFileType = url.pathExtension.lowercased() == "mov" ? .mov : .mp4
        let writer = try AVAssetWriter(outputURL: url, fileType: type)
        writer.metadata = metadataItems(metadata)
        // H.264 needs even dimensions; the encoder input is cropped to them.
        let (w, h) = (clip.width / 2 * 2, clip.height / 2 * 2)
        let bitsPerPixel = 0.04 + 0.26 * min(max(quality, 0), 1)
        let bitrate = Int(Double(w * h) * clip.fps * bitsPerPixel)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: codec == .hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bitrate],
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        guard writer.canAdd(videoInput) else { throw MediaError.writer("video input rejected") }
        writer.add(videoInput)

        var audioInput: AVAssetWriterInput?
        var track = audio ?? clip.audio
        // A clip of at most one sample is the "no soundtrack" placeholder (Video Decode of a silent model).
        if let a = track, a.samples.dim(-1) <= 1 { track = nil }
        if var a = track {
            if a.sampleRate > 48000 { a = MediaOps.resample(a, to: 48000) }
            a = MediaOps.fit(a, seconds: clip.duration)
            track = a
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: aacSettings(a))
            input.expectsMediaDataInRealTime = false
            guard writer.canAdd(input) else { throw MediaError.writer("audio input rejected") }
            writer.add(input)
            audioInput = input
        }

        guard writer.startWriting() else { throw MediaError.writer(writer.error.map { "\($0)" } ?? "start failed") }
        writer.startSession(atSourceTime: .zero)

        // Frames as BGRA bytes, converted once.
        let bgra = (MLX.clip(clip.frames[0..., ..<h, ..<w, 0...], min: 0, max: 1) * 255).round().asType(.uint8)
        let rgbaFull = concatenated([bgra[.ellipsis, 2..<3], bgra[.ellipsis, 1..<2], bgra[.ellipsis, 0..<1],
                                     MLXArray.full([clip.frameCount, h, w, 1], values: MLXArray(UInt8(255)))], axis: 3)
        eval(rgbaFull)
        let frameBytes = rgbaFull.asArray(UInt8.self)
        let frameSize = w * h * 4
        let fps = clip.fps
        let timescale = CMTimeScale(max(600, Int(fps * 100)))

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await feed(videoInput, queue: "video") { index in
                    guard index < clip.frameCount else { return false }
                    guard let pool = adaptor.pixelBufferPool else { throw MediaError.writer("no pixel buffer pool") }
                    var pb: CVPixelBuffer?
                    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
                    guard let pb else { throw MediaError.writer("pixel buffer") }
                    CVPixelBufferLockBaseAddress(pb, [])
                    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
                    let rowBytes = CVPixelBufferGetBytesPerRow(pb)
                    frameBytes.withUnsafeBufferPointer { src in
                        for row in 0..<h {
                            memcpy(base + row * rowBytes, src.baseAddress! + index * frameSize + row * w * 4, w * 4)
                        }
                    }
                    CVPixelBufferUnlockBaseAddress(pb, [])
                    let t = CMTime(value: CMTimeValue((Double(index) * Double(timescale) / fps).rounded()), timescale: timescale)
                    return adaptor.append(pb, withPresentationTime: t)
                }
            }
            if let audioInput, let track {
                let buffers = try sampleBuffers(track)
                group.addTask {
                    try await feed(audioInput, queue: "audio") { index in
                        guard index < buffers.count else { return false }
                        return audioInput.append(buffers[index])
                    }
                }
            }
            try await group.waitForAll()
        }
        await writer.finishWriting()
        if writer.status != .completed { throw MediaError.writer(writer.error.map { "\($0)" } ?? "status \(writer.status.rawValue)") }
    }

    /// Calls `next(i)` for i = 0, 1, … whenever the input wants data, until it returns false; then finishes the input.
    static func feed(_ input: AVAssetWriterInput, queue: String, _ next: @escaping (Int) throws -> Bool) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            var index = 0
            var done = false
            input.requestMediaDataWhenReady(on: DispatchQueue(label: "mlxcomfy.write.\(queue)")) {
                guard !done else { return }
                do {
                    while input.isReadyForMoreMediaData {
                        if try next(index) { index += 1 } else {
                            done = true
                            input.markAsFinished()
                            cont.resume()
                            return
                        }
                    }
                } catch {
                    done = true
                    input.markAsFinished()
                    cont.resume(throwing: error)
                }
            }
        }
    }

    static func aacSettings(_ a: AudioClip) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: a.sampleRate, AVNumberOfChannelsKey: min(a.channels, 2),
         AVEncoderBitRateKey: a.channels > 1 ? 192_000 : 128_000]
    }

    /// Interleaved float32 PCM sample buffers of about 4096 frames.
    static func sampleBuffers(_ clip: AudioClip) throws -> [CMSampleBuffer] {
        let a = MediaOps.channels(clip, min(clip.channels, 2))
        let channels = a.channels
        let interleaved = MLX.clip(a.samples, min: -1, max: 1).transposed(1, 0).asType(.float32)
        eval(interleaved)
        let data = interleaved.asArray(Float.self)
        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(a.sampleRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(4 * channels), mFramesPerPacket: 1, mBytesPerFrame: UInt32(4 * channels),
            mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        guard let format else { throw MediaError.writer("audio format") }
        var out: [CMSampleBuffer] = []
        let chunk = 4096
        var start = 0
        while start < a.frameCount {
            let n = min(chunk, a.frameCount - start)
            let bytes = n * channels * 4
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: bytes, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: bytes, flags: 0,
                                               blockBufferOut: &block)
            guard let block else { throw MediaError.writer("audio block") }
            data.withUnsafeBytes { raw in
                _ = CMBlockBufferReplaceDataBytes(with: raw.baseAddress! + start * channels * 4, blockBuffer: block,
                                                  offsetIntoDestination: 0, dataLength: bytes)
            }
            var sample: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: nil, dataBuffer: block, formatDescription: format, sampleCount: n,
                presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: CMTimeScale(a.sampleRate)),
                packetDescriptions: nil, sampleBufferOut: &sample)
            guard let sample else { throw MediaError.writer("audio sample buffer") }
            out.append(sample)
            start += n
        }
        return out
    }

    // MARK: writing audio

    public static func writeAudio(_ clip: AudioClip, to url: URL, format: AudioFormat = .wav,
                                  metadata: [String: String] = [:]) async throws {
        try? FileManager.default.removeItem(at: url)
        switch format {
        case .wav:
            try wavData(clip, metadata: metadata).write(to: url, options: .atomic)
        case .m4a:
            var a = MediaOps.channels(clip, min(clip.channels, 2))
            if a.sampleRate > 48000 { a = MediaOps.resample(a, to: 48000) }
            let writer = try AVAssetWriter(outputURL: url, fileType: .m4a)
            writer.metadata = metadataItems(metadata)
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: aacSettings(a))
            input.expectsMediaDataInRealTime = false
            writer.add(input)
            guard writer.startWriting() else { throw MediaError.writer(writer.error.map { "\($0)" } ?? "start failed") }
            writer.startSession(atSourceTime: .zero)
            let buffers = try sampleBuffers(a)
            try await feed(input, queue: "m4a") { i in i < buffers.count ? input.append(buffers[i]) : false }
            await writer.finishWriting()
            if writer.status != .completed { throw MediaError.writer(writer.error.map { "\($0)" } ?? "m4a failed") }
        }
    }

    /// 16-bit PCM WAV with an `mlxc` chunk holding the metadata JSON.
    public static func wavData(_ clip: AudioClip, metadata: [String: String] = [:]) -> Data {
        let x = (MLX.clip(clip.samples, min: -1, max: 1).transposed(1, 0) * 32767).round().asType(.int16)
        eval(x)
        let pcm = x.asArray(Int16.self)
        var d = Data()
        func u32(_ v: UInt32) { var le = v.littleEndian; d.append(Data(bytes: &le, count: 4)) }
        func u16(_ v: UInt16) { var le = v.littleEndian; d.append(Data(bytes: &le, count: 2)) }
        let channels = clip.channels
        let dataBytes = pcm.count * 2
        var meta = Data(encodeMetadata(metadata).utf8)
        if meta.count % 2 == 1 { meta.append(0) }
        let metaChunk = metadata.isEmpty ? 0 : 8 + meta.count
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(4 + 24 + 8 + dataBytes + metaChunk))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(UInt16(channels)); u32(UInt32(clip.sampleRate))
        u32(UInt32(clip.sampleRate * channels * 2)); u16(UInt16(channels * 2)); u16(16)
        if !metadata.isEmpty { d.append(contentsOf: Array("mlxc".utf8)); u32(UInt32(meta.count)); d.append(meta) }
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(dataBytes))
        pcm.withUnsafeBytes { d.append(contentsOf: $0) }
        return d
    }

    static func readWAVChunk(_ url: URL) -> [String: String] {
        guard let d = try? Data(contentsOf: url, options: .mappedIfSafe), d.count > 12,
              d.prefix(4) == Data("RIFF".utf8) else { return [:] }
        var i = 12
        while i + 8 <= d.count {
            let id = String(decoding: d[(d.startIndex + i)..<(d.startIndex + i + 4)], as: UTF8.self)
            let size = Int(d[(d.startIndex + i + 4)..<(d.startIndex + i + 8)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
            if id == "mlxc", i + 8 + size <= d.count {
                let s = String(decoding: d[(d.startIndex + i + 8)..<(d.startIndex + i + 8 + size)], as: UTF8.self)
                return decodeMetadata(s.trimmingCharacters(in: CharacterSet(charactersIn: "\0")))
            }
            i += 8 + size + (size % 2)
        }
        return [:]
    }

    // MARK: reading

    /// Decodes an audio file (or a video's soundtrack) to float samples at its own rate.
    public static func readAudio(_ url: URL, maxSeconds: Double? = nil) async throws -> AudioClip {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { throw MediaError.noTrack(url) }
        let descs = try await track.load(.formatDescriptions)
        let asbd = descs.first.flatMap { CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee }
        let rate = Int(asbd?.mSampleRate ?? 48000)
        let channels = max(1, min(Int(asbd?.mChannelsPerFrame ?? 2), 2))
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false,
            AVNumberOfChannelsKey: channels, AVSampleRateKey: rate,
        ])
        reader.add(output)
        guard reader.startReading() else { throw MediaError.reader(reader.error.map { "\($0)" } ?? "start") }
        let limit = maxSeconds.map { Int($0 * Double(rate)) * channels }
        let samples: [Float] = try await blockingRead {
            var samples: [Float] = []
            while let buffer = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(buffer) {
                let length = CMBlockBufferGetDataLength(block)
                var chunk = [Float](repeating: 0, count: length / 4)
                chunk.withUnsafeMutableBytes {
                    _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!)
                }
                samples += chunk
                if let limit, samples.count >= limit { samples = Array(samples.prefix(limit)); break }
            }
            return samples
        }
        let frames = samples.count / channels
        let x = MLXArray(Array(samples.prefix(frames * channels)), [frames, channels]).transposed(1, 0)
        return AudioClip(samples: x, sampleRate: rate)
    }

    /// AVAssetReader's sample loop blocks its thread. Run on Swift's shared async pool, a history full of audio
    /// thumbnails can tie up every pool thread and stall all other async work, so the loops run here instead
    /// (at most four at a time).
    static let readQueue = DispatchQueue(label: "mlxcomfy.media-read", qos: .userInitiated, attributes: .concurrent)
    static let readSlots = DispatchSemaphore(value: 4)

    static func blockingRead<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            readQueue.async {
                readSlots.wait()
                defer { readSlots.signal() }
                cont.resume(with: Result { try work() })
            }
        }
    }

    /// Decodes a video to frames (at most `maxFrames`), with its soundtrack if it has one.
    public static func readVideo(_ url: URL, maxFrames: Int? = nil) async throws -> VideoClip {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw MediaError.noTrack(url) }
        let fps = Double(try await track.load(.nominalFrameRate))
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(output)
        guard reader.startReading() else { throw MediaError.reader(reader.error.map { "\($0)" } ?? "start") }
        let (frames, count, w, h) = try await blockingRead { () -> ([Float], Int, Int, Int) in
        var frames: [Float] = []
        var count = 0
        var (w, h) = (0, 0)
        while maxFrames.map({ count < $0 }) ?? true, let buffer = output.copyNextSampleBuffer(),
              let pb = CMSampleBufferGetImageBuffer(buffer) {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            w = CVPixelBufferGetWidth(pb)
            h = CVPixelBufferGetHeight(pb)
            let row = CVPixelBufferGetBytesPerRow(pb)
            let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            frames.reserveCapacity(frames.count + w * h * 3)
            for y in 0..<h {
                let p = base + y * row
                for x in 0..<w {
                    frames.append(Float(p[x * 4 + 2]) / 255)
                    frames.append(Float(p[x * 4 + 1]) / 255)
                    frames.append(Float(p[x * 4]) / 255)
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            count += 1
        }
        return (frames, count, w, h)
        }
        guard count > 0 else { throw MediaError.reader("no frames") }
        let audio = try? await readAudio(url, maxSeconds: Double(count) / max(fps, 1))
        return VideoClip(frames: MLXArray(frames, [count, h, w, 3]), fps: fps > 0 ? Double(fps) : 24, audio: audio)
    }

    /// Duration in seconds (0 if unreadable).
    public static func duration(_ url: URL) async -> Double {
        (try? await AVURLAsset(url: url).load(.duration).seconds) ?? 0
    }

    /// The first frame of a video, for thumbnails.
    public static func posterFrame(_ url: URL, maxSize: Int = 480) async -> CGImage? {
        let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: maxSize, height: maxSize)
        return try? await gen.image(at: .zero).image
    }

    /// Peak level per bucket (0…1) for drawing a waveform.
    public static func waveform(_ url: URL, buckets: Int = 200) async -> [Float] {
        guard let clip = try? await readAudio(url, maxSeconds: 600) else { return [] }
        return waveform(clip, buckets: buckets)
    }

    public static func waveform(_ clip: AudioClip, buckets: Int) -> [Float] {
        let n = clip.frameCount
        guard n > 0, buckets > 0 else { return [] }
        let per = max(1, n / buckets)
        let used = per * min(buckets, n / per)
        guard used > 0 else { return [] }
        let peaks = abs(clip.samples[0..., ..<used]).max(axis: 0).reshaped(-1, per).max(axis: 1)
        eval(peaks)
        return peaks.asArray(Float.self)
    }
}
