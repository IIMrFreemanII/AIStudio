import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import Agent

/// Records one window to an MP4 with ScreenCaptureKit and AVAssetWriter.
final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let file: URL
    private var stream: SCStream!
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let queue = DispatchQueue(label: "uicontrol.recorder")
    private let lock = NSLock()
    // Touched on `queue` while the stream runs; read after it stops.
    private var firstPTS: CMTime?
    private var firstWall: Date?
    private var streamError: Error?
    private var finishTask: Task<UIRecording, Error>?
    private var timer: Task<Void, Never>?

    /// Whether `finish()` has begun (by `stopRecording` or the time limit).
    var isFinished: Bool { lock.withLock { finishTask != nil } }

    static func start(window: SCWindow, filter: SCContentFilter, file: URL, maxSeconds: Int) async throws -> Recorder {
        let config = Capture.configuration(for: window, filter: filter)
        config.minimumFrameInterval = CMTime(value: 1, timescale: 15)
        let r = try Recorder(file: file, filter: filter, config: config)
        do {
            try r.stream.addStreamOutput(r, type: .screen, sampleHandlerQueue: r.queue)
            guard r.writer.startWriting() else {
                throw ToolError("couldn't start writing the recording: \(r.writer.error?.localizedDescription ?? "unknown error")")
            }
            try await r.stream.startCapture()
        } catch let e as ToolError {
            throw e
        } catch {
            r.writer.cancelWriting()
            throw ToolError("couldn't start recording: \(error.localizedDescription)")
        }
        r.timer = Task { [weak r] in
            try? await Task.sleep(nanoseconds: UInt64(max(1, maxSeconds)) * 1_000_000_000)
            guard !Task.isCancelled, let r else { return }
            _ = try? await r.finish()
        }
        return r
    }

    private init(file: URL, filter: SCContentFilter, config: SCStreamConfiguration) throws {
        self.file = file
        try? FileManager.default.removeItem(at: file)
        writer = try AVAssetWriter(outputURL: file, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: config.width,
            AVVideoHeightKey: config.height,
        ])
        input.expectsMediaDataInRealTime = true
        writer.add(input)
        super.init()
        stream = SCStream(filter: filter, configuration: config, delegate: self)
    }

    /// Stops and finishes the file. Safe to call again: later calls get the same result.
    func finish() async throws -> UIRecording {
        let task: Task<UIRecording, Error> = lock.withLock {
            if let t = finishTask { return t }
            let t = Task { try await self.doFinish() }
            finishTask = t
            return t
        }
        return try await task.value
    }

    private func doFinish() async throws -> UIRecording {
        timer?.cancel()
        try? await stream.stopCapture()
        queue.sync {}   // let queued frames drain
        guard let first = firstPTS, let wall = firstWall else {
            writer.cancelWriting()
            throw ToolError("the recording got no frames (is the window visible?)")
        }
        let seconds = Date().timeIntervalSince(wall)
        // A still window sends few frames: end the session by the clock so the video is as long as the recording.
        writer.endSession(atSourceTime: first + CMTime(seconds: seconds, preferredTimescale: 600))
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed {
            throw ToolError("writing the recording failed: \(writer.error?.localizedDescription ?? "unknown error")")
        }
        if let e = streamError, writer.status != .completed {
            throw ToolError("the recording stopped: \(e.localizedDescription)")
        }
        return UIRecording(file: file, seconds: seconds, keyframes: await Self.keyframes(file, fallbackSeconds: seconds))
    }

    /// Four JPEGs at even points through the video.
    static func keyframes(_ file: URL, fallbackSeconds: Double) async -> [String] {
        let asset = AVURLAsset(url: file)
        var duration = ((try? await asset.load(.duration))?.seconds) ?? 0
        if !duration.isFinite || duration <= 0 { duration = fallbackSeconds }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: Capture.maxSide, height: Capture.maxSide)
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        var out: [String] = []
        for i in 0..<4 {
            let t = CMTime(seconds: duration * (Double(i) + 0.5) / 4, preferredTimescale: 600)
            if let (image, _) = try? await gen.image(at: t), let url = Capture.jpegDataURL(image, maxSide: Capture.maxSide) {
                out.append(url)
            }
        }
        return out
    }

    // MARK: SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid, writer.status == .writing else { return }
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let raw = attachments.first?[.status] as? Int, SCFrameStatus(rawValue: raw) == .complete else { return }
        let pts = sampleBuffer.presentationTimeStamp
        if firstPTS == nil {
            firstPTS = pts
            firstWall = Date()
            writer.startSession(atSourceTime: pts)
        }
        if input.isReadyForMoreMediaData { input.append(sampleBuffer) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) { streamError = error }
}
