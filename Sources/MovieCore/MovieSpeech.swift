import Accelerate
import AVFoundation
import Foundation

/// Where someone speaks in a clip's own sound. MiniMax H3 is asked to say each line at a time, but it picks its own
/// (a one-word line asked for at 0.4 s came at 4.6 s), so the subtitles and the music's dip go where the voice is.
///
/// A voiced frame is one with a clear pitch in the speaking range: wind, waves and creaks have none, or sit above it.
/// Frames close together make one stretch; a lone frame doesn't count.
public enum MovieSpeech {
    public static let rate = 16_000.0
    /// The analysis window (100 ms) and its step (50 ms), in samples.
    public static let window = 1600, hop = 800
    /// The pitch range searched, Hz: a low man's voice to a child's.
    public static let lowest = 70.0, highest = 330.0
    /// Quieter frames are skipped (about −46 dBFS RMS).
    public static let quiet: Float = 0.005
    /// How alike the frame is to itself one pitch period later (1 = a pure tone).
    public static let voicedAt: Float = 0.6
    /// Stretches closer than this join; shorter ones are dropped.
    public static let join = 0.35, shortest = 0.15

    /// The voiced stretches of `samples` (mono at `rate`), in order.
    public static func spans(samples: [Float], rate: Double = rate) -> [VoiceSpan] {
        let minLag = Int(rate / highest), maxLag = Int(rate / lowest), shortestLag = Int(rate / 500)
        guard samples.count >= window, maxLag < window else { return [] }
        var frame = [Float](repeating: 0, count: window + maxLag)
        var corr = [Float](repeating: 0, count: maxLag + 1)
        var voiced: [(Double, Double)] = []
        var i = 0
        while i + window <= samples.count {
            samples.withUnsafeBufferPointer { s in
                frame.withUnsafeMutableBufferPointer { f in
                    for j in 0..<window { f[j] = s[i + j] }
                }
            }
            var mean: Float = 0
            vDSP_meanv(frame, 1, &mean, vDSP_Length(window))
            var neg = -mean
            vDSP_vsadd(frame, 1, &neg, &frame, 1, vDSP_Length(window))
            var rms: Float = 0
            vDSP_rmsqv(frame, 1, &rms, vDSP_Length(window))
            if rms >= quiet {
                // corr[k] = Σ frame[n + k] · frame[n]: the zero padding after the window ends each sum there.
                vDSP_conv(frame, 1, frame, 1, &corr, 1, vDSP_Length(maxLag + 1), vDSP_Length(window))
                // The strongest period from 500 Hz down: a whistle above the range (also alike at twice its
                // period) or a smooth rumble (most alike at the shortest lag) peaks below `minLag`, and doesn't count.
                var best: Float = 0, lag = 0
                for k in shortestLag...maxLag where corr[k] > best { (best, lag) = (corr[k], k) }
                if corr[0] > 0, lag >= minLag, best / corr[0] > voicedAt {
                    voiced.append((Double(i) / rate, Double(i + window) / rate))
                }
            }
            i += hop
        }
        var out: [VoiceSpan] = []
        var singles: [Bool] = []
        for (a, b) in voiced {
            if let last = out.last, a - last.end < join {
                out[out.count - 1].end = max(last.end, b)
                singles[singles.count - 1] = false
            } else {
                out.append(VoiceSpan(start: a, end: b))
                singles.append(true)
            }
        }
        return zip(out, singles).filter { !$1 && $0.end - $0.start >= shortest }.map(\.0)
    }

    /// The voiced stretches of a clip's (or a sound file's) audio; none when it has no audio track.
    public static func spans(in url: URL) async throws -> [VoiceSpan] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { return [] }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
        ])
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length,
                                                                          destination: $0.baseAddress!) }
            samples += chunk
        }
        if reader.status == .failed { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
        return spans(samples: samples)
    }
}

/// A stretch of a clip where someone speaks, in seconds from the clip file's start.
public struct VoiceSpan: Codable, Equatable, Sendable {
    public var start: Double
    public var end: Double
    public var length: Double { end - start }
    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}
