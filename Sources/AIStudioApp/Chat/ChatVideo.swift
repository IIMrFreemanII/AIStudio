import AVFoundation
import AppKit
import CryptoKit
import Foundation
import ImageIO
import LLMClient

/// Videos attached to chats: made smaller on this Mac, saved with the chat as `<sha256>.mp4` (the name the GPU server
/// stores its upload under) plus a poster frame for the thumbnail, `<sha256>.mp4.jpg`.
extension AppModel {
    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]
    /// Longer videos are cut to this many seconds (the model looks at a few dozen frames of it anyway).
    static let maxVideoSeconds = 600.0

    enum VideoError: LocalizedError {
        case noVideoTrack, exportFailed(String)
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return "it has no video track"
            case .exportFailed(let why): return why
            }
        }
    }

    /// Re-encodes a video at most 960×540 (H.264, MP4), at most `maxVideoSeconds` long, into the chat's folder.
    /// Returns its file name.
    nonisolated static func saveVideo(_ url: URL, into folder: URL) async -> Result<String, Error> {
        do {
            let asset = AVURLAsset(url: url)
            guard try await !asset.loadTracks(withMediaType: .video).isEmpty else { throw VideoError.noVideoTrack }
            let duration = try await asset.load(.duration)
            let preset = await AVAssetExportSession.compatibility(ofExportPreset: AVAssetExportPreset960x540,
                                                                   with: asset, outputFileType: .mp4)
                ? AVAssetExportPreset960x540 : AVAssetExportPresetMediumQuality
            guard let session = AVAssetExportSession(asset: asset, presetName: preset) else {
                throw VideoError.exportFailed("this Mac can't re-encode it")
            }
            if duration.seconds > maxVideoSeconds {
                session.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: maxVideoSeconds, preferredTimescale: 600))
            }
            session.shouldOptimizeForNetworkUse = true
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
            defer { try? FileManager.default.removeItem(at: tmp) }
            try await session.export(to: tmp, as: .mp4)
            let data = try Data(contentsOf: tmp, options: .mappedIfSafe)
            let name = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() + ".mp4"
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let dest = folder.appendingPathComponent(name)
            if !FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.moveItem(at: tmp, to: dest)
            }
            await savePoster(of: dest, to: posterURL(dest))
            return .success(name)
        } catch {
            return .failure(error)
        }
    }

    /// Where a video's poster frame is saved.
    nonisolated static func posterURL(_ video: URL) -> URL { video.appendingPathExtension("jpg") }

    nonisolated static func savePoster(of video: URL, to dest: URL) async {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: video))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 256, height: 256)
        guard let (image, _) = try? await generator.image(at: CMTime(seconds: 0.5, preferredTimescale: 600)),
              let out = CGImageDestinationCreateWithURL(dest as CFURL, "public.jpeg" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(out, image, nil)
        CGImageDestinationFinalize(out)
    }
}
