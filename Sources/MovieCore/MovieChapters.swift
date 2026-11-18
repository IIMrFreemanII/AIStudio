import AVFoundation
import CoreMedia
import Foundation

/// Chapters in an MP4: a 3GPP text track (tx3g) of the chapter titles, which the video track names as its chapter
/// list. QuickTime Player, AVFoundation and ffmpeg-based players all read it. The export session can't write one,
/// so it's added afterwards by copying the file's tracks as they are (nothing is re-encoded).
public enum MovieChapters {
    public enum ChapterError: Error, CustomStringConvertible {
        case failed(String)
        public var description: String {
            switch self { case .failed(let why): "Couldn't add the chapters: \(why)" }
        }
    }

    /// Adds `chapters` (in order, the first at 0) to the MP4 at `url`, in place.
    public static func write(_ chapters: [MovieTimeline.Chapter], into url: URL, language: String = "en") async throws {
        guard !chapters.isEmpty else { return }
        let asset = AVURLAsset(url: url)
        let end = try await asset.load(.duration).seconds
        let temp = url.deletingLastPathComponent().appendingPathComponent(".chapters-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: temp) }

        let reader = try AVAssetReader(asset: asset)
        let writer = try AVAssetWriter(outputURL: temp, fileType: .mp4)
        writer.shouldOptimizeForNetworkUse = true
        var pumps: [Pump] = []
        var video: AVAssetWriterInput?
        for track in try await asset.load(.tracks) where track.mediaType == .video || track.mediaType == .audio {
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw ChapterError.failed("can't read a \(track.mediaType.rawValue) track") }
            reader.add(output)
            let (formats, transform) = try await track.load(.formatDescriptions, .preferredTransform)
            let input = AVAssetWriterInput(mediaType: track.mediaType, outputSettings: nil, sourceFormatHint: formats.first)
            input.expectsMediaDataInRealTime = false
            if track.mediaType == .video {
                input.transform = transform
                if video == nil { video = input }
            }
            guard writer.canAdd(input) else { throw ChapterError.failed("can't copy a \(track.mediaType.rawValue) track") }
            writer.add(input)
            pumps.append(Pump(input: input) { output.copyNextSampleBuffer() })
        }
        guard let video else { throw ChapterError.failed("the film has no video track") }

        let format = try textFormat()
        let text = AVAssetWriterInput(mediaType: .text, outputSettings: nil, sourceFormatHint: format)
        text.marksOutputTrackAsEnabled = false
        text.languageCode = Locale(identifier: language).language.languageCode?.identifier(.alpha3) ?? "eng"
        text.extendedLanguageTag = language
        guard writer.canAdd(text) else { throw ChapterError.failed("an MP4 can't take a text track here") }
        writer.add(text)
        let kind = AVAssetTrack.AssociationType.chapterList.rawValue
        guard video.canAddTrackAssociation(withTrackOf: text, type: kind) else {
            throw ChapterError.failed("the video track can't name a chapter list")
        }
        video.addTrackAssociation(withTrackOf: text, type: kind)
        var samples = try chapters.enumerated().compactMap { i, c -> CMSampleBuffer? in
            let next = i + 1 < chapters.count ? chapters[i + 1].start : end
            guard next > c.start else { return nil }
            return try sample(c.title, start: c.start, duration: next - c.start, format: format)
        }.makeIterator()
        pumps.append(Pump(input: text) { samples.next() })

        guard reader.startReading() else { throw ChapterError.failed(reader.error?.localizedDescription ?? "can't read the film") }
        guard writer.startWriting() else { throw ChapterError.failed(writer.error?.localizedDescription ?? "can't write") }
        writer.startSession(atSourceTime: .zero)
        // Every track side by side: the writer interleaves them by time, so feeding one at a time would stall.
        await withTaskGroup(of: Void.self) { group in
            for (i, pump) in pumps.enumerated() {
                group.addTask { await pump.run(on: DispatchQueue(label: "movie.chapters.\(i)")) }
            }
        }
        if reader.status == .failed {
            writer.cancelWriting()
            throw ChapterError.failed(reader.error?.localizedDescription ?? "reading the film failed")
        }
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw ChapterError.failed(writer.error?.localizedDescription ?? "writing the film failed")
        }
        _ = try FileManager.default.replaceItemAt(url, withItemAt: temp)
    }

    /// The chapters an MP4 (or any movie file) has, in order.
    public static func read(_ url: URL) async throws -> [MovieTimeline.Chapter] {
        let asset = AVURLAsset(url: url)
        let locales = try await asset.load(.availableChapterLocales)
        guard !locales.isEmpty else { return [] }
        let groups = try await asset.loadChapterMetadataGroups(bestMatchingPreferredLanguages: locales.map(\.identifier))
        var out: [MovieTimeline.Chapter] = []
        for g in groups {
            let title = try await g.items.first?.load(.stringValue) ?? ""
            out.append(MovieTimeline.Chapter(start: g.timeRange.start.seconds, title: title))
        }
        return out
    }

    // MARK: text samples

    /// A plain tx3g description: white sans-serif on black (only the titles matter; the track isn't shown).
    static func textFormat() throws -> CMFormatDescription {
        func color(_ v: Int) -> [CFString: Any] {
            [kCMTextFormatDescriptionColor_Red: v, kCMTextFormatDescriptionColor_Green: v,
             kCMTextFormatDescriptionColor_Blue: v, kCMTextFormatDescriptionColor_Alpha: 255]
        }
        let extensions: [CFString: Any] = [
            kCMTextFormatDescriptionExtension_DisplayFlags: 0,
            kCMTextFormatDescriptionExtension_BackgroundColor: color(0),
            // A box of no size is refused as an invalid format.
            kCMTextFormatDescriptionExtension_DefaultTextBox: [
                kCMTextFormatDescriptionRect_Top: 0, kCMTextFormatDescriptionRect_Left: 0,
                kCMTextFormatDescriptionRect_Bottom: 20, kCMTextFormatDescriptionRect_Right: 64,
            ],
            kCMTextFormatDescriptionExtension_DefaultStyle: [
                kCMTextFormatDescriptionStyle_StartChar: 0, kCMTextFormatDescriptionStyle_EndChar: 0,
                kCMTextFormatDescriptionStyle_Font: 1, kCMTextFormatDescriptionStyle_FontFace: 0,
                kCMTextFormatDescriptionStyle_ForegroundColor: color(255), kCMTextFormatDescriptionStyle_FontSize: 12,
            ],
            kCMTextFormatDescriptionExtension_HorizontalJustification: 0,
            kCMTextFormatDescriptionExtension_VerticalJustification: 0,
            kCMTextFormatDescriptionExtension_FontTable: ["1": "Sans-Serif"],
        ]
        var format: CMFormatDescription?
        let status = CMFormatDescriptionCreate(allocator: nil, mediaType: kCMMediaType_Text,
                                               mediaSubType: kCMTextFormatType_3GText,
                                               extensions: extensions as CFDictionary, formatDescriptionOut: &format)
        guard status == noErr, let format else { throw ChapterError.failed("no text format (\(status))") }
        return format
    }

    /// One title: its UTF-8 bytes after a 16-bit big-endian length, as tx3g samples are.
    static func sample(_ title: String, start: Double, duration: Double, format: CMFormatDescription) throws
        -> CMSampleBuffer
    {
        let utf8 = Array(title.utf8.prefix(65_535))
        let bytes = [UInt8(utf8.count >> 8), UInt8(utf8.count & 0xff)] + utf8
        var block: CMBlockBuffer?
        var status = CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: bytes.count,
                                                        blockAllocator: nil, customBlockSource: nil, offsetToData: 0,
                                                        dataLength: bytes.count, flags: 0, blockBufferOut: &block)
        guard status == noErr, let block else { throw ChapterError.failed("no sample buffer (\(status))") }
        status = CMBlockBufferReplaceDataBytes(with: bytes, blockBuffer: block, offsetIntoDestination: 0,
                                               dataLength: bytes.count)
        guard status == noErr else { throw ChapterError.failed("no sample data (\(status))") }
        let scale: CMTimeScale = 600
        var timing = CMSampleTimingInfo(duration: CMTime(seconds: duration, preferredTimescale: scale),
                                        presentationTimeStamp: CMTime(seconds: start, preferredTimescale: scale),
                                        decodeTimeStamp: .invalid)
        var size = bytes.count
        var out: CMSampleBuffer?
        status = CMSampleBufferCreate(allocator: nil, dataBuffer: block, dataReady: true, makeDataReadyCallback: nil,
                                      refcon: nil, formatDescription: format, sampleCount: 1, sampleTimingEntryCount: 1,
                                      sampleTimingArray: &timing, sampleSizeEntryCount: 1, sampleSizeArray: &size,
                                      sampleBufferOut: &out)
        guard status == noErr, let out else { throw ChapterError.failed("no sample (\(status))") }
        return out
    }

    /// Feeds one writer input from `next` whenever it's ready, until `next` runs out.
    private final class Pump: @unchecked Sendable {
        let input: AVAssetWriterInput
        let next: () -> CMSampleBuffer?

        init(input: AVAssetWriterInput, next: @escaping () -> CMSampleBuffer?) {
            self.input = input
            self.next = next
        }

        func run(on queue: DispatchQueue) async {
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                var finished = false
                input.requestMediaDataWhenReady(on: queue) { [self] in
                    guard !finished else { return }
                    while input.isReadyForMoreMediaData {
                        guard let sample = next(), input.append(sample) else {
                            finished = true
                            input.markAsFinished()
                            done.resume()
                            return
                        }
                    }
                }
            }
        }
    }
}
