import Foundation
import MLX
import Testing
@testable import StableAudio

@Test func pingPongScheduleIsLinearFromOneToZero() {
    #expect(StableAudioPipeline.schedule(steps: 4) == [1, 0.75, 0.5, 0.25, 0])
    #expect(StableAudioPipeline.schedule(steps: 8).count == 9)
}

/// T5's bidirectional relative-position buckets, checked against values from HF transformers (32 buckets, max 128).
@Test func t5RelativePositionBuckets() throws {
    let url = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/AIStudio/models/stable-audio-open-small/t5")
    guard FileManager.default.fileExists(atPath: url.appendingPathComponent("model.safetensors").path) else { return }
    let t5 = try T5Encoder(directory: url)
    let b = t5.buckets(200).asArray(Int32.self)
    func at(_ q: Int, _ k: Int) -> Int32 { b[q * 200 + k] }
    #expect(at(0, 0) == 0)
    #expect(at(0, 1) == 17 && at(1, 0) == 1)       // ±1 is exact; keys ahead get the upper half
    #expect(at(0, 7) == 23 && at(0, 8) == 24)      // last exact bucket, first logarithmic one
    #expect(at(0, 199) == 31 && at(199, 0) == 15)  // beyond max_distance: the last bucket
}
