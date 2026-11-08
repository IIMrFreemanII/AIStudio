import Foundation
import MLX
import Testing
@testable import MLXDiffusionCore

func tempURL(_ ext: String) -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
}

@Test func mixPadsResamplesAndOffsets() {
    let a = MediaOps.tone(frequency: 440, seconds: 1, sampleRate: 48000, channels: 2)
    let b = MediaOps.tone(frequency: 220, seconds: 0.5, sampleRate: 24000, channels: 1)
    let m = MediaOps.mix(a, b, gain: 0.5, offset: 0.75)
    #expect(m.sampleRate == 48000)
    #expect(m.channels == 2)
    #expect(m.frameCount == 48000 + 12000)  // b ends 0.75 + 0.5 = 1.25 s in
    #expect(m.samples.max().item(Float.self) <= 1)
    // Before the offset the mix is exactly a.
    #expect(abs(m.samples[0..., ..<30000] - a.samples[0..., ..<30000]).max().item(Float.self) < 1e-6)
}

@Test func fitTrimsAndPads() {
    let t = MediaOps.tone(frequency: 300, seconds: 2, sampleRate: 16000, channels: 1)
    #expect(MediaOps.fit(t, seconds: 1).frameCount == 16000)
    let padded = MediaOps.fit(t, seconds: 3, offset: 0.5)
    #expect(padded.frameCount == 48000)
    #expect(abs(padded.samples[0..., ..<8000]).max().item(Float.self) == 0)
}

@Test func resampleKeepsDuration() {
    let t = MediaOps.tone(frequency: 300, seconds: 1, sampleRate: 22050, channels: 2)
    let r = MediaOps.resample(t, to: 44100)
    #expect(r.frameCount == 44100)
    #expect(abs(r.duration - 1) < 1e-3)
}

@Test func wavRoundTripKeepsSamplesAndMetadata() async throws {
    let url = tempURL("wav")
    defer { try? FileManager.default.removeItem(at: url) }
    let t = MediaOps.tone(frequency: 440, seconds: 0.5, sampleRate: 24000, channels: 2)
    try await MediaIO.writeAudio(t, to: url, format: .wav, metadata: ["mlxcomfy.graph": "{\"nodes\":[]}"])
    let back = try await MediaIO.readAudio(url)
    #expect(back.sampleRate == 24000)
    #expect(back.channels == 2)
    #expect(back.frameCount == t.frameCount)
    #expect(abs(back.samples - t.samples).max().item(Float.self) < 1.0 / 16000)
    #expect(await MediaIO.readMetadata(url)["mlxcomfy.graph"] == "{\"nodes\":[]}")
}

@Test func m4aRoundTripKeepsDurationAndMetadata() async throws {
    let url = tempURL("m4a")
    defer { try? FileManager.default.removeItem(at: url) }
    let t = MediaOps.tone(frequency: 440, seconds: 1, sampleRate: 44100, channels: 2)
    try await MediaIO.writeAudio(t, to: url, format: .m4a, metadata: ["k": "v"])
    let back = try await MediaIO.readAudio(url)
    #expect(back.sampleRate == 44100)
    #expect(abs(back.duration - 1) < 0.1)
    #expect(await MediaIO.readMetadata(url) == ["k": "v"])
}

@Test func videoRoundTripWithSoundtrackAndMetadata() async throws {
    let url = tempURL("mp4")
    defer { try? FileManager.default.removeItem(at: url) }
    let clip = MediaOps.testPattern(width: 96, height: 64, frames: 24, fps: 12)
    let tone = MediaOps.tone(frequency: 330, seconds: 3, sampleRate: 48000)  // longer than the 2 s video
    try await MediaIO.writeVideo(clip, audio: tone, to: url, metadata: ["mlxcomfy.graph": "{}"])
    let back = try await MediaIO.readVideo(url)
    #expect(back.frameCount == 24)
    #expect(back.width == 96 && back.height == 64)
    #expect(abs(back.fps - 12) < 0.01)
    // Lossy, but colours survive roughly.
    #expect(abs(back.frames - clip.frames).mean().item(Float.self) < 0.06)
    let audio = try #require(back.audio)
    #expect(abs(audio.duration - 2) < 0.1)  // trimmed to the video
    #expect(await MediaIO.readMetadata(url)["mlxcomfy.graph"] == "{}")
    #expect(await MediaIO.posterFrame(url) != nil)
}

@Test func waveformPeaks() {
    let t = MediaOps.tone(frequency: 100, seconds: 1, sampleRate: 8000, channels: 1, amplitude: 0.5)
    let w = MediaIO.waveform(t, buckets: 20)
    #expect(w.count == 20)
    #expect(w.max()! <= 0.5001 && w[10] > 0.45)
}
