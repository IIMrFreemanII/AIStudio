import Foundation
import MLX
import MLXDiffusionCore
import Testing
@testable import LTX2

@Test func scheduleShorteningKeepsTheEnd() {
    #expect(LTXSchedule.shortened(LTXSchedule.distilled, steps: nil, keepStart: true) == LTXSchedule.distilled)
    #expect(LTXSchedule.shortened(LTXSchedule.distilled, steps: 3, keepStart: true) == [1.0, 0.725, 0.421875, 0.0])
    #expect(LTXSchedule.shortened(LTXSchedule.stage2, steps: 2, keepStart: false) == [0.725, 0.421875, 0.0])
    #expect(LTXSchedule.shortened(LTXSchedule.distilled, steps: 20, keepStart: true).count == 9)
}

@Test func frameConditionPinsTheFirstFrame() {
    let tpf = 4
    let s = LTXSampler.noisedState(shape: [1, 3 * tpf, 2], initial: nil, positions: MLXArray.zeros([1, 3 * tpf, 3]),
                                   seed: 1, sigma: 1,
                                   conditions: [LTXFrameCondition(latentFrame: 0, tokens: MLXArray.ones([1, tpf, 2]))],
                                   tokensPerFrame: tpf, scalarBlend: true)
    #expect(s.latent[0, 0..<tpf].asType(.float32).allClose(MLXArray.ones([tpf, 2])).item(Bool.self))
    #expect(s.mask[0, 0..<tpf].sum().item(Float.self) == 0)
    #expect(s.mask[0, tpf...].sum().item(Float.self) == Float(2 * tpf))
    #expect(!s.isUniform)
}

@Test func eulerReachesTheEstimateAtSigmaZero() {
    let x = MLXArray([Float(2)]), x0 = MLXArray([Float(0.5)])
    #expect(LTXSampler.euler(x, x0, 0.5, 0).item(Float.self) == 0.5)
    #expect(abs(LTXSampler.euler(x, x0, 1, 0.5).item(Float.self) - 1.25) < 1e-6)
}

/// The VideoToolbox round trip should degrade a photo about as much as libx264 CRF 33 (PSNR ~30 dB).
@Test func h264RoundTripMatchesCRF33Strength() throws {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../outputs/txt2img_00001.png")
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    let rgb = (ImageArray.array(from: try ImageArray.load(url))[0..., 0..., 0..<3] * 255).round()
    let out = H264RoundTrip.apply(rgb)
    #expect(out.shape == rgb.shape)
    let mse = ((out - rgb) * (out - rgb)).mean().item(Float.self)
    let psnr = 10 * log10(255 * 255 / mse)
    #expect(psnr > 27 && psnr < 34, "PSNR \(psnr)")
}
