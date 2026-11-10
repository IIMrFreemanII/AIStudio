import Foundation
import MLX
import Testing
@testable import MLXDiffusionCore

@Test func sigmasAppendTerminalZero() {
    var s = FlowMatchEuler()
    s.setSigmas([1.0, 0.5])
    #expect(s.sigmas == [1.0, 0.5, 0])
    #expect(s.stepCount == 2)
    #expect(s.timestep(1) == 500)
}

@Test func linspaceMatchesReference() {
    var s = FlowMatchEuler()
    s.setSteps(4)  // np.linspace(1, 0.25, 4) = [1, 0.75, 0.5, 0.25]
    let expected: [Float] = [1, 0.75, 0.5, 0.25, 0]
    for (a, b) in zip(s.sigmas, expected) { #expect(abs(a - b) < 1e-6) }
}

@Test func pngTextRoundTrips() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("pngtext-\(UInt32.random(in: 0...UInt32.max)).png")
    defer { try? FileManager.default.removeItem(at: url) }
    let img = MLX.MLXArray.zeros([8, 8, 3])
    let meta = ["aistudio.graph": "{\"nodes\":[],\"version\":1}", "note": "ünïcödé ✓"]
    try ImageArray.savePNG(img, to: url, text: meta)
    #expect(ImageArray.readPNGText(url) == meta)
    // The image itself must still decode.
    let back = ImageArray.array(from: try ImageArray.load(url))
    #expect(back.shape == [8, 8, 4])
}
