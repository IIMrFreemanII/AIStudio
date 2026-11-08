import MLX
import Testing
@testable import MLXDiffusionCore

/// 32×32 mask with a white square at rows/cols 12..<20.
func squareMask() -> MLXArray {
    var v = [Float](repeating: 0, count: 32 * 32)
    for r in 12..<20 { for c in 12..<20 { v[r * 32 + c] = 1 } }
    return MLXArray(v, [32, 32])
}

@Test func growDilatesAndErodesBySquare() {
    let m = squareMask()
    let g = MaskOps.grow(m, pixels: 2)
    #expect(g.sum().item(Float.self) == 12 * 12)
    #expect(g[10, 10].item(Float.self) == 1)
    #expect(g[9, 10].item(Float.self) == 0)
    let s = MaskOps.grow(m, pixels: -2)
    #expect(s.sum().item(Float.self) == 4 * 4)
    #expect(MaskOps.grow(m, pixels: 0).sum().item(Float.self) == 64)
}

@Test func blurKeepsRangeAndSoftensEdge() {
    let b = MaskOps.blur(squareMask(), radius: 6)
    #expect(b.min().item(Float.self) >= 0)
    #expect(b.max().item(Float.self) <= 1)
    // The centre stays mostly white, the edge is half-way, far pixels stay black.
    #expect(b[16, 16].item(Float.self) > 0.8)
    let edge = b[16, 12].item(Float.self)
    #expect(edge > 0.3 && edge < 0.7)
    #expect(b[0, 0].item(Float.self) < 1e-4)
    // Zero radius is the identity.
    #expect(abs(MaskOps.blur(squareMask(), radius: 0) - squareMask()).max().item(Float.self) == 0)
}

@Test func blurOfFullMaskStaysFull() {
    // Edge padding: a mask touching the border is not darkened there.
    let b = MaskOps.blur(MLXArray.ones([20, 20]), radius: 8)
    #expect(abs(b - 1).max().item(Float.self) < 1e-5)
}

@Test func padBuildsBorderMaskAndKeepsInterior() {
    let img = MLXRandom.uniform(0 ..< 1, [4, 6, 4], key: MLXRandom.key(7))
    let (out, mask) = MaskOps.pad(img, left: 2, top: 1, right: 3, bottom: 0, fill: .gray)
    #expect(out.shape == [5, 11, 4])
    #expect(mask.shape == [5, 11])
    #expect(mask.sum().item(Float.self) == Float(5 * 11 - 4 * 6))
    #expect(abs(out[1 ..< 5, 2 ..< 8] - img).max().item(Float.self) == 0)
    #expect(out[0, 0, 0].item(Float.self) == 0.5)
    #expect(out[0, 0, 3].item(Float.self) == 1)
    let (edge, _) = MaskOps.pad(img, left: 2, top: 0, right: 0, bottom: 0, fill: .edge)
    #expect(abs(edge[0..., 0] - img[0..., 0]).max().item(Float.self) == 0)
}

@Test func tileStartsCoverWithOverlap() {
    #expect(Tiling.starts(length: 50, tile: 64, overlap: 8) == [0])
    let s = Tiling.starts(length: 160, tile: 64, overlap: 8)
    #expect(s.first == 0)
    #expect(s.last == 160 - 64)
    for (a, b) in zip(s, s.dropFirst()) { #expect(a + 64 - b >= 8) }
}

@Test func rampsSumToOneInOverlap() {
    // Two tiles of 10 starting at 0 and 6: overlap 6..<10.
    let a = Tiling.ramp(start: 0, size: 10, prevEnd: nil, nextStart: 6)
    let b = Tiling.ramp(start: 6, size: 10, prevEnd: 10, nextStart: nil)
    for p in 6..<10 { #expect(abs(a[p] + b[p - 6] - 1) < 1e-6) }
    #expect(a[0] == 1)
    #expect(b[9] == 1)
}

@Test func tiledIdentityReproducesInput() {
    // With an identity "decoder" (×2 nearest upsample), blending tiles must give back the untiled result.
    let x = MLXRandom.normal([1, 40, 52, 3], key: MLXRandom.key(3))
    let up = { (t: MLXArray) -> MLXArray in
        let (h, w, c) = (t.dim(1), t.dim(2), t.dim(3))
        return broadcast(t.reshaped(1, h, 1, w, 1, c), to: [1, h, 2, w, 2, c]).reshaped(1, 2 * h, 2 * w, c)
    }
    let tiled = Tiling.run(x, cell: 1, outPerCell: 2, tile: 16, overlap: 4, up)
    #expect(tiled.shape == [1, 80, 104, 3])
    #expect(abs(tiled - up(x)).max().item(Float.self) < 1e-5)
    // Encoder direction: 2×2 average pooling over 2-pixel cells.
    let down = { (t: MLXArray) -> MLXArray in
        t.reshaped(1, t.dim(1) / 2, 2, t.dim(2) / 2, 2, t.dim(3)).mean(axes: [2, 4])
    }
    let enc = Tiling.run(up(x), cell: 2, outPerCell: 1, tile: 16, overlap: 4, down)
    #expect(abs(enc - x).max().item(Float.self) < 1e-5)
}
