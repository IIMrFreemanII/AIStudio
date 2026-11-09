import MLX

/// Unpadded stride-1 2D convolution of `x` `[N, H, W, C]`, computed in horizontal bands when the unrolled patch
/// buffer would be large (each band reads the `kh − 1` rows below it, so the result is identical).
///
/// MLX unrolls a convolution's patches for the whole input at once (`H·W·kh·kw·C` elements), which at full image
/// resolution costs gigabytes; bands of about `budget` unrolled elements keep that bounded.
public func bandedConv2d(_ x: MLXArray, _ w: MLXArray, budget: Int = 64 << 20) -> MLXArray {
    let (n, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let kh = w.dim(1), kw = w.dim(2)
    let hOut = h - kh + 1
    let rowCost = n * (wd - kw + 1) * c * kh * kw  // unrolled elements per output row
    guard rowCost * hOut > budget else { return conv2d(x, w) }
    let band = max(8, budget / max(rowCost, 1))
    var parts: [MLXArray] = []
    var r = 0
    while r < hOut {
        let r1 = min(hOut, r + band)
        let o = conv2d(x[0..., r ..< (r1 + kh - 1)], w)
        eval(o)
        parts.append(o)
        r = r1
    }
    return concatenated(parts, axis: 1)
}
