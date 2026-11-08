import Foundation
import MLX

/// Mask and canvas helpers for inpainting and outpainting. Masks are `[H, W]` float32 in `[0, 1]`, white = regenerate.
public enum MaskOps {
    /// Separable convolution of a `[H, W]` map with a 1-D kernel along both axes, with edge padding (so a mask that
    /// touches the border stays fully on there).
    static func separable(_ m: MLXArray, kernel k: [Float]) -> MLXArray {
        let r = k.count / 2
        let x = m.asType(.float32).expandedDimensions(axes: [0, 3])  // [1, H, W, 1]
        let kx = MLXArray(k, [1, 1, k.count, 1])
        let ky = MLXArray(k, [1, k.count, 1, 1])
        var y = padded(x, widths: [0, 0, .init((r, r)), 0], mode: .edge)
        y = conv2d(y, ky)
        y = padded(y, widths: [0, .init((r, r)), 0, 0], mode: .edge)
        y = conv2d(y, kx)
        return y[0, 0..., 0..., 0]
    }

    /// Normalised Gaussian kernel for `sigma` (radius 3σ).
    static func gaussianKernel(sigma: Float) -> [Float] {
        let r = max(1, Int((3 * sigma).rounded(.up)))
        let k = (-r...r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let s = k.reduce(0, +)
        return k.map { $0 / s }
    }

    /// Gaussian blur; `radius` is the visible feather width in pixels (σ = radius / 3).
    public static func blur(_ m: MLXArray, radius: Float) -> MLXArray {
        guard radius >= 1 else { return m.asType(.float32) }
        return clip(separable(m, kernel: gaussianKernel(sigma: radius / 3)), min: 0, max: 1)
    }

    /// Dilates (`pixels` > 0) or erodes (< 0) the mask by a square of that radius.
    /// Dilation: every pixel within `pixels` of any non-zero pixel becomes 1; erosion is dilation of the inverse.
    public static func grow(_ m: MLXArray, pixels: Int) -> MLXArray {
        let mf = m.asType(.float32)
        if pixels == 0 { return mf }
        if pixels < 0 { return 1 - grow(1 - mf, pixels: -pixels) }
        let k = [Float](repeating: 1, count: 2 * pixels + 1)
        let reach = separable(mf, kernel: k)
        return MLX.where(reach .> 1e-4, MLXArray(Float(1)), mf)
    }

    /// Grow then blur: the usual "expand and soften" step before compositing.
    public static func feather(_ m: MLXArray, grow pixels: Int, radius: Float) -> MLXArray {
        let out = blur(grow(m, pixels: pixels), radius: radius)
        eval(out)
        return out
    }

    public enum Fill: String, CaseIterable { case edge, gray }

    /// Pads an RGBA `[H, W, 4]` image and returns it with a mask that is 1 over the new border.
    public static func pad(_ image: MLXArray, left: Int, top: Int, right: Int, bottom: Int, fill: Fill)
        -> (image: MLXArray, mask: MLXArray)
    {
        let (l, t, r, b) = (max(0, left), max(0, top), max(0, right), max(0, bottom))
        let (h, w) = (image.dim(0), image.dim(1))
        let x = image.asType(.float32)
        let widths: [IntOrPair] = [.init((t, b)), .init((l, r)), 0]
        var out: MLXArray
        switch fill {
        case .edge:
            out = padded(x, widths: widths, mode: .edge)
        case .gray:
            let rgb = padded(x[0..., 0..., ..<3], widths: widths, value: MLXArray(Float(0.5)))
            let a = padded(x[0..., 0..., 3...], widths: widths, value: MLXArray(Float(1)))
            out = concatenated([rgb, a], axis: 2)
        }
        let inner = MLXArray.zeros([h, w])
        let mask = padded(inner, widths: [.init((t, b)), .init((l, r))], value: MLXArray(Float(1)))
        eval(out, mask)
        return (out, mask)
    }
}

/// Overlapping tiles along one axis and the linear blend weights for them.
public enum Tiling {
    /// Start offsets of tiles of `tile` cells covering `length` with at least `overlap` cells shared between
    /// neighbours. One tile when it fits.
    public static func starts(length: Int, tile: Int, overlap: Int) -> [Int] {
        guard length > tile else { return [0] }
        let stride = max(1, tile - overlap)
        let n = Int((Double(length - tile) / Double(stride)).rounded(.up)) + 1
        // Spread the tiles evenly so every overlap is about the same width.
        return (0..<n).map { i in Int((Double(i) * Double(length - tile) / Double(n - 1)).rounded()) }
    }

    /// Weights for a tile of `size` starting at `start` in `length`: a linear ramp over the part shared with the
    /// previous / next tile, flat 1 elsewhere (and at the outer borders).
    public static func ramp(start: Int, size: Int, prevEnd: Int?, nextStart: Int?) -> [Float] {
        (0..<size).map { i in
            let p = start + i
            var w: Float = 1
            if let prevEnd, p < prevEnd {
                let span = Float(prevEnd - start)
                w = min(w, (Float(p - start) + 0.5) / span)
            }
            if let nextStart, p >= nextStart {
                let span = Float(start + size - nextStart)
                w = min(w, (Float(start + size - p) - 0.5) / span)
            }
            return w
        }
    }

    /// Runs `f` on overlapping tiles of a `[1, H, W, C]` input and blends the outputs linearly in the overlaps.
    ///
    /// Sizes are in cells: one input cell is `cell` pixels and maps to `outPerCell` output pixels (a VAE decoder
    /// over latents: 1 → 16; an encoder over pixels: 16 → 1). `tile` and `overlap` are in cells, so tile edges stay
    /// on the cell grid.
    public static func run(_ x: MLXArray, cell: Int, outPerCell: Int, tile: Int, overlap: Int,
                           _ f: (MLXArray) -> MLXArray) -> MLXArray {
        let (H, W) = (x.dim(1) / cell, x.dim(2) / cell)
        let rows = starts(length: H, tile: tile, overlap: overlap)
        let cols = starts(length: W, tile: tile, overlap: overlap)
        if rows.count == 1 && cols.count == 1 { return f(x) }
        let th = min(tile, H), tw = min(tile, W)
        let o = { (v: Int) in v * outPerCell }
        var acc: MLXArray?
        var wsum: MLXArray?
        for (ri, r0) in rows.enumerated() {
            let wr = ramp(start: o(r0), size: o(th),
                          prevEnd: ri > 0 ? o(rows[ri - 1] + th) : nil,
                          nextStart: ri + 1 < rows.count ? o(rows[ri + 1]) : nil)
            for (ci, c0) in cols.enumerated() {
                let wc = ramp(start: o(c0), size: o(tw),
                              prevEnd: ci > 0 ? o(cols[ci - 1] + tw) : nil,
                              nextStart: ci + 1 < cols.count ? o(cols[ci + 1]) : nil)
                let y = f(x[0..., (r0 * cell) ..< ((r0 + th) * cell), (c0 * cell) ..< ((c0 + tw) * cell), 0...])
                    .asType(.float32)
                let wmap = MLXArray(wr, [1, wr.count, 1, 1]) * MLXArray(wc, [1, 1, wc.count, 1])
                let widths: [IntOrPair] = [0, .init((o(r0), o(H) - o(r0) - o(th))),
                                           .init((o(c0), o(W) - o(c0) - o(tw))), 0]
                let contrib = padded(y * wmap, widths: widths)
                let wfull = padded(wmap, widths: widths)
                acc = acc.map { $0 + contrib } ?? contrib
                wsum = wsum.map { $0 + wfull } ?? wfull
                eval(acc!, wsum!)
            }
        }
        return acc! / maximum(wsum!, MLXArray(Float(1e-6)))
    }
}
