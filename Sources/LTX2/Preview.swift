import Foundation
import MLX

/// A cheap sampler preview: a linear map from the 128 latent channels to RGB, applied to the middle latent frame.
/// Fitted by least squares against decoded clips (`mlxcomfy-cli generate-video --fit-preview`).
public enum LTXPreview {
    /// Weights `[128, 3]` (row-major) followed by the bias `[3]`; empty until fitted.
    static let coefficients: [Float] = LTXPreviewCoefficients.values

    /// RGB `[h, w, 3]` in `[0, 1]` of latent frame `F/2`, or nil without coefficients.
    public static func rgb(tokens: MLXArray, frames: Int, height: Int, width: Int) -> MLXArray? {
        guard coefficients.count == 128 * 3 + 3, tokens.dim(1) >= frames * height * width else { return nil }
        let w = MLXArray(Array(coefficients[0..<(128 * 3)]), [128, 3])
        let b = MLXArray(Array(coefficients[(128 * 3)...]), [3])
        let f = frames / 2, n = height * width
        let mid = tokens[0, (f * n)..<((f + 1) * n), 0...].asType(.float32)
        let out = clip(matmul(mid, w) + b, min: 0, max: 1).reshaped(height, width, 3)
        eval(out)
        return out
    }

    /// Least-squares fit of tokens `[1, F*H*W, 128]` against decoded frames `[T, 32H, 32W, 3]` in `[0, 1]`: each
    /// latent frame is matched with the pixel frames it covers (the first covers one), pooled to the latent grid.
    /// Returns `[128*3 + 3]` coefficients (weights then bias).
    public static func fit(tokens: MLXArray, frames F: Int, height H: Int, width W: Int, pixels: MLXArray) -> [Float] {
        var targets: [MLXArray] = []
        for i in 0..<F {
            let lo = i == 0 ? 0 : 8 * i - 7, hi = i == 0 ? 1 : 8 * i + 1
            let px = pixels[lo..<min(hi, pixels.dim(0))].asType(.float32).mean(axis: 0)  // [32H, 32W, 3]
            targets.append(px.reshaped(H, 32, W, 32, 3).mean(axes: [1, 3]).reshaped(H * W, 3))
        }
        let y = concatenated(targets, axis: 0)                                 // [N, 3]
        let x = tokens[0, 0..<(F * H * W), 0...].asType(.float32)              // [N, 128]
        let xb = concatenated([x, MLXArray.ones([x.dim(0), 1])], axis: 1)      // [N, 129]
        let xtx = matmul(xb.T, xb) + 1e-3 * MLXArray.eye(129)
        let sol = matmul(MLXLinalg.inv(xtx, stream: .cpu), matmul(xb.T, y))   // [129, 3]
        eval(sol)
        let a = sol.asArray(Float.self)
        return Array(a[0..<(128 * 3)]) + Array(a[(128 * 3)...])
    }
}
