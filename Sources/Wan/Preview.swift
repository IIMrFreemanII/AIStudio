import Foundation
import MLX

/// Fast step previews: a linear map from the 16 latent channels to RGB, fitted on a decoded clip with
/// `aistudio-cli generate-video --fit-preview` (Wan writes `WanPreviewCoefficients.swift`).
public enum WanPreview {
    static let coefficients: [Float] = WanPreviewCoefficients.values

    /// RGB `[h, w, 3]` in `[0, 1]` of the middle latent frame of `latents` `[16, T, h, w]`, or nil until fitted.
    public static func rgb(_ latents: MLXArray) -> MLXArray? {
        guard coefficients.count == 16 * 3 + 3 else { return nil }
        let w = MLXArray(Array(coefficients[0..<48]), [16, 3]), b = MLXArray(Array(coefficients[48...]), [3])
        let mid = latents[0..., latents.dim(1) / 2].transposed(1, 2, 0).asType(.float32)  // [h, w, 16]
        let out = clip(matmul(mid, w) + b, min: 0, max: 1)
        eval(out)
        return out
    }

    /// Least-squares fit of latents `[16, T, h, w]` against the decoded frames `[1+4(T−1), 8h, 8w, 3]` in `[0, 1]`:
    /// latent frame 0 covers pixel frame 0, frame i covers pixel frames 4i−3…4i; pixels are pooled to the latent grid.
    /// Returns `[16·3 + 3]` coefficients (weights, then bias).
    public static func fit(latents: MLXArray, pixels: MLXArray) -> [Float] {
        let (T, h, w) = (latents.dim(1), latents.dim(2), latents.dim(3))
        var xs: [MLXArray] = [], ys: [MLXArray] = []
        for i in 0..<T {
            let lo = i == 0 ? 0 : 4 * i - 3, hi = 4 * i + 1
            let px = pixels[lo..<min(hi, pixels.dim(0))].asType(.float32).mean(axis: 0)
            ys.append(px.reshaped(h, 8, w, 8, 3).mean(axes: [1, 3]).reshaped(h * w, 3))
            xs.append(latents[0..., i].transposed(1, 2, 0).reshaped(h * w, 16).asType(.float32))
        }
        let x = concatenated([concatenated(xs, axis: 0), MLXArray.ones([T * h * w, 1])], axis: 1)  // [N, 17]
        let y = concatenated(ys, axis: 0)
        let xtx = matmul(x.T, x) + 1e-3 * MLXArray.eye(17)
        let sol = matmul(MLX.inv(xtx, stream: .cpu), matmul(x.T, y))  // [17, 3]
        eval(sol)
        return sol[0..<16].reshaped(-1).asArray(Float.self) + sol[16].asArray(Float.self)
    }
}
