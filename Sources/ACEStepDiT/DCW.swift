import Foundation
import MLX

/// Differential Correction in Wavelet domain (DCW, CVPR 2026), as applied by ACE-Step's sampler after each step:
/// a single-level Haar DWT along time of both the stepped latent `x_next` and the predicted clean sample
/// `denoised = x - v·t`, then `band += λ·(band_x − band_denoised)` for the selected band(s), then the inverse DWT.
///
/// Port of `acestep/models/common/dcw_correction.py` + `dcw_primitives.py` (pytorch_wavelets `DWT1DForward(J=1,
/// mode="zero", wave="haar")`): odd `T` is zero-padded by one frame on the right and the result trimmed back.
/// Only the Haar basis is implemented (the turbo default).
public struct DCWOptions: Sendable {
    public enum Mode: String, Sendable { case low, high, double, pix }

    public var enabled: Bool
    public var mode: Mode
    /// Low-band scaler (or the single scaler for `high` / `pix`).
    public var scaler: Float
    /// High-band scaler, used only in `double` mode.
    public var highScaler: Float

    public init(enabled: Bool = true, mode: Mode = .double, scaler: Float = 0.05, highScaler: Float = 0.02) {
        self.enabled = enabled
        self.mode = mode
        self.scaler = scaler
        self.highScaler = highScaler
    }

    /// The official pipeline default for turbo (`GenerationParams`: enabled, "double", 0.05 / 0.02, "haar").
    public static let turboDefault = DCWOptions()
    public static let off = DCWOptions(enabled: false)

    public var isActive: Bool {
        guard enabled else { return false }
        return mode == .double ? (scaler != 0 || highScaler != 0) : scaler != 0
    }

    /// Corrected `xNext` (`[B, T, C]`) given `denoised` and the step's current time `t` (before the step).
    /// Schedules: low `t·scaler`; high `(1−t)·scaler`; double low `t·scaler` + high `(1−t)·highScaler`; pix `scaler`.
    public func apply(_ xNext: MLXArray, denoised: MLXArray, t: Float) -> MLXArray {
        guard isActive else { return xNext }
        let lowS = t * scaler, highS = (1 - t) * scaler, doubleHighS = (1 - t) * highScaler
        let x = xNext.asType(.float32), y = denoised.asType(.float32)
        if mode == .pix { return (x + scaler * (x - y)).asType(xNext.dtype) }
        let (ls, hs): (Float, Float)
        switch mode {
        case .low: (ls, hs) = (lowS, 0)
        case .high: (ls, hs) = (0, highS)
        default: (ls, hs) = (lowS, doubleHighS)
        }
        if ls == 0 && hs == 0 { return xNext }
        var (xl, xh) = haarDWT(x)
        let (yl, yh) = haarDWT(y)
        if ls != 0 { xl = xl + ls * (xl - yl) }
        if hs != 0 { xh = xh + hs * (xh - yh) }
        return haarIDWT(xl, xh, length: x.dim(1)).asType(xNext.dtype)
    }
}

let invSqrt2 = Float(1 / 2.0.squareRoot())

/// Single-level orthonormal Haar DWT along axis 1 of `[B, T, C]`; returns (low, high), each `[B, ceil(T/2), C]`.
func haarDWT(_ x: MLXArray) -> (MLXArray, MLXArray) {
    var x = x
    let B = x.dim(0), C = x.dim(2)
    if x.dim(1) % 2 == 1 { x = concatenated([x, zeros([B, 1, C], dtype: x.dtype)], axis: 1) }
    let p = x.reshaped(B, -1, 2, C).split(parts: 2, axis: 2)
    let even = p[0].squeezed(axis: 2), odd = p[1].squeezed(axis: 2)
    return ((even + odd) * invSqrt2, (even - odd) * invSqrt2)
}

/// Inverse of ``haarDWT(_:)``, trimmed to `length` frames.
func haarIDWT(_ low: MLXArray, _ high: MLXArray, length: Int) -> MLXArray {
    let even = (low + high) * invSqrt2, odd = (low - high) * invSqrt2
    let B = low.dim(0), C = low.dim(2)
    let r = stacked([even, odd], axis: 2).reshaped(B, -1, C)
    return r.dim(1) == length ? r : r[0..., 0..<length, 0...]
}
