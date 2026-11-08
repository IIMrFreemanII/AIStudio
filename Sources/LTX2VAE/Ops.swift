import Foundation
import MLX
import MLXDiffusionCore

// Building blocks shared by the LTX-2 video VAE encoder/decoder and the latent upsampler.
// Every tensor is channels-last: video activations are `[B, F, H, W, C]`.
// The MLX-converted checkpoints already store kernels in MLX layout (`[O, kd, kh, kw, I]` for 3-D convs,
// `[O, kh, kw, I]` for 2-D convs), so nothing is transposed on load.

/// 3-D convolution with the LTX temporal padding rule (`Conv3dBlock` in the reference).
///
/// Causal: the first frame is replicated `kt - 1` times in front. Non-causal: the first and the last frame are each
/// replicated `(kt - 1) / 2` times. Spatial padding is symmetric zeros.
struct LTXCausalConv3d {
    let weight: MLXArray
    let bias: MLXArray
    let causal: Bool
    let spatialPad: Int

    init(store: WeightStore, prefix: String, causal: Bool, spatialPad: Int = 1, dtype: DType? = nil) throws {
        var w = try store.get(prefix + ".weight")
        guard w.ndim == 5 else { throw WeightError.badShape(prefix + ".weight", w.shape) }
        var b = try store.get(prefix + ".bias")
        if let dtype {
            w = w.asType(dtype)
            b = b.asType(dtype)
        }
        weight = w
        bias = b
        self.causal = causal
        self.spatialPad = spatialPad
    }

    var kt: Int { weight.dim(1) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        let k = kt
        if causal {
            if k > 1 {
                let first = repeated(x[0..., 0 ..< 1], count: k - 1, axis: 1)
                x = concatenated([first, x], axis: 1)
            }
        } else {
            let p = (k - 1) / 2
            if p > 0 {
                let f = x.dim(1)
                let first = repeated(x[0..., 0 ..< 1], count: p, axis: 1)
                let last = repeated(x[0..., (f - 1) ..< f], count: p, axis: 1)
                x = concatenated([first, x, last], axis: 1)
            }
        }
        // Explicit zero pad then an unpadded conv, as the reference does: MLX picks a different (numerically
        // slightly different) kernel when the conv pads internally.
        if spatialPad > 0 {
            let p = IntOrPair((spatialPad, spatialPad))
            x = padded(x, widths: [0, 0, p, p, 0])
        }
        return conv3d(x, weight) + bias
    }
}

/// Parameter-free RMS norm over channels (`PixelNorm`, eps 1e-8).
func ltxPixelNorm(_ x: MLXArray) -> MLXArray {
    rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: 1e-8)
}

/// `nn.GroupNorm(groups, C, pytorch_compatible=True)` on a channels-last tensor of any rank (statistics span every
/// non-batch position of the group, frames included), then the affine `weight * x + bias`.
func ltxGroupNorm(_ x: MLXArray, groups: Int, weight: MLXArray, bias: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let shape = x.shape
    let b = shape[0]
    let c = shape[shape.count - 1]
    let gs = c / groups
    var y = x.reshaped(b, -1, groups, gs).transposed(0, 2, 1, 3).reshaped(b, groups, -1)
    y = layerNorm(y, eps: eps)
    y = y.reshaped(b, groups, -1, gs).transposed(0, 2, 1, 3).reshaped(shape)
    return weight * y + bias
}

/// Depth-to-space: `[B, D, H, W, C·tf·sf·sf] → [B, D·tf, H·sf, W·sf, C]` with channel order `(c, t, h, w)`.
func ltxPixelShuffle3d(_ x: MLXArray, spatial sf: Int, temporal tf: Int) -> MLXArray {
    let (b, d, h, w, ct) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let c = ct / (sf * sf * tf)
    return x.reshaped(b, d, h, w, c, tf, sf, sf)
        .transposed(0, 1, 5, 2, 6, 3, 7, 4)
        .reshaped(b, d * tf, h * sf, w * sf, c)
}

/// 2-D depth-to-space on `[N, H, W, C·f·f]` with channel order `(c, h, w)`.
func ltxPixelShuffle2d(_ x: MLXArray, factor f: Int) -> MLXArray {
    let (n, h, w, ct) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = ct / (f * f)
    return x.reshaped(n, h, w, c, f, f).transposed(0, 1, 4, 2, 5, 3).reshaped(n, h * f, w * f, c)
}

/// Final VAE unpatchify: `[B, F, H, W, C·p·p] → [B, F, H·p, W·p, C]` with channel order `(c, w, h)` (width factor
/// before height factor, unlike ``ltxPixelShuffle3d``).
func ltxUnpatchify(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w, ct) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let c = ct / (p * p)
    return x.reshaped(b, f, h, w, c, p, p)
        .transposed(0, 1, 2, 6, 3, 5, 4)
        .reshaped(b, f, h * p, w * p, c)
}

/// Inverse of ``ltxUnpatchify``: `[B, F, H, W, C] → [B, F, H/p, W/p, C·p·p]`.
func ltxPatchify(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    return x.reshaped(b, f, h / p, p, w / p, p, c)
        .transposed(0, 1, 2, 4, 6, 5, 3)
        .reshaped(b, f, h / p, w / p, c * p * p)
}

/// Space-to-depth: `[B, D·st, H·sh, W·sw, C] → [B, D, H, W, C·st·sh·sw]` with channel order `(c, t, h, w)`.
func ltxSpaceToDepth(_ x: MLXArray, st: Int, sh: Int, sw: Int) -> MLXArray {
    let (b, df, hf, wf, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let (d, h, w) = (df / st, hf / sh, wf / sw)
    return x.reshaped(b, d, st, h, sh, w, sw, c)
        .transposed(0, 1, 3, 5, 7, 2, 4, 6)
        .reshaped(b, d, h, w, c * st * sh * sw)
}

/// Pre-activation residual block: `x + conv2(silu(norm(conv1(silu(norm(x))))))`.
struct LTXResBlock3d {
    let conv1: LTXCausalConv3d
    let conv2: LTXCausalConv3d

    init(store: WeightStore, prefix: String, causal: Bool) throws {
        conv1 = try LTXCausalConv3d(store: store, prefix: prefix + ".conv1.conv", causal: causal)
        conv2 = try LTXCausalConv3d(store: store, prefix: prefix + ".conv2.conv", causal: causal)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1(siluAct(ltxPixelNorm(x)))
        y = conv2(siluAct(ltxPixelNorm(y)))
        return y + x
    }
}

/// `res_blocks.{i}` of a `ResBlockStage`; the block count is read from the checkpoint.
struct LTXResStage {
    let blocks: [LTXResBlock3d]

    init(store: WeightStore, prefix: String, causal: Bool) throws {
        var bs: [LTXResBlock3d] = []
        while store.has("\(prefix).res_blocks.\(bs.count).conv1.conv.weight") {
            bs.append(try LTXResBlock3d(store: store, prefix: "\(prefix).res_blocks.\(bs.count)", causal: causal))
        }
        guard !bs.isEmpty else { throw WeightError.missing("\(prefix).res_blocks.0.conv1.conv.weight") }
        blocks = bs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for b in blocks { y = b(y) }
        return y
    }
}

/// Strips `prefix` from every key (the LTX packs prefix each tensor with the file stem, e.g. `vae_decoder.`).
func ltxStripPrefix(_ store: WeightStore, _ prefix: String) -> WeightStore {
    guard store.keys.contains(where: { $0.hasPrefix(prefix) }) else { return store }
    var out: [String: MLXArray] = [:]
    for (k, v) in store.arrays {
        out[k.hasPrefix(prefix) ? String(k.dropFirst(prefix.count)) : k] = v
    }
    return WeightStore(arrays: out, metadata: store.metadata)
}
