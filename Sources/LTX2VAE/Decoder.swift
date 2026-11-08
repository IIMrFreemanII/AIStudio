import Foundation
import MLX
import MLXDiffusionCore

/// LTX-2.3 video VAE decoder (`vae_decoder.safetensors`, reference `VideoDecoder`).
///
/// Input: a **normalized** latent `[B, F, H, W, 128]` (channels-last, as the transformer produces it). The decoder
/// denormalizes internally with its own per-channel statistics (`per_channel_statistics.{mean,std}` in
/// `vae_decoder.safetensors`; ``mean`` / ``std``). Output: `[B, 1 + 8·(F−1), 32·H, 32·W, 3]`.
///
/// LTX-2.3 configuration (`embedded_config.json` → `vae`): non-causal decoder (first/last frame replicate padding in
/// time), zero spatial padding, no timestep conditioning, no decode-time noise. The conv decoder is deterministic;
/// the reference default pipeline decodes with it (`video_decoder = "conv"`).
public final class LTXVideoDecoder {
    /// Per-channel latent mean / std, `[128]`.
    public let mean: MLXArray
    public let std: MLXArray
    /// Compute dtype (the checkpoint dtype, bf16).
    public let dtype: DType
    /// Debug hook: receives named intermediates (`conv_in`, `up0` … `up8`) during ``decode(_:materializeStages:)``.
    public var debugTrace: ((String, MLXArray) -> Void)?

    let convIn: LTXCausalConv3d
    let stages: [LTXResStage]  // up_blocks 0, 2, 4, 6, 8
    let upsamplers: [LTXCausalConv3d]  // up_blocks 1, 3, 5, 7
    let upsampleFactors: [(spatial: Int, temporal: Int)] = [(2, 2), (2, 2), (1, 2), (2, 1)]
    let convOut: LTXCausalConv3d

    /// Loads `vae_decoder.safetensors` (keys may carry the `vae_decoder.` file-stem prefix).
    public convenience init(url: URL, causal: Bool = false) throws {
        try self.init(store: WeightStore(url: url), causal: causal)
    }

    public init(store raw: WeightStore, causal: Bool = false) throws {
        let store = ltxStripPrefix(raw, "vae_decoder.")
        convIn = try LTXCausalConv3d(store: store, prefix: "conv_in.conv", causal: causal)
        stages = try [0, 2, 4, 6, 8].map { try LTXResStage(store: store, prefix: "up_blocks.\($0)", causal: causal) }
        upsamplers = try [1, 3, 5, 7].map {
            try LTXCausalConv3d(store: store, prefix: "up_blocks.\($0).conv.conv", causal: causal)
        }
        convOut = try LTXCausalConv3d(store: store, prefix: "conv_out.conv", causal: causal)
        mean = try store.get("per_channel_statistics.mean")
        std = try store.get("per_channel_statistics.std")
        dtype = convIn.weight.dtype
    }

    /// `x * std + mean` over the channel axis.
    public func denormalize(_ latent: MLXArray) -> MLXArray { latent * std + mean }

    /// Decodes a normalized latent `[B, F, H, W, 128]` in one pass to pixels `[B, T, 32H, 32W, 3]` in roughly
    /// `[-1, 1]` (not clipped), returned in the latent's dtype. Peak memory grows with the clip volume; see
    /// ``decodeTiled(_:tiling:onChunk:)`` for long or large clips.
    public func decode(_ latent: MLXArray, materializeStages: Bool = false) -> MLXArray {
        let outDtype = latent.dtype
        var x = latent.dtype == dtype ? latent : latent.asType(dtype)
        x = denormalize(x)
        x = convIn(x)
        debugTrace?("conv_in", x)
        for i in 0..<4 {
            x = stages[i](x)
            debugTrace?("up\(2 * i)", x)
            x = upsamplers[i](x)
            let (sf, tf) = upsampleFactors[i]
            x = ltxPixelShuffle3d(x, spatial: sf, temporal: tf)
            if tf > 1 { x = x[0..., 1...] }  // drop the first frame after every temporal ×2
            debugTrace?("up\(2 * i + 1)", x)
            if materializeStages { eval(x) }
        }
        x = stages[4](x)
        debugTrace?("up8", x)
        x = convOut(siluAct(ltxPixelNorm(x)))
        x = ltxUnpatchify(x, patch: 4)
        return x.dtype == outDtype ? x : x.asType(outDtype)
    }

    /// Tiled decode (reference `tiled_decode`): cuts the latent into overlapping tiles, decodes each, blends overlaps
    /// with trapezoidal masks and hands over finished frames in temporal order as `[B, t, 32H, 32W, 3]` chunks in
    /// roughly `[-1, 1]` (float32 when tiled, latent dtype when `tiling` is nil). `onChunk` gets each chunk and the
    /// index of its first frame. Chunks concatenated along axis 1 give the whole clip.
    public func decodeTiled(_ latent: MLXArray, tiling: LTXVAETiling?,
                            onChunk: (_ chunk: MLXArray, _ firstFrame: Int) throws -> Void) rethrows {
        guard let tiling else {
            let px = decode(latent)
            eval(px)
            MLX.GPU.clearCache()
            try onChunk(px, 0)
            return
        }
        let (b, f, h, w) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        let outH = h * 32, outW = w * 32
        let tiles = ltxDecodeTiles(frames: f, height: h, width: w, tiling: tiling)

        // group consecutive tiles sharing a temporal output range
        var groups: [[LTXDecodeTile]] = []
        for t in tiles {
            if let last = groups.last?.first, last.outT == t.outT { groups[groups.count - 1].append(t) } else { groups.append([t]) }
        }

        var prevBuf: MLXArray?
        var prevW: MLXArray?
        var prevRange: Range<Int>?
        for group in groups {
            let cur = group[0].outT
            let tlen = cur.count
            var buf = MLXArray.zeros([b, tlen, outH, outW, 3], dtype: .float32)
            var wts = MLXArray.zeros([b, tlen, outH, outW, 3], dtype: .float32)
            for tile in group {
                let decoded = decode(latent[0..., tile.inF, tile.inH, tile.inW], materializeStages: true)
                eval(decoded)
                MLX.GPU.clearCache()
                let mask = tile.blendMask
                let offset = tile.outT.lowerBound - cur.lowerBound
                let n = min(tile.outT.count, decoded.dim(1), tlen - offset)
                let dec = decoded[0..., 0 ..< n]
                let m = mask.dim(1) > 1 ? mask[0..., 0 ..< n] : mask
                let tr = offset ..< (offset + n)
                buf[0..., tr, tile.outH, tile.outW, 0...] = buf[0..., tr, tile.outH, tile.outW, 0...] + dec * m
                wts[0..., tr, tile.outH, tile.outW, 0...] = wts[0..., tr, tile.outH, tile.outW, 0...] + m
                eval(buf, wts)
                MLX.GPU.clearCache()
            }
            if let pb = prevBuf, let pw = prevW, let pr = prevRange {
                if pr.upperBound > cur.lowerBound {
                    let overlap = pr.upperBound - cur.lowerBound
                    let ps = cur.lowerBound - pr.lowerBound
                    let merged = pb[0..., ps...] + buf[0..., 0 ..< overlap]
                    let mergedW = pw[0..., ps...] + wts[0..., 0 ..< overlap]
                    prevBuf = concatenated([pb[0..., 0 ..< ps], merged], axis: 1)
                    prevW = concatenated([pw[0..., 0 ..< ps], mergedW], axis: 1)
                    buf = concatenated([merged, buf[0..., overlap...]], axis: 1)
                    wts = concatenated([mergedW, wts[0..., overlap...]], axis: 1)
                }
                let yieldLen = cur.lowerBound - pr.lowerBound
                if yieldLen > 0, let pb2 = prevBuf, let pw2 = prevW {
                    let chunk = (pb2 / maximum(pw2, MLXArray(Float(1e-8))))[0..., 0 ..< yieldLen]
                    eval(chunk)
                    try onChunk(chunk, pr.lowerBound)
                }
            }
            prevBuf = buf
            prevW = wts
            prevRange = cur
        }
        if let pb = prevBuf, let pw = prevW, let pr = prevRange {
            let chunk = pb / maximum(pw, MLXArray(Float(1e-8)))
            eval(chunk)
            try onChunk(chunk, pr.lowerBound)
        }
    }

    /// Pipeline-default decode of a batch-1 normalized latent `[1, F, H, W, 128]` to frames `[T, 32H, 32W, 3]`
    /// float32 in `[0, 1]` (reference `decode_and_stream`: `clip(x, -1, 1)`, then `(x + 1) / 2`; the reference then
    /// truncates `x * 255` to uint8). `tiling` defaults to the reference's automatic choice
    /// (``LTXVAETiling/auto(frames:height:width:frameRate:budgetBytes:)``, budget = half of unified memory).
    /// `onFrames`, if set, receives each finished chunk of frames and the index of its first frame as soon as it is
    /// ready; the whole clip is returned only when `collect` is true (else an empty array, and memory stays bounded by
    /// one temporal tile). MLX's allocator cache is disabled for the duration, as in the reference.
    @discardableResult
    public func decodeFrames(_ latent: MLXArray, tiling: LTXDecodeTiling = .auto(), collect: Bool = true,
                             onFrames: ((_ frames: MLXArray, _ firstFrame: Int) throws -> Void)? = nil) rethrows
        -> MLXArray
    {
        let chosen: LTXVAETiling?
        switch tiling {
        case .untiled: chosen = nil
        case .fixed(let t): chosen = t
        case .auto(let budget, let fps):
            chosen = LTXVAETiling.auto(frames: latent.dim(1), height: latent.dim(2), width: latent.dim(3),
                                       frameRate: fps, budgetBytes: budget)
        }
        var parts: [MLXArray] = []
        // As the reference (`decode_cache_limit`): no allocator cache during the decode, so the large short-lived
        // activations of each tile are returned to the system instead of parking in MLX's free list.
        let savedCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = savedCacheLimit }
        try decodeTiled(latent, tiling: chosen) { chunk, first in
            let frames = ((clip(chunk.asType(.float32), min: -1, max: 1) + 1) / 2)[0]
            eval(frames)
            try onFrames?(frames, first)
            if collect { parts.append(frames) }
        }
        guard collect else { return MLXArray.zeros([0]) }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
    }
}

/// How ``LTXVideoDecoder/decodeFrames(_:tiling:collect:onFrames:)`` tiles.
public enum LTXDecodeTiling: Sendable {
    /// Single pass.
    case untiled
    /// These tiles.
    case fixed(LTXVAETiling)
    /// The reference's budget ladder (`budgetBytes` nil = half of unified memory / `LTX2_VAE_DECODE_BUDGET_GB`).
    case auto(budgetBytes: Int? = nil, frameRate: Double = 24)
}
