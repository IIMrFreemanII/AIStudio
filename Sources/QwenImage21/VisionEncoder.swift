import Foundation
import MLX
import MLXDiffusionCore

/// The Qwen3-VL vision tower (`model.visual`), for single images.
///
/// Patches are processed in 2×2 spatial-merge-block order (block row, block col, in-row, in-col), exactly as the
/// Qwen2-VL image processor emits them, so the merger can fold every 4 consecutive patches into one LLM token.
public final class Qwen3VisionTower {
    public struct Config {
        public var hidden = 1152
        public var heads = 16
        public var depth = 27
        public var intermediate = 4304
        public var patch = 16
        public var merge = 2
        public var outHidden = 4096
        public var numPositions = 2304
        public var deepstackIndexes = [8, 16, 24]
        public var ropeTheta: Float = 10000

        public init(textEncoderConfig url: URL) throws {
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            guard let v = obj["vision_config"] as? [String: Any] else { return }
            hidden = v["hidden_size"] as? Int ?? hidden
            heads = v["num_heads"] as? Int ?? heads
            depth = v["depth"] as? Int ?? depth
            intermediate = v["intermediate_size"] as? Int ?? intermediate
            patch = v["patch_size"] as? Int ?? patch
            merge = v["spatial_merge_size"] as? Int ?? merge
            outHidden = v["out_hidden_size"] as? Int ?? outHidden
            numPositions = v["num_position_embeddings"] as? Int ?? numPositions
            deepstackIndexes = v["deepstack_visual_indexes"] as? [Int] ?? deepstackIndexes
        }

        public var headDim: Int { hidden / heads }
        /// Pixel multiple an input image must have (one merged token covers `patch * merge` pixels).
        public var pixelMultiple: Int { patch * merge }
    }

    struct Block {
        let norm1W, norm1B, norm2W, norm2B: MLXArray
        let qkv, proj, fc1, fc2: Lin
    }

    struct Merger {
        let normW, normB: MLXArray
        let fc1, fc2: Lin
        let postShuffleNorm: Bool
    }

    public let config: Config
    let patchWeight: MLXArray  // [hidden, 3*p*p], temporal axis summed (both frames are the same image)
    let patchBias: MLXArray
    let posEmbed: MLXArray  // [numPositions, hidden]
    let blocks: [Block]
    let merger: Merger
    let deepstackMergers: [Merger]

    init(store: WeightStore, config: Config, dtype: DType) throws {
        self.config = config
        let p = "model.visual."
        let w = try store.get(p + "patch_embed.proj.weight").asType(.float32)  // [hidden, 3, t, p, p]
        patchWeight = w.sum(axis: 2).reshaped(config.hidden, -1).asType(dtype)
        patchBias = try store.get(p + "patch_embed.proj.bias").asType(dtype)
        posEmbed = try store.get(p + "pos_embed.weight").asType(dtype)
        blocks = try (0..<config.depth).map { i in
            let b = "\(p)blocks.\(i)."
            return Block(
                norm1W: try store.get(b + "norm1.weight").asType(dtype), norm1B: try store.get(b + "norm1.bias").asType(dtype),
                norm2W: try store.get(b + "norm2.weight").asType(dtype), norm2B: try store.get(b + "norm2.bias").asType(dtype),
                qkv: try store.linear(b + "attn.qkv", dtype: dtype), proj: try store.linear(b + "attn.proj", dtype: dtype),
                fc1: try store.linear(b + "mlp.linear_fc1", dtype: dtype),
                fc2: try store.linear(b + "mlp.linear_fc2", dtype: dtype))
        }
        func loadMerger(_ m: String, post: Bool) throws -> Merger {
            Merger(normW: try store.get(m + "norm.weight").asType(dtype), normB: try store.get(m + "norm.bias").asType(dtype),
                   fc1: try store.linear(m + "linear_fc1", dtype: dtype), fc2: try store.linear(m + "linear_fc2", dtype: dtype),
                   postShuffleNorm: post)
        }
        merger = try loadMerger(p + "merger.", post: false)
        deepstackMergers = try config.deepstackIndexes.indices.map { try loadMerger("\(p)deepstack_merger_list.\($0).", post: true) }
    }

    /// Output of the tower for one image.
    public struct Output {
        /// Merged visual tokens `[n, outHidden]`, raster order over the merged grid.
        public let embeddings: MLXArray
        /// Per-deepstack-layer features `[n, outHidden]`, same order.
        public let deepstack: [MLXArray]
        /// Merged grid size (LLM tokens = gridH * gridW).
        public let gridH: Int
        public let gridW: Int
    }

    /// `rgb`: `[H, W, 3]` in `[0, 1]`, H and W multiples of ``Config/pixelMultiple``.
    public func callAsFunction(_ rgb: MLXArray) -> Output {
        let c = config
        let (H, W) = (rgb.dim(0), rgb.dim(1))
        let gh = H / c.patch, gw = W / c.patch, m = c.merge
        let n = gh * gw

        // Patchify in merge-block order: [gh/m, m, p, gw/m, m, p, C] → [gh/m, gw/m, m, m, C, p, p].
        let x = ((rgb.asType(.float32) - 0.5) / 0.5).asType(patchWeight.dtype)
            .reshaped(gh / m, m, c.patch, gw / m, m, c.patch, 3)
            .transposed(0, 3, 1, 4, 6, 2, 5)
            .reshaped(n, 3 * c.patch * c.patch)
        var h = matmul(x, patchWeight.T) + patchBias

        // Patch (row, col) in the same order.
        var rows = [Int32](), cols = [Int32]()
        rows.reserveCapacity(n); cols.reserveCapacity(n)
        for br in 0..<(gh / m) {
            for bc in 0..<(gw / m) {
                for ir in 0..<m {
                    for ic in 0..<m {
                        rows.append(Int32(br * m + ir))
                        cols.append(Int32(bc * m + ic))
                    }
                }
            }
        }
        h = h + interpolatedPositions(rows: rows, cols: cols, gh: gh, gw: gw)

        let rope = visionRope(rows: rows, cols: cols)
        var deep: [MLXArray] = []
        for (i, b) in blocks.enumerated() {
            h = h + attention(b, layerNorm(h, weight: b.norm1W, bias: b.norm1B, eps: 1e-6), rope: rope)
            h = h + b.fc2(geluTanh(b.fc1(layerNorm(h, weight: b.norm2W, bias: b.norm2B, eps: 1e-6))))
            if let k = c.deepstackIndexes.firstIndex(of: i) { deep.append(merge(deepstackMergers[k], h)) }
        }
        let out = merge(merger, h)
        eval([out] + deep)
        return Output(embeddings: out, deepstack: deep, gridH: gh / m, gridW: gw / m)
    }

    /// Bilinear (align-corners, border-clamped) resampling of the learned square position grid.
    func interpolatedPositions(rows: [Int32], cols: [Int32], gh: Int, gw: Int) -> MLXArray {
        let side = Int(Double(config.numPositions).squareRoot())
        func taps(_ i: Int32, _ size: Int) -> (Int, Int, Float, Float) {
            let src = Float(i) * Float(side - 1) / Float(max(size - 1, 1))
            let f = src.rounded(.down)
            let t = src - f
            let a = min(max(Int(f), 0), side - 1), b = min(max(Int(f) + 1, 0), side - 1)
            return (a, b, 1 - t, t)
        }
        var idx = [Int32](), wts = [Float]()
        idx.reserveCapacity(rows.count * 4); wts.reserveCapacity(rows.count * 4)
        for k in rows.indices {
            let (r0, r1, wr0, wr1) = taps(rows[k], gh)
            let (c0, c1, wc0, wc1) = taps(cols[k], gw)
            idx += [Int32(r0 * side + c0), Int32(r0 * side + c1), Int32(r1 * side + c0), Int32(r1 * side + c1)]
            wts += [wr0 * wc0, wr0 * wc1, wr1 * wc0, wr1 * wc1]
        }
        let gathered = posEmbed[MLXArray(idx)].reshaped(rows.count, 4, config.hidden).asType(.float32)
        let w = MLXArray(wts, [rows.count, 4, 1])
        return (gathered * w).sum(axis: 1).asType(posEmbed.dtype)
    }

    /// 2D RoPE: the first half of the head's frequencies follows rows, the second half columns (rotate-half form).
    func visionRope(rows: [Int32], cols: [Int32]) -> (MLXArray, MLXArray) {
        let spatial = config.headDim / 2
        let inv = MLXArray((0 ..< spatial / 2).map { 1 / pow(config.ropeTheta, Float(2 * $0) / Float(spatial)) })
        let fr = MLXArray(rows).asType(.float32).expandedDimensions(axis: 1) * inv
        let fc = MLXArray(cols).asType(.float32).expandedDimensions(axis: 1) * inv
        let half = concatenated([fr, fc], axis: 1)  // [n, headDim/2]
        let full = concatenated([half, half], axis: 1)  // [n, headDim]
        return (cos(full).expandedDimensions(axis: 1), sin(full).expandedDimensions(axis: 1))
    }

    func attention(_ b: Block, _ x: MLXArray, rope: (MLXArray, MLXArray)) -> MLXArray {
        let n = x.dim(0), c = config
        let qkv = b.qkv(x).reshaped(n, 3, c.heads, c.headDim)
        let q = applyRotateHalf(qkv[0..., 0], rope).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = applyRotateHalf(qkv[0..., 1], rope).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = qkv[0..., 2].transposed(1, 0, 2).expandedDimensions(axis: 0)
        let o = scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(c.headDim)), mask: nil)
        return b.proj(o[0].transposed(1, 0, 2).reshaped(n, c.hidden))
    }

    func merge(_ m: Merger, _ x: MLXArray) -> MLXArray {
        let groups = x.dim(0) / (config.merge * config.merge)
        var y: MLXArray
        if m.postShuffleNorm {
            y = layerNorm(x.reshaped(groups, -1), weight: m.normW, bias: m.normB, eps: 1e-6)
        } else {
            y = layerNorm(x, weight: m.normW, bias: m.normB, eps: 1e-6).reshaped(groups, -1)
        }
        return m.fc2(geluErf(m.fc1(y)))
    }
}

/// `x * cos + rotate_half(x) * sin` in float32; `rope` broadcasts over the head axis.
func applyRotateHalf(_ x: MLXArray, _ rope: (MLXArray, MLXArray)) -> MLXArray {
    let d = x.dim(-1)
    let xf = x.asType(.float32)
    let x1 = xf[.ellipsis, ..<(d / 2)]
    let x2 = xf[.ellipsis, (d / 2)...]
    let rotated = concatenated([-x2, x1], axis: -1)
    return (xf * rope.0 + rotated * rope.1).asType(x.dtype)
}

/// Exact (erf) GELU, as `nn.GELU()`.
func geluErf(_ x: MLXArray) -> MLXArray { 0.5 * x * (1 + erf(x / Float(2).squareRoot())) }
