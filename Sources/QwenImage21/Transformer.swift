import Foundation
import MLX
import MLXDiffusionCore

/// `QwenImage21Transformer2DModel`: a single-stream, block-causal DiT.
///
/// The joint sequence is `[prefix (text + condition images), target image]`. Attention is causal except inside image
/// blocks, so prefix tokens never attend to the target, and with `causal_condition` the prefix is modulated from
/// `t = 0`. The prefix activations are therefore step-independent: ``encodePrefix(_:)`` runs them once and caches
/// every block's keys/values, and ``denoise(latents:prefix:sigma:latentHeight:latentWidth:)`` only runs the target
/// image tokens, attending to `[cached prefix, target]` without a mask. This is exactly the reference's KV-cache path.
public final class QwenImage21Transformer {
    public struct Config {
        public var inChannels = 64
        public var outChannels = 64
        public var numLayers = 32
        public var headDim = 128
        public var numHeads = 32
        public var contextInDim = 4096
        public var mlpRatio = 3
        public var axesDims = [16, 56, 56]
        public var eps: Float = 1e-6
        public var causalCondition = true

        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            inChannels = o["in_channels"] as? Int ?? inChannels
            outChannels = o["out_channels"] as? Int ?? inChannels
            numLayers = o["num_layers"] as? Int ?? numLayers
            headDim = o["attention_head_dim"] as? Int ?? headDim
            numHeads = o["num_attention_heads"] as? Int ?? numHeads
            contextInDim = o["context_in_dim"] as? Int ?? contextInDim
            mlpRatio = o["mlp_ratio"] as? Int ?? mlpRatio
            axesDims = o["axes_dims_rope"] as? [Int] ?? axesDims
            eps = Float(o["eps"] as? Double ?? Double(eps))
            causalCondition = o["causal_condition"] as? Bool ?? causalCondition
            precondition((o["patch_size"] as? Int ?? 1) == 1, "only patch_size 1 is supported")
            precondition(causalCondition, "the prefix cache requires causal_condition")
        }

        public var innerDim: Int { headDim * numHeads }
    }

    struct Block {
        var q, k, v, o: Lin
        let normQ, normK: MLXArray
        var proj, gate, out: Lin

        mutating func mapLinears(_ f: (Lin) -> Lin) {
            q = f(q); k = f(k); v = f(v); o = f(o); proj = f(proj); gate = f(gate); out = f(out)
        }
    }

    /// Per-block keys/values of the prefix, `[1, heads, Lp, headDim]`, plus the prefix length and RoPE cursor.
    public struct PrefixCache {
        public let keys: [MLXArray]
        public let values: [MLXArray]
        public let length: Int
        /// The frame position the target image block sits at (end of the prefix on the frame axis).
        public let nextPosition: Int
    }

    /// Description of the prefix sequence: text runs and condition image blocks (in latent tokens).
    public enum Segment {
        case text(MLXArray)  // [1, n, contextInDim] encoder features
        case image(MLXArray, height: Int, width: Int)  // [1, h*w, inChannels] normalised VAE latents
    }

    public let config: Config
    let txtNorm: MLXArray
    private(set) var txtIn1, txtIn2: Lin
    private(set) var imgIn: Lin
    private(set) var time1, time2: Lin
    private(set) var modLinear: Lin
    private(set) var blocks: [Block]
    private(set) var normOutLinear: Lin
    private(set) var projOut: Lin
    let ropeInvFreqs: [MLXArray]

    private init(copying o: QwenImage21Transformer) {
        config = o.config; txtNorm = o.txtNorm; txtIn1 = o.txtIn1; txtIn2 = o.txtIn2; imgIn = o.imgIn
        time1 = o.time1; time2 = o.time2; modLinear = o.modLinear; blocks = o.blocks
        normOutLinear = o.normOutLinear; projOut = o.projOut; ropeInvFreqs = o.ropeInvFreqs
    }

    /// Names of every linear layer (checkpoint key prefixes), for matching LoRA files.
    public var linearNames: Set<String> {
        var names: [String] = [txtIn1, txtIn2, imgIn, time1, time2, modLinear, normOutLinear, projOut].map(\.name)
        for b in blocks { names += [b.q, b.k, b.v, b.o, b.proj, b.gate, b.out].map(\.name) }
        return Set(names)
    }

    /// A transformer sharing these weights with `adapters` added to the matching layers; returns the match count.
    public func adding(_ adapters: [String: LoRAAdapter]) -> (QwenImage21Transformer, Int) {
        let t = QwenImage21Transformer(copying: self)
        var matched = 0
        func f(_ l: Lin) -> Lin {
            guard let a = adapters[l.name] else { return l }
            matched += 1
            return l.adding(a)
        }
        t.txtIn1 = f(t.txtIn1); t.txtIn2 = f(t.txtIn2); t.imgIn = f(t.imgIn); t.time1 = f(t.time1)
        t.time2 = f(t.time2); t.modLinear = f(t.modLinear); t.normOutLinear = f(t.normOutLinear)
        t.projOut = f(t.projOut)
        for i in t.blocks.indices { t.blocks[i].mapLinears(f) }
        return (t, matched)
    }

    /// - Parameter dequantize: expand 8-bit block weights to bf16 at load (≈2× the memory, faster matmuls).
    public init(directory: URL, dequantize: Bool = false) throws {
        config = try Config(url: directory.appendingPathComponent("config.json"))
        let store = try WeightStore(directory: directory)
        let s = LinearSource(store: store, dense: dequantize)
        txtNorm = try store.get("txt_in.text_norm.weight").asType(.float32) + 1  // zero-centred RMSNorm weight
        txtIn1 = try s.linear("txt_in.in_layer")
        txtIn2 = try s.linear("txt_in.out_layer")
        imgIn = try s.linear("img_in")
        time1 = try s.linear("time_text_embed.timestep_embedder.linear_1")
        time2 = try s.linear("time_text_embed.timestep_embedder.linear_2")
        modLinear = try s.linear("modulation.1")
        blocks = try (0..<config.numLayers).map { i in
            let p = "transformer_blocks.\(i)."
            return Block(
                q: try s.linear(p + "attn.to_q"), k: try s.linear(p + "attn.to_k"),
                v: try s.linear(p + "attn.to_v"), o: try s.linear(p + "attn.to_out.0"),
                normQ: try store.get(p + "attn.norm_q.weight"), normK: try store.get(p + "attn.norm_k.weight"),
                proj: try s.linear(p + "img_mlp.proj"), gate: try s.linear(p + "img_mlp.gate_layer"),
                out: try s.linear(p + "img_mlp.out"))
        }
        normOutLinear = try s.linear("norm_out.linear")
        projOut = try s.linear("proj_out")
        ropeInvFreqs = config.axesDims.map { d in
            let exps = (0 ..< d / 2).map { Float(2 * $0) / Float(d) }
            return MLXArray(exps.map { 1 / pow(Float(10000), $0) })
        }
    }

    // MARK: modulation

    struct Modulation {
        let scale1, gate1, scale2, gate2: MLXArray  // [1, 1, D] each
        let outScale: MLXArray  // [1, 1, D]
    }

    /// Sinusoidal embedding with `cos` first; `sigma` in [0, 1] (the reference multiplies by 1000 twice over).
    func timestepEmbedding(sigma: Float) -> MLXArray {
        let half = 128
        let freqs = (0..<half).map { exp(-log(Float(10000)) * Float($0) / Float(half)) }
        let t = sigma * 1000
        let args = freqs.map { t * $0 }
        let emb = MLXArray(args.map(cos) + args.map(sin)).reshaped(1, 256).asType(.bfloat16)
        return time2(siluAct(time1(emb)))
    }

    func modulation(sigma: Float) -> Modulation {
        let temb = timestepEmbedding(sigma: sigma)
        let m = modLinear(siluAct(temb))  // [1, 4D]
        let d = config.innerDim
        func part(_ i: Int) -> MLXArray { m[0..., (i * d) ..< ((i + 1) * d)].reshaped(1, 1, d) }
        let outScale = normOutLinear(siluAct(temb)).reshaped(1, 1, d)
        return Modulation(scale1: part(0), gate1: tanh(part(1)), scale2: part(2), gate2: tanh(part(3)),
                          outScale: outScale)
    }

    // MARK: RoPE

    /// Complex rotation tables `(cos, sin)` of shape `[S, 1, headDim/2]` for 3-axis integer positions.
    func rope(frame: [Int32], height: [Int32], width: [Int32]) -> (MLXArray, MLXArray) {
        let axes = [frame, height, width]
        let angles = concatenated(zip(axes, ropeInvFreqs).map { pos, inv in
            MLXArray(pos).asType(.float32).expandedDimensions(axis: 1) * inv.expandedDimensions(axis: 0)
        }, axis: 1)
        return (cos(angles).expandedDimensions(axis: 1), sin(angles).expandedDimensions(axis: 1))
    }

    /// Applies interleaved-pair complex rotation to `x: [1, S, H, D]`.
    func applyRope(_ x: MLXArray, _ r: (MLXArray, MLXArray)) -> MLXArray {
        let shape = x.shape
        let xf = x.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let re = xf[.ellipsis, 0]
        let im = xf[.ellipsis, 1]
        let (c, s) = r
        let outRe = re * c - im * s
        let outIm = re * s + im * c
        return stacked([outRe, outIm], axis: -1).reshaped(shape).asType(x.dtype)
    }

    /// Spatial positions of an image block, centred on zero: `-(n - n/2) ..< n/2`.
    static func centred(_ n: Int) -> [Int32] { (0..<n).map { Int32($0 - (n - n / 2)) } }

    // MARK: blocks

    func qkv(_ b: Block, _ h: MLXArray, rope r: (MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
        let (n, s) = (h.dim(0), h.dim(1))
        let c = config
        var q = b.q(h).reshaped(n, s, c.numHeads, c.headDim)
        var k = b.k(h).reshaped(n, s, c.numHeads, c.headDim)
        let v = b.v(h).reshaped(n, s, c.numHeads, c.headDim)
        q = applyRope(rmsNorm(q, weight: b.normQ, eps: c.eps), r)
        k = applyRope(rmsNorm(k, weight: b.normK, eps: c.eps), r)
        return (q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3))
    }

    func attentionOut(_ b: Block, _ a: MLXArray) -> MLXArray {
        let (n, s) = (a.dim(0), a.dim(2))
        return b.o(a.transposed(0, 2, 1, 3).reshaped(n, s, config.innerDim))
    }

    func mlp(_ b: Block, _ x: MLXArray) -> MLXArray { b.out(siluAct(b.gate(x)) * b.proj(x)) }

    // MARK: public API

    /// Runs the prefix (text and optional condition images) once and caches per-block keys/values.
    public func encodePrefix(_ segments: [Segment]) -> PrefixCache {
        let c = config
        var parts: [MLXArray] = []
        var frame: [Int32] = [], hIdx: [Int32] = [], wIdx: [Int32] = []
        var imageIds: [Int32] = []  // -1 text, else block id
        var position: Int32 = 0
        var blockId: Int32 = 0
        for seg in segments {
            switch seg {
            case .text(let feats):
                let n = feats.dim(1)
                parts.append(txtIn2(geluTanh(txtIn1(textNorm(feats)))))
                let pos = (0..<n).map { position + Int32($0) }
                frame += pos; hIdx += pos; wIdx += pos
                imageIds += Array(repeating: -1, count: n)
                position += Int32(n)
            case .image(let lat, let h, let w):
                parts.append(imgIn(lat.asType(.bfloat16)))
                frame += Array(repeating: position, count: h * w)
                hIdx += Self.centred(h).flatMap { Array(repeating: $0, count: w) }
                wIdx += Array((0..<h).map { _ in Self.centred(w) }.joined())
                imageIds += Array(repeating: blockId, count: h * w)
                blockId += 1
                position += Int32(max(h, w))
            }
        }
        var x = concatenated(parts, axis: 1)
        let L = x.dim(1)
        let r = rope(frame: frame, height: hIdx, width: wIdx)
        // Block-causal mask: causal, plus bidirectional inside each image block.
        let ids = MLXArray(imageIds)
        let q = MLXArray(Array(Int32(0) ..< Int32(L)))
        let causal = q.expandedDimensions(axis: 1) .>= q.expandedDimensions(axis: 0)
        let sameBlock = (ids.expandedDimensions(axis: 1) .== ids.expandedDimensions(axis: 0))
            .&& (ids.expandedDimensions(axis: 1) .>= 0)
        let mask = MLX.where(causal .|| sameBlock, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(.bfloat16)

        let mod = modulation(sigma: 0)
        let scale = 1 / sqrt(Float(c.headDim))
        var keys: [MLXArray] = [], values: [MLXArray] = []
        for b in blocks {
            let h = layerNorm(x, eps: c.eps) * (1 + mod.scale1)
            let (qq, kk, vv) = qkv(b, h, rope: r)
            keys.append(kk); values.append(vv)
            let a = scaledDotProductAttention(queries: qq, keys: kk, values: vv, scale: scale, mask: mask)
            x = x + mod.gate1 * attentionOut(b, a)
            x = x + mod.gate2 * mlp(b, layerNorm(x, eps: c.eps) * (1 + mod.scale2))
        }
        eval(keys + values)
        return PrefixCache(keys: keys, values: values, length: L, nextPosition: Int(position))
    }

    /// Predicts the flow velocity for target latents `[1, h*w, 64]` at noise level `sigma`.
    public func denoise(latents: MLXArray, prefix: PrefixCache, sigma: Float, latentHeight h: Int,
                        latentWidth w: Int) -> MLXArray {
        let c = config
        var x = imgIn(latents.asType(.bfloat16))
        let pos = Int32(prefix.nextPosition)
        let r = rope(frame: Array(repeating: pos, count: h * w),
                     height: Self.centred(h).flatMap { Array(repeating: $0, count: w) },
                     width: Array((0..<h).map { _ in Self.centred(w) }.joined()))
        let mod = modulation(sigma: sigma)
        let scale = 1 / sqrt(Float(c.headDim))
        for (i, b) in blocks.enumerated() {
            let hN = layerNorm(x, eps: c.eps) * (1 + mod.scale1)
            let (q, k, v) = qkv(b, hN, rope: r)
            let keys = concatenated([prefix.keys[i], k], axis: 2)
            let vals = concatenated([prefix.values[i], v], axis: 2)
            let a = scaledDotProductAttention(queries: q, keys: keys, values: vals, scale: scale, mask: nil)
            x = x + mod.gate1 * attentionOut(b, a)
            x = x + mod.gate2 * mlp(b, layerNorm(x, eps: c.eps) * (1 + mod.scale2))
        }
        x = layerNorm(x, eps: c.eps) * (1 + mod.outScale)
        return projOut(x)
    }

    struct LinearSource {
        let store: WeightStore
        let dense: Bool
        func linear(_ prefix: String) throws -> Lin {
            dense ? try store.denseLinear(prefix) : try store.linear(prefix)
        }
    }

    func textNorm(_ x: MLXArray) -> MLXArray {
        rmsNorm(x.asType(.float32), weight: txtNorm, eps: config.eps).asType(.bfloat16)
    }
}
