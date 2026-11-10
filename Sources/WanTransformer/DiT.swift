import Foundation
import MLX
import MLXDiffusionCore

/// One Wan 2.1/2.2 diffusion transformer (one expert of the A14B MoE), as the official `WanModel`:
/// patchified latents → 40 blocks of (AdaLN self-attention with 3D RoPE, text cross-attention, AdaLN GELU FFN)
/// → modulated output head → unpatchify.
///
/// Precision follows the official autocast: the residual stream and the modulation are float32, every linear
/// layer and attention runs in the weights' dtype (bf16, or bf16-scaled MLX quantized weights), RoPE in float32.
/// Keys are the mlx-video conversion's (`patch_embedding_proj`, `text_embedding_0`, `ffn.fc1`, …); LoRA names use
/// the official module names (`blocks.N.ffn.0`, …) so official/Comfy LoRAs match.
public final class WanDiT {
    public struct Config: Sendable {
        public var dim = 5120, ffnDim = 13_824, freqDim = 256, textDim = 4096, textLen = 512
        public var inDim = 16, outDim = 16, numHeads = 40, numLayers = 40
        public var patch = (t: 1, h: 2, w: 2)
        public var eps: Float = 1e-6
        public var headDim: Int { dim / numHeads }
        public init() {}

        /// Reads an official / mlx-video `config.json`.
        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            dim = o["dim"] as? Int ?? dim
            ffnDim = o["ffn_dim"] as? Int ?? ffnDim
            freqDim = o["freq_dim"] as? Int ?? freqDim
            textDim = o["text_dim"] as? Int ?? textDim
            textLen = o["text_len"] as? Int ?? textLen
            inDim = o["in_dim"] as? Int ?? inDim
            outDim = o["out_dim"] as? Int ?? outDim
            numHeads = o["num_heads"] as? Int ?? numHeads
            numLayers = o["num_layers"] as? Int ?? numLayers
            if let p = o["patch_size"] as? [Int], p.count == 3 { patch = (p[0], p[1], p[2]) }
            eps = Float(o["eps"] as? Double ?? Double(eps))
        }
    }

    struct Block {
        let modulation: MLXArray  // [1, 6, dim] float32
        var q, k, v, o: Lin
        let normQ, normK: MLXArray
        var cq, ck, cv, co: Lin
        let cNormQ, cNormK: MLXArray
        let norm3Weight, norm3Bias: MLXArray
        var fc1, fc2: Lin

        mutating func mapLinears(_ f: (Lin) -> Lin) {
            q = f(q); k = f(k); v = f(v); o = f(o)
            cq = f(cq); ck = f(ck); cv = f(cv); co = f(co)
            fc1 = f(fc1); fc2 = f(fc2)
        }
        var linears: [Lin] { [q, k, v, o, cq, ck, cv, co, fc1, fc2] }
    }

    public let config: Config
    /// Activation dtype of the linear layers.
    public let dtype: DType
    let patchEmbed: Lin
    let text0, text1, time0, time1, timeProj: Lin
    let baseBlocks: [Block]
    var blocks: [Block]
    let head: Lin
    let headModulation: MLXArray  // [1, 2, dim] float32
    /// Per-axis RoPE inverse frequencies (temporal 22, height 21, width 21 pairs for head dim 128).
    let ropeInvFreq: (t: [Double], h: [Double], w: [Double])
    let timeInvFreq: MLXArray

    public init(store: WeightStore, config: Config) throws {
        self.config = config
        func lin(_ k: String, name: String? = nil) throws -> Lin {
            var l = try store.linear(k)
            l.name = name ?? k
            return l
        }
        patchEmbed = try lin("patch_embedding_proj")
        dtype = patchEmbed.isQuantized ? patchEmbed.scales!.dtype : patchEmbed.weight.dtype
        text0 = try lin("text_embedding_0"); text1 = try lin("text_embedding_1")
        time0 = try lin("time_embedding_0"); time1 = try lin("time_embedding_1")
        timeProj = try lin("time_projection")
        let dt = dtype
        baseBlocks = try (0..<config.numLayers).map { i in
            let p = "blocks.\(i)."
            return Block(
                modulation: try store.get(p + "modulation").asType(.float32),
                q: try lin(p + "self_attn.q"), k: try lin(p + "self_attn.k"), v: try lin(p + "self_attn.v"),
                o: try lin(p + "self_attn.o"),
                normQ: try store.get(p + "self_attn.norm_q.weight").asType(dt),
                normK: try store.get(p + "self_attn.norm_k.weight").asType(dt),
                cq: try lin(p + "cross_attn.q"), ck: try lin(p + "cross_attn.k"), cv: try lin(p + "cross_attn.v"),
                co: try lin(p + "cross_attn.o"),
                cNormQ: try store.get(p + "cross_attn.norm_q.weight").asType(dt),
                cNormK: try store.get(p + "cross_attn.norm_k.weight").asType(dt),
                norm3Weight: try store.get(p + "norm3.weight"), norm3Bias: try store.get(p + "norm3.bias"),
                fc1: try lin(p + "ffn.fc1", name: p + "ffn.0"), fc2: try lin(p + "ffn.fc2", name: p + "ffn.2"))
        }
        blocks = baseBlocks
        head = try lin("head.head")
        headModulation = try store.get("head.modulation").asType(.float32)

        let d = config.headDim
        func inv(_ axisDim: Int) -> [Double] {
            stride(from: 0, to: axisDim, by: 2).map { 1 / pow(10_000, Double($0) / Double(axisDim)) }
        }
        ropeInvFreq = (inv(d - 4 * (d / 6)), inv(2 * (d / 6)), inv(2 * (d / 6)))
        let half = config.freqDim / 2
        timeInvFreq = MLXArray((0..<half).map { Float(pow(10_000, -Double($0) / Double(half))) })
    }

    // MARK: LoRA

    /// Official names of every adaptable linear layer (`blocks.N.self_attn.q`, `blocks.N.ffn.0`, …).
    public var linearNames: Set<String> { Set(baseBlocks.flatMap { $0.linears.map(\.name) }) }

    /// Adds `adapters` (keyed by official names) to the base weights, replacing any previous ones.
    /// Returns how many layers matched.
    @discardableResult
    public func setAdapters(_ adapters: [String: LoRAAdapter]) -> Int { setAdapters([adapters]) }

    /// Several LoRAs at once (Lightning plus your own), each added to the layers it has.
    @discardableResult
    public func setAdapters(_ sets: [[String: LoRAAdapter]]) -> Int {
        var matched = 0
        blocks = baseBlocks.map { b in
            var b = b
            b.mapLinears { l in
                var l = l
                for adapters in sets {
                    guard let a = adapters[l.name] else { continue }
                    matched += 1
                    l = l.adding(a)
                }
                return l
            }
            return b
        }
        return matched
    }

    public func clearAdapters() { blocks = baseBlocks }

    // MARK: conditioning, computed once per generation

    /// Text tokens `[L, textDim]` (unpadded encoder output) → `[1, textLen, dim]`, zero-padded before the MLP.
    public func embedText(_ context: MLXArray) -> MLXArray {
        var c = context
        let pad = config.textLen - c.dim(0)
        if pad > 0 { c = concatenated([c, MLXArray.zeros([pad, c.dim(1)], dtype: c.dtype)], axis: 0) }
        let h = text1(geluTanh(text0(c.expandedDimensions(axis: 0))))
        return h.asType(dtype)
    }

    /// Cross-attention keys and values `[1, heads, textLen, headDim]` for every block.
    public func crossKV(_ text: MLXArray) -> [(MLXArray, MLXArray)] {
        let H = config.numHeads, D = config.headDim, L = text.dim(1)
        return blocks.map { b in
            let k = MLXFast.rmsNorm(b.ck(text), weight: b.cNormK, eps: config.eps)
            let v = b.cv(text)
            return (k.reshaped(1, L, H, D).transposed(0, 2, 1, 3), v.reshaped(1, L, H, D).transposed(0, 2, 1, 3))
        }
    }

    /// RoPE `(cos, sin)`, each `[S, 1, headDim/2]` float32, for a patch grid of `f × h × w`.
    public func rope(frames f: Int, height h: Int, width w: Int) -> (MLXArray, MLXArray) {
        let (it, ih, iw) = ropeInvFreq
        let half = it.count + ih.count + iw.count
        var cosv = [Float](repeating: 0, count: f * h * w * half), sinv = cosv
        var idx = 0
        for fi in 0..<f {
            for hi in 0..<h {
                for wi in 0..<w {
                    for (pos, freqs) in [(fi, it), (hi, ih), (wi, iw)] {
                        for fr in freqs {
                            let a = Double(pos) * fr
                            cosv[idx] = Float(cos(a)); sinv[idx] = Float(sin(a))
                            idx += 1
                        }
                    }
                }
            }
        }
        let s = f * h * w
        return (MLXArray(cosv, [s, 1, half]), MLXArray(sinv, [s, 1, half]))
    }

    /// Interleaved-pair rotation of `x` `[B, S, H, D]` (float32).
    static func applyRope(_ x: MLXArray, _ cs: (MLXArray, MLXArray)) -> MLXArray {
        let (B, S, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let p = x.reshaped(B, S, H, D / 2, 2)
        let re = p[.ellipsis, 0], im = p[.ellipsis, 1]
        let (c, s) = cs
        return stacked([re * c - im * s, re * s + im * c], axis: -1).reshaped(B, S, H, D)
    }

    // MARK: forward

    /// Velocity prediction `[outDim, F, H, W]` (float32) for latents `x` `[C, F, H, W]` (C = inDim; for I2V the
    /// caller concatenates noise + mask + image latents) at scheduler timestep `t` (0–1000).
    public func callAsFunction(_ x: MLXArray, timestep t: Float, crossKV: [(MLXArray, MLXArray)],
                               rope: (MLXArray, MLXArray)) -> MLXArray {
        let c = config
        let (C, F, Hh, Ww) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (pt, ph, pw) = c.patch
        let (fo, ho, wo) = (F / pt, Hh / ph, Ww / pw)
        let S = fo * ho * wo, Hn = c.numHeads, D = c.headDim

        // Patchify: [C, F, H, W] → [S, C·pt·ph·pw] (C slowest, matching the Conv3d kernel layout).
        let patches = x.reshaped(C, fo, pt, ho, ph, wo, pw).transposed(1, 3, 5, 0, 2, 4, 6).reshaped(S, -1)
        var h = patchEmbed(patches.asType(.float32)).asType(dtype).expandedDimensions(axis: 0)  // [1, S, dim]

        // Time embedding (float32) and its 6-way modulation.
        let sinus = MLXArray(t) * timeInvFreq
        let tEmb = concatenated([cos(sinus), sin(sinus)], axis: -1).expandedDimensions(axis: 0)  // [1, freqDim]
        let e = time1(siluAct(time0(tEmb.asType(.float32)))).asType(.float32)  // [1, dim]
        let e0 = timeProj(siluAct(e)).asType(.float32).reshaped(1, 1, 6, c.dim)

        for (i, b) in blocks.enumerated() {
            let mod = b.modulation + e0  // [1, 1, 6, dim]
            let m = (0..<6).map { mod[0..., 0..., $0, 0...] }  // each [1, 1, dim]

            // Self-attention.
            let xm = (MLXFast.layerNorm(h, weight: nil, bias: nil, eps: c.eps) * (1 + m[1]) + m[0]).asType(dtype)
            var q = MLXFast.rmsNorm(b.q(xm), weight: b.normQ, eps: c.eps).reshaped(1, S, Hn, D)
            var k = MLXFast.rmsNorm(b.k(xm), weight: b.normK, eps: c.eps).reshaped(1, S, Hn, D)
            let v = b.v(xm).reshaped(1, S, Hn, D).transposed(0, 2, 1, 3)
            q = Self.applyRope(q.asType(.float32), rope).asType(dtype).transposed(0, 2, 1, 3)
            k = Self.applyRope(k.asType(.float32), rope).asType(dtype).transposed(0, 2, 1, 3)
            var a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(D).squareRoot(),
                                                      mask: .none)
            h = h + b.o(a.transposed(0, 2, 1, 3).reshaped(1, S, c.dim)) * m[2]

            // Text cross-attention (no modulation).
            let xc = MLXFast.layerNorm(h, weight: b.norm3Weight, bias: b.norm3Bias, eps: c.eps).asType(dtype)
            let cq = MLXFast.rmsNorm(b.cq(xc), weight: b.cNormQ, eps: c.eps).reshaped(1, S, Hn, D).transposed(0, 2, 1, 3)
            let (ck, cv) = crossKV[i]
            a = MLXFast.scaledDotProductAttention(queries: cq, keys: ck, values: cv, scale: 1 / Float(D).squareRoot(),
                                                  mask: .none)
            h = h + b.co(a.transposed(0, 2, 1, 3).reshaped(1, S, c.dim))

            // Feed-forward.
            let xf = (MLXFast.layerNorm(h, weight: nil, bias: nil, eps: c.eps) * (1 + m[4]) + m[3]).asType(dtype)
            h = h + b.fc2(geluTanh(b.fc1(xf))) * m[5]
        }

        // Output head, modulated by the (unprojected) time embedding.
        let hm = headModulation.expandedDimensions(axis: 1) + e.reshaped(1, 1, 1, c.dim)  // [1, 1, 2, dim]
        let shift = hm[0..., 0..., 0, 0...], scale = hm[0..., 0..., 1, 0...]
        let xo = MLXFast.layerNorm(h, weight: nil, bias: nil, eps: c.eps) * (1 + scale) + shift
        let out = head(xo).asType(.float32)[0]  // [S, pt·ph·pw·outDim]

        // Unpatchify: [S, pt, ph, pw, C] → [C, F, H, W].
        return out.reshaped(fo, ho, wo, pt, ph, pw, c.outDim).transposed(6, 0, 3, 1, 4, 2, 5)
            .reshaped(c.outDim, F, Hh, Ww)
    }
}
