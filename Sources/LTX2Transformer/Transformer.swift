import Foundation
import MLX
import MLXDiffusionCore

// MARK: - small building blocks

/// `nn.Linear` semantics: `addmm(bias, x, Wᵀ)` for dense layers, `quantized_matmul + bias` for quantized ones.
@inline(__always)
func linear(_ l: Lin, _ x: MLXArray) -> MLXArray {
    if !l.isQuantized, let b = l.bias, l.adapters.isEmpty { return addMM(b, x, l.weight.T) }
    return l(x)
}

/// Casts a layer's float parameters (bias, dense weight, quantization scales/biases) to `dtype`.
func castLinear(_ l: Lin, _ dtype: DType) -> Lin {
    var c = Lin(weight: l.isQuantized ? l.weight : l.weight.asType(dtype), bias: l.bias?.asType(dtype),
                scales: l.scales?.asType(dtype), qbiases: l.qbiases?.asType(dtype), bits: l.bits,
                groupSize: l.groupSize)
    c.name = l.name
    return c
}

private let siluCompiled: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in x * sigmoid(x) }
private let geluApproxCompiled: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
    0.5 * x * (1 + tanh(Float(0.7978845608028654) * (x + 0.044715 * pow(x, Float(3)))))
}

/// `mx.fast.rms_norm(x, weight=None)`: an all-ones weight in `x`'s dtype gives the identical kernel result.
private final class OnesCache: @unchecked Sendable {
    private var cache: [String: MLXArray] = [:]
    private let lock = NSLock()
    func ones(_ n: Int, _ dtype: DType) -> MLXArray {
        let key = "\(n)-\(dtype)"
        lock.lock(); defer { lock.unlock() }
        if let a = cache[key] { return a }
        let a = MLXArray.ones([n], dtype: dtype)
        cache[key] = a
        return a
    }
}
private let onesCache = OnesCache()

@inline(__always)
func plainRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    rmsNorm(x, weight: onesCache.ones(x.dim(-1), x.dtype), eps: eps)
}

/// Sinusoidal timestep embedding `[cos, sin]` (mlx_arsenal `get_timestep_embedding`, flip_sin_to_cos).
func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Double = 10000) -> MLXArray {
    let half = dim / 2
    let exponent = (MLXArray((0 ..< half).map { Float($0) }) * Float(-Foundation.log(maxPeriod))) / Float(half)
    let freqs = exp(exponent)
    let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
    return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}

/// `AdaLayerNormSingle`: timestep MLP + `num_params`-way modulation projection.
struct AdaLN {
    let l1, l2, proj: Lin
    let numParams: Int

    init(_ s: WeightStore, _ p: String, numParams: Int) throws {
        l1 = try s.linear(p + ".emb.timestep_embedder.linear1")
        l2 = try s.linear(p + ".emb.timestep_embedder.linear2")
        proj = try s.linear(p + ".linear")
        self.numParams = numParams
    }

    /// Returns `(params [M, P*dim], embedded [M, dim])`.
    func callAsFunction(_ tEmb: MLXArray) -> (MLXArray, MLXArray) {
        let embedded = linear(l2, siluCompiled(linear(l1, tEmb)))
        return (linear(proj, siluCompiled(embedded)), embedded)
    }
}

/// AdaLN output that is either per-sample (`[B, P*dim]`) or per-token, kept as distinct rows + an inverse index.
enum AdaParams {
    case scalar(MLXArray)
    case perToken(unique: MLXArray, inverse: MLXArray, batch: Int, tokens: Int)

    /// `_unpack_adaln`: `P` arrays broadcastable to `[B, N, dim]` with the block's table rows added.
    func unpack(_ table: MLXArray, _ p: Int, _ dim: Int, total: Int) -> [MLXArray] {
        switch self {
        case .scalar(let a):
            let r = a.reshaped(-1, total, dim)[0..., ..<p, 0...] + table[..<p].expandedDimensions(axis: 0)
            return (0 ..< p).map { r[0..., $0].expandedDimensions(axis: 1) }
        case .perToken(let u, let inv, let b, let n):
            let packed = u.reshaped(-1, total, dim)[0..., ..<p, 0...] + table[..<p].expandedDimensions(axis: 0)
            return (0 ..< p).map { take(packed[0..., $0], inv, axis: 0).reshaped(b, n, dim) }
        }
    }

    /// The full `[B, N or 1, dim]` tensor of a 1-parameter head (the output block's embedded timestep).
    func gathered(dim: Int) -> MLXArray {
        switch self {
        case .scalar(let a): return a.expandedDimensions(axis: 1)
        case .perToken(let u, let inv, let b, let n): return take(u, inv, axis: 0).reshaped(b, n, dim)
        }
    }
}

/// Multi-head attention with RMS q/k-norm over the full inner dim, optional SPLIT RoPE and a per-head gate.
struct LTXAttention {
    var q, k, v, out, gate: Lin
    var qNorm, kNorm: MLXArray
    let heads, headDim: Int
    let eps: Float
    var computeDType: DType?

    init(_ s: WeightStore, _ p: String, heads: Int, headDim: Int, eps: Float) throws {
        q = try s.linear(p + ".to_q"); k = try s.linear(p + ".to_k"); v = try s.linear(p + ".to_v")
        out = try s.linear(p + ".to_out"); gate = try s.linear(p + ".to_gate_logits")
        qNorm = try s.get(p + ".q_norm.weight"); kNorm = try s.get(p + ".k_norm.weight")
        self.heads = heads; self.headDim = headDim; self.eps = eps
    }

    mutating func setComputeDType(_ d: DType) {
        computeDType = d
        q = castLinear(q, d); k = castLinear(k, d); v = castLinear(v, d); out = castLinear(out, d)
        gate = castLinear(gate, d); qNorm = qNorm.asType(d); kNorm = kNorm.asType(d)
    }

    func callAsFunction(_ x0: MLXArray, context: MLXArray? = nil, rope: LTXRope? = nil, ropeK: LTXRope? = nil)
        -> MLXArray
    {
        var x = x0
        var ctx = context
        if let d = computeDType { x = x.asType(d); ctx = ctx?.asType(d) }
        let kv = ctx ?? x
        let (b, n) = (x.dim(0), x.dim(1))
        var qq = rmsNorm(linear(q, x), weight: qNorm, eps: eps)
        var kk = rmsNorm(linear(k, kv), weight: kNorm, eps: eps)
        let vv = linear(v, kv).reshaped(b, -1, heads, headDim).transposed(0, 2, 1, 3)
        qq = qq.reshaped(b, -1, heads, headDim).transposed(0, 2, 1, 3)
        kk = kk.reshaped(b, -1, heads, headDim).transposed(0, 2, 1, 3)
        if let rope {
            qq = applyRopeSplit(qq, rope)
            kk = applyRopeSplit(kk, ropeK ?? rope)
        }
        var o = scaledDotProductAttention(queries: qq, keys: kk, values: vv,
                                          scale: Float(1 / Double(headDim).squareRoot()), mask: nil)
        let g = 2.0 * sigmoid(linear(gate, x))  // [B, N, H]
        o = o * g.transposed(0, 2, 1).expandedDimensions(axis: -1)
        o = o.transposed(0, 2, 1, 3).reshaped(b, n, heads * headDim)
        return linear(out, o)
    }
}

/// GELU(tanh) MLP.
struct LTXFeedForward {
    var projIn, projOut: Lin
    var computeDType: DType?

    init(_ s: WeightStore, _ p: String) throws {
        projIn = try s.linear(p + ".proj_in"); projOut = try s.linear(p + ".proj_out")
    }

    mutating func setComputeDType(_ d: DType) {
        computeDType = d
        projIn = castLinear(projIn, d); projOut = castLinear(projOut, d)
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let x = computeDType.map { x0.asType($0) } ?? x0
        return linear(projOut, geluApproxCompiled(linear(projIn, x)))
    }
}

/// `BasicAVTransformerBlock`: video/audio self-attn, text cross-attn (with cross-attention AdaLN),
/// bidirectional audio↔video cross-attn with its own AdaLN, and feed-forwards.
struct LTXBlock {
    var attn1, audioAttn1, attn2, audioAttn2, a2v, v2a: LTXAttention
    var ff, audioFF: LTXFeedForward
    let sst, audioSst, promptSst, audioPromptSst, sstA2vVideo, sstA2vAudio: MLXArray
    let eps: Float

    init(_ s: WeightStore, _ p: String, c: LTXTransformerConfig) throws {
        attn1 = try LTXAttention(s, p + "attn1", heads: c.videoHeads, headDim: c.videoHeadDim, eps: c.normEps)
        audioAttn1 = try LTXAttention(s, p + "audio_attn1", heads: c.audioHeads, headDim: c.audioHeadDim,
                                      eps: c.normEps)
        attn2 = try LTXAttention(s, p + "attn2", heads: c.videoHeads, headDim: c.videoHeadDim, eps: c.normEps)
        audioAttn2 = try LTXAttention(s, p + "audio_attn2", heads: c.audioHeads, headDim: c.audioHeadDim,
                                      eps: c.normEps)
        a2v = try LTXAttention(s, p + "audio_to_video_attn", heads: c.avCrossHeads, headDim: c.avCrossHeadDim,
                               eps: c.normEps)
        v2a = try LTXAttention(s, p + "video_to_audio_attn", heads: c.avCrossHeads, headDim: c.avCrossHeadDim,
                               eps: c.normEps)
        ff = try LTXFeedForward(s, p + "ff")
        audioFF = try LTXFeedForward(s, p + "audio_ff")
        sst = try s.get(p + "scale_shift_table")
        audioSst = try s.get(p + "audio_scale_shift_table")
        promptSst = try s.get(p + "prompt_scale_shift_table")
        audioPromptSst = try s.get(p + "audio_prompt_scale_shift_table")
        sstA2vVideo = try s.get(p + "scale_shift_table_a2v_ca_video")
        sstA2vAudio = try s.get(p + "scale_shift_table_a2v_ca_audio")
        eps = c.normEps
    }

    mutating func setComputeDType(_ d: DType) {
        attn1.setComputeDType(d); audioAttn1.setComputeDType(d); attn2.setComputeDType(d)
        audioAttn2.setComputeDType(d); a2v.setComputeDType(d); v2a.setComputeDType(d)
        ff.setComputeDType(d); audioFF.setComputeDType(d)
    }

    var parameters: [MLXArray] {
        var a: [MLXArray] = [sst, audioSst, promptSst, audioPromptSst, sstA2vVideo, sstA2vAudio]
        for at in [attn1, audioAttn1, attn2, audioAttn2, a2v, v2a] {
            for l in [at.q, at.k, at.v, at.out, at.gate] { a += [l.weight] + [l.bias, l.scales, l.qbiases].compactMap { $0 } }
            a += [at.qNorm, at.kNorm]
        }
        for f in [ff, audioFF] {
            for l in [f.projIn, f.projOut] { a += [l.weight] + [l.bias, l.scales, l.qbiases].compactMap { $0 } }
        }
        return a
    }

    func callAsFunction(video vh0: MLXArray, audio ah0: MLXArray, m: LTXTransformer.Modulation) -> (MLXArray, MLXArray) {
        var vh = vh0
        var ah = ah0
        let vdim = vh.dim(-1)
        let adim = ah.dim(-1)
        let rms = { (x: MLXArray) in plainRMSNorm(x, eps: self.eps) }

        let v = m.video.unpack(sst, 9, vdim, total: 9)
        let a = m.audio.unpack(audioSst, 9, adim, total: 9)
        let avV = m.avVideo.unpack(sstA2vVideo, 4, vdim, total: 4)
        let avA = m.avAudio.unpack(sstA2vAudio, 4, adim, total: 4)
        let gateA2v = (m.a2vGate + sstA2vVideo[4]).expandedDimensions(axis: 1)
        let gateV2a = (m.v2aGate + sstA2vAudio[4]).expandedDimensions(axis: 1)

        // 1-2. self-attention
        vh = vh + attn1(rms(vh) * (1.0 + v[1]) + v[0], rope: m.videoRope) * v[2]
        ah = ah + audioAttn1(rms(ah) * (1.0 + a[1]) + a[0], rope: m.audioRope) * a[2]

        // 3-4. text cross-attention (AdaLN 6-8, prompt table modulates the context)
        let vp = AdaParams.scalar(m.videoPrompt).unpack(promptSst, 2, vdim, total: 2)
        let vText = m.videoText * (1.0 + vp[1]) + vp[0]
        vh = vh + attn2(rms(vh) * (1.0 + v[7]) + v[6], context: vText) * v[8]
        let ap = AdaParams.scalar(m.audioPrompt).unpack(audioPromptSst, 2, adim, total: 2)
        let aText = m.audioText * (1.0 + ap[1]) + ap[0]
        ah = ah + audioAttn2(rms(ah) * (1.0 + a[7]) + a[6], context: aText) * a[8]

        // 5-6. audio↔video cross-attention (shared pre-norms); tables: scale_a2v, shift_a2v, scale_v2a, shift_v2a
        let vn3 = rms(vh)
        let an3 = rms(ah)
        let a2vOut = a2v(vn3 * (1.0 + avV[0]) + avV[1], context: an3 * (1.0 + avA[0]) + avA[1],
                         rope: m.videoCrossRope, ropeK: m.audioCrossRope) * gateA2v
        vh = vh + a2vOut
        let v2aOut = v2a(an3 * (1.0 + avA[2]) + avA[3], context: vn3 * (1.0 + avV[2]) + avV[3],
                         rope: m.audioCrossRope, ropeK: m.videoCrossRope) * gateV2a
        ah = ah + v2aOut

        // 7-8. feed-forward (AdaLN 3-5)
        vh = vh + ff(rms(vh) * (1.0 + v[4]) + v[3]) * v[5]
        ah = ah + audioFF(rms(ah) * (1.0 + a[4]) + a[3]) * a[5]
        return (vh, ah)
    }
}

// MARK: - model

/// LTX-2.3 audio-video diffusion transformer (`LTXModel`): returns velocity predictions for both modalities.
///
/// Mirrors the reference's default (no compute-dtype) path: inputs are cast to bfloat16, the AdaLN modulation and
/// therefore the residual stream run in float32 (the F32 scale-shift tables promote), quantized int4 block weights
/// are used as-is.
public final class LTXTransformer {
    public let config: LTXTransformerConfig
    let patchify, audioPatchify, projOut, audioProjOut: Lin
    let outSst, audioOutSst: MLXArray
    let adaln, audioAdaln, promptAdaln, audioPromptAdaln: AdaLN
    let avVideoAdaln, avAudioAdaln, a2vGateAdaln, v2aGateAdaln: AdaLN
    var blocks: [LTXBlock]
    /// Evaluate the lazy graph every N blocks (reference `LTX2_DIT_EVAL_EVERY`, default 8); 0 disables.
    public var evalEvery = 8

    /// Per-call conditioning shared by all blocks.
    struct Modulation {
        let video, audio, avVideo, avAudio: AdaParams
        let a2vGate, v2aGate, videoPrompt, audioPrompt: MLXArray
        let videoText, audioText: MLXArray
        let videoRope, audioRope, videoCrossRope, audioCrossRope: LTXRope
    }

    /// Loads `transformer-<variant>.safetensors` (int4, group 64) from `directory` (memory-mapped).
    ///
    /// - Parameter computeDType: like the reference `set_compute_dtype`: run the inside of every attention and
    ///   feed-forward module in this dtype (residual stream and AdaLN stay float32). `nil` = reference default.
    public convenience init(directory: URL, variant: String = "distilled-1.1", computeDType: DType? = nil) throws {
        let url = directory.appendingPathComponent("transformer-\(variant).safetensors")
        let store = try WeightStore(url: url)
        store.bitsOverride = 4
        try self.init(store: store, config: LTXTransformerConfig(directory: directory), computeDType: computeDType)
    }

    /// Builds the model from an already-loaded store (keys with or without the `transformer.` prefix).
    public init(store s: WeightStore, config: LTXTransformerConfig, computeDType: DType? = nil) throws {
        self.config = config
        let p = s.has("transformer.patchify_proj.weight") ? "transformer." : ""
        patchify = try s.linear(p + "patchify_proj")
        audioPatchify = try s.linear(p + "audio_patchify_proj")
        projOut = try s.linear(p + "proj_out")
        audioProjOut = try s.linear(p + "audio_proj_out")
        outSst = try s.get(p + "scale_shift_table")
        audioOutSst = try s.get(p + "audio_scale_shift_table")
        adaln = try AdaLN(s, p + "adaln_single", numParams: 9)
        audioAdaln = try AdaLN(s, p + "audio_adaln_single", numParams: 9)
        promptAdaln = try AdaLN(s, p + "prompt_adaln_single", numParams: 2)
        audioPromptAdaln = try AdaLN(s, p + "audio_prompt_adaln_single", numParams: 2)
        avVideoAdaln = try AdaLN(s, p + "av_ca_video_scale_shift_adaln_single", numParams: 4)
        avAudioAdaln = try AdaLN(s, p + "av_ca_audio_scale_shift_adaln_single", numParams: 4)
        a2vGateAdaln = try AdaLN(s, p + "av_ca_a2v_gate_adaln_single", numParams: 1)
        v2aGateAdaln = try AdaLN(s, p + "av_ca_v2a_gate_adaln_single", numParams: 1)
        blocks = try (0 ..< config.numLayers).map { i in
            try LTXBlock(s, p + "transformer_blocks.\(i).", c: config)
        }
        if let d = computeDType {
            for i in blocks.indices {
                blocks[i].setComputeDType(d)
                eval(blocks[i].parameters)
            }
        }
    }

    func embedScalar(_ t: MLXArray) -> MLXArray {
        timestepEmbedding(t * Float(config.timestepScaleMultiplier), dim: config.timestepEmbeddingDim)
    }

    /// Per-token AdaLN on the distinct timestep values only, expanded per block (reference `PerTokenAdaLNParams`).
    func perToken(_ timesteps: MLXArray, _ modules: [AdaLN]) -> [(AdaParams, AdaParams)] {
        let (b, n) = (timesteps.dim(0), timesteps.dim(1))
        let flat = (timesteps * Float(config.timestepScaleMultiplier)).reshaped(-1)
        let bits = flat.asType(.float32).asArray(Float.self).map(\.bitPattern)
        var index: [UInt32: Int32] = [:]
        var firsts: [Int32] = []
        var inverse = [Int32](repeating: 0, count: bits.count)
        for (i, bp) in bits.enumerated() {
            if let u = index[bp] { inverse[i] = u; continue }
            let u = Int32(firsts.count)
            index[bp] = u
            firsts.append(Int32(i))
            inverse[i] = u
        }
        let reps = take(flat, MLXArray(firsts), axis: 0)
        let emb = timestepEmbedding(reps, dim: config.timestepEmbeddingDim)
        let inv = MLXArray(inverse)
        return modules.map { mod in
            let (params, embedded) = mod(emb)
            return (.perToken(unique: params, inverse: inv, batch: b, tokens: n),
                    .perToken(unique: embedded, inverse: inv, batch: b, tokens: n))
        }
    }

    func outputBlock(_ x: MLXArray, _ embedded: MLXArray, _ table: MLXArray, _ proj: Lin) -> MLXArray {
        // embedded: [B, N or 1, dim]; table: [2, dim]
        let ssv = table.expandedDimensions(axes: [0, 1]) + embedded.expandedDimensions(axis: 2)
        let shift = ssv[0..., 0..., 0]
        let scale = ssv[0..., 0..., 1]
        let h = layerNorm(x, eps: config.normEps) * (1.0 + scale) + shift
        return linear(proj, h)
    }

    /// One forward pass; returns velocities `(video [B, Nv, 128], audio [B, Na, 128])` in float32.
    ///
    /// - Parameters:
    ///   - video: patchified video latent `[B, Nv, 128]`; `audio`: patchified audio latent `[B, Na, 128]`.
    ///   - sigma: `[B]` noise level (the pipeline passes bfloat16).
    ///   - videoTimesteps/audioTimesteps: optional per-token timesteps `[B, N]` (`denoise_mask * sigma`, float32);
    ///     conditioning tokens get 0.
    ///   - videoText: `[B, Nt, 4096]`, audioText: `[B, Nt, 2048]` (connector outputs).
    ///   - videoPositions: `[B, Nv, 3]`, audioPositions: `[B, Na, 1]` (see ``ltxVideoPositions(frames:height:width:frameRate:)``).
    ///   - videoSigma/audioSigma: optional per-modality sigma (frozen streams pass 0); default `sigma`.
    public func callAsFunction(
        video: MLXArray, audio: MLXArray, sigma: MLXArray,
        videoTimesteps: MLXArray? = nil, audioTimesteps: MLXArray? = nil,
        videoText: MLXArray, audioText: MLXArray,
        videoPositions: MLXArray, audioPositions: MLXArray,
        videoSigma: MLXArray? = nil, audioSigma: MLXArray? = nil,
        blockHook: ((Int, MLXArray, MLXArray) -> Void)? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let c = config
        var vh = linear(patchify, video.asType(.bfloat16))
        var ah = linear(audioPatchify, audio.asType(.bfloat16))
        let vText = videoText.asType(.bfloat16)
        let aText = audioText.asType(.bfloat16)

        let t = sigma.asType(.bfloat16)
        let tEmb = embedScalar(t)
        let avFactor = Float(c.avCaTimestepScaleMultiplier / c.timestepScaleMultiplier)
        let tEmbGate = timestepEmbedding(t * Float(c.timestepScaleMultiplier) * avFactor, dim: c.timestepEmbeddingDim)
        func modalitySigma(_ s: MLXArray?) -> (MLXArray, MLXArray) {
            guard let s else { return (tEmb, tEmbGate) }
            let sb = s.asType(.bfloat16)
            return (embedScalar(sb),
                    timestepEmbedding(sb * Float(c.timestepScaleMultiplier) * avFactor, dim: c.timestepEmbeddingDim))
        }
        let (videoTEmb, videoTEmbGate) = modalitySigma(videoSigma)
        let (audioTEmb, audioTEmbGate) = modalitySigma(audioSigma)

        let videoAda: AdaParams, avVideo: AdaParams, videoEmbedded: MLXArray
        if let vt = videoTimesteps {
            let r = perToken(vt, [adaln, avVideoAdaln])
            videoAda = r[0].0; avVideo = r[1].0; videoEmbedded = r[0].1.gathered(dim: c.videoDim)
        } else {
            let (pp, e) = adaln(tEmb)
            videoAda = .scalar(pp); videoEmbedded = e.expandedDimensions(axis: 1)
            avVideo = .scalar(avVideoAdaln(tEmb).0)
        }
        let a2vGate = a2vGateAdaln(audioTEmbGate).0
        let videoPrompt = promptAdaln(videoTEmb).0

        let audioAda: AdaParams, avAudio: AdaParams, audioEmbedded: MLXArray
        if let at = audioTimesteps {
            let r = perToken(at, [audioAdaln, avAudioAdaln])
            audioAda = r[0].0; avAudio = r[1].0; audioEmbedded = r[0].1.gathered(dim: c.audioDim)
        } else {
            let (pp, e) = audioAdaln(tEmb)
            audioAda = .scalar(pp); audioEmbedded = e.expandedDimensions(axis: 1)
            avAudio = .scalar(avAudioAdaln(tEmb).0)
        }
        let v2aGate = v2aGateAdaln(videoTEmbGate).0
        let audioPrompt = audioPromptAdaln(audioTEmb).0

        let dp = c.doublePrecisionRope
        let crossMax = [max(c.maxPos[0], c.audioMaxPos[0])]
        let m = Modulation(
            video: videoAda, audio: audioAda, avVideo: avVideo, avAudio: avAudio,
            a2vGate: a2vGate, v2aGate: v2aGate, videoPrompt: videoPrompt, audioPrompt: audioPrompt,
            videoText: vText, audioText: aText,
            videoRope: LTXRope(positions: videoPositions, heads: c.videoHeads, headDim: c.videoHeadDim,
                               theta: c.ropeTheta, maxPos: Array(c.maxPos.prefix(videoPositions.dim(-1))),
                               doublePrecision: dp),
            audioRope: LTXRope(positions: audioPositions, heads: c.audioHeads, headDim: c.audioHeadDim,
                               theta: c.ropeTheta, maxPos: c.audioMaxPos, doublePrecision: dp),
            videoCrossRope: LTXRope(positions: videoPositions[0..., 0..., 0 ..< 1], heads: c.avCrossHeads,
                                    headDim: c.avCrossHeadDim, theta: c.ropeTheta, maxPos: crossMax,
                                    doublePrecision: dp),
            audioCrossRope: LTXRope(positions: audioPositions[0..., 0..., 0 ..< 1], heads: c.avCrossHeads,
                                    headDim: c.avCrossHeadDim, theta: c.ropeTheta, maxPos: crossMax,
                                    doublePrecision: dp))

        for (i, block) in blocks.enumerated() {
            (vh, ah) = block(video: vh, audio: ah, m: m)
            if let blockHook { blockHook(i, vh, ah) }
            if evalEvery > 0, (i + 1) % evalEvery == 0 { eval(vh, ah) }
        }

        let vOut = outputBlock(vh, videoEmbedded, outSst, projOut)
        let aOut = outputBlock(ah, audioEmbedded, audioOutSst, audioProjOut)
        return (vOut, aOut)
    }
}

/// `X0Model`: turns the transformer's velocity into a denoised estimate, `x0 = x_t − σ·v` in float32.
///
/// With per-token timesteps the per-token σ is used, so conditioning tokens (σ = 0) are returned unchanged.
public final class LTXX0Model {
    public let transformer: LTXTransformer

    public init(_ transformer: LTXTransformer) { self.transformer = transformer }

    /// Same arguments as ``LTXTransformer/callAsFunction(video:audio:sigma:videoTimesteps:audioTimesteps:videoText:audioText:videoPositions:audioPositions:videoSigma:audioSigma:blockHook:)``;
    /// outputs keep the dtype of `video`/`audio`.
    public func callAsFunction(
        video: MLXArray, audio: MLXArray, sigma: MLXArray,
        videoTimesteps: MLXArray? = nil, audioTimesteps: MLXArray? = nil,
        videoText: MLXArray, audioText: MLXArray,
        videoPositions: MLXArray, audioPositions: MLXArray,
        videoSigma: MLXArray? = nil, audioSigma: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let (vv, av) = transformer(
            video: video, audio: audio, sigma: sigma, videoTimesteps: videoTimesteps, audioTimesteps: audioTimesteps,
            videoText: videoText, audioText: audioText, videoPositions: videoPositions,
            audioPositions: audioPositions, videoSigma: videoSigma, audioSigma: audioSigma)
        func sig(_ ts: MLXArray?) -> MLXArray {
            if let ts { return ts.expandedDimensions(axis: -1).asType(.float32) }
            return sigma.reshaped(-1, 1, 1).asType(.float32)
        }
        let vx0 = (video.asType(.float32) - sig(videoTimesteps) * vv.asType(.float32)).asType(video.dtype)
        let ax0 = (audio.asType(.float32) - sig(audioTimesteps) * av.asType(.float32)).asType(audio.dtype)
        return (vx0, ax0)
    }
}
