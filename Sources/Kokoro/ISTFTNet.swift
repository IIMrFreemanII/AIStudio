import Accelerate
import Foundation
import MLX
import MLXDiffusionCore

/// The source of the harmonic generator's Gaussian noise (`SineGen`'s `randn_like(sine_waves)`, `[L, 9]`).
///
/// This is the only random draw that reaches the output: `SineGen`'s `torch.rand` initial phase only touches
/// sample 0, which the following ×1/300 linear downsample never reads, and `SourceModuleHnNSF`'s own noise branch is
/// discarded by the generator.
public enum KokoroNoise {
    /// Standard normal noise from MLX's RNG (seeded when a seed is given).
    case random(seed: UInt64?)
    /// No noise (deterministic; note the STFT phase of a noiseless unvoiced source is ill-conditioned).
    case zeros
    /// Explicit standard-normal tensor `[L, 9]` (or `[1, L, 9]`), `L = 600 · frames`.
    case given(MLXArray)
}

/// StyleTTS2 `AdaINResBlock1` (HiFi-GAN resblock with AdaIN and Snake activations `x + sin²(αx)/α`).
struct AdaINResBlock1 {
    struct Unit {
        let n1: AdaIN1d, n2: AdaIN1d, c1: KConv, c2: KConv
        let a1: MLXArray, ia1: MLXArray, a2: MLXArray, ia2: MLXArray
    }
    let units: [Unit]

    init(_ w: KWeights, _ k: String, kernel: Int, dilations: [Int]) throws {
        units = try dilations.enumerated().map { i, d in
            let a1 = try w.get(k + ".alpha1.\(i)").reshaped(-1), a2 = try w.get(k + ".alpha2.\(i)").reshaped(-1)
            return Unit(n1: try AdaIN1d(w, k + ".adain1.\(i)"), n2: try AdaIN1d(w, k + ".adain2.\(i)"),
                        c1: try w.conv(k + ".convs1.\(i)", padding: (kernel * d - d) / 2, dilation: d),
                        c2: try w.conv(k + ".convs2.\(i)", padding: (kernel - 1) / 2),
                        a1: a1, ia1: 1 / a1, a2: a2, ia2: 1 / a2)
        }
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var x = x
        for u in units {
            var t = u.n1(x, s)
            var sn = sin(u.a1 * t)
            t = u.c1(t + u.ia1 * (sn * sn))
            t = u.n2(t, s)
            sn = sin(u.a2 * t)
            t = u.c2(t + u.ia2 * (sn * sn))
            x = t + x
        }
        return x
    }
}

/// iSTFTNet `Generator` with the hn-NSF harmonic source and `TorchSTFT` (the default `disable_complex=False` path:
/// `torch.stft`/`torch.istft`, center=True, reflect padding, periodic Hann): pocketfft forward, DFT-matmul inverse.
struct Generator {
    let ups: [KConvT]
    let resblocks: [AdaINResBlock1]
    let noiseConvs: [KConv]
    let noiseRes: [AdaINResBlock1]
    let convPost: KConv
    let lLinear: (w: MLXArray, b: MLXArray)  // [9], [1]
    let kernels: Int
    let nFFT: Int, hop: Int
    let upsampleScale: Int  // prod(upsample_rates) · hop = 300
    let harmonics = 9
    let sampleRate: Float = 24000

    init(_ w: KWeights, rates: [Int], kernelsUp: [Int], initialChannels: Int, resKernels: [Int],
         resDilations: [[Int]], nFFT: Int, hop: Int) throws {
        let p = "decoder.generator."
        ups = try zip(rates, kernelsUp).enumerated().map { i, rk in
            try w.convTransposed(p + "ups.\(i)", stride: rk.0, padding: (rk.1 - rk.0) / 2)
        }
        var rb: [AdaINResBlock1] = []
        var nc: [KConv] = [], nr: [AdaINResBlock1] = []
        for i in 0 ..< rates.count {
            for (j, (k, d)) in zip(resKernels, resDilations).enumerated() {
                rb.append(try AdaINResBlock1(w, p + "resblocks.\(i * resKernels.count + j)", kernel: k, dilations: d))
            }
            if i + 1 < rates.count {
                let s = rates[(i + 1)...].reduce(1, *)
                nc.append(try w.conv(p + "noise_convs.\(i)", stride: s, padding: (s + 1) / 2))
                nr.append(try AdaINResBlock1(w, p + "noise_res.\(i)", kernel: 7, dilations: [1, 3, 5]))
            } else {
                nc.append(try w.conv(p + "noise_convs.\(i)"))
                nr.append(try AdaINResBlock1(w, p + "noise_res.\(i)", kernel: 11, dilations: [1, 3, 5]))
            }
        }
        resblocks = rb; noiseConvs = nc; noiseRes = nr
        kernels = resKernels.count
        convPost = try w.conv(p + "conv_post", padding: 3, dtype: .float32)
        lLinear = (try w.get(p + "m_source.l_linear.weight").reshaped(-1), try w.get(p + "m_source.l_linear.bias"))
        self.nFFT = nFFT
        self.hop = hop
        upsampleScale = rates.reduce(1, *) * hop
    }

    // MARK: harmonic source

    /// `SineGen` sines (before amplitude/uv) and the voiced flags, computed on the CPU so the phase is bit-identical
    /// to PyTorch's float32 arithmetic: the phase reaches ~1e6 rad for long inputs (one float32 ulp there is 0.06 rad),
    /// so it must follow torch's exact op order — float64-accumulated cumsum, fused source-index and lerp in the ×300
    /// linear upsample.
    func sineSource(f0 curve: [Float]) -> (sines: [Float], uv: [Float]) {
        let n2 = curve.count, S = upsampleScale, L = n2 * S, H = harmonics
        let scale = Float(1.0 / Double(S))
        @inline(__always) func nearest(_ n: Int) -> Int { min(Int((Float(n) * scale).rounded(.down)), n2 - 1) }
        // rad = (f0·h / sr) % 1, linear ×1/S downsample (reads samples S·d + S/2 - 1 and S·d + S/2: same block),
        // cumsum (float64 accumulator), ·2·π, ·S.
        var phase = [Float](repeating: 0, count: n2 * H)
        var acc = [Double](repeating: 0, count: H)
        for d in 0 ..< n2 {
            let f = curve[nearest(S * d + S / 2 - 1)]
            for h in 0 ..< H {
                var r = (f * Float(h + 1) / sampleRate).truncatingRemainder(dividingBy: 1)
                if r != 0 && r < 0 { r += 1 }
                acc[h] += Double(r)
                phase[d * H + h] = (Float(acc[h]) * 2 * Float(Double.pi)) * Float(S)  // Float.pi rounds toward 0
            }
        }
        var arg = [Float](repeating: 0, count: L * H)
        var uv = [Float](repeating: 0, count: L)
        arg.withUnsafeMutableBufferPointer { out in
            phase.withUnsafeBufferPointer { ph in
                for n in 0 ..< L {
                    let src = max(Float(-0.5).addingProduct(scale, Float(n) + 0.5), 0)
                    let i0 = Int(src)
                    let l1 = src - Float(i0), l0 = 1 - l1
                    let i1 = min(i0 + 1, n2 - 1)
                    for h in 0 ..< H {
                        out[n * H + h] = (ph[i1 * H + h] * l1).addingProduct(ph[i0 * H + h], l0)
                    }
                    uv[n] = curve[nearest(n)] > 10 ? 1 : 0
                }
            }
        }
        // sin in float64 (correctly rounded to float32 — what torch's float32 sin returns ~95% of the time).
        var argD = arg.map { Double($0) }
        var count = Int32(L * H)
        vvsin(&argD, argD, &count)
        return (argD.map { Float($0) }, uv)
    }

    // MARK: STFT

    /// `torch.hann_window(n, periodic=True)` as torch computes it in float32 (slightly asymmetric), for n = 20;
    /// other sizes use the exact formula.
    static func torchWindow(_ n: Int) -> [Float] {
        if n == 20 {
            let bits: [UInt32] = [
                0x0000_0000, 0x3cc8_78f0, 0x3dc3_910c, 0x3e53_0dd0, 0x3eb0_e444, 0x3f00_0000, 0x3f27_8ddf,
                0x3f4b_3c8c, 0x3f67_8ddf, 0x3f79_bc39, 0x3f80_0000, 0x3f79_bc38, 0x3f67_8dde, 0x3f4b_3c8a,
                0x3f27_8ddf, 0x3f00_0000, 0x3eb0_e441, 0x3e53_0dc8, 0x3dc3_90fc, 0x3cc8_78f0,
            ]
            return bits.map { Float(bitPattern: $0) }
        }
        return (0 ..< n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) }
    }

    /// `torch.stft(center=True, pad_mode='reflect')` magnitude and phase. `x`: `[L]` → `[1, L/hop + 1, 2·(nFFT/2+1)]`.
    func stft(_ x: MLXArray) -> MLXArray {
        let L = x.dim(0), N = nFFT, half = N / 2
        var idx = [Int32]()
        idx.reserveCapacity(L + N)
        for i in stride(from: half, to: 0, by: -1) { idx.append(Int32(i)) }
        for i in 0 ..< L { idx.append(Int32(i)) }
        for i in 0 ..< half { idx.append(Int32(L - 2 - i)) }
        let padded = take(x, MLXArray(idx))
        let nf = L / hop + 1, r = N / hop
        let blocks = padded.reshaped(-1, hop)
        let frames = concatenated((0 ..< r).map { blocks[$0 ..< $0 + nf] }, axis: 1)  // [nf, N]
        // pocketfft on the CPU stream — the FFT torch's CPU backend uses — gives bit-identical spectra, so even the
        // phase of near-zero bins (which the network does see) matches; a DFT matmul does not.
        let fw = frames * MLXArray(Generator.torchWindow(N))
        let X = MLXFFT.rfft(fw, axis: 1, stream: .cpu)
        let re = X.realPart(stream: .cpu), im = X.imaginaryPart(stream: .cpu)
        let mag = sqrt(re * re + im * im)
        return concatenated([mag, atan2(im, re)], axis: 1).expandedDimensions(axis: 0)
    }

    /// `torch.istft(center=True)` of `mag · e^{i·phase}`. `[1, nf, bins]` each → `[hop · (nf - 1)]`.
    func istft(mag: MLXArray, phase: MLXArray) -> MLXArray {
        let N = nFFT, half = N / 2, bins = N / 2 + 1, nf = mag.dim(1), r = N / hop
        let re = (mag * cos(phase))[0], im = (mag * sin(phase))[0]
        let w = Generator.torchWindow(N)
        // irfft (imaginary parts of DC/Nyquist ignored) with the synthesis window folded in.
        var cr = [Float](repeating: 0, count: bins * N), ci = [Float](repeating: 0, count: bins * N)
        for k in 0 ..< bins {
            for n in 0 ..< N {
                let a = 2 * Double.pi * Double((k * n) % N) / Double(N)
                let scale = (k == 0 || k == half) ? 1.0 : 2.0
                cr[k * N + n] = Float(scale * cos(a) * Double(w[n]) / Double(N))
                ci[k * N + n] = (k == 0 || k == half) ? 0 : Float(-scale * sin(a) * Double(w[n]) / Double(N))
            }
        }
        let frames = (matmul(re, MLXArray(cr, [bins, N])) + matmul(im, MLXArray(ci, [bins, N])))
            .reshaped(nf, r, hop)
        var y = MLXArray.zeros([nf + r - 1, hop])
        for c in 0 ..< r {
            y = y + padded(frames[0..., c, 0...], widths: [IntOrPair((c, r - 1 - c)), IntOrPair(0)])
        }
        // window envelope Σ w² over the overlapping frames
        let total = (nf + r - 1) * hop
        var env = [Float](repeating: 0, count: total)
        for b in 0 ..< nf + r - 1 {
            for c in 0 ..< r where b - c >= 0 && b - c < nf {
                for j in 0 ..< hop { env[b * hop + j] += w[c * hop + j] * w[c * hop + j] }
            }
        }
        let out = y.reshaped(-1)[half ..< total - half]
        return out / MLXArray(Array(env[half ..< total - half]))
    }

    // MARK: forward

    /// `x`: decoder features `[1, 2F, 512]`, `s`: `[1, 128]`, `f0`: F0 curve `[2F]` → audio `[600 · F]`.
    func callAsFunction(_ x: MLXArray, _ s: MLXArray, f0 curve: [Float], noise: KokoroNoise,
                        inject: [String: MLXArray] = [:], taps: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var clock = Date()
        func mark(_ label: String, _ a: MLXArray...) {
            guard KokoroModel.profile else { return }
            eval(a)
            print("      \(label.padding(toLength: 10, withPad: " ", startingAt: 0)) \(String(format: "%7.1f ms  peak %5.0f MB", Date().timeIntervalSince(clock) * 1000, Double(Memory.peakMemory) / 1e6))")
            MLX.GPU.resetPeakMemory()
            clock = Date()
        }
        let (sines, uvA) = sineSource(f0: curve)
        mark("sine-cpu")
        let L = uvA.count, H = harmonics
        let uv = MLXArray(uvA, [L, 1])
        let sw = MLXArray(sines, [L, H]) * Float(0.1) * uv
        var sineWaves = sw
        let noiseAmp = uv * Float(0.003) + ((1 - uv) * Float(0.1)) / 3
        switch noise {
        case .zeros: break
        case .given(let n):
            precondition(n.size == L * H, "KokoroNoise.given: expected \(L)×\(H) values, got \(n.shape)")
            sineWaves = sw + noiseAmp * n.reshaped(L, H).asType(.float32)
        case .random(let seed):
            let key = seed.map { MLXRandom.key($0) }
            sineWaves = sw + noiseAmp * MLXRandom.normal([L, H], key: key)
        }
        taps?("sine_wavs", sineWaves)
        // l_linear (9 → 1) + tanh on the CPU with a float64 tanh: MLX's GPU tanh is 1 ulp off torch's correctly
        // rounded value, and with zero noise the unvoiced source is the constant tanh(bias), whose STFT phase is
        // pure rounding noise that the network sees.
        let swA = sineWaves.asArray(Float.self)
        let lw = lLinear.w.asArray(Float.self), lb = lLinear.b.item(Float.self)
        var hs = [Float](repeating: 0, count: L)
        swA.withUnsafeBufferPointer { p in
            for n in 0 ..< L {
                var acc: Float = 0
                for h in 0 ..< H { acc += p[n * H + h] * lw[h] }
                hs[n] = Float(Foundation.tanh(Double(acc + lb)))
            }
        }
        var harSource = MLXArray(hs)  // [L]
        if let h = inject["har_source"] { harSource = h.reshaped(-1) }
        taps?("har_source", harSource)
        var har = stft(harSource)  // [1, nf, 22]
        taps?("har", har)
        if let h = inject["har"] { har = h }
        mark("stft", har)
        var x = x
        for i in 0 ..< ups.count {
            x = leaky(x, 0.1)
            let xs = noiseRes[i](noiseConvs[i](har), s)
            x = ups[i](x)
            if i == ups.count - 1 { x = concatenated([x[0..., 1 ..< 2], x], axis: 1) }  // ReflectionPad1d((1, 0))
            x = x + xs
            eval(x)  // evaluating per resblock bounds the live intermediates at long lengths
            var acc = resblocks[i * kernels](x, s)
            for j in 1 ..< kernels {
                eval(acc)
                acc = acc + resblocks[i * kernels + j](x, s)
            }
            x = acc / Float(kernels)
            taps?("gen_x\(i)", x)
            if let h = inject["gen_x\(i)"] { x = h }
            eval(x)
            mark("stage \(i)")
        }
        let post = convPost(leaky(x, 0.01).asType(.float32))
        taps?("post", post)
        let bins = nFFT / 2 + 1
        return istft(mag: exp(post[0..., 0..., ..<bins]), phase: sin(post[0..., 0..., bins...]))
    }
}

/// iSTFTNet `Decoder`: F0/N stride-2 convs, AdainResBlk1d encode, 4 decode blocks (the last upsamples ×2) with
/// the ASR residual and prosody re-concatenated, then the generator.
struct Decoder {
    let encode: AdainResBlk1d
    let decode: [AdainResBlk1d]
    let f0Conv: KConv, nConv: KConv, asrRes: KConv
    let generator: Generator

    init(_ w: KWeights, config c: KokoroConfig) throws {
        let p = "decoder."
        encode = try AdainResBlk1d(w, p + "encode")
        decode = try (0 ..< 4).map { try AdainResBlk1d(w, p + "decode.\($0)", upsample: $0 == 3) }
        f0Conv = try w.conv(p + "F0_conv", stride: 2, padding: 1, dtype: .float32)
        nConv = try w.conv(p + "N_conv", stride: 2, padding: 1, dtype: .float32)
        asrRes = try w.conv(p + "asr_res.0")
        generator = try Generator(
            w, rates: c.upsampleRates, kernelsUp: c.upsampleKernelSizes, initialChannels: c.upsampleInitialChannel,
            resKernels: c.resblockKernelSizes, resDilations: c.resblockDilationSizes, nFFT: c.nFFT, hop: c.hop)
    }

    /// Features for the generator: `asr` `[1, F, 512]`, `f0`/`n` `[1, 2F]`, `s` `[1, 128]` → `[1, 2F, 512]`.
    func features(asr: MLXArray, f0: MLXArray, n: MLXArray, s: MLXArray) -> MLXArray {
        let dt = asr.dtype
        let F0 = f0Conv(f0.expandedDimensions(axis: -1)).asType(dt)
        let N = nConv(n.expandedDimensions(axis: -1)).asType(dt)
        var x = encode(concatenated([asr, F0, N], axis: 2), s)
        let res = asrRes(asr)
        for (i, b) in decode.enumerated() {
            x = b(concatenated([x.asType(dt), res.asType(dt), F0, N], axis: 2), s)
            if i == 1 { eval(x) }
        }
        return x
    }
}
