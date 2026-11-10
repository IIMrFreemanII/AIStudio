import Foundation
import MLX
import MLXDiffusionCore

/// The subset of Kokoro's `config.json` the model needs.
public struct KokoroConfig: Decodable {
    struct ISTFTNet: Decodable {
        let upsample_kernel_sizes: [Int], upsample_rates: [Int], gen_istft_hop_size: Int, gen_istft_n_fft: Int
        let resblock_dilation_sizes: [[Int]], resblock_kernel_sizes: [Int], upsample_initial_channel: Int
    }
    struct PLBert: Decodable {
        let hidden_size: Int, num_attention_heads: Int, num_hidden_layers: Int, max_position_embeddings: Int
    }
    let istftnet: ISTFTNet
    let plbert: PLBert
    let hidden_dim: Int, n_layer: Int, style_dim: Int, text_encoder_kernel_size: Int, n_token: Int
    public let vocab: [String: Int]

    var upsampleRates: [Int] { istftnet.upsample_rates }
    var upsampleKernelSizes: [Int] { istftnet.upsample_kernel_sizes }
    var upsampleInitialChannel: Int { istftnet.upsample_initial_channel }
    var resblockKernelSizes: [Int] { istftnet.resblock_kernel_sizes }
    var resblockDilationSizes: [[Int]] { istftnet.resblock_dilation_sizes }
    var nFFT: Int { istftnet.gen_istft_n_fft }
    var hop: Int { istftnet.gen_istft_hop_size }
}

/// Kokoro-82M (StyleTTS2 + iSTFTNet): phoneme tokens + a 256-d voice style → 24 kHz mono speech.
/// A port of `kokoro.model.KModel.forward_with_tokens` (hexgrad/kokoro, Apache-2.0).
///
/// ```swift
/// let model = try KokoroModel(directory: dir)               // config.json + kokoro-v1_0.pth (+ voices/)
/// let ps = "həlˈO wˈɜɹld."                                   // misaki phonemes
/// let pack = try model.loadVoice("af_heart")               // [510, 1, 256]
/// let out = model(tokens: model.tokens(ps), refS: KokoroModel.refS(pack, phonemes: ps), seed: 0)
/// // out.audio: [600 · Σdurations] float32 at 24 kHz; out.durations: frames per token (incl. the 2 pad tokens)
/// ```
public final class KokoroModel {
    public static let sampleRate = 24000
    /// Audio samples per predicted duration frame (2 F0 frames × 300).
    public static let samplesPerFrame = 600
    /// Maximum phonemes per call (the PL-BERT context is 512 tokens including the two pad tokens).
    public static let maxPhonemes = 510

    public let config: KokoroConfig
    /// Phoneme (one Unicode scalar) → token id.
    public let vocab: [String: Int]
    public let dtype: DType
    let directory: URL
    let bert: Albert
    let bertEncoder: Lin
    let predictor: ProsodyPredictor
    let textEncoder: TextEncoder
    let decoder: Decoder
    private var voices: [String: MLXArray] = [:]
    /// Development aid: when true, `forward` evaluates at stage boundaries and prints per-stage wall times.
    public static var profile = false
    private let lock = NSLock()

    /// Loads `config.json` and `kokoro-v1_0.pth` (or the first `*.pth`) from `directory`. `dtype` applies to the
    /// convolution/linear weights (LSTMs, norms, the harmonic source and the iSTFT always run in float32).
    /// Use float32: bfloat16 is no faster here and perturbs durations/F0 noticeably (≈1e-2 relative).
    public init(directory: URL, dtype: DType = .float32) throws {
        self.directory = directory
        self.dtype = dtype
        config = try JSONDecoder().decode(KokoroConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        vocab = config.vocab
        var pth = directory.appendingPathComponent("kokoro-v1_0.pth")
        if !FileManager.default.fileExists(atPath: pth.path),
           let any = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .first(where: { $0.pathExtension == "pth" }) {
            pth = any
        }
        let w = KWeights(try TorchPickle.load(url: pth), dtype: dtype)
        bert = try Albert(w, heads: config.plbert.num_attention_heads, layers: config.plbert.num_hidden_layers)
        bertEncoder = try w.linear("bert_encoder")
        predictor = try ProsodyPredictor(w, layers: config.n_layer)
        textEncoder = try TextEncoder(w, depth: config.n_layer, kernel: config.text_encoder_kernel_size)
        decoder = try Decoder(w, config: config)
    }

    /// `[0, vocab[p] for each phoneme (unknown ones dropped), 0]`, phonemes being Unicode scalars (as Python's
    /// `str` iteration). Truncates to ``maxPhonemes`` known phonemes.
    public func tokens(_ phonemes: String) -> [Int] {
        var ids = [0]
        for u in phonemes.unicodeScalars {
            if let id = vocab[String(u)] { ids.append(id) }
            if ids.count > KokoroModel.maxPhonemes { break }
        }
        ids.append(0)
        return ids
    }

    /// `voices/<name>.pt` → `[510, 1, 256]` float32 (cached).
    public func loadVoice(_ name: String) throws -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let v = voices[name] { return v }
        // Voice names are file stems; a graph from a remote client must not reach outside `voices/`.
        guard !name.isEmpty, !name.contains("/"), !name.contains("..") else { throw WeightError.missing("voice \(name)") }
        let url = directory.appendingPathComponent("voices/\(name).pt")
        guard let v = try TorchPickle.load(url: url).values.first else { throw WeightError.missing("voice \(name)") }
        let a = v.asType(.float32)
        eval(a)
        voices[name] = a
        return a
    }

    /// The style row for an utterance: `pack[len(phonemes) - 1]` → `[1, 256]` (as `KPipeline`, where the length is
    /// the phoneme string's length in Unicode scalars, unknown symbols included).
    public static func refS(_ pack: MLXArray, phonemes: String) -> MLXArray {
        refS(pack, phonemeCount: phonemes.unicodeScalars.count)
    }

    public static func refS(_ pack: MLXArray, phonemeCount n: Int) -> MLXArray {
        pack[max(0, min(n, pack.dim(0)) - 1)].reshaped(1, -1)
    }

    /// Synthesizes `tokens` (from ``tokens(_:)``) with style `refS` `[1, 256]` (`[:, :128]` → decoder, `[:, 128:]` →
    /// prosody). Returns float32 audio `[600 · Σdurations]` at 24 kHz and the per-token frame durations.
    /// `speed` divides the predicted durations. `seed` seeds the harmonic source noise (nil: MLX's global RNG).
    public func callAsFunction(tokens: [Int], refS: MLXArray, speed: Float = 1, seed: UInt64? = nil)
        -> (audio: MLXArray, durations: [Int])
    {
        forward(tokens: tokens, refS: refS, speed: speed, noise: .random(seed: seed))
    }

    /// Only the duration branch (PL-BERT → duration encoder → duration LSTM): frames per token (600 samples each)
    /// and the unrounded values. Cheap; useful for timestamps.
    public func predictDurations(tokens: [Int], refS: MLXArray, speed: Float = 1) -> (durations: [Int], raw: [Float]) {
        let ids = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
        let sPro = refS.reshaped(1, -1).asType(.float32)[0..., 128...]
        let d = predictor.durationEncoder(bertEncoder(bert(ids)).asType(.float32), sPro)
        let raw = predictor.durations(d, speed: speed).asArray(Float.self)
        return (raw.map { max(1, Int($0.rounded(.toNearestOrEven))) }, raw)
    }

    /// Full forward with control over the noise and, for verification, hooks: `tap` receives intermediates
    /// (channels-last, e.g. `bert` `[1,T,768]`, `d_en` `[1,T,512]`, `d` `[1,T,640]`, `dur_float` `[T]`, `en`
    /// `[1,F,640]`, `F0_pred`/`N_pred` `[1,2F]`, `t_en`, `asr`, `dec_x` `[1,2F,512]`, `sine_wavs` `[L,9]`, `har`,
    /// `gen_x0`, `gen_x1`, `post`); `inject` replaces the same-named stage outputs downstream (plus `pred_dur` as an
    /// int32 `[T]`) so modules can be checked in isolation — `tap` still sees what the module itself computed.
    public func forward(tokens: [Int], refS: MLXArray, speed: Float = 1, noise: KokoroNoise,
                        inject: [String: MLXArray] = [:], tap: ((String, MLXArray) -> Void)? = nil)
        -> (audio: MLXArray, durations: [Int])
    {
        // A stage's own output goes to `tap`; the forward continues with the injected value when there is one.
        func stage(_ name: String, _ f: () -> MLXArray) -> MLXArray {
            if inject[name] != nil && tap == nil { return inject[name]! }
            let v = f()
            tap?(name, v)
            return inject[name] ?? v
        }
        var clock = Date()
        func mark(_ label: String, _ a: MLXArray...) {
            guard KokoroModel.profile else { return }
            eval(a)
            print("    \(label.padding(toLength: 12, withPad: " ", startingAt: 0)) \(String(format: "%7.1f ms  peak %5.0f MB", Date().timeIntervalSince(clock) * 1000, Double(Memory.peakMemory) / 1e6))")
            MLX.GPU.resetPeakMemory()
            clock = Date()
        }
        let ids = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
        let refS = refS.reshaped(1, -1).asType(.float32)
        let sPro = refS[0..., 128...], sDec = refS[0..., ..<128]
        let bertOut = stage("bert") { bert(ids) }
        let dEn = stage("d_en") { bertEncoder(bertOut).asType(.float32) }
        mark("bert", dEn)
        let d = stage("d") { predictor.durationEncoder(dEn, sPro) }
        mark("dur-encoder", d)
        var durations: [Int]
        if let pd = inject["pred_dur"] {
            durations = pd.asType(.int32).asArray(Int32.self).map(Int.init)
        } else {
            let df = stage("dur_float") { predictor.durations(d, speed: speed) }
            durations = df.asArray(Float.self).map { max(1, Int($0.rounded(.toNearestOrEven))) }
        }
        var idx = [Int32]()
        for (i, n) in durations.enumerated() { idx.append(contentsOf: repeatElement(Int32(i), count: n)) }
        let align = MLXArray(idx)
        mark("durations")
        let en = stage("en") { take(d, align, axis: 1) }
        let f0n = inject["F0_pred"] != nil && inject["N_pred"] != nil && tap == nil ? nil : predictor.f0n(en, sPro)
        let f0 = stage("F0_pred") { f0n!.0 }
        let n = stage("N_pred") { f0n!.1 }
        mark("F0/N", f0, n)
        let tEn = stage("t_en") { textEncoder(ids) }
        let asr = stage("asr") { take(tEn, align, axis: 1).asType(dtype) }
        mark("text-encoder", asr)
        let x = stage("dec_x") { decoder.features(asr: asr, f0: f0, n: n, s: sDec) }
        mark("decoder", x)
        let curve = f0.asType(.float32).reshaped(-1).asArray(Float.self)
        let audio = decoder.generator(x, sDec, f0: curve, noise: noise, inject: inject, taps: tap)
        eval(audio)
        mark("generator")
        return (audio, durations)
    }
}
