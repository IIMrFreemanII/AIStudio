import Foundation
import MLX
import MLXDiffusionCore

/// Text generation with the Qwen3-VL language model that Qwen-Image already loads as its text encoder.
///
/// The checkpoint keeps the model's `lm_head` (untied from the embeddings) and final norm; they are only needed
/// here, so they're loaded on first use (8-bit quantized, about 0.6 GB) and the image path never pays for them.
extension Qwen3TextEncoder {
    struct LMHead {
        let norm: MLXArray
        let proj: Lin
    }

    public struct SamplingOptions: Sendable {
        public var maxTokens = 320
        /// 0 = greedy.
        public var temperature: Float = 0.7
        public var topP: Float = 0.9
        /// > 1 discourages repeating tokens already generated.
        public var repetitionPenalty: Float = 1.05
        public var seed: UInt64 = 0
        public init(maxTokens: Int = 320, temperature: Float = 0.7, topP: Float = 0.9, seed: UInt64 = 0) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.seed = seed
        }
    }

    /// Whether the LM head is resident (text generation has been used).
    public var isGeneratorLoaded: Bool {
        lmHeadLock.lock(); defer { lmHeadLock.unlock() }
        return lmHeadStorage != nil
    }

    func lmHead() throws -> LMHead {
        lmHeadLock.lock(); defer { lmHeadLock.unlock() }
        if let h = lmHeadStorage { return h }
        // Safetensors are memory-mapped lazily: only the two tensors read here are materialised.
        let store = try WeightStore(directory: directory)
        let norm = try store.get("model.language_model.norm.weight").asType(dtype)
        let w = try store.get("lm_head.weight").asType(dtype)
        let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 8)
        eval(norm, wq, scales, biases ?? norm)
        let head = LMHead(norm: norm, proj: Lin(weight: wq, scales: scales, qbiases: biases, bits: 8, groupSize: 64))
        lmHeadStorage = head
        return head
    }

    func logits(_ hidden: MLXArray, head: LMHead) -> MLXArray {
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]  // [1, 1, D]
        return head.proj(rmsNorm(last, weight: head.norm, eps: config.rmsEps)).reshaped(-1).asType(.float32)
    }

    /// Generates tokens after `tokens` (a full chat prompt ending in the assistant turn).
    ///
    /// - Parameters:
    ///   - visual: image inputs for the prompt (as built for the feature path).
    ///   - stop: token ids that end generation (not included in the result).
    ///   - onToken: called with all generated ids so far; return `false` to stop early.
    public func generate(tokens: [Int], visual: VisualInputs?, stop: Set<Int>, options: SamplingOptions,
                         onToken: ([Int]) -> Bool) throws -> [Int] {
        let head = try lmHead()
        let cache = KVCache()
        let L = tokens.count
        let positions: RotaryPositions = visual.map { .mrope(mropeTables($0.positions)) } ?? .sequential(offset: 0)
        var h = forward(embeddings(for: tokens, visual: visual), positions: positions,
                        mask: L > 1 ? causalMask(L, dtype: embed.dtype) : nil, cache: cache, visual: visual)
        cache.nextPosition = visual.map { v in Int(v.positions.map { max($0.0, $0.1, $0.2) }.max() ?? -1) + 1 } ?? L

        var out: [Int] = []
        for step in 0..<options.maxTokens {
            try Task.checkCancellation()
            let next = sample(logits(h, head: head), generated: out, options: options, step: step)
            if stop.contains(next) { break }
            out.append(next)
            if !onToken(out) { break }
            // Generated text has equal positions on every mRoPE axis, so plain RoPE at the running position.
            h = forward(embeddings(for: [next], visual: nil), positions: .sequential(offset: cache.nextPosition),
                        mask: nil, cache: cache, visual: nil)
            cache.nextPosition += 1
        }
        return out
    }

    func sample(_ logits: MLXArray, generated: [Int], options o: SamplingOptions, step: Int) -> Int {
        var l = logits
        if o.repetitionPenalty != 1, !generated.isEmpty {
            // CTRL-style: shrink positive logits, grow negative ones, for tokens already produced.
            let ids = MLXArray(Array(Set(generated)).map { Int32($0) })
            let picked = l[ids]
            let penalised = MLX.where(picked .> 0, picked / o.repetitionPenalty, picked * o.repetitionPenalty)
            l[ids] = penalised
        }
        if o.temperature <= 0 { return argMax(l).item(Int.self) }
        let probs = softmax(l / o.temperature)
        let order = argSort(-probs)
        let sorted = probs[order]
        // Keep the smallest prefix whose mass reaches top-p (always at least the top token).
        let keep = (cumsum(sorted) - sorted) .< o.topP
        let filtered = MLX.where(keep, log(sorted), MLXArray(-Float.infinity))
        let pick = MLXRandom.categorical(filtered, key: MLXRandom.key(o.seed &+ UInt64(step)))
        return order[pick].item(Int.self)
    }
}

/// The prompt assistant on Qwen-Image: Qwen3-VL in its normal chat format (not the image-generation template).
extension QwenImage21Pipeline: TextGenerating {
    public func generateText(system: String, user: String, images: [MLXArray], options: TextGenerationOptions,
                             onText: (String) -> Bool) throws -> String {
        let vision = String(repeating: "<|vision_start|><|image_pad|><|vision_end|>", count: images.count)
        let chat = "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(vision)\(user)<|im_end|>\n"
            + "<|im_start|>assistant\n"
        var ids = tokenizer.encode(text: chat, addSpecialTokens: false)
        var visual: Qwen3TextEncoder.VisualInputs?
        if !images.isEmpty {
            let visions = try images.map { textEncoder.vision(try conditioningRGB($0).rgb) }
            (ids, visual) = { let e = Self.expandImagePads(ids, visions: visions); return (e.ids, e.visual) }()
        }
        let stop = Set(["<|im_end|>", "<|endoftext|>"].compactMap { tokenizer.convertTokenToId($0) })
        var sampling = Qwen3TextEncoder.SamplingOptions(maxTokens: options.maxTokens, temperature: options.temperature,
                                                        seed: options.seed)
        if options.temperature <= 0 { sampling.repetitionPenalty = 1 }
        let out = try textEncoder.generate(tokens: ids, visual: visual, stop: stop, options: sampling) { generated in
            onText(tokenizer.decode(tokens: generated))
        }
        return tokenizer.decode(tokens: out)
    }
}
