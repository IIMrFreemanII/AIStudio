import Foundation
import MLX

/// One low-rank update applied as a side branch: `y += scale · (x Aᵀ) Bᵀ`.
public struct LoRAAdapter {
    public let down: MLXArray  // A: [rank, in]
    public let up: MLXArray  // B: [out, rank]
    public let scale: Float  // strength · alpha / rank

    public init(down: MLXArray, up: MLXArray, scale: Float) {
        self.down = down
        self.up = up
        self.scale = scale
    }
}

public enum LoRAError: Error, CustomStringConvertible {
    case noMatches(URL, sample: [String])
    case unsupported(String)

    public var description: String {
        switch self {
        case .noMatches(let u, let s):
            return "\(u.lastPathComponent): no LoRA weights match this model (keys look like \(s.joined(separator: ", ")))"
        case .unsupported(let m): return "this model does not support LoRA (\(m))"
        }
    }
}

/// A parsed LoRA file, keyed by the target model's linear-layer names (e.g. `transformer_blocks.0.attn.to_q`).
public struct LoRAFile {
    public struct Pair {
        public var down: MLXArray?
        public var up: MLXArray?
        public var alpha: Float?
        /// The file's tensor keys this pair came from (for reporting unmatched ones).
        public var keys: [String] = []
        public init(down: MLXArray? = nil, up: MLXArray? = nil, alpha: Float? = nil) {
            self.down = down; self.up = up; self.alpha = alpha
        }
    }

    public let url: URL
    public let pairs: [String: Pair]
    public let unmatchedKeys: [String]

    public static let keyPrefixes = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
                                     "unet.", "lora_unet_", "lora_transformer_"]
    static let suffixes: [(String, Int)] = [  // 0 = down, 1 = up, 2 = alpha
        (".lora_A.weight", 0), (".lora_B.weight", 1), (".lora_down.weight", 0), (".lora_up.weight", 1),
        (".lora.down.weight", 0), (".lora.up.weight", 1), (".alpha", 2), (".lora_alpha", 2),
    ]

    /// Groups a LoRA file's tensors by module (the key without its LoRA suffix and with the known prefixes removed).
    /// Keys that aren't LoRA tensors are returned separately.
    public static func groups(_ arrays: [String: MLXArray]) -> (pairs: [String: Pair], other: [String]) {
        var pairs: [String: Pair] = [:]
        var other: [String] = []
        for (key, value) in arrays {
            guard let (suffix, kind) = suffixes.first(where: { key.hasSuffix($0.0) }) else {
                other.append(key); continue
            }
            var name = String(key.dropLast(suffix.count))
            for p in keyPrefixes where name.hasPrefix(p) { name = String(name.dropFirst(p.count)) }
            var pair = pairs[name] ?? Pair()
            pair.keys.append(key)
            switch kind {
            case 0: pair.down = value
            case 1: pair.up = value
            default: pair.alpha = value.asType(.float32).item(Float.self)
            }
            pairs[name] = pair
        }
        return (pairs, other)
    }

    /// Parses PEFT (`lora_A`/`lora_B`), kohya/Comfy (`lora_down`/`lora_up` + `.alpha`) and `lora_unet_…` underscore
    /// naming, mapping keys onto `moduleNames` (the model's linear layers).
    public init(url: URL, moduleNames: Set<String>) throws {
        let (arrays, _) = try loadArraysAndMetadata(url: url)
        let underscored = Dictionary(moduleNames.map { ($0.replacingOccurrences(of: ".", with: "_"), $0) },
                                     uniquingKeysWith: { a, _ in a })
        let (groups, other) = Self.groups(arrays)
        var pairs: [String: Pair] = [:]
        var unmatched = other
        for (name, pair) in groups {
            guard let target = moduleNames.contains(name) ? name : underscored[name] else {
                unmatched += pair.keys; continue
            }
            pairs[target] = pair
        }
        try self.init(url: url, pairs: pairs, unmatchedKeys: unmatched, sample: Array(arrays.keys.sorted().prefix(3)))
    }

    /// A LoRA from already-mapped pairs (model-specific key mappings build these). Pairs missing a matrix are dropped.
    public init(url: URL, pairs: [String: Pair], unmatchedKeys: [String], sample: [String] = []) throws {
        self.url = url
        self.pairs = pairs.filter { $0.value.down != nil && $0.value.up != nil }
        self.unmatchedKeys = unmatchedKeys.sorted()
        if self.pairs.isEmpty { throw LoRAError.noMatches(url, sample: sample) }
    }

    /// Adapters at `strength`, cast to `dtype`.
    public func adapters(strength: Float, dtype: DType = .bfloat16) -> [String: LoRAAdapter] {
        pairs.mapValues { p in
            let down = p.down!, up = p.up!
            let rank = Float(down.dim(0))
            let scale = strength * (p.alpha ?? rank) / rank
            return LoRAAdapter(down: down.asType(dtype), up: up.asType(dtype), scale: scale)
        }
    }
}

/// What applying a LoRA did, for display.
public struct LoRAReport: CustomStringConvertible {
    public let file: String
    public let matchedLayers: Int
    public let unmatchedKeys: Int
    public init(file: String, matchedLayers: Int, unmatchedKeys: Int) {
        self.file = file
        self.matchedLayers = matchedLayers
        self.unmatchedKeys = unmatchedKeys
    }
    public var description: String {
        "\(file): \(matchedLayers) layers" + (unmatchedKeys > 0 ? ", \(unmatchedKeys) keys ignored" : "")
    }
}
