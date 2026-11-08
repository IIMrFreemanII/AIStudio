import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

/// LTX-2.3 prompt encoder (`PromptEncoder.encode` of the reference, Gemma 3 path):
/// prompt → Gemma 3 12B hidden states of all 49 layers → per-token RMS stack → video/audio projection →
/// video/audio embeddings connectors.
///
/// Output: `video` `[1, maxLength, 4096]` and `audio` `[1, maxLength, 2048]` bfloat16 (maxLength = 1024). The first
/// `tokenCount` rows are the prompt tokens (BOS first); the remaining rows are the connectors' learnable registers
/// (the reference replaces padding with registers), so every row is valid and no text mask goes to the DiT.
public final class LTXTextEncoder {
    public let tokenizer: Tokenizer
    public let gemma: LTXGemma3Model
    public let connector: LTXTextConnector
    /// Padded sequence length (`LTX2_GEMMA_MAX_LENGTH`, default 1024). Longer prompts keep their last tokens.
    public let maxLength: Int
    public let padTokenId: Int
    /// Run Gemma over the full left-padded sequence with the reference's causal + padding mask, exactly like the
    /// reference. Off by default: padded rows never reach the output (they're zeroed, then replaced by registers)
    /// and padding keys are masked out, so running Gemma on the real tokens only (at RoPE positions
    /// `maxLength - n ..< maxLength`) gives the same embeddings, much faster.
    public var computePaddedRows = false
    /// Run the 188160 → 4096/2048 projection over all `maxLength` rows (zero rows for padding) like the reference,
    /// rather than over the real tokens only. Costs a few hundred ms but matches the reference's matmul tiling
    /// exactly; the connectors amplify even ~1e-4 input differences to ~5e-3.
    public var projectPaddedRows = true

    /// - Parameters:
    ///   - gemmaDirectory: mlx-community `gemma-3-12b-it-4bit` folder (config, tokenizer, safetensors).
    ///   - ltxDirectory: LTX-2.3 folder with `connector.safetensors` and `embedded_config.json`.
    public init(gemmaDirectory: URL, ltxDirectory: URL, dtype: DType = .bfloat16, maxLength: Int = 1024) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: gemmaDirectory)
        padTokenId = tokenizer.convertTokenToId("<pad>") ?? 0
        self.maxLength = maxLength
        gemma = try LTXGemma3Model(directory: gemmaDirectory, dtype: dtype)
        let cfg = ltxDirectory.appendingPathComponent("embedded_config.json")
        connector = try LTXTextConnector(
            url: ltxDirectory.appendingPathComponent("connector.safetensors"),
            ltxConfig: FileManager.default.fileExists(atPath: cfg.path) ? cfg : nil,
            embeddingDim: gemma.config.hiddenSize, dtype: dtype)
    }

    /// Token ids of a prompt as the reference builds them: `encode(prompt.strip())` with special tokens (BOS),
    /// truncated from the left to `maxLength`. Unpadded.
    public func tokenize(_ prompt: String) -> [Int] {
        let ids = tokenizer.encode(text: prompt.trimmingCharacters(in: .whitespacesAndNewlines))
        return ids.count > maxLength ? Array(ids.suffix(maxLength)) : ids
    }

    /// Gemma hidden states (embedding output + 48 layers) of the real tokens only: 49 arrays `[1, n, 3840]`.
    public func hiddenStates(tokens ids: [Int]) -> [MLXArray] {
        let n = ids.count, L = maxLength
        if computePaddedRows {
            let padded = [Int](repeating: padTokenId, count: L - n) + ids
            let valid = [Int32](repeating: 0, count: L - n) + [Int32](repeating: 1, count: n)
            let states = gemma.allHiddenStates(tokens: MLXArray(padded.map(Int32.init)).reshaped(1, L),
                                               mask: ltxGemmaMask(valid: valid))
            return states.map { $0[0..., (L - n)...] }
        }
        let mask = n > 1 ? ltxGemmaMask(valid: [Int32](repeating: 1, count: n)) : nil
        return gemma.allHiddenStates(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, n),
                                     positionOffset: L - n, mask: mask)
    }

    /// Encodes a prompt to `(video [1, maxLength, 4096], audio [1, maxLength, 2048])`, evaluated.
    public func encode(_ prompt: String) throws -> (video: MLXArray, audio: MLXArray) {
        encode(tokens: tokenize(prompt))
    }

    /// Same as ``encode(_:)`` from token ids (as returned by ``tokenize(_:)``).
    public func encode(tokens ids: [Int]) -> (video: MLXArray, audio: MLXArray) {
        precondition(!ids.isEmpty && ids.count <= maxLength)
        let n = ids.count
        var states = hiddenStates(tokens: ids)
        let x = ltxPerTokenRMSStack(states, paddedLength: maxLength, keepPadding: projectPaddedRows || computePaddedRows)
        states.removeAll()
        eval(x)
        var (pv, pa) = connector.project(x)
        if x.dim(1) > n {
            pv = pv[0..., (maxLength - n)...]
            pa = pa[0..., (maxLength - n)...]
        }
        let v = connector.video(connector.video.fillWithRegisters(pv, length: maxLength))
        let a = connector.audio(connector.audio.fillWithRegisters(pa, length: maxLength))
        eval(v, a)
        return (v, a)
    }
}
