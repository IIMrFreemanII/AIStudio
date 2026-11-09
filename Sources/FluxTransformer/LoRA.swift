import Foundation
import MLX
import MLXDiffusionCore

/// Flux LoRA files come in three layouts:
/// - diffusers / PEFT and ai-toolkit: `transformer.transformer_blocks.N.attn.to_q.lora_A.weight` — the model's own names;
/// - kohya with diffusers names: `lora_transformer_single_transformer_blocks_N_attn_to_q.lora_down.weight`;
/// - kohya / BFL original names: `lora_unet_double_blocks_N_img_attn_qkv.lora_down.weight` (or
///   `diffusion_model.double_blocks.N.img_attn.qkv.lora_A.weight`), where q, k, v (and in single blocks the MLP
///   input) are one fused projection: its up matrix is split by rows onto the separate layers, all sharing the down
///   matrix, which gives exactly the fused update.
/// Text-encoder keys (`lora_te…`) don't apply to the transformer and are reported as ignored.
extension FluxTransformer {
    /// A diffusers target of a BFL-named module: the layer names and their row counts (in order), and whether the
    /// two halves are swapped (BFL's final modulation is `(shift, scale)`, diffusers' `(scale, shift)`).
    struct BFLTarget {
        var names: [String]
        var swapHalves = false
    }

    func bflTarget(_ name: String) -> BFLTarget? {
        let n = name.replacingOccurrences(of: ".", with: "_")
        let parts = n.split(separator: "_").map(String.init)
        func rest(_ k: Int) -> String { parts.dropFirst(k).joined(separator: "_") }
        if parts.count >= 4, parts[0] == "double", parts[1] == "blocks", let i = Int(parts[2]) {
            let p = "transformer_blocks.\(i)."
            let img = parts[3] == "img"
            switch rest(4) {
            case "attn_qkv":
                return BFLTarget(names: img ? ["attn.to_q", "attn.to_k", "attn.to_v"].map { p + $0 }
                    : ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"].map { p + $0 })
            case "attn_proj": return BFLTarget(names: [p + (img ? "attn.to_out.0" : "attn.to_add_out")])
            case "mlp_0": return BFLTarget(names: [p + (img ? "ff.net.0.proj" : "ff_context.net.0.proj")])
            case "mlp_2": return BFLTarget(names: [p + (img ? "ff.net.2" : "ff_context.net.2")])
            case "mod_lin": return BFLTarget(names: [p + (img ? "norm1.linear" : "norm1_context.linear")])
            default: return nil
            }
        }
        if parts.count >= 4, parts[0] == "single", parts[1] == "blocks", let i = Int(parts[2]) {
            let p = "single_transformer_blocks.\(i)."
            switch rest(3) {
            case "linear1": return BFLTarget(names: ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"].map { p + $0 })
            case "linear2": return BFLTarget(names: [p + "proj_out"])
            case "modulation_lin": return BFLTarget(names: [p + "norm.linear"])
            default: return nil
            }
        }
        let top: [String: String] = [
            "img_in": "x_embedder", "txt_in": "context_embedder",
            "time_in_in_layer": "time_text_embed.timestep_embedder.linear_1",
            "time_in_out_layer": "time_text_embed.timestep_embedder.linear_2",
            "vector_in_in_layer": "time_text_embed.text_embedder.linear_1",
            "vector_in_out_layer": "time_text_embed.text_embedder.linear_2",
            "guidance_in_in_layer": "time_text_embed.guidance_embedder.linear_1",
            "guidance_in_out_layer": "time_text_embed.guidance_embedder.linear_2",
            "final_layer_linear": "proj_out",
        ]
        if let t = top[n] { return BFLTarget(names: [t]) }
        if n == "final_layer_adaLN_modulation_1" { return BFLTarget(names: ["norm_out.linear"], swapHalves: true) }
        return nil
    }

    /// Parses a Flux LoRA in any of the supported layouts onto this transformer's layers.
    public func loraFile(_ url: URL) throws -> LoRAFile {
        let (arrays, _) = try loadArraysAndMetadata(url: url)
        let names = linearNames
        let widths = linearOutFeatures
        let underscored = Dictionary(names.map { ($0.replacingOccurrences(of: ".", with: "_"), $0) },
                                     uniquingKeysWith: { a, _ in a })
        let (groups, other) = LoRAFile.groups(arrays)
        var pairs: [String: LoRAFile.Pair] = [:]
        var unmatched = other
        for (name, pair) in groups {
            if let direct = names.contains(name) ? name : underscored[name] {
                pairs[direct] = pair
                continue
            }
            guard let target = bflTarget(name), let up = pair.up,
                  target.names.allSatisfy({ widths[$0] != nil }) else {
                unmatched += pair.keys; continue
            }
            let rows = target.names.map { widths[$0]! }
            guard rows.reduce(0, +) == up.dim(0) else { unmatched += pair.keys; continue }
            var offset = 0
            for (t, r) in zip(target.names, rows) {
                var u = up[offset ..< offset + r]
                if target.swapHalves {
                    let h = r / 2
                    u = concatenated([u[h...], u[..<h]], axis: 0)
                }
                pairs[t] = LoRAFile.Pair(down: pair.down, up: u, alpha: pair.alpha)
                offset += r
            }
        }
        return try LoRAFile(url: url, pairs: pairs, unmatchedKeys: unmatched,
                            sample: Array(arrays.keys.sorted().prefix(3)))
    }
}
