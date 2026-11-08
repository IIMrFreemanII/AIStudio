import Foundation
import MLX
import MLXDiffusionCore

/// Single-layer bidirectional `nn.LSTM(batch_first=True)` for batch 1 (PyTorch gate order i, f, g, o).
///
/// The input projections of both directions are one matmul; the recurrence runs in a single Metal kernel with one
/// threadgroup per direction (one thread per gate row, `h` kept in threadgroup memory), so a sequence of any length
/// costs one dispatch instead of a graph of `T` steps.
struct BiLSTM {
    let wih: MLXArray  // [In, 2·4H]  (forward | reverse), transposed for x @ W
    let bias: MLXArray  // [2·4H]  (b_ih + b_hh)
    let whhT: MLXArray  // [2, H, 4H]
    let hidden: Int

    init(_ w: KWeights, _ k: String) throws {
        let wf = try w.get(k + ".weight_ih_l0"), wr = try w.get(k + ".weight_ih_l0_reverse")
        hidden = wf.dim(0) / 4
        wih = concatenated([wf, wr], axis: 0).T
        bias = concatenated([try w.get(k + ".bias_ih_l0") + w.get(k + ".bias_hh_l0"),
                             try w.get(k + ".bias_ih_l0_reverse") + w.get(k + ".bias_hh_l0_reverse")], axis: 0)
        whhT = stacked([try w.get(k + ".weight_hh_l0").T, try w.get(k + ".weight_hh_l0_reverse").T], axis: 0)
        eval(wih, bias, whhT)
    }

    /// `x`: `[1, T, In]` → `[1, T, 2H]` (forward and reverse hidden states concatenated, as torch).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let T = x.dim(1)
        let xp = (matmul(x[0].asType(.float32), wih) + bias).reshaped(T, 2, 4 * hidden)
        let out = BiLSTM.kernel(
            [xp, whhT], template: [("H", hidden)],
            grid: (2 * 4 * hidden, 1, 1), threadGroup: (4 * hidden, 1, 1),
            outputShapes: [[T, 2, hidden]], outputDTypes: [.float32])[0]
        return out.reshaped(1, T, 2 * hidden)
    }

    static let kernel = MLXFast.metalKernel(
        name: "kokoro_bilstm", inputNames: ["xp", "whh"], outputNames: ["out"],
        source: """
            const uint dir = threadgroup_position_in_grid.x;
            const uint j = thread_position_in_threadgroup.x;
            constexpr int G = 4 * H;
            threadgroup float hs[H];
            threadgroup float gs[G];
            const int T = xp_shape[0];
            const device float* W = whh + dir * (H * G) + j;
            float c = 0.0f;
            if (j < H) hs[j] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int s = 0; s < T; ++s) {
                const int t = dir == 0 ? s : T - 1 - s;
                float a0 = xp[(t * 2 + dir) * G + j], a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int k = 0; k < H; k += 4) {
                    a0 = fma(W[(k + 0) * G], hs[k + 0], a0);
                    a1 = fma(W[(k + 1) * G], hs[k + 1], a1);
                    a2 = fma(W[(k + 2) * G], hs[k + 2], a2);
                    a3 = fma(W[(k + 3) * G], hs[k + 3], a3);
                }
                gs[j] = (a0 + a1) + (a2 + a3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (j < H) {
                    const float ig = 1.0f / (1.0f + metal::precise::exp(-gs[j]));
                    const float fg = 1.0f / (1.0f + metal::precise::exp(-gs[H + j]));
                    const float gg = metal::precise::tanh(gs[2 * H + j]);
                    const float og = 1.0f / (1.0f + metal::precise::exp(-gs[3 * H + j]));
                    c = fg * c + ig * gg;
                    const float h = og * metal::precise::tanh(c);
                    hs[j] = h;
                    out[(t * 2 + dir) * H + j] = h;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """)
}
