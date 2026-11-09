import Foundation
import MLX
import MLXDiffusionCore

/// The FLUX.1 autoencoder (diffusers `AutoencoderKL`: 16 latent channels, 8× downsampling, GroupNorm-32 ResNet
/// blocks with one mid-block attention, no quant convs). Computed in float32, in NHWC; large 3×3 convolutions run
/// in horizontal bands so full-resolution layers don't materialise gigabyte patch buffers.
public final class FluxVAE {
    public struct Config: Sendable {
        public var latentChannels = 16
        public var blockOutChannels = [128, 256, 512, 512]
        public var layersPerBlock = 2
        public var groups = 32
        public var scalingFactor: Float = 0.3611
        public var shiftFactor: Float = 0.1159

        public init() {}

        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            if let v = o["latent_channels"] as? Int { latentChannels = v }
            if let v = o["block_out_channels"] as? [Int] { blockOutChannels = v }
            if let v = o["layers_per_block"] as? Int { layersPerBlock = v }
            if let v = o["norm_num_groups"] as? Int { groups = v }
            if let v = o["scaling_factor"] as? Double { scalingFactor = Float(v) }
            if let v = o["shift_factor"] as? Double { shiftFactor = Float(v) }
        }
    }

    struct Conv {
        let weight: MLXArray  // [O, kh, kw, I]
        let bias: MLXArray
        init(_ s: WeightStore, _ p: String) throws {
            weight = try s.get(p + ".weight").asType(.float32).transposed(0, 2, 3, 1)
            bias = try s.get(p + ".bias").asType(.float32)
        }
        /// Stride 1, "same" padding.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let k = weight.dim(1)
            guard k > 1 else { return conv2d(x, weight) + bias }
            let p = k / 2
            let padded = MLX.padded(x, widths: [.init(0), .init(p), .init(p), .init(0)])
            return bandedConv2d(padded, weight) + bias
        }
    }

    struct Norm {
        let weight, bias: MLXArray
        let groups: Int
        init(_ s: WeightStore, _ p: String, groups: Int) throws {
            weight = try s.get(p + ".weight").asType(.float32)
            bias = try s.get(p + ".bias").asType(.float32)
            self.groups = groups
        }
        /// GroupNorm (eps 1e-6) over `[N, H, W, C]`.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let g = x.reshaped(n, h * w, groups, c / groups)
            let mean = g.mean(axes: [1, 3], keepDims: true)
            let v = ((g - mean) * (g - mean)).mean(axes: [1, 3], keepDims: true)
            return ((g - mean) * rsqrt(v + 1e-6)).reshaped(n, h, w, c) * weight + bias
        }
    }

    struct ResBlock {
        let norm1, norm2: Norm
        let conv1, conv2: Conv
        let shortcut: Conv?
        init(_ s: WeightStore, _ p: String, groups: Int) throws {
            norm1 = try Norm(s, p + ".norm1", groups: groups)
            norm2 = try Norm(s, p + ".norm2", groups: groups)
            conv1 = try Conv(s, p + ".conv1")
            conv2 = try Conv(s, p + ".conv2")
            shortcut = s.has(p + ".conv_shortcut.weight") ? try Conv(s, p + ".conv_shortcut") : nil
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = conv1(siluAct(norm1(x)))
            eval(h)
            h = conv2(siluAct(norm2(h)))
            let out = (shortcut.map { $0(x) } ?? x) + h
            eval(out)
            return out
        }
    }

    struct Attention {
        let norm: Norm
        let q, k, v, o: Lin
        init(_ s: WeightStore, _ p: String, groups: Int) throws {
            norm = try Norm(s, p + ".group_norm", groups: groups)
            q = try s.linear(p + ".to_q", dtype: .float32)
            k = try s.linear(p + ".to_k", dtype: .float32)
            v = try s.linear(p + ".to_v", dtype: .float32)
            o = try s.linear(p + ".to_out.0", dtype: .float32)
        }
        /// Single-head attention over all positions, plus the residual.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let t = norm(x).reshaped(n, 1, h * w, c)
            let a = MLXFast.scaledDotProductAttention(queries: q(t), keys: k(t), values: v(t),
                                                      scale: 1 / Float(c).squareRoot(), mask: .none)
            let out = x + o(a).reshaped(n, h, w, c)
            eval(out)
            return out
        }
    }

    public let config: Config
    // decoder
    let decConvIn, decConvOut: Conv
    let decMid: (ResBlock, Attention, ResBlock)
    let upBlocks: [(resnets: [ResBlock], upsample: Conv?)]
    let decNormOut: Norm
    // encoder (optional)
    let encConvIn, encConvOut: Conv?
    let encMid: (ResBlock, Attention, ResBlock)?
    let downBlocks: [(resnets: [ResBlock], downsample: MLXArray?, downsampleBias: MLXArray?)]
    let encNormOut: Norm?

    public init(store s: WeightStore, config: Config = Config(), encoder: Bool = true) throws {
        self.config = config
        let g = config.groups
        decConvIn = try Conv(s, "decoder.conv_in")
        decMid = (try ResBlock(s, "decoder.mid_block.resnets.0", groups: g),
                  try Attention(s, "decoder.mid_block.attentions.0", groups: g),
                  try ResBlock(s, "decoder.mid_block.resnets.1", groups: g))
        upBlocks = try (0 ..< config.blockOutChannels.count).map { i in
            let p = "decoder.up_blocks.\(i)"
            let res = try (0 ... config.layersPerBlock).map { j in try ResBlock(s, "\(p).resnets.\(j)", groups: g) }
            let up = s.has("\(p).upsamplers.0.conv.weight") ? try Conv(s, "\(p).upsamplers.0.conv") : nil
            return (res, up)
        }
        decNormOut = try Norm(s, "decoder.conv_norm_out", groups: g)
        decConvOut = try Conv(s, "decoder.conv_out")
        if encoder {
            encConvIn = try Conv(s, "encoder.conv_in")
            downBlocks = try (0 ..< config.blockOutChannels.count).map { i in
                let p = "encoder.down_blocks.\(i)"
                let res = try (0 ..< config.layersPerBlock).map { j in try ResBlock(s, "\(p).resnets.\(j)", groups: g) }
                let dw = s.optional("\(p).downsamplers.0.conv.weight")?.asType(.float32).transposed(0, 2, 3, 1)
                let db = s.optional("\(p).downsamplers.0.conv.bias")?.asType(.float32)
                return (res, dw, db)
            }
            encMid = (try ResBlock(s, "encoder.mid_block.resnets.0", groups: g),
                      try Attention(s, "encoder.mid_block.attentions.0", groups: g),
                      try ResBlock(s, "encoder.mid_block.resnets.1", groups: g))
            encNormOut = try Norm(s, "encoder.conv_norm_out", groups: g)
            encConvOut = try Conv(s, "encoder.conv_out")
        } else {
            encConvIn = nil; encConvOut = nil; encMid = nil; downBlocks = []; encNormOut = nil
        }
    }

    public convenience init(directory: URL, encoder: Bool = true) throws {
        let cfgURL = directory.appendingPathComponent("config.json")
        let cfg = FileManager.default.fileExists(atPath: cfgURL.path) ? try Config(url: cfgURL) : Config()
        try self.init(store: WeightStore(directory: directory), config: cfg, encoder: encoder)
    }

    public var hasEncoder: Bool { encConvIn != nil }

    /// Raw (unscaled) latents `[N, h, w, 16]` → images `[N, 8h, 8w, 3]` in `[−1, 1]` (not clipped).
    public func decode(_ z: MLXArray) -> MLXArray {
        var x = decConvIn(z.asType(.float32))
        x = decMid.2(decMid.1(decMid.0(x)))
        for (res, up) in upBlocks {
            for r in res { x = r(x) }
            if let up {
                let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
                x = broadcast(x.reshaped(n, h, 1, w, 1, c), to: [n, h, 2, w, 2, c]).reshaped(n, 2 * h, 2 * w, c)
                x = up(x)
                eval(x)
            }
        }
        return decConvOut(siluAct(decNormOut(x)))
    }

    /// Images `[N, H, W, 3]` in `[−1, 1]` → the latent distribution's mean `[N, H/8, W/8, 16]` (unscaled).
    public func encode(_ image: MLXArray) -> MLXArray {
        guard let encConvIn, let encMid, let encNormOut, let encConvOut else {
            preconditionFailure("FluxVAE loaded without its encoder")
        }
        var x = encConvIn(image.asType(.float32))
        for (res, dw, db) in downBlocks {
            for r in res { x = r(x) }
            if let dw, let db {
                // pad (0, 1, 0, 1), then a 3×3 stride-2 convolution
                let p = MLX.padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
                x = conv2d(p, dw, stride: .init(2)) + db
                eval(x)
            }
        }
        x = encMid.2(encMid.1(encMid.0(x)))
        let moments = encConvOut(siluAct(encNormOut(x)))
        return moments[.ellipsis, ..<config.latentChannels]
    }

    /// Model-space latents (`(z − shift) · scale`) from raw ones, and back.
    public func normalize(_ z: MLXArray) -> MLXArray { (z - config.shiftFactor) * config.scalingFactor }
    public func denormalize(_ z: MLXArray) -> MLXArray { z / config.scalingFactor + config.shiftFactor }
}
