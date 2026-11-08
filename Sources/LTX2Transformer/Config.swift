import Foundation

/// Hyperparameters of the LTX-2.3 audio-video DiT, mirroring the reference `LTXModelConfig.from_checkpoint_config`.
public struct LTXTransformerConfig {
    public var numLayers = 48
    public var videoDim = 4096
    public var audioDim = 2048
    public var videoHeads = 32
    public var audioHeads = 32
    public var videoHeadDim = 128
    public var audioHeadDim = 64
    public var avCrossHeads = 32
    public var avCrossHeadDim = 64
    public var videoInChannels = 128
    public var audioInChannels = 128
    public var timestepEmbeddingDim = 256
    public var timestepScaleMultiplier: Double = 1000
    /// The reference dataclass default is 1; every LTX-2.3 checkpoint config supplies 1000.
    public var avCaTimestepScaleMultiplier: Double = 1
    public var ropeTheta: Double = 10000
    public var maxPos: [Int] = [20, 2048, 2048]
    public var audioMaxPos: [Int] = [20]
    public var normEps: Float = 1e-6
    /// `frequencies_precision == "float64"`: build the RoPE frequency grid in Double.
    public var doublePrecisionRope = false

    public init() {}

    /// Reads `embedded_config.json` (its `"transformer"` section), falling back to `config.json`.
    public init(directory: URL) throws {
        let fm = FileManager.default
        var dict: [String: Any]?
        for name in ["embedded_config.json", "config.json"] {
            let url = directory.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else { continue }
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            dict = (o["transformer"] as? [String: Any]) ?? o
            break
        }
        self.init()
        guard let t = dict else { return }
        func int(_ k: String, _ d: Int) -> Int { (t[k] as? NSNumber)?.intValue ?? d }
        func dbl(_ k: String, _ d: Double) -> Double { (t[k] as? NSNumber)?.doubleValue ?? d }
        func ints(_ k: String, _ d: [Int]) -> [Int] { (t[k] as? [NSNumber])?.map(\.intValue) ?? d }
        numLayers = int("num_layers", numLayers)
        videoDim = int("cross_attention_dim", videoDim)
        audioDim = int("audio_cross_attention_dim", audioDim)
        videoHeads = int("num_attention_heads", videoHeads)
        audioHeads = int("audio_num_attention_heads", audioHeads)
        videoHeadDim = int("attention_head_dim", videoHeadDim)
        audioHeadDim = int("audio_attention_head_dim", audioHeadDim)
        avCrossHeads = int("audio_num_attention_heads", avCrossHeads)
        avCrossHeadDim = int("audio_attention_head_dim", avCrossHeadDim)
        videoInChannels = int("in_channels", videoInChannels)
        audioInChannels = int("audio_in_channels", audioInChannels)
        timestepScaleMultiplier = dbl("timestep_scale_multiplier", timestepScaleMultiplier)
        avCaTimestepScaleMultiplier = dbl("av_ca_timestep_scale_multiplier", avCaTimestepScaleMultiplier)
        ropeTheta = dbl("positional_embedding_theta", ropeTheta)
        maxPos = ints("positional_embedding_max_pos", maxPos)
        audioMaxPos = ints("audio_positional_embedding_max_pos", audioMaxPos)
        normEps = Float(dbl("norm_eps", Double(normEps)))
        doublePrecisionRope = (t["frequencies_precision"] as? String) == "float64"
        let ropeType = (t["rope_type"] as? String ?? "split").lowercased()
        precondition(ropeType == "split", "only SPLIT rope is supported (got \(ropeType))")
        precondition(!(t["share_ff"] as? Bool ?? false), "share_ff is not supported")
    }
}
