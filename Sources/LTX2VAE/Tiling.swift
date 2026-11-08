import Foundation
import MLX

// Port of `ltx_core_mlx/model/video_vae/tiling.py` (decode side) plus the automatic tile ladder of `video_vae.py`
// (`_compute_decode_tiling`). Latent tiles overlap; decoded tiles are blended with trapezoidal masks.

/// Spatial tile size and overlap in output pixels (both multiples of 32; tile ≥ 64; overlap < tile).
public struct LTXSpatialTiling: Equatable, Sendable {
    public var tileSizeInPixels: Int
    public var tileOverlapInPixels: Int

    public init(tileSizeInPixels: Int, tileOverlapInPixels: Int = 0) {
        precondition(tileSizeInPixels >= 64 && tileSizeInPixels % 32 == 0, "tile size must be >= 64 and divisible by 32")
        precondition(tileOverlapInPixels % 32 == 0 && tileOverlapInPixels < tileSizeInPixels,
                     "overlap must be divisible by 32 and smaller than the tile")
        self.tileSizeInPixels = tileSizeInPixels
        self.tileOverlapInPixels = tileOverlapInPixels
    }
}

/// Temporal tile size and overlap in output frames (both multiples of 8; tile ≥ 16; overlap < tile).
public struct LTXTemporalTiling: Equatable, Sendable {
    public var tileSizeInFrames: Int
    public var tileOverlapInFrames: Int

    public init(tileSizeInFrames: Int, tileOverlapInFrames: Int = 0) {
        precondition(tileSizeInFrames >= 16 && tileSizeInFrames % 8 == 0, "tile size must be >= 16 and divisible by 8")
        precondition(tileOverlapInFrames % 8 == 0 && tileOverlapInFrames < tileSizeInFrames,
                     "overlap must be divisible by 8 and smaller than the tile")
        self.tileSizeInFrames = tileSizeInFrames
        self.tileOverlapInFrames = tileOverlapInFrames
    }
}

/// Combined spatial + temporal tiling for the video VAE decoder (`TilingConfig`).
public struct LTXVAETiling: Equatable, Sendable {
    public var spatial: LTXSpatialTiling?
    public var temporal: LTXTemporalTiling?

    public init(spatial: LTXSpatialTiling? = nil, temporal: LTXTemporalTiling? = nil) {
        self.spatial = spatial
        self.temporal = temporal
    }

    /// Upstream `TilingConfig.default()`: 768 px tiles with 64 px overlap, 80-frame tiles with 24 frames overlap.
    public static let `default` = LTXVAETiling(
        spatial: LTXSpatialTiling(tileSizeInPixels: 768, tileOverlapInPixels: 64),
        temporal: LTXTemporalTiling(tileSizeInFrames: 80, tileOverlapInFrames: 24))

    /// Short summary such as `frames=40/8 px=512/32`.
    public var summary: String {
        var parts: [String] = []
        if let t = temporal { parts.append("frames=\(t.tileSizeInFrames)/\(t.tileOverlapInFrames)") }
        if let s = spatial { parts.append("px=\(s.tileSizeInPixels)/\(s.tileOverlapInPixels)") }
        return parts.isEmpty ? "untiled" : parts.joined(separator: " ")
    }

    // MARK: automatic choice (reference `_compute_decode_tiling`)

    /// Peak activation bytes per output pixel-frame used by the reference's budget estimate.
    public static let bytesPerPixelFrame = 750
    static let accumulatorBytesPerPixel = 4 * 3 * 4
    static let spatialLadder: [(Int, Int)] = [(768, 64), (512, 32), (256, 32)]

    /// Reference decode budget: half of unified memory (`LTX2_VAE_DECODE_BUDGET_GB` is honoured too).
    public static var defaultBudgetBytes: Int {
        if let s = ProcessInfo.processInfo.environment["LTX2_VAE_DECODE_BUDGET_GB"], let gb = Double(s) {
            return Int(gb * 1024 * 1024 * 1024)
        }
        return Int(ProcessInfo.processInfo.physicalMemory / 2)
    }

    static func spatialTilePixels(_ latentAxis: Int, longSide: Int, _ cfg: LTXSpatialTiling) -> Int {
        let tileLat = cfg.tileSizeInPixels / 32
        let overlapLat = cfg.tileOverlapInPixels / 32
        let adjusted = max(max(2, overlapLat + 1), pyRound(Double(tileLat * latentAxis) / Double(longSide)))
        return min(latentAxis, adjusted) * 32
    }

    /// Reference estimate of the decode peak for a latent of `frames × height × width` cells.
    public static func estimatePeakBytes(frames f: Int, height h: Int, width w: Int, tiling: LTXVAETiling?) -> Int {
        let fpx = 8 * f - 7, hpx = 32 * h, wpx = 32 * w
        guard let tiling else { return bytesPerPixelFrame * fpx * hpx * wpx }
        var tf = fpx, th = hpx, tw = wpx
        if let t = tiling.temporal { tf = min(fpx, t.tileSizeInFrames) }
        if let s = tiling.spatial {
            let long = max(h, w)
            th = spatialTilePixels(h, longSide: long, s)
            tw = spatialTilePixels(w, longSide: long, s)
        }
        return bytesPerPixelFrame * tf * th * tw + tf * hpx * wpx * accumulatorBytesPerPixel
    }

    static func temporalTile(_ frames: Int, frameRate: Double) -> LTXTemporalTiling {
        let oneSecond = max(8, (Int(frameRate) / 8) * 8)
        let overlap = min(oneSecond, (Int(Double(frames) * 0.3) / 8) * 8)
        return LTXTemporalTiling(tileSizeInFrames: frames, tileOverlapInFrames: overlap)
    }

    /// The tiling the reference picks for a latent of `frames × height × width` cells: `nil` (untiled) when the
    /// estimate fits `budgetBytes`, else the first rung that fits of temporal tiles 80→40 frames, then spatial tiles
    /// 768→512→256 px at 40 frames, then temporal tiles down to 16 frames at 256 px (the smallest rung if none fits).
    public static func auto(frames f: Int, height h: Int, width w: Int, frameRate: Double = 24,
                            budgetBytes: Int? = nil) -> LTXVAETiling? {
        let budget = budgetBytes ?? defaultBudgetBytes
        if estimatePeakBytes(frames: f, height: h, width: w, tiling: nil) <= budget { return nil }
        let fpx = 8 * f - 7
        let temporalSizes = stride(from: 80, through: 16, by: -8).filter { $0 < fpx }
        let preferred = temporalSizes.filter { $0 >= 40 }
        let smallest = temporalSizes.filter { $0 < 40 }
        var candidates = preferred.map { LTXVAETiling(temporal: temporalTile($0, frameRate: frameRate)) }
        let baseTemporal = preferred.last.map { temporalTile($0, frameRate: frameRate) }
        for (px, ov) in spatialLadder {
            candidates.append(LTXVAETiling(spatial: LTXSpatialTiling(tileSizeInPixels: px, tileOverlapInPixels: ov),
                                           temporal: baseTemporal))
        }
        let (lastPx, lastOv) = spatialLadder.last!
        for n in smallest {
            candidates.append(LTXVAETiling(spatial: LTXSpatialTiling(tileSizeInPixels: lastPx, tileOverlapInPixels: lastOv),
                                           temporal: temporalTile(n, frameRate: frameRate)))
        }
        for c in candidates where estimatePeakBytes(frames: f, height: h, width: w, tiling: c) <= budget { return c }
        return candidates.last
    }
}

/// Python's `round()` (banker's rounding) for the tile-size arithmetic.
@inline(__always) func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

// MARK: - intervals and tiles

struct LTXIntervals {
    var starts: [Int]
    var ends: [Int]
    var leftRamps: [Int]
    var rightRamps: [Int]

    static func whole(_ n: Int) -> LTXIntervals { .init(starts: [0], ends: [n], leftRamps: [0], rightRamps: [0]) }

    /// `split_with_symmetric_overlaps`.
    static func symmetric(_ n: Int, size: Int, overlap: Int) -> LTXIntervals {
        if n <= size { return whole(n) }
        let amount = (n + size - 2 * overlap - 1) / (size - overlap)
        let starts = (0..<amount).map { $0 * (size - overlap) }
        var ends = starts.map { $0 + size }
        ends[amount - 1] = n
        return .init(starts: starts, ends: ends, leftRamps: [0] + Array(repeating: overlap, count: amount - 1),
                     rightRamps: Array(repeating: overlap, count: amount - 1) + [0])
    }

    /// `split_temporal_latents`: non-first tiles start one latent frame earlier with a one-longer left ramp.
    static func temporalLatents(_ n: Int, size: Int, overlap: Int) -> LTXIntervals {
        if n <= size { return whole(n) }
        var iv = symmetric(n, size: size, overlap: overlap)
        for i in 1..<iv.starts.count {
            iv.starts[i] -= 1
            iv.leftRamps[i] += 1
        }
        return iv
    }
}

/// `compute_trapezoidal_mask_1d` (float32).
func ltxTrapezoidalMask(_ length: Int, rampLeft: Int, rampRight: Int, leftStartsFrom0: Bool) -> MLXArray {
    precondition(length > 0)
    let rl = max(0, min(rampLeft, length))
    let rr = max(0, min(rampRight, length))
    var mask = MLXArray.ones([length], dtype: .float32)
    if rl > 0 {
        let n = leftStartsFrom0 ? rl + 1 : rl + 2
        var fadeIn = MLXArray.linspace(Float(0), Float(1), count: n)[0 ..< (n - 1)]
        if !leftStartsFrom0 { fadeIn = fadeIn[1...] }
        mask = concatenated([mask[0 ..< rl] * fadeIn, mask[rl...]])
    }
    if rr > 0 {
        let fadeOut = MLXArray.linspace(Float(1), Float(0), count: rr + 2)[1 ..< (rr + 1)]
        mask = concatenated([mask[0 ..< (length - rr)], mask[(length - rr)...] * fadeOut])
    }
    return clip(mask, min: 0, max: 1)
}

/// One decode tile: latent ranges to cut, pixel ranges to place the result at, and 1-D blend masks
/// (`nil` = the axis is not tiled, weight 1).
struct LTXDecodeTile {
    var inF: Range<Int>, inH: Range<Int>, inW: Range<Int>
    var outT: Range<Int>, outH: Range<Int>, outW: Range<Int>
    var maskT: MLXArray?, maskH: MLXArray?, maskW: MLXArray?

    /// Blend mask broadcastable to `[1, t, h, w, 1]` (same product order as the reference).
    var blendMask: MLXArray {
        let one = MLXArray.ones([1], dtype: .float32)
        let t = (maskT ?? one).reshaped(1, -1, 1, 1, 1)
        let h = (maskH ?? one).reshaped(1, 1, -1, 1, 1)
        let w = (maskW ?? one).reshaped(1, 1, 1, -1, 1)
        return t * h * w
    }
}

/// `prepare_tiles_for_decoding` for a latent of `f × h × w` cells, ordered frames-outer like the reference.
func ltxDecodeTiles(frames f: Int, height h: Int, width w: Int, tiling: LTXVAETiling?) -> [LTXDecodeTile] {
    let fpx = 8 * f - 7
    // per axis: input ranges, output ranges, masks
    typealias Axis = (ins: [Range<Int>], outs: [Range<Int>], masks: [MLXArray?])

    func untiled(_ n: Int, out: Int) -> Axis { ([0 ..< n], [0 ..< out], [nil]) }

    func spatialAxis(_ n: Int, long: Int, _ cfg: LTXSpatialTiling) -> Axis {
        let tile = cfg.tileSizeInPixels / 32
        let overlap = cfg.tileOverlapInPixels / 32
        let adjusted = max(max(2, overlap + 1), pyRound(Double(tile * n) / Double(long)))
        let iv = LTXIntervals.symmetric(n, size: adjusted, overlap: overlap)
        var a: Axis = ([], [], [])
        for i in iv.starts.indices {
            let (s, e) = (iv.starts[i], iv.ends[i])
            a.ins.append(s ..< e)
            a.outs.append(s * 32 ..< e * 32)
            a.masks.append(ltxTrapezoidalMask((e - s) * 32, rampLeft: iv.leftRamps[i] * 32,
                                              rampRight: iv.rightRamps[i] * 32, leftStartsFrom0: false))
        }
        return a
    }

    var axH = untiled(h, out: h * 32)
    var axW = untiled(w, out: w * 32)
    var axF = untiled(f, out: fpx)
    if let s = tiling?.spatial {
        let long = max(h, w)
        axH = spatialAxis(h, long: long, s)
        axW = spatialAxis(w, long: long, s)
    }
    if let t = tiling?.temporal {
        let iv = LTXIntervals.temporalLatents(f, size: t.tileSizeInFrames / 8, overlap: t.tileOverlapInFrames / 8)
        var a: Axis = ([], [], [])
        for i in iv.starts.indices {
            let (b, e) = (iv.starts[i], iv.ends[i])
            let start = b * 8, stop = 1 + (e - 1) * 8
            let lr = iv.leftRamps[i] == 0 ? 0 : 1 + (iv.leftRamps[i] - 1) * 8
            a.ins.append(b ..< e)
            a.outs.append(start ..< stop)
            a.masks.append(ltxTrapezoidalMask(stop - start, rampLeft: lr, rampRight: iv.rightRamps[i] * 8,
                                              leftStartsFrom0: true))
        }
        axF = a
    }
    var tiles: [LTXDecodeTile] = []
    for i in axF.ins.indices {
        for j in axH.ins.indices {
            for k in axW.ins.indices {
                tiles.append(LTXDecodeTile(
                    inF: axF.ins[i], inH: axH.ins[j], inW: axW.ins[k],
                    outT: axF.outs[i], outH: axH.outs[j], outW: axW.outs[k],
                    maskT: axF.masks[i], maskH: axH.masks[j], maskW: axW.masks[k]))
            }
        }
    }
    return tiles
}
