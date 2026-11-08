import Foundation
import MLX
import MLXDiffusionCore
import StableAudio

// Verification probe for ACE-Step's Oobleck VAE (diffusers layout) and silence latent against ace_ref.py dumps.
// usage: ace-probe-vae <dump-dir>

let work = URL(fileURLWithPath: CommandLine.arguments[1])
let models = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/MLXComfyUI/models/ace-step-1.5")

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    guard a.shape == b.shape else {
        print("  \(name) SHAPE MISMATCH swift \(a.shape) ref \(b.shape)")
        return
    }
    let x = a.asType(.float32), y = b.asType(.float32)
    let d = abs(x - y)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    print(String(format: "  %-14@ %@  maxabs %.4g (ref max %.3g)  rel-L2 %.3g", name as NSString, "\(a.shape)" as NSString,
                 d.max().item(Float.self), abs(y).max().item(Float.self), rel))
}

let dump = try loadArrays(url: work.appendingPathComponent("dump.safetensors"))

let silenceURL = models.appendingPathComponent("acestep-v15-turbo/silence_latent.pt")
let n = try TorchArchive.float32Count(url: silenceURL)
let silence = try TorchArchive.float32(url: silenceURL, shape: [1, 64, n / 64]).transposed(0, 2, 1)
print("silence latent")
report("silence", silence, dump["silence_latent"]!)

let store = try WeightStore(url: models.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
for dt in [DType.float32, .bfloat16] {
    let dec = try OobleckDecoder(store: store, prefix: "decoder.", strides: [2, 4, 4, 6, 10], dtype: dt, layout: .diffusers)
    print("oobleck \(dt), hop \(dec.hopLength)")
    var t = Date()
    let out0 = dec(dump["vae_in0"]!.transposed(0, 2, 1))
    eval(out0)
    print(String(format: "  chunk0 %.2f s", Date().timeIntervalSince(t)))
    report("chunk0", out0, dump["vae_out0"]!.transposed(0, 2, 1))
    t = Date()
    let lat = dump["gen_target_latents"]!
    let full = dec.decodeTiled(lat, chunk: 256, overlap: 64)
    eval(full)
    print(String(format: "  tiled 20 s %.2f s, peak %.2f GB", Date().timeIntervalSince(t),
                 Double(Memory.peakMemory) / 1e9))
    let ref = dump["audio"]!  // [2, N], after the pipeline's peak normalization
    let mine = full[0].transposed(1, 0)
    let scale = (abs(ref).max() / abs(mine).max()).item(Float.self)
    print(String(format: "  output scale ref/swift %.4f", scale))
    report("tiled·scale", mine * scale, ref)
}
