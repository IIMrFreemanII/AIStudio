import Flux
import Foundation
import MLX
import MLXDiffusionCore
@testable import ModelImport
import Testing

/// Opt-in checks on the installed models (never modified: imports go to a temporary folder). Run with
/// `TEST_RUNNER_MLXCOMFY_REAL_MODELS=1 xcodebuild test … -only-testing:ModelImportTests/RealModelTests`;
/// `TEST_RUNNER_MLXCOMFY_REAL_OUT=<dir>` keeps the generated images.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXCOMFY_REAL_MODELS"] != nil))
struct RealModelTests {
    var models: URL { ModelPaths.defaultDirectory }

    /// Re-imports FLUX.1 dev's own 24 GB diffusers transformer as a "fine-tune" of itself: the result must equal
    /// the app's 8-bit cache tensor for tensor, memory must stay far below the model's size, and the imported
    /// folder must generate exactly the base's image for the same seed.
    @Test func fluxDevReimportMatchesTheCacheAndTheBase() async throws {
        let base = models.appendingPathComponent("flux-1-dev")
        let cache = FluxPipeline.transformerCacheURL(base)
        try #require(FileManager.default.fileExists(atPath: cache.path), "build the base's q8 cache first")
        let tmp = TempDir("real-flux")
        let dest = tmp.url.appendingPathComponent("flux-1-dev-reimport")

        Memory.peakMemory = 0
        let t0 = Date()
        let r = try ModelImporter.importTransformer(from: base.appendingPathComponent("transformer"), base: base,
                                                    into: dest)
        let peak = Memory.peakMemory
        print(String(format: "import: %.0f s, %.2f GB written, MLX peak %.2f GB", Date().timeIntervalSince(t0),
                     Double(r.bytesWritten) / 1e9, Double(peak) / 1e9))
        #expect(peak < 4_000_000_000, "conversion should not hold the model in memory")

        let mine = try loadArrays(url: dest.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        let theirs = try loadArrays(url: cache)
        #expect(Set(mine.keys) == Set(theirs.keys))
        var different: [String] = []
        for k in mine.keys.sorted() where theirs[k] != nil {
            let a = mine[k]!, b = theirs[k]!
            if a.shape != b.shape || a.dtype != b.dtype || !arrayEqual(a, b).item(Bool.self) { different.append(k) }
            Memory.clearCache()
        }
        #expect(different.isEmpty, "differs from the cache: \(different.prefix(5))")

        // The assembled folder loads in the real pipeline and generates the base's image.
        func image(_ dir: URL) async throws -> MLXArray {
            let p = try await FluxPipeline(directory: dir)
            let img = try p.generate(prompt: "a red fox sitting in fresh snow, golden hour", width: 512, height: 512,
                                     seed: 7, steps: 4)
            eval(img)
            return img
        }
        let a = try await image(dest)
        Memory.clearCache()
        let b = try await image(base)
        #expect(!FileManager.default.fileExists(atPath: dest.appendingPathComponent(".mlxcomfy-cache").path))
        #expect(maxDiff(a, b) == 0)
        if let out = ProcessInfo.processInfo.environment["MLXCOMFY_REAL_OUT"] {
            try ImageArray.savePNG(a, to: URL(fileURLWithPath: out).appendingPathComponent("flux-reimport.png"))
        }
    }
}
