import Foundation
import Testing
@testable import MLXDiffusionCore

/// A folder source that fails after `limit` bytes of its first stream.
final class FlakySource: ByteSource, @unchecked Sendable {
    let base: FolderByteSource
    let limit: Int
    var failed = false
    init(base: FolderByteSource, limit: Int) { self.base = base; self.limit = limit }

    func list() async throws -> [RemoteFile] { try await base.list() }

    func open(_ file: RemoteFile, from offset: Int64) -> AsyncThrowingStream<Data, Error> {
        let inner = base.open(file, from: offset)
        let limit = limit
        let shouldFail = !failed
        failed = true
        return AsyncThrowingStream { cont in
            Task {
                var sent = 0
                do {
                    for try await d in inner {
                        if shouldFail && sent + d.count > limit {
                            cont.yield(d.prefix(limit - sent))
                            throw URLError(.networkConnectionLost)
                        }
                        cont.yield(d)
                        sent += d.count
                    }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }
}

func makeRepo() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("repo-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("transformer"), withIntermediateDirectories: true)
    var big = Data(count: 300_000)
    for i in 0..<big.count { big[i] = UInt8(truncatingIfNeeded: i &* 31 &+ 7) }
    try big.write(to: root.appendingPathComponent("transformer/model.safetensors"))
    try Data("{\"a\":1}".utf8).write(to: root.appendingPathComponent("config.json"))
    try Data("# readme".utf8).write(to: root.appendingPathComponent("README.md"))
    return root
}

@Test func parsesHFTreeAndSelectsByGlob() throws {
    let json = """
        [{"type":"directory","path":"vae"},
         {"type":"file","path":"vae/config.json","size":512},
         {"type":"file","path":"vae/model.safetensors","size":134,"lfs":{"oid":"abc123","size":644000000}},
         {"type":"file","path":"README.md","size":2000}]
        """
    let files = try HFAPI.parseTree(Data(json.utf8))
    #expect(files.count == 3)
    #expect(files[1] == RemoteFile(path: "vae/model.safetensors", size: 644_000_000, sha256: "abc123"))
    let picked = HFAPI.select(files, include: [], exclude: ["*.md"])
    #expect(picked.map(\.path) == ["vae/config.json", "vae/model.safetensors"])
    #expect(HFAPI.select(files, include: ["*.safetensors"], exclude: []).count == 1)
}

@Test func downloadsVerifiesAndSkipsFinishedFiles() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let source = FolderByteSource(root: repo)
    let files = HFAPI.select(try await source.list(), include: [], exclude: ["*.md"])
    var last = ModelDownloader.Progress(done: 0, total: 0, file: "")
    try await ModelDownloader.download(files, to: dest, from: source) { last = $0 }
    #expect(last.done == last.total && last.total == 300_000 + 7)
    #expect(try Data(contentsOf: dest.appendingPathComponent("transformer/model.safetensors"))
        == Data(contentsOf: repo.appendingPathComponent("transformer/model.safetensors")))
    #expect(!FileManager.default.fileExists(atPath: dest.appendingPathComponent("README.md").path))
    #expect(ModelDownloader.remainingBytes(files, in: dest) == 0)
    // A second run has nothing to fetch.
    var calls = 0
    try await ModelDownloader.download(files, to: dest, from: source) { _ in calls += 1 }
    #expect(calls == 0)
}

@Test func interruptedDownloadResumesFromThePart() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let base = FolderByteSource(root: repo)
    let files = HFAPI.select(try await base.list(), include: ["*.safetensors"], exclude: [])
    let flaky = FlakySource(base: base, limit: 100_000)
    await #expect(throws: URLError.self) { try await ModelDownloader.download(files, to: dest, from: flaky) }
    let part = dest.appendingPathComponent("transformer/model.safetensors.part")
    #expect(ModelDownloader.size(of: part) == 100_000)
    #expect(ModelDownloader.remainingBytes(files, in: dest) == 200_000)
    var first: Int64 = -1
    try await ModelDownloader.download(files, to: dest, from: flaky) { p in if first < 0 { first = p.done } }
    #expect(first > 100_000)  // resumed, not restarted
    #expect(try ModelDownloader.sha256(of: dest.appendingPathComponent("transformer/model.safetensors")) == files[0].sha256)
    #expect(!FileManager.default.fileExists(atPath: part.path))
}

@Test func corruptedPartFailsTheChecksumAndIsDiscarded() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let source = FolderByteSource(root: repo)
    let files = HFAPI.select(try await source.list(), include: ["*.safetensors"], exclude: [])
    let part = dest.appendingPathComponent("transformer/model.safetensors.part")
    try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(repeating: 0xFF, count: 1000).write(to: part)  // wrong bytes
    await #expect(throws: DownloadError.checksumMismatch("transformer/model.safetensors")) {
        try await ModelDownloader.download(files, to: dest, from: source)
    }
    #expect(!FileManager.default.fileExists(atPath: part.path))
    try await ModelDownloader.download(files, to: dest, from: source)  // clean retry succeeds
}

/// The LTX-2.3 entry fetches exactly what the pipeline loads from the 4-bit repo (not the other transformer
/// variants or the LoRAs), plus the whole Gemma repo into `text_encoder/`.
@Test func ltxCatalogEntrySelectsThePipelineFiles() throws {
    let entry = try #require(ModelCatalog.entries.first { $0.id == "ltx-2.3" })
    #expect(entry.parts.count == 2 && entry.parts[1].subfolder == "text_encoder")
    let repoFiles = [".gitattributes", "LICENSE", "README.md", "audio_vae.safetensors", "config.json",
                     "connector.safetensors", "embedded_config.json", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                     "ltx-2.3-22b-distilled-lora-384.safetensors", "quantize_config.json",
                     "spatial_upscaler_x1_5_v1_0.safetensors", "spatial_upscaler_x1_5_v1_0_config.json",
                     "spatial_upscaler_x2_v1_1.safetensors", "spatial_upscaler_x2_v1_1_config.json", "split_model.json",
                     "temporal_upscaler_x2_v1_0.safetensors", "temporal_upscaler_x2_v1_0_config.json",
                     "transformer-dev.safetensors", "transformer-distilled-1.1.safetensors",
                     "transformer-distilled.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
                     "vocoder.safetensors"].map { RemoteFile(path: $0, size: 1) }
    let picked = Set(HFAPI.select(repoFiles, include: entry.include, exclude: entry.exclude).map(\.path))
    let weights = picked.filter { $0.hasSuffix(".safetensors") }
    #expect(weights == ["transformer-distilled-1.1.safetensors", "connector.safetensors", "vae_decoder.safetensors",
                        "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
                        "spatial_upscaler_x2_v1_1.safetensors"])
    #expect(picked.isSuperset(of: ["config.json", "embedded_config.json", "quantize_config.json",
                                   "spatial_upscaler_x2_v1_1_config.json", "LICENSE"]))
    #expect(entry.manifest?["class"] == "LTX2Pipeline")
    let gemma = HFAPI.select([".gitattributes", "config.json", "model-00001-of-00002.safetensors", "tokenizer.json"]
        .map { RemoteFile(path: $0, size: 1) }, include: entry.parts[1].include, exclude: entry.parts[1].exclude)
    #expect(gemma.count == 3)
}

/// The ACE-Step entry fetches the turbo DiT, VAE, text encoder and 1.7B planner, but not the repo's Python model code
/// or the duplicate top-level config.
@Test func aceStepCatalogEntrySelectsThePipelineFiles() throws {
    let entry = try #require(ModelCatalog.entries.first { $0.id == "ace-step-1.5" })
    #expect(entry.supported && entry.kind == .music && entry.manifest?["class"] == "ACEStepPipeline")
    let repoFiles = [".gitattributes", "README.md", "config.json",
                     "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/model.safetensors",
                     "Qwen3-Embedding-0.6B/tokenizer.json", "acestep-5Hz-lm-1.7B/added_tokens.json",
                     "acestep-5Hz-lm-1.7B/model.safetensors", "acestep-5Hz-lm-1.7B/tokenizer.json",
                     "acestep-v15-turbo/config.json", "acestep-v15-turbo/configuration_acestep_v15.py",
                     "acestep-v15-turbo/model.safetensors", "acestep-v15-turbo/modeling_acestep_v15_turbo.py",
                     "acestep-v15-turbo/silence_latent.pt", "vae/config.json",
                     "vae/diffusion_pytorch_model.safetensors"].map { RemoteFile(path: $0, size: 1) }
    let picked = Set(HFAPI.select(repoFiles, include: entry.include, exclude: entry.exclude).map(\.path))
    #expect(picked == ["README.md", "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/model.safetensors",
                       "Qwen3-Embedding-0.6B/tokenizer.json", "acestep-5Hz-lm-1.7B/added_tokens.json",
                       "acestep-5Hz-lm-1.7B/model.safetensors", "acestep-5Hz-lm-1.7B/tokenizer.json",
                       "acestep-v15-turbo/config.json", "acestep-v15-turbo/model.safetensors",
                       "acestep-v15-turbo/silence_latent.pt", "vae/config.json",
                       "vae/diffusion_pytorch_model.safetensors"])
}

/// The Kokoro entry fetches the model and the English (American and British) voices only.
@Test func kokoroCatalogEntrySelectsEnglishVoices() throws {
    let entry = try #require(ModelCatalog.entries.first { $0.id == "kokoro-82m" })
    #expect(entry.supported && entry.kind == .speech && entry.manifest?["class"] == "KokoroPipeline")
    let repoFiles = [".gitattributes", "README.md", "VOICES.md", "config.json", "kokoro-v1_0.pth", "samples/af_heart_0.wav",
                     "eval/TTS_Arena-2025-02-26.jpeg", "voices/af_heart.pt", "voices/am_adam.pt", "voices/bf_emma.pt",
                     "voices/bm_george.pt", "voices/jf_alpha.pt", "voices/zf_xiaobei.pt", "voices/ef_dora.pt"]
        .map { RemoteFile(path: $0, size: 1) }
    let picked = Set(HFAPI.select(repoFiles, include: entry.include, exclude: entry.exclude).map(\.path))
    #expect(picked == ["README.md", "VOICES.md", "config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/am_adam.pt",
                       "voices/bf_emma.pt", "voices/bm_george.pt"])
}
