import CryptoKit
import Foundation
import GraphRuntime
import MLX
import MLXComfyProtocol
import MLXDiffusionCore
@testable import NodesRemote
@testable import NodesStandard
import Testing

/// A small valid safetensors file (header length, JSON header, data).
func safetensorsData(bytes: Int = 200_000, seed: UInt8 = 1) -> Data {
    let header = Data("{\"w\":{\"dtype\":\"U8\",\"shape\":[\(bytes)],\"data_offsets\":[0,\(bytes)]}}".utf8)
    var d = Data()
    var n = UInt64(header.count).littleEndian
    d.append(Data(bytes: &n, count: 8))
    d.append(header)
    d.append(Data((0..<bytes).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) }))
    return d
}

func sha256Hex(_ d: Data) -> String { FileHasher.hex(SHA256.hash(data: d)) }

// MARK: - no network

@Suite struct LoRALinkParsingTests {
    @Test func civitaiAndHuggingFaceLinkForms() throws {
        #expect(try LoRASources.parse("https://civitai.com/models/200255/hands-xl?modelVersionId=804967") == .civitaiVersion(804967))
        #expect(try LoRASources.parse("https://civitai.com/models/200255") == .civitaiModel(200255))
        #expect(try LoRASources.parse("https://www.civitai.com/models/200255/some-slug") == .civitaiModel(200255))
        #expect(try LoRASources.parse("https://civitai.com/api/download/models/804967?type=Model") == .civitaiVersion(804967))
        #expect(try LoRASources.parse(" https://huggingface.co/XLabs-AI/flux-lora-collection/blob/main/anime_lora.safetensors ")
                == .huggingFaceFile(repo: "XLabs-AI/flux-lora-collection", revision: "main", path: "anime_lora.safetensors"))
        #expect(try LoRASources.parse("https://hf.co/a/b/resolve/abc123/sub/dir/x.safetensors?download=true")
                == .huggingFaceFile(repo: "a/b", revision: "abc123", path: "sub/dir/x.safetensors"))
        #expect(try LoRASources.parse("https://huggingface.co/a/b") == .huggingFaceRepo(repo: "a/b", revision: "main"))
        #expect(try LoRASources.parse("https://huggingface.co/a/b/tree/v2") == .huggingFaceRepo(repo: "a/b", revision: "v2"))
        for bad in ["https://huggingface.co/datasets/a/b", "https://example.com/x.safetensors", "civitai 123",
                    "https://civitai.com/images/5"] {
            #expect(throws: LoRASourceError.self) { try LoRASources.parse(bad) }
        }
    }

    @Test func civitaiVersionPicksThePrimarySafetensorsFile() throws {
        let json = """
        {"id": 804967, "name": "v2", "baseModel": "Flux.1 D", "trainedWords": ["Detailed hand", " ", "Perfect hand"],
         "model": {"name": "Hands"},
         "files": [
           {"name": "training.zip", "type": "Training Data", "sizeKB": 10, "downloadUrl": "https://civitai.com/api/download/models/804967?type=Training%20Data"},
           {"name": "Hand v2 fp8.safetensors", "type": "Model", "primary": false, "sizeKB": 100, "hashes": {"SHA256": "AA"}, "downloadUrl": "https://civitai.com/api/download/models/804967?format=fp8"},
           {"name": "Hand v2.safetensors", "type": "Model", "primary": true, "sizeKB": 335747.4375,
            "hashes": {"SHA256": "82B169C950ABCDEF"}, "downloadUrl": "https://civitai.com/api/download/models/804967"}
         ]}
        """
        let r = try LoRASources.civitaiRemote(Data(json.utf8))
        #expect(r.fileName == "Hand v2.safetensors" && r.sha256 == "82b169c950abcdef")
        #expect(r.bytes == Int64((335747.4375 * 1024).rounded()) && r.baseModel == "Flux.1 D")
        #expect(r.triggerWords == ["Detailed hand", "Perfect hand"] && r.title == "Hands · v2")
        #expect(r.downloadURL.absoluteString == "https://civitai.com/api/download/models/804967")
        #expect(r.cacheName == "civitai-804967-Hand_v2.safetensors")
        #expect(r.kind == .lora && LoRASources.family(forBaseModel: r.baseModel!) == "flux.1")
    }

    @Test func civitaiCheckpointsPickSafetensorsInTheBestPrecision() throws {
        func version(_ files: String) -> Data {
            Data("""
            {"id": 7, "baseModel": "Flux.1 D", "model": {"name": "Ultra", "type": "Checkpoint"}, "files": [\(files)]}
            """.utf8)
        }
        let ultra = try LoRASources.civitaiRemote(version("""
            {"name": "a_q4.gguf", "type": "Model", "metadata": {"format": "GGUF", "fp": "fp8"}, "downloadUrl": "https://civitai.com/a"},
            {"name": "b_fp8.safetensors", "type": "Model", "metadata": {"format": "SafeTensor", "fp": "fp8"}, "sizeKB": 1, "downloadUrl": "https://civitai.com/b"},
            {"name": "c.safetensors", "type": "Model", "primary": true, "metadata": {"format": "SafeTensor", "fp": "fp16"}, "sizeKB": 2, "downloadUrl": "https://civitai.com/c"}
            """))
        #expect(ultra.kind == .model && ultra.fileName == "c.safetensors")
        let fp8Only = try LoRASources.civitaiRemote(version("""
            {"name": "n.safetensors", "type": "Model", "primary": true, "metadata": {"format": "SafeTensor", "fp": "nf4"}, "downloadUrl": "https://civitai.com/n"},
            {"name": "f.safetensors", "type": "Model", "metadata": {"format": "SafeTensor", "fp": "fp8"}, "downloadUrl": "https://civitai.com/f"}
            """))
        #expect(fp8Only.fileName == "f.safetensors")
        #expect(throws: LoRASourceError.self) {
            try LoRASources.civitaiRemote(version("""
                {"name": "q.gguf", "type": "Model", "metadata": {"format": "GGUF"}, "downloadUrl": "https://civitai.com/q"}
                """))
        }
    }

    @Test func civitaiPreviewImageIsTheFirstSafeOneAtThumbnailSize() throws {
        let json = """
        {"id": 1, "files": [{"name": "a.safetensors", "type": "Model", "downloadUrl": "https://civitai.com/api/download/models/1"}],
         "images": [{"type": "image", "nsfwLevel": 4, "url": "https://image.civitai.com/x/a/original=true/1.jpeg"},
                    {"type": "video", "nsfwLevel": 1, "url": "https://image.civitai.com/x/b/original=true/2.mp4"},
                    {"type": "image", "nsfwLevel": 1, "url": "https://image.civitai.com/x/c/original=true/3.jpeg"}]}
        """
        let r = try LoRASources.civitaiRemote(Data(json.utf8))
        #expect(r.previewImage?.absoluteString == "https://image.civitai.com/x/c/width=320/3.jpeg")
    }

    @Test func civitaiPagesPickTheNewestVersionForTheModel() throws {
        let page = Data("""
        {"modelVersions": [{"id": 3, "baseModel": "ZImageBase"}, {"id": 2, "baseModel": "Flux.1 D"},
                           {"id": 1, "baseModel": "SDXL 1.0"}]}
        """.utf8)
        #expect(try LoRASources.latestCivitaiVersion(page) == 3)
        #expect(try LoRASources.latestCivitaiVersion(page, fits: LoRASources.civitaiBaseModels(for: "flux.1")) == 2)
        #expect(try LoRASources.latestCivitaiVersion(page, fits: LoRASources.civitaiBaseModels(for: "qwen-image-2.1")) == 3)
        #expect(LoRASources.civitaiBaseModels(for: "wan-2.2-t2v")?("Wan Video 2.2 T2V-A14B") == true)
    }

    @Test func wanExpertsFromNames() {
        #expect(LoRASources.expert(of: "Wan2.2_breast_slider_v2_low_noise.safetensors") == "low")
        #expect(LoRASources.expert(of: "wan2.2_polyhedron_all_high_noise_FULL.safetensors") == "high")
        #expect(LoRASources.expert(of: "v2.0 - Low (Breast)") == "low")
        #expect(LoRASources.expert(of: "Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors") == "high")
        #expect(LoRASources.expert(of: "detail_slider.safetensors") == nil)
        #expect(LoRASources.swapExpert("v2.0 - Low (Breast)") == "v2.0 - High (Breast)")
        #expect(LoRASources.swapExpert("x_HIGH_fp8.safetensors") == "x_LOW_fp8.safetensors")
        #expect(LoRASources.swapExpert("high_noise_model") == "low_noise_model")
    }

    @Test func civitaiWanVersionWithBothFilesIsOnePair() throws {
        let version = Data("""
        {"id": 7, "name": "v1", "baseModel": "Wan Video 2.2 T2V-A14B", "model": {"name": "Motion", "type": "LORA"},
         "files": [
          {"name": "motion_low_noise.safetensors", "type": "Model", "sizeKB": 1000, "primary": true,
           "downloadUrl": "https://civitai.com/api/download/models/7?fileId=2", "hashes": {"SHA256": "BB"}},
          {"name": "motion_high_noise.safetensors", "type": "Model", "sizeKB": 2000,
           "downloadUrl": "https://civitai.com/api/download/models/7?fileId=1", "hashes": {"SHA256": "AA"}}]}
        """.utf8)
        let r = try LoRASources.civitaiRemote(version)
        #expect(r.fileName == "motion_high_noise.safetensors" && r.expert == "high" && r.sha256 == "aa")
        #expect(r.companion?.fileName == "motion_low_noise.safetensors" && r.companion?.sha256 == "bb")
        #expect(r.totalBytes == Int64(3000 * 1024))
        let low = try #require(r.companionRemote)
        #expect(low.expert == "low" && low.companion == nil && low.cacheName == "civitai-7-motion_low_noise.safetensors")
        // A FLUX LoRA with "high" in its name is no Wan expert.
        let flux = Data("""
        {"id": 8, "name": "v1", "baseModel": "Flux.1 D", "files": [{"name": "highres.safetensors", "type": "Model",
         "downloadUrl": "https://civitai.com/api/download/models/8"}]}
        """.utf8)
        #expect(try LoRASources.civitaiRemote(flux).expert == nil)
    }

    @Test func civitaiWanSiblingVersion() {
        let page = Data("""
        {"modelVersions": [
          {"id": 3, "name": "v2.0 - High (Breast)", "files": [{"name": "Wan2.2_breast_slider_v2_high_noise.safetensors"}]},
          {"id": 2, "name": "v2.0 - Low (Breast)", "files": [{"name": "Wan2.2_breast_slider_v2_low_noise.safetensors"}]},
          {"id": 1, "name": "v1.0 HN", "files": [{"name": "slider_v1_HIGH.safetensors"}]},
          {"id": 0, "name": "v1.0 LN", "files": [{"name": "slider_v1_LOW.safetensors"}]}]}
        """.utf8)
        #expect(LoRASources.civitaiSibling(page, versionID: 2, versionName: "v2.0 - Low (Breast)",
                                          fileName: "Wan2.2_breast_slider_v2_low_noise.safetensors") == 3)
        #expect(LoRASources.civitaiSibling(page, versionID: 1, versionName: "v1.0 HN",
                                          fileName: "slider_v1_HIGH.safetensors") == 0)
        #expect(LoRASources.civitaiSibling(page, versionID: 9, versionName: "only", fileName: "other.safetensors") == nil)
    }

    @Test func aWanPairGoesToTheServerAsTwoFiles() throws {
        var remote = LoRARemote(host: .civitai, downloadURL: URL(string: "https://civitai.com/api/download/models/7")!,
                                fileName: "m_high.safetensors", baseModel: "Wan Video 2.2 T2V-A14B", cacheName: "h")
        remote.expert = "high"
        remote.companion = LoRAFileRef(downloadURL: URL(string: "https://civitai.com/api/download/models/7?f=2")!,
                                       fileName: "m_low.safetensors", bytes: 1, sha256: nil, cacheName: "l")
        var params: [String: Any] = [:]
        let pair = CloudLoRA(source: .link(remote), strength: 0.8, baseModel: remote.baseModel)
        let single = CloudLoRA(source: .file(URL(fileURLWithPath: "/x.safetensors")), strength: 1, expert: "low")
        let (files, links) = try CloudLoRA.attach([pair, single], family: "wan-2.2-i2v", to: &params)
        #expect(links["lora1"]?.fileName == "m_high.safetensors" && links["lora1_low"]?.fileName == "m_low.safetensors")
        #expect(files["lora2"] != nil && params["lora2_expert"] as? String == "low")
        #expect(params["lora1_expert"] as? String == "both" && params["lora1_strength"] as? Double == 0.8)
        // FLUX ignores the expert fields.
        var flux: [String: Any] = [:]
        _ = try CloudLoRA.attach([single], family: "flux.1", to: &flux)
        #expect(flux["lora1_expert"] == nil)
    }

    @Test func cloudLoRAsMustFitTheModelTheyFeed() throws {
        func lora(_ base: String?) -> CloudLoRA {
            CloudLoRA(source: .file(URL(fileURLWithPath: "/x.safetensors")), strength: 1, baseModel: base)
        }
        #expect(throws: LoRASourceError.wrongBaseModel("SDXL 1.0", expected: "FLUX.1")) {
            try lora("SDXL 1.0").checkFits("flux.1")
        }
        #expect(throws: LoRASourceError.wrongBaseModel("Qwen", expected: "FLUX.1")) { try lora("Qwen").checkFits("flux.1") }
        #expect(throws: LoRASourceError.wrongBaseModel("Flux.1 D", expected: "Qwen-Image")) {
            try lora("Flux.1 D").checkFits("qwen-image")
        }
        try lora("Flux.1 D").checkFits("flux.1")
        try lora("Qwen").checkFits("qwen-image")
        try lora(nil).checkFits("qwen-image")  // a file from disk: no claim to check
        var params: [String: Any] = [:]
        let (files, links) = try CloudLoRA.attach([lora("Qwen"), lora(nil)], family: "qwen-image", to: &params)
        #expect(files.keys.sorted() == ["lora1", "lora2"] && links.isEmpty)
        #expect(params["lora2_strength"] as? Double == 1)
        #expect(throws: RemoteNodeError.self) {
            try CloudLoRA.attach(Array(repeating: lora(nil), count: 5), family: "qwen-image", to: &params)
        }
    }

    @Test func huggingFaceTreeAndRedirectHeaders() throws {
        let tree = """
        [{"type": "file", "path": "README.md", "size": 10},
         {"type": "file", "path": "lora.safetensors", "size": 134, "lfs": {"oid": "0a83a924b822", "size": 22431400}},
         {"type": "directory", "path": "samples"}]
        """
        let files = try LoRASources.huggingFaceSafetensors(Data(tree.utf8))
        #expect(files.count == 1 && files[0].path == "lora.safetensors" && files[0].bytes == 22431400)
        let sha = String(repeating: "ab", count: 32)
        let r = LoRASources.huggingFaceRemote(repo: "XLabs-AI/flux-lora-collection", revision: "main",
                                              path: "realism_lora.safetensors",
                                              headers: ["X-Linked-Etag": "\"\(sha.uppercased())\"", "X-Linked-Size": "22431400"])
        #expect(r.sha256 == sha && r.bytes == 22431400)
        #expect(r.downloadURL.absoluteString
                == "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/realism_lora.safetensors")
        #expect(r.cacheName == "XLabs-AI--flux-lora-collection--realism_lora.safetensors")
        #expect(LoRASourceError.severalFiles(["a.safetensors", "b.safetensors"]).description.contains("a.safetensors, b.safetensors"))
    }

    @Test func statusLines() {
        #expect(TransferStatus.line(verb: "Uploading", sent: 230_000_000, total: 612_000_000, seconds: 0.5, name: "d.safetensors")
                == "Uploading 230 / 612 MB (d.safetensors)")
        #expect(TransferStatus.line(verb: "Uploading", sent: 230_000_000, total: 612_000_000, seconds: 115, name: "d")
                == "Uploading 230 / 612 MB · 2.0 MB/s · 4 min left (d)")
        // A resumed upload: only this attempt's bytes count toward the speed.
        #expect(TransferStatus.line(verb: "Uploading", sent: 600_000_000, total: 612_000_000, start: 500_000_000,
                                    seconds: 10) == "Uploading 600 / 612 MB · 10 MB/s · 2 s left")
        #expect(TransferStatus.line(verb: "Downloading", sent: 1_500_000_000, total: 4_000_000_000, seconds: 30)
                == "Downloading 1.5 / 4.0 GB · 50 MB/s · 50 s left")
        #expect(TransferStatus.line(verb: "Uploading", sent: 300_000, total: 850_000, seconds: 2) == "Uploading 300 / 850 KB · 150 KB/s · 4 s left")
        #expect(TransferStatus.timeLeft(4000) == "1 h 07 min left")
        #expect(TransferStatus.line(verb: "Downloading", sent: 5_000_000, total: nil, seconds: 0) == "Downloading 5.0 MB")
    }

    @Test func safetensorsHeaderCheck() {
        #expect(LoRASources.looksLikeSafetensors(safetensorsData(bytes: 10)))
        #expect(!LoRASources.looksLikeSafetensors(Data("<html><body>sign in".utf8)))
        #expect(!LoRASources.looksLikeSafetensors(Data([1, 2])))
    }
}

// MARK: - against the GPU test server and its file server

extension RemoteNodesTests {
    @Test(.enabled(if: hasServer)) func uploadReportsProgressAndSkipsKnownFiles() async throws {
        server.install()
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let file = out.appendingPathComponent("big.safetensors")
        let data = safetensorsData(bytes: 8 << 20, seed: UInt8.random(in: 0...255))
        try data.write(to: file)
        let client = GPUJobClient(baseURL: server.baseURL, token: server.token)
        let seen = Locked<[(Int64, Int64)]>([])
        _ = try await client.uploadFile(file, ext: "lora") { sent, total, _ in seen.withLock { $0.append((sent, total)) } }
        let first = seen.withLock { $0 }
        #expect(first.count >= 2 && first.last?.0 == first.last?.1 && first.last?.1 == Int64(data.count))
        #expect(zip(first, first.dropFirst()).allSatisfy { $0.0 <= $1.0 })
        seen.withLock { $0.removeAll() }
        _ = try await client.uploadFile(file, ext: "lora") { sent, total, _ in seen.withLock { $0.append((sent, total)) } }
        #expect(seen.withLock { $0.isEmpty })  // the server had it: nothing sent
    }

    @Test(.enabled(if: hasServer)) func cloudLoRAFromTheLibraryIsDownloadedByTheServer() async throws {
        let data = safetensorsData(bytes: 500_000, seed: UInt8.random(in: 0...255))
        try data.write(to: server.files.appendingPathComponent("linked.safetensors"))
        LoRASources.directHosts = ["127.0.0.1"]
        let savedModels = ModelPaths.directory
        ModelPaths.directory = out.appendingPathComponent("library-models")
        defer { LoRASources.directHosts = []; ModelPaths.directory = savedModels }
        // A library entry from a link (as the Models window adds it), not installed on this Mac.
        let remote = try await LoRASources.resolve(server.fileURL.appendingPathComponent("linked.safetensors").absoluteString)
        try Library.save([LibraryEntry(id: "linked", kind: .lora, family: "flux.1", title: "Linked", link: remote.downloadURL.absoluteString,
                                       source: remote, fileName: remote.fileName, sha256: nil, bytes: nil, baseModel: nil,
                                       triggerWords: ["linked style"])])
        let (result, log) = try await run([
            GraphNode(id: "l1", type: "RemoteLoRA", inputs: ["lora": .value(.string("linked"))]),
            GraphNode(id: "flux", type: "RemoteFlux", inputs: [
                "loras": .link(node: "l1", output: "loras"), "width": .value(.int(256)), "height": .value(.int(256)),
                "steps": .value(.int(2))]),
            GraphNode(id: "show", type: "PreviewImage", inputs: ["image": .link(node: "flux", output: "image")]),
        ])
        #expect(log.statuses.contains { $0.hasPrefix("Asking the server to download linked.safetensors") })
        #expect(log.statuses.contains("Trigger words: linked style"))
        guard case .object(let o)? = result.outputs["show"]?["image"], let image = o as? MLXArray else {
            Issue.record("no image")
            return
        }
        #expect(abs(image[0..., 0..., ..<3][0] - 0.25).max().item(Float.self) < 0.01)  // one LoRA arrived
        // The server keeps it under its SHA-256, like an upload.
        let client = GPUJobClient(baseURL: server.baseURL, token: server.token)
        #expect(try await client.has(sha256Hex(data) + ".lora"))
    }

    @Test func libraryEntriesSaveLoadAndGetUniqueIDs() throws {
        let models = out.appendingPathComponent("lib-\(UUID().uuidString)")
        let a = LibraryEntry(id: Library.newID(for: "Hands XL · Hand F1D", existing: []), kind: .lora, family: "flux.1",
                             title: "Hands XL", link: nil, source: nil, fileName: "a.safetensors", sha256: "ab", bytes: 1,
                             baseModel: nil, triggerWords: [])
        #expect(a.id == "hands-xl-hand-f1d" && a.serverName == "ab.lora")
        #expect(Library.newID(for: "Hands XL · Hand F1D", existing: [a]) == "hands-xl-hand-f1d-2")
        try Library.save([a], models: models)
        try Library.update(a.id, models: models) { $0.title = "Renamed" }
        #expect(Library.load(models: models).map(\.title) == ["Renamed"])
        #expect(Library.item("nope", models: models) == nil && Library.localURL(a, models: models) == nil)
        #expect(throws: LibraryError.missing("nope")) { try LibraryError.missing("nope").throwIt() }
    }

    @Test(.enabled(if: hasServer)) func libraryInstallsALinkOnThisMacOnce() async throws {
        let data = safetensorsData(bytes: 300_000, seed: UInt8.random(in: 0...255))
        try data.write(to: server.files.appendingPathComponent("libmac.safetensors"))
        let models = out.appendingPathComponent("lib-mac-\(UUID().uuidString)")
        let remote = LoRARemote(host: .direct, downloadURL: server.fileURL.appendingPathComponent("libmac.safetensors"),
                                fileName: "libmac.safetensors", sha256: sha256Hex(data), cacheName: "test--libmac.safetensors")
        try Library.save([LibraryEntry(id: "m", kind: .lora, family: nil, title: "M", link: nil, source: remote,
                                       fileName: remote.fileName, sha256: remote.sha256, bytes: nil, baseModel: nil,
                                       triggerWords: [])], models: models)
        let url = try await Library.installLocally("m", models: models) { _ in }
        #expect(try Data(contentsOf: url) == data)
        #expect(Library.item("m", models: models)?.localFile == "loras/test--libmac.safetensors")
        let again = try await Library.installLocally("m", models: models) { _ in Issue.record("downloaded twice") }
        #expect(again == url)
    }

    @Test(.enabled(if: hasServer)) func linkPreviewProbesLocallyAndOnTheServer() async throws {
        server.install()
        let data = safetensorsData(bytes: 100_000, seed: UInt8.random(in: 0...255))
        try data.write(to: server.files.appendingPathComponent("probe.safetensors"))
        try Data("<html>not a lora</html>".utf8).write(to: server.files.appendingPathComponent("page.safetensors"))
        let models = out.appendingPathComponent("probe-models")
        func remote(_ name: String, sha: String? = nil) -> LoRARemote {
            LoRARemote(host: .direct, downloadURL: server.fileURL.appendingPathComponent(name), fileName: name,
                       sha256: sha, cacheName: "probe--\(name)")
        }
        #expect(await LoRADownloads.probe(remote("probe.safetensors"), models: models) == .available(withKey: false))
        #expect(await LoRADownloads.probe(remote("page.safetensors"), models: models)
                == .unavailable("the link isn't a .safetensors file"))
        guard case .unavailable(let why) = await LoRADownloads.probe(remote("missing.safetensors"), models: models) else {
            Issue.record("expected unavailable"); return
        }
        #expect(why.contains("404"))
        let sha = sha256Hex(data)
        _ = try await LoRADownloads.fetch(remote("probe.safetensors", sha: sha), models: models) { _ in }
        #expect(await LoRADownloads.probe(remote("probe.safetensors", sha: sha), models: models) == .stored)
        #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("loras/probe--page.safetensors").path))
        // The cloud node asks the GPU server instead.
        #expect(await checkLinkOnServer(remote("probe.safetensors", sha: sha), serverID: "") == .available(withKey: false))
        guard case .unavailable(let serverWhy) = await checkLinkOnServer(remote("page.safetensors"), serverID: "") else {
            Issue.record("expected unavailable"); return
        }
        #expect(serverWhy.contains("isn't a .safetensors file"))
    }

    @Test(.enabled(if: hasServer)) func localLoRADownloadsOnceAndChecksTheHash() async throws {
        let data = safetensorsData(bytes: 3 << 20, seed: UInt8.random(in: 0...255))
        try data.write(to: server.files.appendingPathComponent("local.safetensors"))
        let models = out.appendingPathComponent("models")
        let link = server.fileURL.appendingPathComponent("local.safetensors")
        var remote = LoRARemote(host: .direct, downloadURL: link, fileName: "local.safetensors",
                                bytes: Int64(data.count), sha256: sha256Hex(data), cacheName: "test--local.safetensors")
        let statuses = Locked<[String]>([])
        let file = try await LoRADownloads.fetch(remote, models: models) { s in statuses.withLock { $0.append(s) } }
        #expect(try Data(contentsOf: file) == data && file.deletingLastPathComponent().lastPathComponent == "loras")
        #expect(statuses.withLock { $0.contains { $0.hasPrefix("Downloading ") && $0.contains("MB") } })
        #expect((try? String(contentsOf: file.appendingPathExtension("sha256"), encoding: .utf8)) == sha256Hex(data))
        statuses.withLock { $0.removeAll() }
        _ = try await LoRADownloads.fetch(remote, models: models) { s in statuses.withLock { $0.append(s) } }
        #expect(statuses.withLock { $0.isEmpty })  // found, not downloaded again
        // A file that doesn't match its published hash is thrown away.
        remote.sha256 = String(repeating: "0", count: 64)
        remote.cacheName = "test--other.safetensors"
        await #expect(throws: LoRASourceError.hashMismatch("local.safetensors")) {
            _ = try await LoRADownloads.fetch(remote, models: models) { _ in }
        }
        #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("loras/test--other.safetensors").path))
        #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("loras/test--other.safetensors.partial").path))
    }
}

final class Locked<T>: @unchecked Sendable {
    private var value: T
    private let lock = NSLock()
    init(_ v: T) { value = v }
    func withLock<R>(_ body: (inout T) -> R) -> R { lock.withLock { body(&value) } }
}

// MARK: - the real sites (opt in: MLXCOMFY_NETWORK_TESTS=1; with xcodebuild TEST_RUNNER_MLXCOMFY_NETWORK_TESTS=1)

@Suite(.enabled(if: ProcessInfo.processInfo.environment["MLXCOMFY_NETWORK_TESTS"] == "1"))
struct LoRALinkNetworkTests {
    @Test func huggingFaceFileRepoAndDownload() async throws {
        let r = try await LoRASources.resolve("https://huggingface.co/XLabs-AI/flux-lora-collection/blob/main/realism_lora.safetensors")
        #expect(r.sha256 == "0a83a924b822b70b5e458d27935ebfa7713edaee04ff9f194209525354031eca" && r.bytes == 22431400)
        await #expect(throws: LoRASourceError.self) {
            _ = try await LoRASources.resolve("https://huggingface.co/XLabs-AI/flux-lora-collection")  // several files
        }
        let models = FileManager.default.temporaryDirectory.appendingPathComponent("lora-net-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: models) }
        let statuses = Locked<[String]>([])
        let file = try await LoRADownloads.fetch(r, models: models) { s in statuses.withLock { $0.append(s) } }
        #expect(LoRADownloads.fileSize(file) == 22431400)
        print("[network] HF download statuses:", statuses.withLock { $0.suffix(3) })
    }

    @Test func probesTellWhatNeedsAKey() async throws {
        let saved = LoRASources.tokenProvider
        LoRASources.tokenProvider = { _ in nil }
        defer { LoRASources.tokenProvider = saved }
        let hf = try await LoRASources.resolve("https://huggingface.co/XLabs-AI/flux-lora-collection/blob/main/realism_lora.safetensors")
        let models = FileManager.default.temporaryDirectory.appendingPathComponent("probe-net-\(UUID().uuidString)")
        #expect(await LoRADownloads.probe(hf, models: models) == .available(withKey: false))
        let civitai = try await LoRASources.resolve("https://civitai.com/models/200255?modelVersionId=804967")
        #expect(civitai.previewImage != nil)
        #expect(await LoRADownloads.probe(civitai, models: models) == .needsKey(.civitai))
    }

    @Test func civitaiModelPage() async throws {
        let r = try await LoRASources.resolve("https://civitai.com/models/200255?modelVersionId=804967")
        #expect(r.baseModel == "Flux.1 D" && r.fileName == "Hand v2.safetensors" && r.sha256?.count == 64)
        #expect(r.triggerWords.contains("Perfect hand"))
        // A page with versions for several models: the newest one for FLUX.1, not the newest overall.
        let latest = try await LoRASources.resolve("https://civitai.com/models/200255/hands-xl",
                                                   fits: LoRASources.civitaiBaseModels(for: "flux.1"))
        #expect(latest.host == .civitai && latest.baseModel?.hasPrefix("Flux.1") == true)
        print("[network] Civitai:", r.title ?? "", r.bytes ?? 0, latest.title ?? "", latest.baseModel ?? "")
    }
}

extension LibraryError { func throwIt() throws { throw self } }

extension RemoteNodesTests {
    @Test(.enabled(if: hasServer)) func aFineTuneImportsOnTheServerAndTheFluxNodePicksIt() async throws {
        server.install()
        let data = safetensorsData(bytes: 400_000, seed: UInt8.random(in: 0...255))
        try data.write(to: server.files.appendingPathComponent("finetune.safetensors"))
        let client = GPUJobClient(baseURL: server.baseURL, token: server.token)
        let id = "ft-\(UInt32.random(in: 0...UInt32.max))"
        _ = try await client.importModel(family: "flux.1", id: id, name: "Tiny fine-tune",
                                         url: server.fileURL.appendingPathComponent("finetune.safetensors"),
                                         sha256: sha256Hex(data), repo: nil)
        var installed = false
        for _ in 0..<100 where !installed {
            try await Task.sleep(nanoseconds: 100_000_000)
            installed = try await client.models().families["flux.1"]?.contains { $0.id == id } == true
        }
        #expect(installed)
        #expect(try await client.info().variants?["flux.1"]?.contains { $0.id == id && $0.name == "Tiny fine-tune" } == true)
        let (result, _) = try await run([
            GraphNode(id: "flux", type: "RemoteFlux", inputs: [
                "model": .value(.string(id)), "width": .value(.int(256)), "height": .value(.int(256)),
                "steps": .value(.int(2))]),
            GraphNode(id: "show", type: "PreviewImage", inputs: ["image": .link(node: "flux", output: "image")]),
        ])
        guard case .object(let o)? = result.outputs["show"]?["image"], let image = o as? MLXArray else {
            Issue.record("no image"); return
        }
        // The fake server whitens the right-hand column for a fine-tune.
        #expect(abs(image[0..., 255, ..<3] - 1).max().item(Float.self) < 0.01)
        #expect(try await client.deleteModel(family: "flux.1", id: id))
        #expect(try await client.models().families["flux.1"]?.contains { $0.id == id } != true)
        #expect(LinkLibrary_huggingFaceRepoForTest() == "black-forest-labs/FLUX.1-Krea-dev@main")
    }
}

/// The repo string the app sends for a Hugging Face diffusers repo link (mirrors `LinkLibrary.huggingFaceRepo`).
func LinkLibrary_huggingFaceRepoForTest() -> String? {
    let r = LoRARemote(host: .huggingface, kind: .model,
                       downloadURL: URL(string: "https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/tree/main")!,
                       fileName: "x", cacheName: "x")
    let parts = r.downloadURL.path.split(separator: "/").map(String.init)
    return parts.count >= 4 && parts[2] == "tree" ? "\(parts[0])/\(parts[1])@\(parts[3...].joined(separator: "/"))" : nil
}
