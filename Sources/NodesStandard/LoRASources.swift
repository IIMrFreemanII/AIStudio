import Foundation
import MLXDiffusionCore

// LoRAs from a link: a Civitai model page or a Hugging Face file (or repo) URL becomes a direct download URL plus
// what is known about the file (size, SHA-256, base model, trigger words). The local LoRA node downloads it into the
// models folder (`LoRADownloads`); the Cloud GPU LoRA node has the GPU server download it.

/// A LoRA file behind a link, resolved.
public struct LoRARemote: Sendable, Equatable, Codable {
    public enum Host: String, Sendable, Codable { case civitai, huggingface, direct }
    /// What the file is: a LoRA, or a whole model (checkpoint, diffusers repo).
    public enum Kind: String, Sendable, Codable { case lora, model }
    public var host: Host
    public var kind: Kind = .lora
    public var downloadURL: URL
    public var fileName: String
    public var bytes: Int64?
    /// Lowercase hex, when the site publishes it (Civitai, Hugging Face LFS files).
    public var sha256: String?
    public var title: String?
    /// Civitai's base model ("Flux.1 D", "SDXL 1.0", …).
    public var baseModel: String?
    public var triggerWords: [String] = []
    /// A sample image (Civitai's first safe-for-work one), for the link preview.
    public var previewImage: URL?
    /// The file name it's kept under in `<models>/loras`: stable per source, so a second run finds it.
    public var cacheName: String
    /// Wan 2.2 A14B's two experts: which one this file is for (`high` / `low`); nil for both (or not Wan).
    public var expert: String?
    /// The other expert's file, when the link has a high/low pair (this file is then the `high` one).
    public var companion: LoRAFileRef?

    /// The companion as a remote of its own (the same page, its file), for downloading it the same way.
    public var companionRemote: LoRARemote? {
        guard let c = companion else { return nil }
        var r = self
        r.downloadURL = c.downloadURL
        r.fileName = c.fileName
        r.bytes = c.bytes
        r.sha256 = c.sha256
        r.cacheName = c.cacheName
        r.expert = expert == "high" ? "low" : "high"
        r.companion = nil
        return r
    }

    /// The bytes of both files.
    public var totalBytes: Int64? {
        guard let b = bytes else { return nil }
        return b + (companion?.bytes ?? 0)
    }

    public init(host: Host, kind: Kind = .lora, downloadURL: URL, fileName: String, bytes: Int64? = nil,
                sha256: String? = nil, title: String? = nil, baseModel: String? = nil, triggerWords: [String] = [],
                previewImage: URL? = nil, cacheName: String) {
        self.host = host
        self.kind = kind
        self.downloadURL = downloadURL
        self.fileName = fileName
        self.bytes = bytes
        self.sha256 = sha256
        self.title = title
        self.baseModel = baseModel
        self.triggerWords = triggerWords
        self.previewImage = previewImage
        self.cacheName = cacheName
    }
}

/// One file of a link (the companion of a Wan 2.2 high/low pair).
public struct LoRAFileRef: Sendable, Equatable, Codable {
    public var downloadURL: URL
    public var fileName: String
    public var bytes: Int64?
    public var sha256: String?
    public var cacheName: String

    public init(downloadURL: URL, fileName: String, bytes: Int64?, sha256: String?, cacheName: String) {
        self.downloadURL = downloadURL; self.fileName = fileName; self.bytes = bytes; self.sha256 = sha256
        self.cacheName = cacheName
    }

    public init(_ r: LoRARemote) {
        self.init(downloadURL: r.downloadURL, fileName: r.fileName, bytes: r.bytes, sha256: r.sha256,
                  cacheName: r.cacheName)
    }
}

public enum LoRASourceError: Error, CustomStringConvertible, Equatable {
    case notALink(String)
    case unsupported(String)
    case noSafetensors(String)
    case severalFiles([String])
    case needsKey(LoRARemote.Host)
    case http(Int, String)
    case notSafetensors(String)
    case hashMismatch(String)
    case wrongBaseModel(String, expected: String)
    case unsupportedFormat([String])

    public var description: String {
        switch self {
        case .notALink(let s): return "Not a link: \(s)"
        case .unsupported(let s):
            return "Links from \(s) aren't supported: paste a Civitai model page or a Hugging Face file or repo link"
        case .noSafetensors(let s): return "No .safetensors LoRA file in \(s)"
        case .severalFiles(let files):
            return "This repo has several .safetensors files; paste the link of the one you want: "
                + files.prefix(8).joined(separator: ", ") + (files.count > 8 ? ", …" : "")
        case .needsKey(.civitai): return "This file needs a Civitai API key: add it in the Models window"
        case .needsKey(.huggingface):
            return "This Hugging Face repo is gated: accept its licence on huggingface.co and add your token in the Models window"
        case .needsKey(.direct): return "The server refused the download (it needs a login)"
        case .http(let code, let url): return "HTTP \(code) from \(url)"
        case .notSafetensors(let name): return "\(name) isn't a .safetensors file"
        case .hashMismatch(let name): return "\(name) didn't download intact (its SHA-256 doesn't match): try again"
        case .wrongBaseModel(let base, let expected): return "This LoRA is for \(base), not \(expected)"
        case .unsupportedFormat(let files):
            return "Only GGUF / NF4 / pickle files here (\(files.prefix(3).joined(separator: ", "))): the app needs a .safetensors in bf16, fp16 or fp8"
        }
    }
}

public enum LoRASources {
    /// What a pasted link points at.
    public enum Link: Equatable, Sendable {
        case civitaiVersion(Int)
        case civitaiModel(Int)
        case huggingFaceFile(repo: String, revision: String, path: String)
        case huggingFaceRepo(repo: String, revision: String)
        case direct(URL)
    }

    /// Hosts whose plain file links are accepted as they are (tests use a local server).
    nonisolated(unsafe) public static var directHosts: Set<String> = []

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _tokenProvider: @Sendable (LoRARemote.Host) -> String? = { host in
        let env = ProcessInfo.processInfo.environment
        switch host {
        case .civitai: return env["CIVITAI_TOKEN"].flatMap { $0.isEmpty ? nil : $0 }
        case .huggingface: return env["HF_TOKEN"].flatMap { $0.isEmpty ? nil : $0 }
        case .direct: return nil
        }
    }

    /// The API key for a site, asked for only when a request needs one. The app reads its Keychain; the default
    /// reads `CIVITAI_TOKEN` / `HF_TOKEN`.
    public static var tokenProvider: @Sendable (LoRARemote.Host) -> String? {
        get { lock.withLock { _tokenProvider } }
        set { lock.withLock { _tokenProvider = newValue } }
    }

    // MARK: parsing (no network)

    public static func parse(_ text: String) throws -> Link {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let comps = URLComponents(string: trimmed), let scheme = comps.scheme?.lowercased(),
              scheme == "https" || scheme == "http", var host = comps.host?.lowercased() else {
            throw LoRASourceError.notALink(trimmed)
        }
        if host.hasPrefix("www.") { host.removeFirst(4) }
        let parts = comps.path.split(separator: "/").map { $0.removingPercentEncoding ?? String($0) }
        switch host {
        case "civitai.com", "civitai.green", "civitai.red":
            // /models/<id>[/<slug>]?modelVersionId=<v>   ·   /api/download/models/<v>   ·   /api/v1/model-versions/<v>
            if let v = comps.queryItems?.first(where: { $0.name == "modelVersionId" })?.value.flatMap(Int.init) {
                return .civitaiVersion(v)
            }
            if parts.count >= 2, parts[0] == "models", let id = Int(parts[1]) { return .civitaiModel(id) }
            if parts.count >= 4, parts[0] == "api", parts[1] == "download", parts[2] == "models", let v = Int(parts[3]) {
                return .civitaiVersion(v)
            }
            if parts.count >= 4, parts[0] == "api", parts[1] == "v1", parts[2] == "model-versions", let v = Int(parts[3]) {
                return .civitaiVersion(v)
            }
            throw LoRASourceError.unsupported("this Civitai page")
        case "huggingface.co", "hf.co":
            guard parts.count >= 2, !["datasets", "spaces", "api"].contains(parts[0]) else {
                throw LoRASourceError.unsupported("this Hugging Face page")
            }
            let repo = "\(parts[0])/\(parts[1])"
            if parts.count >= 5, parts[2] == "blob" || parts[2] == "resolve" {
                return .huggingFaceFile(repo: repo, revision: parts[3], path: parts[4...].joined(separator: "/"))
            }
            if parts.count >= 4, parts[2] == "tree" { return .huggingFaceRepo(repo: repo, revision: parts[3]) }
            if parts.count == 2 { return .huggingFaceRepo(repo: repo, revision: "main") }
            throw LoRASourceError.unsupported("this Hugging Face page")
        default:
            if directHosts.contains(host), let url = comps.url { return .direct(url) }
            throw LoRASourceError.unsupported(host)
        }
    }

    static let civitaiAPI = URL(string: "https://civitai.com/api/v1/")!

    public static func civitaiVersionURL(_ v: Int) -> URL { civitaiAPI.appendingPathComponent("model-versions/\(v)") }
    public static func civitaiModelURL(_ id: Int) -> URL { civitaiAPI.appendingPathComponent("models/\(id)") }

    public static func huggingFaceResolveURL(repo: String, revision: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/")!.appendingPathComponent(path)
    }

    public static func huggingFaceTreeURL(repo: String, revision: String) -> URL {
        var c = URLComponents(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)")!
        c.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        return c.url!
    }

    /// The newest version of a Civitai model (`/api/v1/models/<id>` lists them newest first) — the newest one for a
    /// base model `fits` accepts, when there is one (a page often has versions for several models).
    public static func latestCivitaiVersion(_ data: Data, fits: ((String) -> Bool)? = nil) throws -> Int {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let versions = obj["modelVersions"] as? [[String: Any]], let newest = versions.first?["id"] as? Int else {
            throw LoRASourceError.noSafetensors("this Civitai model")
        }
        guard let fits else { return newest }
        return versions.first { ($0["baseModel"] as? String).map(fits) ?? false }?["id"] as? Int ?? newest
    }

    /// The app's family for a Civitai base model: "Flux.1 D" → `flux.1`, "Qwen" → `qwen-image` (the original 20B
    /// model), "Wan Video 2.2 T2V-A14B" → `wan-2.2-t2v`, … ; nil for families the app doesn't run (SDXL, Pony…).
    public static func family(forBaseModel base: String) -> String? {
        let b = base.lowercased()
        if b.hasPrefix("flux.1") { return "flux.1" }
        if b.contains("qwen") { return b.contains("2.1") ? "qwen-image-2.1" : "qwen-image" }
        if b.contains("wan") && b.contains("2.2") {
            if b.contains("i2v") && !b.contains("ti2v") { return "wan-2.2-i2v" }
            if b.contains("t2v") && !b.contains("ti2v") { return "wan-2.2-t2v" }
        }
        return nil
    }

    /// Readable names for the app's families.
    public static func familyName(_ family: String?) -> String {
        switch family {
        case "flux.1"?: return "FLUX.1"
        case "qwen-image"?: return "Qwen-Image"
        case "qwen-image-2.1"?: return "Qwen-Image 2.1"
        case "wan-2.2-t2v"?: return "Wan 2.2 T2V"
        case "wan-2.2-i2v"?: return "Wan 2.2 I2V"
        default: return "Other"
        }
    }

    /// Which Civitai base models ("Flux.1 D", "Qwen", "Wan Video 2.2 T2V-A14B"…) suit a local model family
    /// (`GenerativeModel.family`); nil when there's no rule.
    public static func civitaiBaseModels(for family: String) -> ((String) -> Bool)? {
        let f = family.lowercased()
        if f.hasPrefix("flux.1") { return { $0.lowercased().hasPrefix("flux.1") } }
        if f.hasPrefix("qwen-image") { return { $0.lowercased().contains("qwen") } }
        if f.hasPrefix("wan-2.2") { return { let b = $0.lowercased(); return b.contains("wan") && b.contains("2.2") } }
        if f.hasPrefix("ltx") { return { $0.lowercased().contains("ltx") } }
        return nil
    }

    /// A Civitai model version (`/api/v1/model-versions/<v>`): its LoRA file (`type` Model, `.safetensors`, the
    /// primary one first), size, SHA-256, base model and trigger words.
    public static func civitaiRemote(_ data: Data) throws -> LoRARemote {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let versionID = obj["id"] as? Int else { throw LoRASourceError.noSafetensors("this Civitai model") }
        let models = (obj["files"] as? [[String: Any]] ?? []).filter { ($0["type"] as? String ?? "Model") == "Model" }
        // safetensors only, and not NF4 (bitsandbytes) or GGUF; bf16/fp16 first, then fp8, then fp32; the primary
        // file first among equals.
        func meta(_ f: [String: Any], _ k: String) -> String { ((f["metadata"] as? [String: Any])?[k] as? String ?? "").lowercased() }
        let files = models.filter {
            ($0["name"] as? String ?? "").lowercased().hasSuffix(".safetensors") && meta($0, "fp") != "nf4"
                && ["", "safetensor"].contains(meta($0, "format"))
        }
        func rank(_ f: [String: Any]) -> Int {
            let fp = ["bf16": 0, "fp16": 0, "fp8": 1, "fp32": 2][meta(f, "fp")] ?? 1
            return fp * 2 + (f["primary"] as? Bool == true ? 0 : 1)
        }
        if files.isEmpty, !models.isEmpty {
            throw LoRASourceError.unsupportedFormat(models.compactMap { $0["name"] as? String })
        }
        // Wan 2.2: a version with a high-noise and a low-noise file is one pair.
        let wan = isWan22(obj["baseModel"] as? String)
        let byExpert = wan ? Dictionary(grouping: files) { expert(of: $0["name"] as? String ?? "") } : [:]
        let paired = byExpert["high"] != nil && byExpert["low"] != nil
        let pool = paired ? byExpert["high"]! : files
        guard let file = pool.min(by: { rank($0) < rank($1) }),
              let name = file["name"] as? String,
              let link = (file["downloadUrl"] as? String).flatMap(URL.init(string:)) else {
            throw LoRASourceError.noSafetensors("this Civitai model")
        }
        func ref(_ f: [String: Any]) -> LoRAFileRef? {
            guard let n = f["name"] as? String, let u = (f["downloadUrl"] as? String).flatMap(URL.init(string:)) else { return nil }
            return LoRAFileRef(downloadURL: u, fileName: n, bytes: (f["sizeKB"] as? Double).map { Int64(($0 * 1024).rounded()) },
                               sha256: ((f["hashes"] as? [String: Any])?["SHA256"] as? String)?.lowercased(),
                               cacheName: "civitai-\(versionID)-" + safeName(n))
        }
        let model = obj["model"] as? [String: Any]
        let title = [model?["name"] as? String, obj["name"] as? String].compactMap { $0 }.joined(separator: " · ")
        // Civitai's model types: LORA, LoCon, DoRA are LoRAs; Checkpoint is a whole model.
        let type = (model?["type"] as? String ?? "LORA").lowercased()
        let kind: LoRARemote.Kind = type == "checkpoint" ? .model : .lora
        let words = (obj["trainedWords"] as? [String] ?? []).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        // Civitai's nsfwLevel: 1 is safe for work. Thumbnail size instead of the original.
        let image = (obj["images"] as? [[String: Any]] ?? []).first {
            ($0["type"] as? String ?? "image") == "image" && ($0["nsfwLevel"] as? Int ?? 99) <= 1
        }?["url"] as? String
        var r = LoRARemote(
            host: .civitai, kind: kind, downloadURL: link, fileName: name,
            bytes: (file["sizeKB"] as? Double).map { Int64(($0 * 1024).rounded()) },
            sha256: ((file["hashes"] as? [String: Any])?["SHA256"] as? String)?.lowercased(),
            title: title.isEmpty ? nil : title, baseModel: obj["baseModel"] as? String, triggerWords: words,
            previewImage: image.flatMap { URL(string: $0.replacingOccurrences(of: "/original=true/", with: "/width=320/")) },
            cacheName: "civitai-\(versionID)-" + safeName(name))
        if paired {
            r.expert = "high"
            r.companion = byExpert["low"]!.min(by: { rank($0) < rank($1) }).flatMap(ref)
        } else if wan {
            r.expert = expert(of: name) ?? expert(of: obj["name"] as? String ?? "")
        }
        return r
    }

    // MARK: Wan 2.2 high/low pairs

    public static func isWan22(_ baseModel: String?) -> Bool {
        let b = (baseModel ?? "").lowercased()
        return b.contains("wan") && b.contains("2.2")
    }

    /// Which Wan 2.2 expert a file or version name is for: `…HighNoise…`, `high_noise_model`, `v2.0 - Low` → high /
    /// low; nil when it doesn't say (or says both).
    public static func expert(of name: String) -> String? {
        let n = name.lowercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: " ", with: "_")
        let high = n.contains("high") || n.contains("_hn_") || n.hasPrefix("hn_") || n.hasSuffix("_hn.safetensors")
        let low = n.contains("low") || n.contains("_ln_") || n.hasPrefix("ln_") || n.hasSuffix("_ln.safetensors")
        return high == low ? nil : high ? "high" : "low"
    }

    /// The name with the expert word swapped (`…_high_noise…` → `…_low_noise…`), keeping its case.
    public static func swapExpert(_ name: String) -> String {
        var out = name
        for (a, b) in [("HIGH", "LOW"), ("High", "Low"), ("high", "low")] {
            if out.contains(a) { return out.replacingOccurrences(of: a, with: b) }
            if out.contains(b) { out = out.replacingOccurrences(of: b, with: a); return out }
        }
        return out
    }

    /// On a Civitai model page, the version with the other expert of `version` (a model that publishes "v2.0 - High"
    /// and "v2.0 - Low" as two versions): the same name with high/low swapped, else a file of that name.
    public static func civitaiSibling(_ modelData: Data, versionID: Int, versionName: String, fileName: String) -> Int? {
        guard let obj = try? JSONSerialization.jsonObject(with: modelData) as? [String: Any],
              let versions = obj["modelVersions"] as? [[String: Any]] else { return nil }
        let wantName = swapExpert(versionName).lowercased(), wantFile = swapExpert(fileName).lowercased()
        let others = versions.filter { ($0["id"] as? Int) != versionID }
        if wantName != versionName.lowercased(),
           let v = others.first(where: { ($0["name"] as? String)?.lowercased() == wantName }) { return v["id"] as? Int }
        if wantFile != fileName.lowercased() {
            return others.first { v in
                (v["files"] as? [[String: Any]] ?? []).contains { ($0["name"] as? String)?.lowercased() == wantFile }
            }?["id"] as? Int
        }
        return nil
    }

    /// Completes a single-expert Wan 2.2 file from Civitai with its other half, when the model page has it.
    static func civitaiPair(_ r: LoRARemote, version: Data, session: URLSession) async -> LoRARemote {
        guard r.expert != nil, r.companion == nil, isWan22(r.baseModel),
              let obj = try? JSONSerialization.jsonObject(with: version) as? [String: Any],
              let versionID = obj["id"] as? Int, let modelID = obj["modelId"] as? Int,
              let page = try? await get(civitaiModelURL(modelID), host: .civitai, session: session),
              let sibling = civitaiSibling(page, versionID: versionID, versionName: obj["name"] as? String ?? "",
                                           fileName: r.fileName),
              let data = try? await get(civitaiVersionURL(sibling), host: .civitai, session: session),
              let other = try? civitaiRemote(data), other.expert != nil, other.expert != r.expert else { return r }
        // The pair as one: the high-noise file first.
        let pair = r.expert == "high" ? (r, other) : (other, r)
        var high = pair.0
        high.companion = LoRAFileRef(pair.1)
        high.expert = "high"
        high.title = r.title
        high.triggerWords = Array(Set(r.triggerWords + other.triggerWords)).sorted()
        high.previewImage = r.previewImage ?? other.previewImage
        return high
    }

    /// The `.safetensors` files of a Hugging Face tree listing: (path, size, SHA-256 when stored in LFS).
    public static func huggingFaceSafetensors(_ data: Data) throws -> [(path: String, bytes: Int64?, sha256: String?)] {
        guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return items.compactMap { item in
            guard item["type"] as? String == "file", let path = item["path"] as? String,
                  path.lowercased().hasSuffix(".safetensors") else { return nil }
            let lfs = item["lfs"] as? [String: Any]
            let size = (lfs?["size"] as? Int).map(Int64.init) ?? (item["size"] as? Int).map(Int64.init)
            return (path, size, lfs?["oid"] as? String)
        }
    }

    /// A Hugging Face file from the headers of its `resolve` URL (the redirect carries the LFS SHA-256 and size).
    public static func huggingFaceRemote(repo: String, revision: String, path: String, headers: [AnyHashable: Any])
        -> LoRARemote {
        func header(_ name: String) -> String? {
            headers.first { ($0.key as? String)?.lowercased() == name }?.value as? String
        }
        let etag = header("x-linked-etag")?.trimmingCharacters(in: CharacterSet(charactersIn: "\"W/"))
        let sha = etag.flatMap { $0.count == 64 && $0.allSatisfy(\.isHexDigit) ? $0.lowercased() : nil }
        return huggingFaceRemote(repo: repo, revision: revision, path: path,
                                 bytes: header("x-linked-size").flatMap { Int64($0) }, sha256: sha)
    }

    static func huggingFaceRemote(repo: String, revision: String, path: String, bytes: Int64?, sha256: String?)
        -> LoRARemote {
        let name = (path as NSString).lastPathComponent
        // A single file over 2 GB is a whole model (checkpoint), not a LoRA.
        return LoRARemote(host: .huggingface, kind: (bytes ?? 0) > 2_000_000_000 ? .model : .lora,
                          downloadURL: huggingFaceResolveURL(repo: repo, revision: revision, path: path),
                          fileName: name, bytes: bytes, sha256: sha256, title: "\(repo) · \(name)",
                          cacheName: safeName(repo.replacingOccurrences(of: "/", with: "--")) + "--" + safeName(name))
    }

    /// Letters, digits, `.`, `-`, `_` only.
    static func safeName(_ s: String) -> String {
        String(s.map { $0.isLetter || $0.isNumber || "._-".contains($0) ? $0 : "_" })
    }

    // MARK: resolving (network)

    /// Turns a pasted link into the file it points at: metadata only, no download.
    /// `fits`: the base models the LoRA is for (Civitai pages with versions for several models pick the newest one
    /// that fits).
    public static func resolve(_ text: String, fits: ((String) -> Bool)? = nil,
                               session: URLSession = .shared) async throws -> LoRARemote {
        switch try parse(text) {
        case .civitaiVersion(let v):
            let data = try await get(civitaiVersionURL(v), host: .civitai, session: session)
            return await civitaiPair(try civitaiRemote(data), version: data, session: session)
        case .civitaiModel(let id):
            let v = try latestCivitaiVersion(try await get(civitaiModelURL(id), host: .civitai, session: session),
                                             fits: fits)
            let data = try await get(civitaiVersionURL(v), host: .civitai, session: session)
            return await civitaiPair(try civitaiRemote(data), version: data, session: session)
        case .huggingFaceRepo(let repo, let revision):
            let tree = try await get(huggingFaceTreeURL(repo: repo, revision: revision), host: .huggingface, session: session)
            if (try? JSONSerialization.jsonObject(with: tree) as? [[String: Any]])?
                .contains(where: { $0["path"] as? String == "model_index.json" }) == true {
                // A diffusers model repo: a whole model.
                return LoRARemote(host: .huggingface, kind: .model,
                                  downloadURL: URL(string: "https://huggingface.co/\(repo)/tree/\(revision)")!,
                                  fileName: repo, title: repo, cacheName: safeName(repo.replacingOccurrences(of: "/", with: "--")))
            }
            let files = try huggingFaceSafetensors(tree)
            guard !files.isEmpty else { throw LoRASourceError.noSafetensors(repo) }
            // Two files, one per Wan 2.2 expert: one pair.
            if files.count == 2, let h = files.first(where: { expert(of: $0.path) == "high" }),
               let l = files.first(where: { expert(of: $0.path) == "low" }) {
                var r = huggingFaceRemote(repo: repo, revision: revision, path: h.path, bytes: h.bytes,
                                          sha256: h.sha256?.lowercased())
                r.expert = "high"
                r.companion = LoRAFileRef(huggingFaceRemote(repo: repo, revision: revision, path: l.path, bytes: l.bytes,
                                                            sha256: l.sha256?.lowercased()))
                r.title = repo
                return r
            }
            guard files.count == 1 else { throw LoRASourceError.severalFiles(files.map(\.path)) }
            return huggingFaceRemote(repo: repo, revision: revision, path: files[0].path, bytes: files[0].bytes,
                                     sha256: files[0].sha256?.lowercased())
        case .huggingFaceFile(let repo, let revision, let path):
            var r = URLRequest(url: huggingFaceResolveURL(repo: repo, revision: revision, path: path))
            r.httpMethod = "HEAD"
            let (_, response) = try await send(r, host: .huggingface, session: session, delegate: NoRedirects())
            var remote = huggingFaceRemote(repo: repo, revision: revision, path: path,
                                           headers: (response as? HTTPURLResponse)?.allHeaderFields ?? [:])
            // A Wan 2.2 expert's file: its other half, when the repo has it under the swapped name.
            if let e = expert(of: (path as NSString).lastPathComponent), case let other = swapExpert(path), other != path {
                var head = URLRequest(url: huggingFaceResolveURL(repo: repo, revision: revision, path: other))
                head.httpMethod = "HEAD"
                if let (_, res) = try? await send(head, host: .huggingface, session: session, delegate: NoRedirects()),
                   let http = res as? HTTPURLResponse, (200..<400).contains(http.statusCode) {
                    let o = huggingFaceRemote(repo: repo, revision: revision, path: other, headers: http.allHeaderFields)
                    let pair = e == "high" ? (remote, o) : (o, remote)
                    remote = pair.0
                    remote.expert = "high"
                    remote.companion = LoRAFileRef(pair.1)
                } else {
                    remote.expert = e
                }
            }
            return remote
        case .direct(let url):
            let name = url.lastPathComponent
            return LoRARemote(host: .direct, downloadURL: url, fileName: name, title: name,
                              cacheName: safeName(url.host ?? "link") + "--" + safeName(name))
        }
    }

    /// Adds the site's key; false when there is none.
    @discardableResult
    public static func authorize(_ r: inout URLRequest, _ host: LoRARemote.Host) -> Bool {
        guard let token = tokenProvider(host) else { return false }
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return true
    }

    /// Sends without a key first (most links are public, and reading the key may ask for Keychain access), then once
    /// more with it when the site says it needs one.
    static func send(_ request: URLRequest, host: LoRARemote.Host, session: URLSession,
                     delegate: URLSessionTaskDelegate? = nil) async throws -> (Data, URLResponse) {
        var (data, response) = try await session.data(for: request, delegate: delegate)
        if let code = (response as? HTTPURLResponse)?.statusCode, code == 401 || code == 403 {
            var r = request
            if authorize(&r, host) { (data, response) = try await session.data(for: r, delegate: delegate) }
        }
        try check(response as? HTTPURLResponse, url: request.url!, host: host)
        return (data, response)
    }

    static func get(_ url: URL, host: LoRARemote.Host, session: URLSession) async throws -> Data {
        try await send(URLRequest(url: url), host: host, session: session).0
    }

    static func check(_ http: HTTPURLResponse?, url: URL, host: LoRARemote.Host) throws {
        guard let code = http?.statusCode, !(200..<400).contains(code) else { return }
        if code == 401 || code == 403 { throw LoRASourceError.needsKey(host) }
        throw LoRASourceError.http(code, url.host ?? url.absoluteString)
    }

    /// Whether a file starts like a safetensors file: an 8-byte little-endian header length, then a JSON object.
    public static func looksLikeSafetensors(_ head: Data) -> Bool {
        guard head.count >= 9 else { return false }
        let n = head.prefix(8).enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * UInt64($1.offset)) }
        return n > 1 && n < 100_000_000 && head[head.startIndex + 8] == UInt8(ascii: "{")
    }
}

/// Keeps a HEAD request on the first response (Hugging Face puts the file's hash and size on its redirect).
final class NoRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? { nil }
}

// MARK: - status lines

/// "Uploading 230 / 612 MB · 1.9 MB/s · 3 min left (disney.safetensors)": the numbers first, so a narrow badge
/// still shows them.
public enum TransferStatus {
    /// `start` is where this attempt began (a resumed upload's offset): the speed counts only bytes moved since.
    public static func line(verb: String, sent: Int64, total: Int64?, start: Int64 = 0, seconds: Double,
                            name: String? = nil) -> String {
        var parts: [String] = []
        if let total, total > 0 {
            parts.append("\(verb) \(sizes(sent, total))")
        } else {
            parts.append("\(verb) \(size(sent))")
        }
        let rate = seconds >= 1 ? Double(sent - start) / seconds : 0
        if rate > 0 {
            parts.append(speed(rate))
            if let total, total > sent { parts.append(timeLeft(Double(total - sent) / rate)) }
        }
        return parts.joined(separator: " · ") + (name.map { " (\($0))" } ?? "")
    }

    /// "230 / 612 MB", "1.2 / 3.4 GB", "300 / 850 KB": both in the total's unit.
    static func sizes(_ a: Int64, _ total: Int64) -> String {
        let (div, unit, digits) = unit(for: total)
        return "\(fmt(Double(a) / div, digits)) / \(fmt(Double(total) / div, digits)) \(unit)"
    }

    static func size(_ b: Int64) -> String {
        let (div, unit, digits) = unit(for: b)
        return "\(fmt(Double(b) / div, digits)) \(unit)"
    }

    static func unit(for b: Int64) -> (Double, String, Int) {
        if b >= 1_000_000_000 { return (1e9, "GB", 1) }
        if b >= 10_000_000 { return (1e6, "MB", 0) }
        if b >= 1_000_000 { return (1e6, "MB", 1) }
        return (1e3, "KB", 0)
    }

    static func fmt(_ v: Double, _ digits: Int) -> String { String(format: "%.\(digits)f", v) }

    static func speed(_ bytesPerSecond: Double) -> String {
        if bytesPerSecond >= 10e6 { return fmt(bytesPerSecond / 1e6, 0) + " MB/s" }
        if bytesPerSecond >= 1e6 { return fmt(bytesPerSecond / 1e6, 1) + " MB/s" }
        return fmt(bytesPerSecond / 1e3, 0) + " KB/s"
    }

    static func timeLeft(_ s: Double) -> String {
        if s < 60 { return "\(max(1, Int(s.rounded(.up)))) s left" }
        let minutes = Int((s / 60).rounded(.up))
        if minutes < 60 { return "\(minutes) min left" }
        return String(format: "%d h %02d min left", minutes / 60, minutes % 60)
    }
}

/// Turns byte counts into status lines at most four times a second (plus the first one).
public final class TransferReporter: @unchecked Sendable {
    let verb: String, name: String?, total: Int64?, start: Int64
    let emit: @Sendable (String) -> Void
    let began = Date()
    private let lock = NSLock()
    private var last: Date?

    public init(verb: String, name: String?, total: Int64?, start: Int64 = 0, emit: @escaping @Sendable (String) -> Void) {
        self.verb = verb
        self.name = name
        self.total = total
        self.start = start
        self.emit = emit
    }

    public func update(_ sent: Int64, force: Bool = false) {
        let now = Date()
        let due = lock.withLock {
            guard force || last.map({ now.timeIntervalSince($0) >= 0.25 }) ?? true else { return false }
            last = now
            return true
        }
        guard due else { return }
        emit(TransferStatus.line(verb: verb, sent: sent, total: total, start: start,
                                 seconds: now.timeIntervalSince(began), name: name))
    }
}

// MARK: - can it be downloaded?

/// Whether a link's file can be fetched, for the link preview.
public enum LoRAAvailability: Equatable, Sendable {
    /// Already in `<models>/loras` (local) or the server's uploads (cloud).
    case stored
    /// Downloadable; `withKey` when only with the site's API key.
    case available(withKey: Bool)
    case needsKey(LoRARemote.Host)
    case unavailable(String)
}

extension LoRADownloads {
    /// The file this link downloads to, when it's already there (and matches the site's hash, if it publishes one).
    public static func existing(_ remote: LoRARemote, models: URL = ModelPaths.directory) -> URL? {
        let dest = folder(models: models).appendingPathComponent(remote.cacheName)
        guard FileManager.default.fileExists(atPath: dest.path) else { return nil }
        guard let want = remote.sha256 else { return remote.bytes == nil || fileSize(dest) == remote.bytes ? dest : nil }
        let recorded = (try? String(contentsOf: dest.appendingPathExtension("sha256"), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return recorded == want ? dest : nil
    }

    /// Asks for the file's first bytes (without the key first, then with it if the site wants one) and checks they
    /// start a safetensors file. Downloads nothing more.
    public static func probe(_ remote: LoRARemote, models: URL = ModelPaths.directory,
                             session: URLSession = .shared) async -> LoRAAvailability {
        if existing(remote, models: models) != nil { return .stored }
        func attempt(_ withKey: Bool) async throws -> (Int, Data) {
            var r = URLRequest(url: remote.downloadURL)
            r.setValue("bytes=0-15", forHTTPHeaderField: "Range")
            if withKey { LoRASources.authorize(&r, remote.host) }
            let (bytes, response) = try await session.bytes(for: r, delegate: KeyOnlyToSameHost())
            defer { bytes.task.cancel() }
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            var head = Data()
            if code == 200 || code == 206 {
                for try await b in bytes {
                    head.append(b)
                    if head.count >= 16 { break }
                }
            }
            return (code, head)
        }
        do {
            var (code, head) = try await attempt(false)
            var withKey = false
            if code == 401 || code == 403 {
                guard LoRASources.tokenProvider(remote.host) != nil else { return .needsKey(remote.host) }
                (code, head) = try await attempt(true)
                withKey = true
                if code == 401 || code == 403 { return .unavailable("the site refused your API key") }
            }
            guard code == 200 || code == 206 else { return .unavailable("HTTP \(code) from \(remote.downloadURL.host ?? "")") }
            guard LoRASources.looksLikeSafetensors(head) else { return .unavailable("the link isn't a .safetensors file") }
            return .available(withKey: withKey)
        } catch {
            return .unavailable(error.localizedDescription)
        }
    }
}

/// Drops the Authorization header when a redirect leaves the site.
final class KeyOnlyToSameHost: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        var next = request
        if request.url?.host != task.originalRequest?.url?.host { next.setValue(nil, forHTTPHeaderField: "Authorization") }
        return next
    }
}

// MARK: - local downloads

/// Downloads a LoRA into `<models>/loras` (or finds it there from an earlier run): resumable, checked against the
/// site's SHA-256 (remembered in a `.sha256` file beside it) and the safetensors header.
public enum LoRADownloads {
    public static func folder(models: URL) -> URL { models.appendingPathComponent("loras") }

    /// `into`: another folder than `<models>/loras` (a fine-tune's checkpoint, downloaded before converting).
    /// A Wan 2.2 pair downloads both files (the companion next to it, under its own cache name); returns the first.
    public static func fetch(_ remote: LoRARemote, models: URL = ModelPaths.directory, into: URL? = nil,
                             status: @escaping @Sendable (String) -> Void) async throws -> URL {
        let url = try await fetchOne(remote, models: models, into: into, status: status)
        if let other = remote.companionRemote {
            _ = try await fetchOne(other, models: models, into: into, status: status)
        }
        return url
    }

    static func fetchOne(_ remote: LoRARemote, models: URL, into: URL?,
                         status: @escaping @Sendable (String) -> Void) async throws -> URL {
        let dir = into ?? folder(models: models)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dest = dir.appendingPathComponent(remote.cacheName)
        let sidecar = dest.appendingPathExtension("sha256")
        if FileManager.default.fileExists(atPath: dest.path) {
            let recorded = (try? String(contentsOf: sidecar, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let want = remote.sha256 else {
                // No published hash (a direct link): the same size is good enough.
                if remote.bytes == nil || fileSize(dest) == remote.bytes { return dest }
                try? FileManager.default.removeItem(at: dest)
                return try await download(remote, to: dest, sidecar: sidecar, status: status)
            }
            if recorded == want { return dest }
            status("Checking \(remote.fileName)")
            let have = try await Task.detached { try ModelDownloader.sha256(of: dest) }.value
            if have == want {
                try? have.write(to: sidecar, atomically: true, encoding: .utf8)
                return dest
            }
            try? FileManager.default.removeItem(at: dest)
        }
        return try await download(remote, to: dest, sidecar: sidecar, status: status)
    }

    static func fileSize(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1)
    }

    static func download(_ remote: LoRARemote, to dest: URL, sidecar: URL,
                         status: @escaping @Sendable (String) -> Void) async throws -> URL {
        let partial = dest.appendingPathExtension("partial")
        var attempt = 0
        var authorized = false  // the site's key only once it asks for one (Civitai does for most files)
        while true {
            let offset = FileManager.default.fileExists(atPath: partial.path) ? max(0, fileSize(partial)) : 0
            do {
                try await ChunkDownload(remote: remote, file: partial, offset: offset, authorized: authorized,
                                        status: status).run()
                break
            } catch LoRASourceError.needsKey(let host) where !authorized {
                guard LoRASources.tokenProvider(host) != nil else { throw LoRASourceError.needsKey(host) }
                authorized = true
            } catch let e as LoRASourceError {
                throw e
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if (error as? URLError)?.code == .cancelled { throw CancellationError() }
                attempt += 1
                if attempt >= 4 { throw error }
                try await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)
            }
        }
        let handle = try FileHandle(forReadingFrom: partial)
        let head = try handle.read(upToCount: 9) ?? Data()
        try? handle.close()
        guard LoRASources.looksLikeSafetensors(head) else {
            try? FileManager.default.removeItem(at: partial)
            throw LoRASourceError.notSafetensors(remote.fileName)
        }
        status("Checking \(remote.fileName)")
        let sha = try await Task.detached { try ModelDownloader.sha256(of: partial) }.value
        if let want = remote.sha256, want != sha {
            try? FileManager.default.removeItem(at: partial)
            throw LoRASourceError.hashMismatch(remote.fileName)
        }
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: partial, to: dest)
        try? sha.write(to: sidecar, atomically: true, encoding: .utf8)
        return dest
    }
}

/// One streamed GET into a file (appending from `offset` with a Range request), reporting progress. Drops the
/// Authorization header when a redirect leaves the site (Civitai and Hugging Face hand out signed CDN links).
final class ChunkDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let remote: LoRARemote, file: URL, offset: Int64, authorized: Bool
    let status: @Sendable (String) -> Void
    private var handle: FileHandle?
    private var written: Int64 = 0
    private var reporter: TransferReporter?
    private var continuation: CheckedContinuation<Void, Error>?
    private var failure: Error?

    init(remote: LoRARemote, file: URL, offset: Int64, authorized: Bool, status: @escaping @Sendable (String) -> Void) {
        self.remote = remote
        self.file = file
        self.offset = offset
        self.authorized = authorized
        self.status = status
    }

    func run() async throws {
        var r = URLRequest(url: remote.downloadURL)
        if authorized { LoRASources.authorize(&r, remote.host) }
        if offset > 0 { r.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: r)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                self.continuation = c
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        var next = request
        if request.url?.host != task.originalRequest?.url?.host { next.setValue(nil, forHTTPHeaderField: "Authorization") }
        completionHandler(next)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        do {
            switch code {
            case 200, 206:
                let resumed = code == 206 && offset > 0
                if !resumed { FileManager.default.createFile(atPath: file.path, contents: nil) }
                let h = try FileHandle(forWritingTo: file)
                if resumed { try h.seekToEnd() } else { try h.truncate(atOffset: 0) }
                handle = h
                written = resumed ? offset : 0
                let expected = response.expectedContentLength > 0 ? response.expectedContentLength + written : nil
                reporter = TransferReporter(verb: "Downloading", name: remote.fileName, total: remote.bytes ?? expected,
                                            start: written, emit: status)
                reporter?.update(written, force: true)
                completionHandler(.allow)
            case 401, 403:
                failure = LoRASourceError.needsKey(remote.host)
                completionHandler(.cancel)
            default:
                failure = LoRASourceError.http(code, remote.downloadURL.host ?? "")
                completionHandler(.cancel)
            }
        } catch {
            failure = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle?.write(contentsOf: data)
            written += Int64(data.count)
            reporter?.update(written)
        } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        handle = nil
        if let e = failure ?? error { continuation?.resume(throwing: e) } else { continuation?.resume() }
        continuation = nil
    }
}
