import CryptoKit
import Foundation

/// A model the app knows how to fetch from Hugging Face.
public struct CatalogEntry: Codable, Hashable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable, CaseIterable { case image, video, music, speech, sound }

    public var id: String
    public var name: String
    public var kind: Kind
    public var repo: String
    public var revision: String
    /// Glob patterns of files to fetch (all files when empty), and patterns to skip.
    public var include: [String]
    public var exclude: [String]
    /// Rough size for the list before the file list is fetched.
    public var approxGB: Double
    public var license: String
    public var licenseURL: String
    /// Needs a Hugging Face token (and the terms accepted on the model page).
    public var gated: Bool
    /// Folder name under the models directory.
    public var folder: String
    /// Whether this version of the app can run it (others are listed as coming later).
    public var supported: Bool
    public var notes: String
    /// The loader class its folder declares (`model_index.json` `_class_name`), to recognise copies elsewhere.
    public var className: String?
    /// More repos fetched into subfolders of the same folder (e.g. a text encoder), each under its own license.
    public var extraParts: [CatalogPart]
    /// Written as `mlxcomfy.json` after the download, for repos that don't declare a loader themselves.
    public var manifest: [String: String]?

    public init(id: String, name: String, kind: Kind, repo: String, revision: String = "main", include: [String] = [],
                exclude: [String] = [], approxGB: Double, license: String, licenseURL: String, gated: Bool = false,
                folder: String, supported: Bool, notes: String = "", className: String? = nil,
                extraParts: [CatalogPart] = [], manifest: [String: String]? = nil) {
        self.id = id; self.name = name; self.kind = kind; self.repo = repo; self.revision = revision
        self.include = include; self.exclude = exclude; self.approxGB = approxGB; self.license = license
        self.licenseURL = licenseURL; self.gated = gated; self.folder = folder; self.supported = supported
        self.notes = notes
        self.className = className
        self.extraParts = extraParts
        self.manifest = manifest
    }

    /// The main repo followed by the extra parts.
    public var parts: [CatalogPart] {
        [CatalogPart(repo: repo, revision: revision, include: include, exclude: exclude, subfolder: "",
                     license: license, licenseURL: licenseURL, gated: gated)] + extraParts
    }
}

/// One repo of a catalog entry, downloaded into `subfolder` ("" = the entry's folder itself).
public struct CatalogPart: Codable, Hashable, Sendable {
    public var repo: String
    public var revision: String
    public var include: [String]
    public var exclude: [String]
    public var subfolder: String
    public var license: String
    public var licenseURL: String
    public var gated: Bool

    public init(repo: String, revision: String = "main", include: [String] = [], exclude: [String] = [],
                subfolder: String, license: String, licenseURL: String, gated: Bool = false) {
        self.repo = repo; self.revision = revision; self.include = include; self.exclude = exclude
        self.subfolder = subfolder; self.license = license; self.licenseURL = licenseURL; self.gated = gated
    }
}

public enum ModelCatalog {
    /// Repos for models not supported yet are confirmed when their port lands.
    public static let entries: [CatalogEntry] = [
        CatalogEntry(id: "qwen-image-2.1-turbo", name: "Qwen-Image 2.1 Turbo (8-bit MLX)", kind: .image,
                     repo: "abenzerps/Qwen-Image-2.1-Turbo-Quantized", exclude: ["*.md", ".gitattributes"],
                     approxGB: 24, license: "Apache-2.0",
                     licenseURL: "https://www.apache.org/licenses/LICENSE-2.0", folder: "qwen-turbo-mlx",
                     supported: true, notes: "Text-to-image, editing, inpainting and outpainting.",
                     className: "QwenImage21Pipeline"),
        CatalogEntry(id: "ltx-2.3", name: "LTX-2.3 Distilled (video + audio, 4-bit MLX)", kind: .video,
                     repo: "dgrauet/ltx-2.3-mlx-q4",
                     include: ["*.json", "LICENSE", "README.md", "transformer-distilled-1.1.safetensors",
                               "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
                               "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"],
                     approxGB: 28.6, license: "LTX-2 Community License",
                     licenseURL: "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2",
                     folder: "ltx-2.3", supported: true,
                     notes: "Text- and image-to-video with a synchronized soundtrack, up to 10 s. Includes the Gemma 3 12B text encoder.",
                     className: "LTX2Pipeline",
                     extraParts: [CatalogPart(repo: "mlx-community/gemma-3-12b-it-4bit", exclude: [".gitattributes"],
                                              subfolder: "text_encoder", license: "Gemma Terms of Use",
                                              licenseURL: "https://ai.google.dev/gemma/terms")],
                     manifest: ["class": "LTX2Pipeline"]),
        CatalogEntry(id: "stable-audio-open-small", name: "Stable Audio Open Small (sound effects)", kind: .sound,
                     repo: "stabilityai/stable-audio-open-small",
                     include: ["model.safetensors", "model_config.json", "LICENSE", "README.md"],
                     exclude: ["tflite-arm-path/*"], approxGB: 2.6, license: "Stability AI Community License",
                     licenseURL: "https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE",
                     gated: true, folder: "stable-audio-open-small", supported: true,
                     notes: "Sound effects, ambience and loops up to about 11 s, stereo 44.1 kHz. Includes the T5-base text encoder.",
                     className: "StableAudioPipeline",
                     extraParts: [CatalogPart(repo: "google-t5/t5-base",
                                              include: ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"],
                                              subfolder: "t5", license: "Apache-2.0",
                                              licenseURL: "https://www.apache.org/licenses/LICENSE-2.0")],
                     manifest: ["class": "StableAudioPipeline"]),
        CatalogEntry(id: "ace-step-1.5", name: "ACE-Step 1.5 Turbo (music)", kind: .music, repo: "ACE-Step/Ace-Step1.5",
                     revision: "19671f406d603126926c1b7e2adc169acbcade22",
                     include: ["README.md", "vae/*", "acestep-v15-turbo/config.json", "acestep-v15-turbo/model.safetensors",
                               "acestep-v15-turbo/silence_latent.pt", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*"],
                     exclude: ["*.py"], approxGB: 10.3, license: "MIT",
                     licenseURL: "https://huggingface.co/ACE-Step/Ace-Step1.5", folder: "ace-step-1.5", supported: true,
                     notes: "Songs with or without lyrics, 10 s to 4 min, stereo 48 kHz. Includes the 1.7B song-planner LM and the Qwen3 text encoder.",
                     className: "ACEStepPipeline", manifest: ["class": "ACEStepPipeline"]),
        CatalogEntry(id: "kokoro-82m", name: "Kokoro 82M (speech)", kind: .speech, repo: "hexgrad/Kokoro-82M",
                     revision: "f3ff3571791e39611d31c381e3a41a3af07b4987",
                     include: ["config.json", "kokoro-v1_0.pth", "README.md", "VOICES.md", "voices/a*.pt", "voices/b*.pt"],
                     approxGB: 0.34, license: "Apache-2.0", licenseURL: "https://www.apache.org/licenses/LICENSE-2.0",
                     folder: "kokoro-82m", supported: true,
                     notes: "English narration (American and British) with 28 voices, 24 kHz. Pronunciation runs on-device in Swift.",
                     className: "KokoroPipeline", manifest: ["class": "KokoroPipeline"]),
    ]
}

/// One file of a repo.
public struct RemoteFile: Codable, Hashable, Sendable {
    public var path: String
    public var size: Int64
    /// SHA-256 for LFS files (the large ones); nil for small text files.
    public var sha256: String?
    public init(path: String, size: Int64, sha256: String? = nil) { self.path = path; self.size = size; self.sha256 = sha256 }
}

public enum DownloadError: Error, CustomStringConvertible, Equatable {
    case http(Int, String)
    case unauthorized(String)
    case rangeIgnored
    case sizeMismatch(String, expected: Int64, got: Int64)
    case checksumMismatch(String)
    case notEnoughSpace(needed: Int64, free: Int64)

    public var description: String {
        switch self {
        case .http(let c, let p): return "HTTP \(c) for \(p)"
        case .unauthorized(let r): return "\(r) needs a Hugging Face token, and its terms accepted on the model page"
        case .rangeIgnored: return "the server doesn't support resuming"
        case .sizeMismatch(let p, let e, let g): return "\(p): expected \(e) bytes, got \(g)"
        case .checksumMismatch(let p): return "\(p) is corrupted (checksum mismatch); it will be downloaded again"
        case .notEnoughSpace(let n, let f):
            return String(format: "not enough disk space: %.1f GB needed, %.1f GB free", Double(n) / 1e9, Double(f) / 1e9)
        }
    }
}

/// Where file bytes come from: Hugging Face over HTTP, or a local folder (tests, mirrors).
public protocol ByteSource: Sendable {
    /// The repo's files.
    func list() async throws -> [RemoteFile]
    /// Bytes of `file` starting at `offset`, in chunks.
    func open(_ file: RemoteFile, from offset: Int64) -> AsyncThrowingStream<Data, Error>
}

public enum HFAPI {
    /// Parses `GET /api/models/{repo}/tree/{rev}?recursive=true`.
    public static func parseTree(_ data: Data) throws -> [RemoteFile] {
        let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        return items.compactMap { item in
            guard item["type"] as? String == "file", let path = item["path"] as? String else { return nil }
            let lfs = item["lfs"] as? [String: Any]
            let size = (lfs?["size"] as? NSNumber)?.int64Value ?? (item["size"] as? NSNumber)?.int64Value ?? 0
            return RemoteFile(path: path, size: size, sha256: lfs?["oid"] as? String)
        }
    }

    /// Files matching `include` (all when empty) and not `exclude`, by shell-style globs on the path.
    public static func select(_ files: [RemoteFile], include: [String], exclude: [String]) -> [RemoteFile] {
        files.filter { f in
            (include.isEmpty || include.contains { glob($0, f.path) }) && !exclude.contains { glob($0, f.path) }
        }
    }

    static func glob(_ pattern: String, _ path: String) -> Bool {
        fnmatch(pattern, path, 0) == 0 || fnmatch(pattern, (path as NSString).lastPathComponent, 0) == 0
    }
}

/// Hugging Face over HTTP, with an optional token, streaming each file with a `Range` header to resume.
public final class HTTPByteSource: ByteSource, @unchecked Sendable {
    let repo: String
    let revision: String
    let token: String?
    let session: URLSession

    public init(repo: String, revision: String = "main", token: String? = nil) {
        self.repo = repo
        self.revision = revision
        self.token = token
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        session = URLSession(configuration: config)
    }

    func request(_ url: URL) -> URLRequest {
        var r = URLRequest(url: url)
        if let token, !token.isEmpty { r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        return r
    }

    public func list() async throws -> [RemoteFile] {
        let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)?recursive=true")!
        let (data, response) = try await session.data(for: request(url))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 401 || code == 403 { throw DownloadError.unauthorized(repo) }
        guard code == 200 else { throw DownloadError.http(code, repo) }
        return try HFAPI.parseTree(data)
    }

    public func open(_ file: RemoteFile, from offset: Int64) -> AsyncThrowingStream<Data, Error> {
        let path = file.path.split(separator: "/").map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        var req = request(URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")!)
        if offset > 0 { req.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        let repo = repo
        return AsyncThrowingStream { cont in
            let delegate = StreamingDelegate(cont: cont, repo: repo, path: file.path, resuming: offset > 0)
            // One session per file: the delegate is retained by the session until it's invalidated.
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            let task = session.dataTask(with: req)
            cont.onTermination = { _ in
                task.cancel()
                session.invalidateAndCancel()
            }
            task.resume()
        }
    }
}

/// Forwards a data task's chunks to a stream, checking the status first.
final class StreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let cont: AsyncThrowingStream<Data, Error>.Continuation
    let repo: String
    let path: String
    let resuming: Bool

    init(cont: AsyncThrowingStream<Data, Error>.Continuation, repo: String, path: String, resuming: Bool) {
        self.cont = cont
        self.repo = repo
        self.path = path
        self.resuming = resuming
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        var error: DownloadError?
        if code == 401 || code == 403 { error = .unauthorized(repo) }
        else if resuming && code == 200 { error = .rangeIgnored }
        else if code != 200 && code != 206 { error = .http(code, path) }
        if let error {
            cont.finish(throwing: error)
            completionHandler(.cancel)
        } else {
            completionHandler(.allow)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) { cont.yield(data) }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error, (error as? URLError)?.code != .cancelled { cont.finish(throwing: error) } else { cont.finish() }
        session.finishTasksAndInvalidate()
    }
}

/// A local folder laid out like a repo (for tests and offline mirrors).
public struct FolderByteSource: ByteSource {
    public let root: URL
    public init(root: URL) { self.root = root }

    public func list() async throws -> [RemoteFile] {
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return [] }
        var out: [RemoteFile] = []
        for case let f as URL in e where (try? f.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            let rel = String(f.standardizedFileURL.path.dropFirst(root.standardizedFileURL.path.count + 1))
            let size = Int64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            out.append(RemoteFile(path: rel, size: size, sha256: try? ModelDownloader.sha256(of: f)))
        }
        return out.sorted { $0.path < $1.path }
    }

    public func open(_ file: RemoteFile, from offset: Int64) -> AsyncThrowingStream<Data, Error> {
        let url = root.appendingPathComponent(file.path)
        return AsyncThrowingStream { cont in
            do {
                let h = try FileHandle(forReadingFrom: url)
                try h.seek(toOffset: UInt64(offset))
                while let d = try h.read(upToCount: 1 << 16), !d.isEmpty { cont.yield(d) }
                try h.close()
                cont.finish()
            } catch {
                cont.finish(throwing: error)
            }
        }
    }
}

/// Downloads files into a folder: `<file>.part` while in progress (resumed from its size next time), verified
/// against the expected size and SHA-256, then moved into place. Finished files are skipped.
public enum ModelDownloader {
    public struct Progress: Sendable {
        public var done: Int64
        public var total: Int64
        public var file: String
        public var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
        public init(done: Int64, total: Int64, file: String) { self.done = done; self.total = total; self.file = file }
    }

    public static func freeSpace(at url: URL) -> Int64 {
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path) && probe.path != "/" { probe.deleteLastPathComponent() }
        let v = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    /// Bytes still to fetch (files already complete, and the parts already on disk, don't count).
    public static func remainingBytes(_ files: [RemoteFile], in dir: URL) -> Int64 {
        files.reduce(0) { sum, f in
            let dest = dir.appendingPathComponent(f.path)
            if size(of: dest) == f.size { return sum }
            return sum + max(0, f.size - max(0, size(of: partURL(dest))))
        }
    }

    static func size(of url: URL) -> Int64 {
        Int64((try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? -1)
    }

    static func partURL(_ dest: URL) -> URL { dest.appendingPathExtension("part") }

    public static func download(_ files: [RemoteFile], to dir: URL, from source: ByteSource,
                                progress: @escaping (Progress) -> Void = { _ in }) async throws {
        let total = files.reduce(0) { $0 + $1.size }
        let needed = remainingBytes(files, in: dir)
        let free = freeSpace(at: dir)
        if needed > free { throw DownloadError.notEnoughSpace(needed: needed, free: free) }
        var done = total - needed
        for f in files {
            try Task.checkCancellation()
            let dest = dir.appendingPathComponent(f.path)
            if size(of: dest) == f.size { continue }
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let part = partURL(dest)
            var offset = max(0, size(of: part))
            if offset > f.size { try? FileManager.default.removeItem(at: part); offset = 0 }
            if offset == 0 { FileManager.default.createFile(atPath: part.path, contents: nil) }
            do {
                try await append(f, from: offset, to: part, source: source) { n in
                    done += n
                    progress(Progress(done: done, total: total, file: f.path))
                }
            } catch DownloadError.rangeIgnored {
                // Start this file over.
                done -= offset
                try? FileManager.default.removeItem(at: part)
                FileManager.default.createFile(atPath: part.path, contents: nil)
                try await append(f, from: 0, to: part, source: source) { n in
                    done += n
                    progress(Progress(done: done, total: total, file: f.path))
                }
            }
            let got = size(of: part)
            guard got == f.size else { throw DownloadError.sizeMismatch(f.path, expected: f.size, got: got) }
            if let expected = f.sha256, try sha256(of: part) != expected.lowercased() {
                try? FileManager.default.removeItem(at: part)
                throw DownloadError.checksumMismatch(f.path)
            }
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: part, to: dest)
        }
    }

    static func append(_ f: RemoteFile, from offset: Int64, to part: URL, source: ByteSource,
                       received: (Int64) -> Void) async throws {
        let handle = try FileHandle(forWritingTo: part)
        defer { try? handle.close() }
        try handle.seekToEnd()
        for try await chunk in source.open(f, from: offset) {
            try Task.checkCancellation()
            try handle.write(contentsOf: chunk)
            received(Int64(chunk.count))
        }
    }

    public static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let d = try h.read(upToCount: 8 << 20), !d.isEmpty { hasher.update(data: d) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
