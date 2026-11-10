import CryptoKit
import Foundation
import MLXComfyProtocol

/// What `GET /v1/info` of a GPU server (server/gpu) says about itself. Asking never starts a GPU.
public struct GPUServerInfo: Codable, Sendable, Equatable {
    public struct Task: Codable, Sendable, Equatable, Identifiable {
        public var id: String
        public var title: String
        public var kind: String
        public var gpu: String
        public var pricePerSecond: Double?
        public var installed: Bool
        public var inputs: [String]
        /// Inputs the task can use when connected (a voice to clone).
        public var optionalInputs: [String]?
        public var params: [String: Param]?
    }

    /// A task parameter's type and limits (its default is the node's business).
    public struct Param: Codable, Sendable, Equatable {
        public var type: String
        public var min: Double?
        public var max: Double?
        /// For strings: the values the server accepts (its built-in voices…).
        public var choices: [String]?
    }

    public var server: String
    public var version: String
    public var api: Int
    public var runner: String
    public var idleSeconds: Int?
    /// Fine-tunes installed from links, per family ("flux.1"): what a task's `model` can pick.
    public var variants: [String: [GPUVariant]]?
    public var tasks: [Task]

    public func task(_ id: String) -> Task? { tasks.first { $0.id == id } }
}

/// A job as the server describes it (`POST /v1/jobs`, `GET /v1/jobs/{id}`).
public struct GPUJob: Codable, Sendable {
    public var id: String
    public var run: UInt32
    public var task: String
    public var state: String
    public var gpuSeconds: Double?
    public var loadSeconds: Double?
    public var costUSD: Double?
    public var error: String?
}

/// A download the GPU server runs for a client (`/v1/fetch`).
public struct GPUFetch: Codable, Sendable {
    public var id: String
    public var state: String  // downloading, done, failed, cancelled
    public var bytes: Int64
    public var total: Int64?
    public var name: String?
    public var error: String?
}

/// A fine-tune installed on the server (`/v1/models`, `/v1/info`).
public struct GPUVariant: Codable, Sendable, Equatable {
    public var id: String
    public var name: String
    public var bytes: Int64?
}

/// Installed fine-tunes and the imports still running ("family/id" → state).
public struct GPUModelListing: Codable, Sendable {
    public struct Import: Codable, Sendable, Equatable {
        public var state: String  // starting, downloading, converting, failed
        public var bytes: Int64?
        public var total: Int64?
        public var error: String?
        /// A fine-tune of several files (Wan 2.2's two experts): which one is downloading, of how many.
        public var file: Int?
        public var files: Int?
    }
    public var families: [String: [GPUVariant]]
    public var imports: [String: Import]
}

/// The server's answer to "could you download this?" (`/v1/fetch/check`).
public struct GPULinkCheck: Codable, Sendable {
    public var stored: Bool
    public var ok: Bool
    public var error: String?
}

/// How a job ended, and what it produced.
public struct GPUJobResult: Sendable {
    public var output: OutputDescriptor
    public var seconds: Double
    public var gpuSeconds: Double
    public var costUSD: Double?
}

/// A client of one GPU server: content-addressed uploads, jobs, their event stream (binary ``WireEvent`` records over
/// a WebSocket that reconnects and replays from the last record it saw), cancel, and the result download.
public final class GPUJobClient: @unchecked Sendable {
    public static let apiVersion = 1

    public enum JobError: Error, CustomStringConvertible {
        case failed(String)
        case lost
        case apiMismatch(Int)
        public var description: String {
            switch self {
            case .failed(let m): return m
            case .lost: return "the server no longer knows this job (it restarted); run it again"
            case .apiMismatch(let v):
                return "this GPU server speaks job API \(v); the app needs \(GPUJobClient.apiVersion) — update one of them"
            }
        }
    }

    public let http: HTTPService

    public init(baseURL: URL, token: String) {
        http = HTTPService(baseURL: baseURL, token: token)
    }

    public func info() async throws -> GPUServerInfo {
        let info: GPUServerInfo = try await http.getJSON("v1/info")
        guard info.api == Self.apiVersion else { throw JobError.apiMismatch(info.api) }
        return info
    }

    /// Uploads PNG bytes (an input image) unless the server has them; returns the server-side name.
    public func uploadPNG(_ data: Data) async throws -> String { try await upload(data, ext: "png") }

    /// Uploads a file's bytes unless the server has them (content-addressed); returns the server-side name
    /// `<sha256>.<ext>`.
    public func upload(_ data: Data, ext: String, progress: UploadProgressHandler? = nil) async throws -> String {
        let name = FileHasher.hex(SHA256.hash(data: data)) + "." + ext
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(name)")
        try data.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        try await http.upload(name: name, file: file, bytes: data.count, progress: progress)
        return name
    }

    /// `(sent, total, start)`: bytes on the server so far, the file's size, and where this attempt began.
    public typealias UploadProgressHandler = @Sendable (_ sent: Int64, _ total: Int64, _ start: Int64) -> Void

    /// Uploads a file from disk unless the server has it, streamed and resumable (LoRAs run to hundreds of MB); returns
    /// the server-side name `<sha256>.<ext>`. The hash is remembered per path, size and modification date.
    public func uploadFile(_ url: URL, ext: String, progress: UploadProgressHandler? = nil) async throws -> String {
        let size = (try url.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        let sha = try await Task.detached { try Self.hasher.sha256(of: url) }.value
        let name = sha + "." + ext
        try await http.upload(name: name, file: url, bytes: size, progress: progress)
        return name
    }

    /// Whether the server already stores an upload of this name.
    public func has(_ name: String) async throws -> Bool {
        let (_, response) = try await http.session.data(for: http.request("v1/files/\(name)", method: "HEAD"))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 401 { throw RemoteClient.ClientError.http(401, "") }
        return code == 200
    }

    /// Removes an upload from the server (a LoRA removed from the library). False when it wasn't there.
    @discardableResult
    public func deleteFile(_ name: String) async throws -> Bool {
        let (data, response) = try await http.session.data(for: http.request("v1/files/\(name)", method: "DELETE"))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 { return false }
        try http.check(response, data)
        return true
    }

    // MARK: server-side downloads (LoRAs from Civitai / Hugging Face links)

    /// Has the server download a file (Civitai, Hugging Face) into its uploads. Answers at once with `done` when a
    /// known SHA-256 is already stored.
    public func fetch(url: URL, sha256: String?, bytes: Int64?, ext: String = "lora") async throws -> GPUFetch {
        var body: [String: Any] = ["url": url.absoluteString, "ext": ext]
        if let sha256 { body["sha256"] = sha256 }
        if let bytes { body["bytes"] = bytes }
        return try await http.postJSON("v1/fetch", body: try JSONSerialization.data(withJSONObject: body))
    }

    /// Whether the server could download a link (with its own keys), without downloading it: for link previews.
    public func checkLink(url: URL, sha256: String?) async throws -> GPULinkCheck {
        var body: [String: Any] = ["url": url.absoluteString]
        if let sha256 { body["sha256"] = sha256 }
        return try await http.postJSON("v1/fetch/check", body: try JSONSerialization.data(withJSONObject: body))
    }

    // MARK: fine-tunes (variants)

    /// Installs a fine-tune from a link on the server: a file (`url`, `sha256`) or a Hugging Face diffusers repo
    /// (`owner/name@revision`). Runs on the server; follow it with `models()`.
    /// One file of a fine-tune; `role` says which Wan 2.2 expert it replaces (`high` / `low`).
    public struct ImportFile: Sendable {
        public var url: URL
        public var sha256: String?
        public var role: String?
        public init(url: URL, sha256: String?, role: String? = nil) { self.url = url; self.sha256 = sha256; self.role = role }
    }

    public func importModel(family: String, id: String, name: String, url: URL?, sha256: String?,
                            repo: String?) async throws -> GPUModelListing.Import {
        try await importModel(family: family, id: id, name: name,
                              files: url.map { [ImportFile(url: $0, sha256: sha256)] } ?? [], repo: repo)
    }

    public func importModel(family: String, id: String, name: String, files: [ImportFile],
                            repo: String?) async throws -> GPUModelListing.Import {
        var body: [String: Any] = ["family": family, "id": id, "name": name]
        if !files.isEmpty {
            body["files"] = files.map { f -> [String: Any] in
                var d: [String: Any] = ["url": f.url.absoluteString]
                if let s = f.sha256 { d["sha256"] = s }
                if let r = f.role { d["role"] = r }
                return d
            }
        }
        if let repo { body["repo"] = repo }
        return try await http.postJSON("v1/models/import", body: try JSONSerialization.data(withJSONObject: body))
    }

    public func models() async throws -> GPUModelListing { try await http.getJSON("v1/models") }

    @discardableResult
    public func deleteModel(family: String, id: String) async throws -> Bool {
        let (data, response) = try await http.session.data(for: http.request("v1/models/\(family)/\(id)", method: "DELETE"))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 { return false }
        try http.check(response, data)
        return true
    }

    public func fetchStatus(_ id: String) async throws -> GPUFetch { try await http.getJSON("v1/fetch/\(id)") }

    public func cancelFetch(_ id: String) async {
        _ = try? await http.session.data(for: http.request("v1/fetch/\(id)", method: "DELETE"))
    }

    private static let hasher = FileHasher(cacheURL: nil)

    public func submit(task: String, params: [String: Any], inputs: [String: String] = [:]) async throws -> GPUJob {
        let body = try JSONSerialization.data(withJSONObject: ["task": task, "params": params, "inputs": inputs])
        return try await http.postJSON("v1/jobs", body: body)
    }

    public func job(_ id: String) async throws -> GPUJob { try await http.getJSON("v1/jobs/\(id)") }

    public func cancel(_ id: String) async {
        _ = try? await http.postJSON("v1/jobs/\(id)/cancel", as: GPUJob.self)
    }

    public func downloadResult(_ id: String, to destination: URL) async throws {
        try await http.download("v1/jobs/\(id)/result", to: destination)
    }

    /// Streams the job's records to `onEvent` until it finishes; returns the result, or throws its failure.
    /// A dropped socket reconnects (up to 6 times, backing off) and resumes after the last record seen.
    public func follow(_ job: GPUJob, onEvent: @escaping (WireEvent) -> Void) async throws -> GPUJobResult {
        var lastSeq: UInt32 = 0
        var output: OutputDescriptor?
        var failures = 0
        while true {
            try Task.checkCancellation()
            do {
                if let end = try await stream(job.id, after: lastSeq, onRecord: { e in
                    lastSeq = max(lastSeq, e.seq)
                    if e.kind == .outputReady { output = try? JSONDecoder().decode(OutputDescriptor.self, from: e.payload) }
                    onEvent(e)
                }) {
                    switch end.kind {
                    case .finished:
                        guard let output else { throw JobError.failed("the server finished without a result") }
                        let values = Self.doubles(end.payload)
                        return GPUJobResult(output: output, seconds: values[safe: 0] ?? 0, gpuSeconds: values[safe: 1] ?? 0,
                                            costUSD: (values[safe: 2] ?? -1) < 0 ? nil : values[safe: 2])
                    case .cancelled: throw CancellationError()
                    default: throw JobError.failed(end.message.isEmpty ? "the job failed" : end.message)
                    }
                }
                failures = 0  // closed without a terminal record (server restarting?): reconnect
            } catch let e as JobError {
                throw e
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                failures += 1
                if failures > 6 { throw error }
            }
            try await Task.sleep(nanoseconds: UInt64(min(failures, 5)) * 700_000_000 + 100_000_000)
        }
    }

    /// One socket session; returns the terminal record, or nil if the socket closed before one.
    private func stream(_ id: String, after: UInt32, onRecord: (WireEvent) -> Void) async throws -> WireEvent? {
        var comps = URLComponents(url: http.baseURL.appendingPathComponent("v1/jobs/\(id)/events"),
                                  resolvingAgainstBaseURL: false)!
        comps.scheme = comps.scheme == "https" ? "wss" : "ws"
        comps.queryItems = [URLQueryItem(name: "after", value: String(after))]
        var request = URLRequest(url: comps.url!)
        request.setValue("Bearer \(http.token)", forHTTPHeaderField: "Authorization")
        let socket = http.session.webSocketTask(with: request)
        socket.maximumMessageSize = 32 << 20
        socket.resume()
        return try await withTaskCancellationHandler {
            defer { socket.cancel(with: .normalClosure, reason: nil) }
            while true {
                let message: URLSessionWebSocketTask.Message
                do {
                    message = try await socket.receive()
                } catch {
                    if socket.closeCode.rawValue == 4404 { throw JobError.lost }
                    if socket.closeCode.rawValue == 4401 { throw RemoteClient.ClientError.http(401, "") }
                    if socket.closeCode == .normalClosure { return nil }
                    throw error
                }
                guard case .data(let data) = message else { continue }
                for e in try WireEvent.decodeAll(data) {
                    onRecord(e)
                    if [.finished, .cancelled, .runFailed].contains(e.kind) { return e }
                }
            }
        } onCancel: {
            socket.cancel(with: .goingAway, reason: nil)
        }
    }

    static func doubles(_ payload: Data) -> [Double] {
        stride(from: 0, to: payload.count - 7, by: 8).map { off in
            payload.withUnsafeBytes { Double(bitPattern: UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: off, as: UInt64.self))) }
        }
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
