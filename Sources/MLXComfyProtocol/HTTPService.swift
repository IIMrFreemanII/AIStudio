import Foundation

/// REST plumbing shared by the clients of the Mac server (``RemoteClient``) and the GPU server: bearer requests,
/// status checks, and content-addressed uploads and downloads that stream from and to disk and resume after a
/// dropped connection. One long-lived `URLSession` per server.
public struct HTTPService: Sendable {
    public let baseURL: URL
    public let token: String
    public let session: URLSession

    public init(baseURL: URL, token: String, session: URLSession? = nil) {
        self.baseURL = baseURL
        self.token = token
        self.session = session ?? Self.makeSession()
    }

    public static func makeSession() -> URLSession {
        let cfg = URLSessionConfiguration.default
        cfg.httpMaximumConnectionsPerHost = 6
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 24 * 3600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        return URLSession(configuration: cfg)
    }

    public func request(_ path: String, method: String = "GET") -> URLRequest {
        var r = URLRequest(url: baseURL.appendingPathComponent(path))
        r.httpMethod = method
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return r
    }

    public func check(_ response: URLResponse, _ data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw RemoteClient.ClientError.http(http.statusCode, Self.errorText(data))
        }
    }

    /// The `error` of a JSON error body, else the body's first 500 bytes.
    static func errorText(_ data: Data?) -> String {
        guard let data, !data.isEmpty else { return "" }
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let e = obj["error"] as? String {
            return e
        }
        return String(decoding: data.prefix(500), as: UTF8.self)
    }

    public func getJSON<T: Decodable>(_ path: String, as: T.Type = T.self) async throws -> T {
        let (data, response) = try await session.data(for: request(path))
        try check(response, data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// POSTs a JSON body (or none) and decodes the JSON reply.
    public func postJSON<T: Decodable>(_ path: String, body: Data? = nil, as: T.Type = T.self) async throws -> T {
        var r = request(path, method: "POST")
        if let body {
            r.httpBody = body
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await session.data(for: r)
        try check(response, data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Uploads `file` as `v1/files/<name>` unless the server has it. Streams from disk (constant memory) and resumes a
    /// partial upload from where the server says it stopped. `progress(sent, total)` counts bytes the server already
    /// has from a broken earlier attempt; `start` is where this attempt began. Returns whether anything was sent.
    @discardableResult
    public func upload(name: String, file: URL, bytes: Int,
                       progress: (@Sendable (_ sent: Int64, _ total: Int64, _ start: Int64) -> Void)? = nil) async throws -> Bool {
        var attempt = 0
        while true {
            let (_, head) = try await session.data(for: request("v1/files/\(name)", method: "HEAD"))
            let status = (head as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 { return attempt > 0 }
            if status == 401 { throw RemoteClient.ClientError.http(401, "") }
            let offset = ((head as? HTTPURLResponse)?.value(forHTTPHeaderField: "X-Upload-Offset")).flatMap(Int.init) ?? 0
            var r = request("v1/files/\(name)", method: "PUT")
            r.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            r.setValue(String(bytes - offset), forHTTPHeaderField: "Content-Length")
            if offset > 0 {
                r.setValue("bytes \(offset)-\(bytes - 1)/\(bytes)", forHTTPHeaderField: "Content-Range")
            }
            guard let stream = InputStream(url: file) else { throw CocoaError(.fileReadNoSuchFile) }
            if offset > 0 { stream.setProperty(NSNumber(value: offset), forKey: .fileCurrentOffsetKey) }
            r.httpBodyStream = stream
            do {
                progress?(Int64(offset), Int64(bytes), Int64(offset))
                let delegate = UploadProgress(file: file, offset: offset, total: bytes, progress: progress)
                let (data, response) = try await session.data(for: r, delegate: delegate)
                try check(response, data)
                progress?(Int64(bytes), Int64(bytes), Int64(offset))
                return true
            } catch let e as RemoteClient.ClientError {
                throw e
            } catch {
                // A dropped connection: ask the server how far it got and continue from there.
                attempt += 1
                if attempt >= 4 { throw error }
                try await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)
            }
        }
    }

    /// Downloads `path` to `destination` (replacing it), resuming after a dropped connection.
    public func download(_ path: String, to destination: URL) async throws {
        let req = request(path)
        var resumeData: Data?
        for attempt in 0..<4 {
            do {
                let (temp, response): (URL, URLResponse)
                if let resumeData { (temp, response) = try await session.download(resumeFrom: resumeData) }
                else { (temp, response) = try await session.download(for: req) }
                try check(response, nil)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                        withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: temp, to: destination)
                return
            } catch let e as RemoteClient.ClientError {
                throw e
            } catch {
                resumeData = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
                if attempt == 3 { throw error }
                try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 500_000_000)
            }
        }
    }
}

/// Reports an upload's bytes as they go out, and hands URLSession a fresh body stream if it must resend.
final class UploadProgress: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    let file: URL, offset: Int, total: Int
    let progress: (@Sendable (Int64, Int64, Int64) -> Void)?

    init(file: URL, offset: Int, total: Int, progress: (@Sendable (Int64, Int64, Int64) -> Void)?) {
        self.file = file
        self.offset = offset
        self.total = total
        self.progress = progress
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        progress?(Int64(offset) + totalBytesSent, Int64(total), Int64(offset))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, needNewBodyStream completionHandler: @escaping (InputStream?) -> Void) {
        let stream = InputStream(url: file)
        if offset > 0 { stream?.setProperty(NSNumber(value: offset), forKey: .fileCurrentOffsetKey) }
        completionHandler(stream)
    }
}
