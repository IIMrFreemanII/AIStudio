import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO
import MLX
import AIStudioProtocol
import MLXDiffusionCore
import NodesStandard

/// Nodes that run a model on a GPU server (server/gpu: Modal or a GPU box) and hand back ordinary images, video and
/// audio, so they mix with local nodes (a cloud Wan clip → local Save Video, a cloud song → local Mix Audio…).
public enum RemoteNodes {
    public static let all: [NodeImplementation.Type] = [
        RemoteWanTextToVideoNode.self, RemoteWanImageToVideoNode.self, RemoteFluxNode.self, RemoteQwenImageNode.self,
        RemoteTextToMusicNode.self, RemoteTextToSoundNode.self, RemoteTextToSpeechNode.self,
    ]

    /// Nodes that only prepare inputs for the job nodes above (no server, no job of their own).
    public static let helpers: [NodeImplementation.Type] = [RemoteLoRANode.self]

    /// The node types that need a GPU server (the app's server picker and presets look for these).
    public static var types: Set<String> { Set(all.map { $0.schema.type }) }

    /// LoRA slots on the server's FLUX and Qwen-Image tasks (`lora1` … `lora4`).
    public static let maxFluxLoRAs = 4

    /// The server task each remote node runs (`/v1/info` describes it: GPU, price, installed).
    public static let taskIDs: [String: String] = [
        "RemoteWanTextToVideo": "wan22-t2v", "RemoteWanImageToVideo": "wan22-i2v", "RemoteFlux": "flux1-dev",
        "RemoteQwenImage": "qwen-image", "RemoteTextToMusic": "ace-step-xl", "RemoteTextToSound": "stable-audio-3", "RemoteTextToSpeech": "qwen3-tts",
    ]

    public static func taskID(forNodeType type: String) -> String? { taskIDs[type] }

    /// Qwen3-TTS's built-in voices (the server's `/v1/info` has the authoritative list) and "designed": a voice
    /// made from the description.
    public static let speechVoices = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna",
                                      "Sohee", "designed"]
    public static let speechLanguages = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French",
                                         "Russian", "Portuguese", "Spanish", "Italian"]

    /// The standard nodes plus the remote ones.
    public static func makeRegistry() -> NodeRegistry {
        let r = StandardNodes.makeRegistry()
        r.register(all)
        r.register(helpers)
        return r
    }

    /// The library family each cloud image node's `model` and LoRAs come from.
    public static let libraryFamilies: [String: String] = [
        "RemoteFlux": "flux.1", "RemoteQwenImage": "qwen-image",
        "RemoteWanTextToVideo": "wan-2.2-t2v", "RemoteWanImageToVideo": "wan-2.2-i2v",
    ]

    /// The base model's name in a cloud node's `model` menu.
    public static func baseTitle(family: String) -> String {
        switch family {
        case "qwen-image": return "Qwen-Image (base)"
        case "wan-2.2-t2v": return "Wan 2.2 T2V (base)"
        case "wan-2.2-i2v": return "Wan 2.2 I2V (base)"
        default: return "FLUX.1 dev (base)"
        }
    }

    /// The `server` input every remote node has: a GPU server's id (empty: the first one).
    static let serverInput = InputSpec("server", .string, default: .string(""))
}

public enum RemoteNodeError: Error, CustomStringConvertible {
    case noServer(String)
    case taskMissing(String, String)
    case noLoRAFile
    case loraMissing(String)
    case tooManyLoRAs(Int)
    case fetchFailed(String, String)
    public var description: String {
        switch self {
        case .noLoRAFile: return "Pick a LoRA in the LoRA (Cloud GPU) node's menu (or add one: Models window › Add from Link)"
        case .loraMissing(let path): return "LoRA file not found: \(path)"
        case .fetchFailed(let name, let why): return "The server couldn't download \(name): \(why)"
        case .tooManyLoRAs(let n):
            return "\(n) LoRAs: a cloud model node takes up to \(RemoteNodes.maxFluxLoRAs)"

        case .noServer(let id):
            return id.isEmpty ? "No GPU server: add one in the Servers panel (+ › GPU server)"
                : "The GPU server this node uses was removed: pick another in the inspector"
        case .taskMissing(let server, let task):
            return "\(server) can't run \(task) yet: run its download step (see docs/remote-server.md)"
        }
    }
}

/// Runs one job and returns the downloaded result file.
func runRemoteJob(task: String, params: [String: Any], images: [String: MLXArray] = [:],
                  audios: [String: AudioClip] = [:], files: [String: URL] = [:], links: [String: LoRARemote] = [:],
                  inputs nodeInputs: NodeInputs, context: ExecutionContext) async throws -> (URL, OutputDescriptor.Kind) {
    let serverID = try nodeInputs.string("server")
    guard let endpoint = RemoteEndpoints.resolve(serverID) else { throw RemoteNodeError.noServer(serverID) }
    let client: GPUJobClient = try await context.resources.get(
        "gpu-client:\(endpoint.id):\(endpoint.baseURL.absoluteString):\(endpoint.token.hashValue)") {
        GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
    }

    var uploaded: [String: String] = [:]
    if !images.isEmpty || !audios.isEmpty || !files.isEmpty {
        context.status("Uploading")
        for (name, image) in images {
            uploaded[name] = try await client.upload(try pngData(image), ext: "png",
                                                     progress: uploadReporter("image", context))
        }
        for (name, clip) in audios {
            uploaded[name] = try await client.upload(MediaIO.wavData(clip), ext: "wav",
                                                     progress: uploadReporter("audio", context))
        }
        for (name, url) in files.sorted(by: { $0.key < $1.key }) {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if size > 50 << 20 { context.status("Checking \(url.lastPathComponent)") }  // hashing: once per file
            uploaded[name] = try await client.uploadFile(url, ext: "lora",
                                                         progress: uploadReporter(url.lastPathComponent, context))
        }
    }
    for (name, link) in links.sorted(by: { $0.key < $1.key }) {
        uploaded[name] = try await serverFetch(link, client: client, context: context)
    }
    context.status("Submitting")
    let job: GPUJob
    do {
        job = try await client.submit(task: task, params: params, inputs: uploaded)
    } catch RemoteClient.ClientError.http(503, let message) {
        throw RemoteNodeError.taskMissing(endpoint.name, message.isEmpty ? task : message)
    }

    let result: GPUJobResult = try await withTaskCancellationHandler {
        try await client.follow(job) { e in
            switch e.kind {
            case .nodeStatus: context.status(e.payload.isEmpty ? nil : e.message)
            case .progress: context.progress(step: Int(e.step), total: Int(e.total))
            case .preview:
                // A CGImage: the app shows it as is (no round trip through an array).
                let image = CGImageSourceCreateWithData(e.payload as CFData, nil)
                    .flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
                context.progress(step: Int(e.step), total: Int(e.total), preview: image)
            default: break
            }
        }
    } onCancel: {
        Task { await client.cancel(job.id) }
    }

    context.status("Downloading result")
    let ext = switch result.output.kind { case .video: "mp4"; case .audio: "wav"; case .image: "png" }
    let url = try StandardNodes.previewFolder().appendingPathComponent("remote-\(job.id).\(ext)")
    try await client.downloadResult(job.id, to: url)
    let load = (try? await client.job(job.id))?.loadSeconds ?? 0
    context.status(costLine(gpuSeconds: result.gpuSeconds, loadSeconds: load, cost: result.costUSD))
    return (url, result.output.kind)
}

/// Status lines for an upload ("Uploading 230 / 612 MB · 1.9 MB/s · 3 min left (name)"), at most four a second.
func uploadReporter(_ name: String, _ context: ExecutionContext) -> GPUJobClient.UploadProgressHandler {
    uploadReporter(name) { context.status($0) }
}

public func uploadReporter(_ name: String, status: @escaping @Sendable (String) -> Void) -> GPUJobClient.UploadProgressHandler {
    let box = ReporterBox()
    return { sent, total, start in
        guard total >= 1 << 20 else {  // small files go in a blink: no byte counts
            if sent == start { status("Uploading \(name)") }
            return
        }
        let r = box.reporter(total: total, start: start) {
            TransferReporter(verb: "Uploading", name: name, total: total, start: start, emit: status)
        }
        r.update(sent, force: sent >= total)
    }
}

/// One reporter per upload attempt (a resumed attempt starts its speed over).
final class ReporterBox: @unchecked Sendable {
    private let lock = NSLock()
    private var current: (start: Int64, reporter: TransferReporter)?

    func reporter(total: Int64, start: Int64, make: () -> TransferReporter) -> TransferReporter {
        lock.withLock {
            if let c = current, c.start == start { return c.reporter }
            let r = make()
            current = (start, r)
            return r
        }
    }
}

/// For a link preview: whether the GPU server (`serverID`, empty: the first) could download this LoRA.
public func checkLinkOnServer(_ link: LoRARemote, serverID: String) async -> LoRAAvailability {
    guard let endpoint = RemoteEndpoints.resolve(serverID) else {
        return .unavailable("no GPU server: add one in the Servers panel")
    }
    do {
        let check = try await GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
            .checkLink(url: link.downloadURL, sha256: link.sha256)
        if check.stored { return .stored }
        return check.ok ? .available(withKey: false) : .unavailable(check.error ?? "the server can't download it")
    } catch {
        return .unavailable("\(endpoint.name): \(error)")
    }
}

/// Has the GPU server download a LoRA from its link (at datacenter speed: nothing goes through this Mac), unless
/// it already has the file. Returns its name among the server's uploads.
func serverFetch(_ link: LoRARemote, client: GPUJobClient, context: ExecutionContext) async throws -> String {
    try await serverFetch(link, client: client) { context.status($0) }
}

public func serverFetch(_ link: LoRARemote, client: GPUJobClient,
                        status: @escaping @Sendable (String) -> Void) async throws -> String {
    if let sha = link.sha256, try await client.has(sha + ".lora") { return sha + ".lora" }
    status("Asking the server to download \(link.fileName)")
    var fetch = try await client.fetch(url: link.downloadURL, sha256: link.sha256, bytes: link.bytes)
    let reporter = TransferReporter(verb: "Server downloading", name: link.fileName, total: link.bytes ?? fetch.total,
                                    start: fetch.bytes, emit: status)
    do {
        while fetch.state == "downloading" {
            try await Task.sleep(nanoseconds: 500_000_000)
            fetch = try await client.fetchStatus(fetch.id)
            reporter.update(fetch.bytes)
        }
    } catch {
        await client.cancelFetch(fetch.id)
        throw error
    }
    guard fetch.state == "done", let name = fetch.name else {
        throw RemoteNodeError.fetchFailed(link.fileName, fetch.error ?? fetch.state)
    }
    return name
}

/// "GPU 12.3 s · ≈ $0.014", with the model loading of a cold start when there was one.
func costLine(gpuSeconds: Double, loadSeconds: Double, cost: Double?) -> String {
    var s = String(format: "GPU %.1f s", gpuSeconds)
    if loadSeconds >= 1 { s += String(format: " + %.0f s loading", loadSeconds) }
    if let cost { s += cost < 0.01 ? String(format: " · ≈ $%.4f", cost) : String(format: " · ≈ $%.3f", cost) }
    return s
}

func pngData(_ image: MLXArray) throws -> Data {
    let cg = try ImageArray.cgImage(from: image)
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else {
        throw CocoaError(.fileWriteUnknown)
    }
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
    return data as Data
}

func loadImage(_ url: URL) throws -> MLXArray { ImageArray.array(from: try ImageArray.load(url)) }

func loadAudio(_ url: URL) async throws -> AudioClip { try await MediaIO.readAudio(url) }

func loadVideo(_ url: URL) async throws -> VideoClip {
    let clip = try await MediaIO.readVideo(url)
    // Unchanged downstream, Save Video copies this file instead of encoding the frames again.
    return VideoClip(frames: clip.frames, fps: clip.fps, audio: clip.audio, sourceURL: url)
}

// MARK: nodes

private func wanInputs(image: Bool) -> [InputSpec] {
    (image ? [InputSpec("image", .image)] : []) + [
        InputSpec("loras", .cloudLoRAs, optional: true),
        RemoteNodes.serverInput,
        InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
        InputSpec("negative", .string, default: .string(""), widget: .text(multiline: true)),
        InputSpec("seed", .int, default: .int(0), widget: .seed),
        InputSpec("width", .int, default: .int(832), widget: .number(min: 256, max: 1280, step: 16)),
        InputSpec("height", .int, default: .int(480), widget: .number(min: 256, max: 1280, step: 16)),
        // Frames at 16 fps; Wan wants 4k+1 (81 = 5 s).
        InputSpec("length", .int, default: .int(81), widget: .number(min: 5, max: 121, step: 4)),
        // The Lightning 4-step LoRAs: about 10× fewer GPU seconds.
        InputSpec("fast", .bool, default: .bool(true), widget: .toggle),
        // 0 = automatic (4 with fast, else 40).
        InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 60, step: 1)),
        // 0 = automatic (none with fast, else Wan's 4 / 3).
        InputSpec("guidance", .float, default: .double(0), widget: .number(min: 0, max: 10, step: 0.5)),
        // A Wan 2.2 fine-tune from the library, installed on the server (empty: Wan 2.2 itself).
        InputSpec("model", .string, default: .string(""), widget: .library("model")),
    ]
}

/// The job's parameters, LoRA files and links.
private func wanJob(_ inputs: NodeInputs, family: String) throws
    -> (params: [String: Any], files: [String: URL], links: [String: LoRARemote]) {
    var params: [String: Any] = [
        "prompt": try inputs.string("prompt"), "negative": try inputs.string("negative"),
        "seed": try inputs.int("seed"), "width": try inputs.int("width"), "height": try inputs.int("height"),
        "length": try inputs.int("length"), "fast": try inputs.bool("fast"), "steps": try inputs.int("steps"),
        "guidance": try inputs.float("guidance"),
    ]
    let model = (try? inputs.string("model")) ?? ""
    params["model"] = model.isEmpty ? "base" : model
    let (files, links) = try CloudLoRA.attach(inputs.optionalObject("loras", as: [CloudLoRA].self) ?? [],
                                              family: family, to: &params)
    return (params, files, links)
}

public struct RemoteWanTextToVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteWanTextToVideo", displayName: "Wan 2.2 Text to Video (Cloud GPU)", category: "remote",
        inputs: wanInputs(image: false), outputs: [OutputSpec("video", .video)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let job = try wanJob(inputs, family: "wan-2.2-t2v")
        let (url, _) = try await runRemoteJob(task: "wan22-t2v", params: job.params, files: job.files, links: job.links,
                                              inputs: inputs, context: context)
        return ["video": .object(try await loadVideo(url))]
    }
}

public struct RemoteWanImageToVideoNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteWanImageToVideo", displayName: "Wan 2.2 Image to Video (Cloud GPU)", category: "remote",
        inputs: wanInputs(image: true), outputs: [OutputSpec("video", .video)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let image: MLXArray = try inputs.object("image")
        let job = try wanJob(inputs, family: "wan-2.2-i2v")
        let (url, _) = try await runRemoteJob(task: "wan22-i2v", params: job.params, images: ["image": image],
                                              files: job.files, links: job.links, inputs: inputs, context: context)
        return ["video": .object(try await loadVideo(url))]
    }
}

// MARK: LoRAs

/// The socket that carries Cloud GPU LoRAs (a chain of LoRA (Cloud GPU) nodes) to a cloud model.
public extension SocketType {
    static let cloudLoRAs = SocketType("CLOUD_LORAS")
}

/// A LoRA for a cloud model: a file on this Mac (uploaded with the job) or a link the server downloads itself.
public struct CloudLoRA: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case file(URL)
        case link(LoRARemote)
    }
    public var source: Source
    public var strength: Double
    /// What Civitai says it was made for ("Flux.1 D", "Qwen"…), checked by the model node it feeds.
    public var baseModel: String? = nil
    /// Wan 2.2: the expert a single file is for (`high` / `low`; nil: both). A link with a companion is a pair.
    public var expert: String? = nil

    /// Refuses a LoRA made for another model family before anything is uploaded or downloaded. (Wan 2.2 T2V and
    /// I2V LoRAs are accepted by either Wan node.)
    public func checkFits(_ family: String) throws {
        guard let base = baseModel, let f = LoRASources.family(forBaseModel: base) else {
            if let base = baseModel { throw LoRASourceError.wrongBaseModel(base, expected: LoRASources.familyName(family)) }
            return
        }
        if f == family || (f.hasPrefix("wan-2.2") && family.hasPrefix("wan-2.2")) { return }
        throw LoRASourceError.wrongBaseModel(base, expected: LoRASources.familyName(family))
    }

    /// The job's `loraN` files and links, and their strengths in `params`.
    public static func attach(_ loras: [CloudLoRA], family: String, to params: inout [String: Any])
        throws -> (files: [String: URL], links: [String: LoRARemote]) {
        guard loras.count <= RemoteNodes.maxFluxLoRAs else { throw RemoteNodeError.tooManyLoRAs(loras.count) }
        var files: [String: URL] = [:]
        var links: [String: LoRARemote] = [:]
        let wan = family.hasPrefix("wan-2.2")
        for (i, lora) in loras.enumerated() {
            try lora.checkFits(family)
            switch lora.source {
            case .file(let url): files["lora\(i + 1)"] = url
            case .link(let remote):
                links["lora\(i + 1)"] = remote
                // A Wan 2.2 pair: the low-noise expert's file goes with it.
                if wan, let low = remote.companionRemote { links["lora\(i + 1)_low"] = low }
            }
            params["lora\(i + 1)_strength"] = lora.strength
            if wan { params["lora\(i + 1)_expert"] = lora.expert ?? "both" }
        }
        return (files, links)
    }
}

/// Adds a LoRA to the LoRAs a Cloud GPU model node (FLUX.1 dev, Qwen-Image, Wan 2.2) uses. Chain several (up to
/// four) to stack them; the model node refuses ones made for another model. A Wan 2.2 LoRA from Civitai or Hugging
/// Face that comes as a high/low pair goes to each expert; a single file to both, or to the one its name says. `lora` picks it from the library (Models window); `path` (a file) is for graphs saved before the library.
/// A LoRA from a link is downloaded by the GPU server itself; one that only exists on this Mac is uploaded with the
/// job, once. The server keeps both, and a warm GPU keeps them loaded.
public struct RemoteLoRANode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteLoRA", displayName: "LoRA (Cloud GPU)", category: "remote",
        inputs: [
            InputSpec("loras", .cloudLoRAs, optional: true),
            InputSpec("lora", .string, default: .string(""), widget: .library("lora")),
            InputSpec("path", .string, default: .string(""), widget: .filePath),
            InputSpec("strength", .float, default: .double(1), widget: .number(min: -2, max: 2, step: 0.05)),
        ],
        outputs: [OutputSpec("loras", .cloudLoRAs)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        var loras = inputs.optionalObject("loras", as: [CloudLoRA].self) ?? []
        let strength = try inputs.float("strength")
        guard strength != 0 else { return ["loras": .object(loras)] }
        let id = (try? inputs.string("lora")) ?? ""
        if !id.isEmpty {
            guard let item = Library.item(id) else { throw LibraryError.missing(id) }
            if let source = item.source {
                loras.append(CloudLoRA(source: .link(source), strength: strength, baseModel: item.baseModel,
                                       expert: source.companion == nil ? source.expert : nil))
            } else if let url = Library.localURL(item) {
                loras.append(CloudLoRA(source: .file(url), strength: strength, baseModel: item.baseModel))
            } else {
                throw LibraryError.notOnThisMac(item.title)
            }
            context.status(StandardNodes.loraSummary(item))
        } else {
            let path = try inputs.string("path")
            guard !path.isEmpty else { throw RemoteNodeError.noLoRAFile }
            let url = try StandardNodes.loraFile(path)
            guard FileManager.default.fileExists(atPath: url.path) else { throw RemoteNodeError.loraMissing(url.path) }
            loras.append(CloudLoRA(source: .file(url), strength: strength))
        }
        return ["loras": .object(loras)]
    }
}

public struct RemoteFluxNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteFlux", displayName: "FLUX.1 dev (Cloud GPU)", category: "remote",
        inputs: [
            InputSpec("loras", .cloudLoRAs, optional: true),
            RemoteNodes.serverInput,
            InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            InputSpec("width", .int, default: .int(1024), widget: .number(min: 256, max: 2048, step: 16)),
            InputSpec("height", .int, default: .int(1024), widget: .number(min: 256, max: 2048, step: 16)),
            InputSpec("steps", .int, default: .int(28), widget: .number(min: 1, max: 60, step: 1)),
            InputSpec("guidance", .float, default: .double(3.5), widget: .number(min: 0, max: 10, step: 0.5)),
            // A FLUX.1 dev fine-tune from the library, installed on the server (empty: FLUX.1 dev itself).
            InputSpec("model", .string, default: .string(""), widget: .library("model")),
        ],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        var params: [String: Any] = [
            "prompt": try inputs.string("prompt"), "seed": try inputs.int("seed"), "width": try inputs.int("width"),
            "height": try inputs.int("height"), "steps": try inputs.int("steps"), "guidance": try inputs.float("guidance"),
        ]
        let model = (try? inputs.string("model")) ?? ""
        params["model"] = model.isEmpty ? "base" : model
        let (files, links) = try CloudLoRA.attach(inputs.optionalObject("loras", as: [CloudLoRA].self) ?? [],
                                                  family: "flux.1", to: &params)
        let (url, _) = try await runRemoteJob(task: "flux1-dev", params: params, files: files, links: links,
                                              inputs: inputs, context: context)
        return ["image": .object(try loadImage(url))]
    }
}

/// Qwen-Image, the original 20B model (strong at text in images and at following long prompts), on a GPU server.
/// `fast` uses its Lightning 8-step LoRA without guidance; the negative prompt needs `cfg` above 1 (the full
/// model's default, 4), which doubles the GPU time per step.
public struct RemoteQwenImageNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteQwenImage", displayName: "Qwen-Image (Cloud GPU)", category: "remote",
        inputs: [
            InputSpec("loras", .cloudLoRAs, optional: true),
            RemoteNodes.serverInput,
            InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("negative", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            // Its training sizes: 1328², 1664×928, 1472×1140, 1584×1056 (and portrait).
            InputSpec("width", .int, default: .int(1328), widget: .number(min: 256, max: 2048, step: 16)),
            InputSpec("height", .int, default: .int(1328), widget: .number(min: 256, max: 2048, step: 16)),
            // The Lightning 8-step LoRA: about 6× fewer GPU seconds.
            InputSpec("fast", .bool, default: .bool(true), widget: .toggle),
            // 0 = automatic (8 with fast, else 50).
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 100, step: 1)),
            // True classifier-free guidance; 0 = automatic (1, none, with fast, else 4).
            InputSpec("cfg", .float, default: .double(0), widget: .number(min: 0, max: 10, step: 0.5)),
            // A Qwen-Image fine-tune from the library, installed on the server (empty: Qwen-Image itself).
            InputSpec("model", .string, default: .string(""), widget: .library("model")),
        ],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        var params: [String: Any] = [
            "prompt": try inputs.string("prompt"), "negative": try inputs.string("negative"),
            "seed": try inputs.int("seed"), "width": try inputs.int("width"), "height": try inputs.int("height"),
            "fast": try inputs.bool("fast"), "steps": try inputs.int("steps"), "cfg": try inputs.float("cfg"),
        ]
        let model = (try? inputs.string("model")) ?? ""
        params["model"] = model.isEmpty ? "base" : model
        let (files, links) = try CloudLoRA.attach(inputs.optionalObject("loras", as: [CloudLoRA].self) ?? [],
                                                  family: "qwen-image", to: &params)
        let (url, _) = try await runRemoteJob(task: "qwen-image", params: params, files: files, links: links,
                                              inputs: inputs, context: context)
        return ["image": .object(try loadImage(url))]
    }
}

// MARK: audio nodes

/// A song with vocals and lyrics (or an instrumental) from ACE-Step 1.5 XL: its 4B planner and 4B DiT, the bigger
/// sibling of the local Text to Music node (same inputs; up to 10 minutes).
public struct RemoteTextToMusicNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteTextToMusic", displayName: "ACE-Step 1.5 XL Music (Cloud GPU)", category: "remote",
        inputs: [
            RemoteNodes.serverInput,
            InputSpec("tags", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("lyrics", .string, default: .string("[Instrumental]"), widget: .text(multiline: true)),
            InputSpec("seconds", .float, default: .double(60), widget: .number(min: 10, max: 600, step: 1)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            // 0 = the turbo model's 8.
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 100, step: 1)),
            InputSpec("planner", .bool, default: .bool(true), widget: .toggle),
            // 0 = auto.
            InputSpec("bpm", .int, default: .int(0), widget: .number(min: 0, max: 300, step: 1)),
            InputSpec("key", .string, default: .string("auto"), widget: .choice(MusicKeys.all)),
            InputSpec("time_signature", .string, default: .string("auto"), widget: .choice(MusicKeys.timeSignatures)),
            InputSpec("language", .string, default: .string("auto"), widget: .choice(MusicKeys.languages)),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let params: [String: Any] = [
            "tags": try inputs.string("tags"), "lyrics": try inputs.string("lyrics"),
            "seconds": try inputs.float("seconds"), "seed": try inputs.int("seed"), "steps": try inputs.int("steps"),
            "planner": try inputs.bool("planner"), "bpm": try inputs.int("bpm"), "key": try inputs.string("key"),
            "time_signature": try inputs.string("time_signature"), "language": try inputs.string("language"),
        ]
        let (url, _) = try await runRemoteJob(task: "ace-step-xl", params: params, inputs: inputs, context: context)
        return ["audio": .object(try await loadAudio(url))]
    }
}

/// Sound effects, ambience and instrumental beds from Stable Audio 3 medium: 44.1 kHz stereo, up to about six
/// minutes (the local Sound Effect model stops at 11 s).
public struct RemoteTextToSoundNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "RemoteTextToSound", displayName: "Stable Audio 3 Sound (Cloud GPU)", category: "remote",
        inputs: [
            RemoteNodes.serverInput,
            InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("seconds", .float, default: .double(10), widget: .number(min: 1, max: 380, step: 0.5)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            // 0 = the distilled model's 8.
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 100, step: 1)),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let params: [String: Any] = [
            "prompt": try inputs.string("prompt"), "seconds": try inputs.float("seconds"),
            "seed": try inputs.int("seed"), "steps": try inputs.int("steps"),
        ]
        let (url, _) = try await runRemoteJob(task: "stable-audio-3", params: params, inputs: inputs, context: context)
        return ["audio": .object(try await loadAudio(url))]
    }
}

/// Speech from Qwen3-TTS (10 languages). Connect a `reference` clip to clone its voice (a few seconds; its
/// transcript in `reference_text` helps); otherwise it speaks with a built-in `voice`, or with `voice` "designed",
/// a new voice made from the `description`. With a built-in voice, the description sets the delivery instead.
public struct RemoteTextToSpeechNode: NodeImplementation {
    /// The longest reference clip sent (more doesn't help a clone; it only slows the upload).
    static let maxReferenceSeconds = 30.0

    public static let schema = NodeSchema(
        type: "RemoteTextToSpeech", displayName: "Qwen3-TTS Speech (Cloud GPU)", category: "remote",
        inputs: [
            InputSpec("reference", .audio, optional: true),
            RemoteNodes.serverInput,
            InputSpec("text", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("language", .string, default: .string("Auto"), widget: .choice(RemoteNodes.speechLanguages)),
            InputSpec("voice", .string, default: .string("Ryan"), widget: .choice(RemoteNodes.speechVoices)),
            InputSpec("description", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("reference_text", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
        ],
        outputs: [OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let params: [String: Any] = [
            "text": try inputs.string("text"), "language": try inputs.string("language"),
            "voice": try inputs.string("voice"), "description": try inputs.string("description"),
            "reference_text": try inputs.string("reference_text"), "seed": try inputs.int("seed"),
        ]
        var audios: [String: AudioClip] = [:]
        if let reference = inputs.optionalObject("reference", as: AudioClip.self) {
            audios["reference"] = Self.referenceClip(reference)
        }
        let (url, _) = try await runRemoteJob(task: "qwen3-tts", params: params, audios: audios, inputs: inputs,
                                              context: context)
        return ["audio": .object(try await loadAudio(url))]
    }

    /// Mono, at most `maxReferenceSeconds`.
    static func referenceClip(_ clip: AudioClip) -> AudioClip {
        var mono = MediaOps.channels(clip, 1)
        let limit = Int(maxReferenceSeconds * Double(mono.sampleRate))
        if mono.frameCount > limit {
            mono = AudioClip(samples: mono.samples[0..., ..<limit], sampleRate: mono.sampleRate)
        }
        return mono
    }
}
