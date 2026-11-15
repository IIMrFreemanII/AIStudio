import CoreGraphics
import Foundation
import GraphRuntime
import MLX
import AIStudioProtocol
import MLXDiffusionCore
@testable import NodesRemote
import NodesStandard
import Testing

/// The real GPU server (server/gpu) in fake-models mode: every GPU task runs a CPU stand-in with the task's own
/// parameters, so the nodes and the client run end to end without models. Needs a Python with the server's
/// dependencies in `AISTUDIO_GPU_PYTHON` (with xcodebuild: `TEST_RUNNER_AISTUDIO_GPU_PYTHON=…`); skipped otherwise.
final class GPUServerFixture: @unchecked Sendable {
    static let python = ProcessInfo.processInfo.environment["AISTUDIO_GPU_PYTHON"]
    static let shared: GPUServerFixture? = python.flatMap { try? GPUServerFixture(python: $0) }

    let process = Process()
    /// A plain file server (Python's http.server) for link downloads: files written to `files` are served at
    /// `fileURL`.
    let fileServer = Process()
    let fileURL: URL
    var files: URL { folder.appendingPathComponent("served") }
    let baseURL: URL
    let token = String(repeating: "f", count: 48)
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("gpu-server-\(UUID().uuidString)")

    init(python: String) throws {
        let port = 18_900 + Int.random(in: 0..<900)
        baseURL = URL(string: "http://127.0.0.1:\(port)")!
        fileURL = URL(string: "http://127.0.0.1:\(port + 1000)")!
        let serverDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("server/gpu")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let tokenFile = folder.appendingPathComponent("token")
        try token.write(to: tokenFile, atomically: true, encoding: .utf8)
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["-m", "aistudio_gpu", "serve", "--port", String(port), "--data", folder.path + "/data",
                             "--models", folder.path + "/models", "--token-file", tokenFile.path]
        process.currentDirectoryURL = serverDir
        var env = ProcessInfo.processInfo.environment
        env["AISTUDIO_FAKE_MODELS"] = "1"
        env["AISTUDIO_FETCH_HOSTS"] = "127.0.0.1"
        env["PYTHONPATH"] = serverDir.path
        process.environment = env
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle(forWritingAtPath: "/dev/null")
        try process.run()
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("served"), withIntermediateDirectories: true)
        fileServer.executableURL = URL(fileURLWithPath: python)
        fileServer.arguments = ["-m", "http.server", String(port + 1000), "--bind", "127.0.0.1",
                                "--directory", folder.appendingPathComponent("served").path]
        fileServer.standardOutput = FileHandle.nullDevice
        fileServer.standardError = FileHandle(forWritingAtPath: "/dev/null")
        try fileServer.run()
        for _ in 0..<150 {
            if let (_, r) = try? awaitSync({ try await URLSession.shared.data(from: self.baseURL.appendingPathComponent("v1/health")) }),
               (r as? HTTPURLResponse)?.statusCode == 200 { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
        process.terminate()
        throw CocoaError(.featureUnsupported)
    }

    deinit {
        process.terminate()
        fileServer.terminate()
    }

    func install() {
        RemoteEndpoints.install([.init(id: "gpu", name: "Test GPU", baseURL: baseURL, token: token)])
    }
}

/// Runs an async closure to completion from synchronous code (fixture start-up only).
func awaitSync<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<T, Error>!
    Task.detached {
        do { result = .success(try await body()) } catch { result = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try result.get()
}

final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [ExecutionEvent] = []
    func add(_ e: ExecutionEvent) { lock.withLock { items.append(e) } }
    var statuses: [String] {
        lock.withLock { items.compactMap { if case .nodeStatus(_, let t?) = $0 { return t }; return nil } }
    }
    var steps: [Int] {
        lock.withLock { items.compactMap { if case .nodeProgress(_, let s, _, nil) = $0 { return s }; return nil } }
    }
    var previews: [CGImage] {
        lock.withLock { items.compactMap { if case .nodeProgress(_, _, _, let p?) = $0 { return (p as! CGImage) }; return nil } }
    }
}

final class RecordBox: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [RemoteJobRecord] = []
    func add(_ r: RemoteJobRecord) { lock.withLock { items.append(r) } }
    var all: [RemoteJobRecord] { lock.withLock { items } }
}

/// One serialized suite: the tests share the global `RemoteEndpoints`.
@Suite(.serialized)
struct RemoteNodesTests {
    static let hasServer = GPUServerFixture.python != nil
    var server: GPUServerFixture { GPUServerFixture.shared! }
    let out = FileManager.default.temporaryDirectory.appendingPathComponent("remote-out-\(UUID().uuidString)")

    /// Runs the graph up to the last node (remote nodes aren't output nodes themselves).
    func run(_ nodes: [GraphNode]) async throws -> (RunResult, EventLog) {
        server.install()
        let log = EventLog()
        let executor = GraphExecutor(registry: RemoteNodes.makeRegistry())
        let result = try await executor.run(Graph(nodes: nodes), targets: [nodes.last!.id]) { log.add($0) }
        return (result, log)
    }

    @Test(.enabled(if: hasServer)) func infoListsTheTasks() async throws {
        let info = try await GPUJobClient(baseURL: server.baseURL, token: server.token).info()
        #expect(info.api == 1 && info.runner == "local")
        #expect(info.task("wan22-t2v")?.gpu == "H100" && info.task("wan22-t2v")?.installed == true)
        #expect(info.task("wan22-i2v")?.inputs == ["image"])
        #expect(info.task("qwen-image-edit")?.inputs == ["image1"] && info.task("qwen-image-edit")?.installed == true)
        // A wrong token is refused.
        await #expect(throws: RemoteClient.ClientError.self) {
            _ = try await GPUJobClient(baseURL: server.baseURL, token: "nope-nope-nope").info()
        }
    }

    @Test(.enabled(if: hasServer)) func wanTextToVideoSavesTheDownloadedFileWithoutReencoding() async throws {
        let (result, log) = try await run([
            GraphNode(id: "wan", type: "RemoteWanTextToVideo", inputs: [
                "prompt": .value(.string("a fox")), "width": .value(.int(320)), "height": .value(.int(256)),
                "length": .value(.int(9)), "seed": .value(.int(3))]),
            GraphNode(id: "save", type: "SaveVideo", inputs: [
                "video": .link(node: "wan", output: "video"), "directory": .value(.string(out.path)),
                "filename_prefix": .value(.string("cloud"))]),
        ])
        #expect(log.steps == [1, 2, 3, 4])  // fast: 4 steps
        #expect(!log.previews.isEmpty)
        #expect(log.statuses.contains("Sampling") && log.statuses.contains("Downloading result"))
        #expect(log.statuses.last?.hasPrefix("GPU ") == true)  // the cost line (no price on the fake server)
        let saved = try #require({ () -> URL? in
            if case .object(let u)? = result.outputs["save"]?["video_url"] { return u as? URL }; return nil
        }())
        let clip = try await MediaIO.readVideo(saved)
        #expect(clip.frameCount == 9 && clip.width == 320 && clip.height == 256)
        #expect(await MediaIO.readMetadata(saved)["aistudio.graph"]?.contains("RemoteWanTextToVideo") == true)
        // The saved file holds the server's samples unchanged (a re-encode would decode differently).
        if case .object(let o)? = result.outputs["save"]?["video"], let v = o as? VideoClip, let src = v.sourceURL {
            let original = try await MediaIO.readVideo(src)
            #expect(abs(original.frames - clip.frames).max().item(Float.self) == 0)
        } else {
            Issue.record("the node's clip should remember its downloaded file")
        }
    }

    @Test(.enabled(if: hasServer)) func wanImageToVideoUploadsTheStartImage() async throws {
        // A solid colour start image: the fake server puts it in the first frame.
        let red = MLXArray.ones([64, 64, 4]) * MLXArray([0.9, 0.1, 0.1, 1.0] as [Float])
        let png = out.appendingPathComponent("start.png")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try ImageArray.savePNG(red, to: png)
        let (result, log) = try await run([
            GraphNode(id: "img", type: "LoadImage", inputs: ["path": .value(.string(png.path))]),
            GraphNode(id: "wan", type: "RemoteWanImageToVideo", inputs: [
                "image": .link(node: "img", output: "image"), "width": .value(.int(256)), "height": .value(.int(256)),
                "length": .value(.int(5))]),
            GraphNode(id: "save", type: "SaveVideo", inputs: [
                "video": .link(node: "wan", output: "video"), "directory": .value(.string(out.path))]),
        ])
        #expect(log.statuses.contains("Uploading"))
        let clip = try #require({ () -> VideoClip? in
            if case .object(let o)? = result.outputs["save"]?["video"] { return o as? VideoClip }; return nil
        }())
        let first = clip.frames[0].mean(axes: [0, 1])
        #expect(first[0].item(Float.self) > 0.7 && first[1].item(Float.self) < 0.3)
    }

    @Test(.enabled(if: hasServer)) func wanImageToVideoEndsOnTheLastImageAndLastFrameChainsClips() async throws {
        // Clip 1 starts red and ends green (the fake server puts the end picture in the last frame); Last Frame
        // hands its green end to clip 2, which starts from it.
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let red = out.appendingPathComponent("red.png"), green = out.appendingPathComponent("green.png")
        try ImageArray.savePNG(MLXArray.ones([64, 64, 4]) * MLXArray([0.9, 0.1, 0.1, 1.0] as [Float]), to: red)
        try ImageArray.savePNG(MLXArray.ones([64, 64, 4]) * MLXArray([0.1, 0.9, 0.1, 1.0] as [Float]), to: green)
        let size: [String: NodeInput] = ["width": .value(.int(256)), "height": .value(.int(256)), "length": .value(.int(5))]
        func previewURL(_ result: RunResult) -> URL? {
            if case .object(let u)? = result.outputs["show"]?["video_url"] { return u as? URL }; return nil
        }
        let (first, _) = try await run([
            GraphNode(id: "start", type: "LoadImage", inputs: ["path": .value(.string(red.path))]),
            GraphNode(id: "end", type: "LoadImage", inputs: ["path": .value(.string(green.path))]),
            GraphNode(id: "one", type: "RemoteWanImageToVideo", inputs: size.merging([
                "image": .link(node: "start", output: "image"), "last_image": .link(node: "end", output: "image"),
            ]) { $1 }),
            GraphNode(id: "show", type: "PreviewVideo", inputs: ["video": .link(node: "one", output: "video")]),
        ])
        let oneURL = try #require(previewURL(first))
        let (second, _) = try await run([
            GraphNode(id: "clip", type: "LoadVideo", inputs: ["path": .value(.string(oneURL.path))]),
            GraphNode(id: "last", type: "LastFrame", inputs: ["video": .link(node: "clip", output: "video")]),
            GraphNode(id: "two", type: "RemoteWanImageToVideo", inputs: size.merging([
                "image": .link(node: "last", output: "image"), "seed": .value(.int(7)),
            ]) { $1 }),
            GraphNode(id: "show", type: "PreviewVideo", inputs: ["video": .link(node: "two", output: "video")]),
        ])
        let one = try await MediaIO.readVideo(oneURL)
        let two = try await MediaIO.readVideo(try #require(previewURL(second)))
        func rgb(_ clip: VideoClip, _ i: Int) -> (Float, Float) {
            let m = clip.frames[i].mean(axes: [0, 1])
            return (m[0].item(Float.self), m[1].item(Float.self))
        }
        #expect(rgb(one, 0).0 > 0.7 && rgb(one, one.frameCount - 1).1 > 0.7, "clip 1: red to green")
        #expect(rgb(two, 0).1 > 0.7 && rgb(two, 0).0 < 0.3, "clip 2 starts where clip 1 ended")
    }

    @Test(.enabled(if: hasServer)) func wanSpeechToVideoFollowsTheAudio() async throws {
        // A 1.5 s tone and a red still: the clip has ceil(1.5 * 16) frames at 16 fps, carries the audio, and starts
        // from the picture.
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let png = out.appendingPathComponent("speaker.png"), wav = out.appendingPathComponent("speech.wav")
        try ImageArray.savePNG(MLXArray.ones([64, 64, 4]) * MLXArray([0.9, 0.1, 0.1, 1.0] as [Float]), to: png)
        let tone = MLX.sin(MLXArray(0..<36000).asType(.float32) * 0.05) * 0.5
        try await MediaIO.writeAudio(AudioClip(samples: MLX.stacked([tone, tone]), sampleRate: 24000), to: wav)
        let (result, log) = try await run([
            GraphNode(id: "img", type: "LoadImage", inputs: ["path": .value(.string(png.path))]),
            GraphNode(id: "voice", type: "LoadAudio", inputs: ["path": .value(.string(wav.path))]),
            GraphNode(id: "s2v", type: "RemoteWanSpeechToVideo", inputs: [
                "image": .link(node: "img", output: "image"), "audio": .link(node: "voice", output: "audio"),
                "width": .value(.int(256)), "height": .value(.int(256))]),
            GraphNode(id: "save", type: "SaveVideo", inputs: [
                "video": .link(node: "s2v", output: "video"), "directory": .value(.string(out.path))]),
        ])
        #expect(log.statuses.contains("Uploading"))
        let clip = try #require({ () -> VideoClip? in
            if case .object(let o)? = result.outputs["save"]?["video"] { return o as? VideoClip }; return nil
        }())
        #expect(clip.frameCount == 24 && clip.fps == 16)
        let audio = try #require(clip.audio)
        #expect(abs(audio.duration - 1.5) < 0.1)
        let first = clip.frames[0].mean(axes: [0, 1])
        #expect(first[0].item(Float.self) > 0.7 && first[1].item(Float.self) < 0.3)
    }

    @Test(.enabled(if: hasServer)) func fluxReturnsAnImage() async throws {
        let (result, _) = try await run([
            GraphNode(id: "flux", type: "RemoteFlux", inputs: [
                "prompt": .value(.string("a lighthouse")), "width": .value(.int(320)), "height": .value(.int(256)),
                "steps": .value(.int(6))]),
            GraphNode(id: "show", type: "PreviewImage", inputs: ["image": .link(node: "flux", output: "image")]),
        ])
        guard case .object(let o)? = result.outputs["show"]?["image"], let image = o as? MLXArray else {
            Issue.record("no image")
            return
        }
        #expect(image.shape[0] == 256 && image.shape[1] == 320)
    }

    @Test(.enabled(if: hasServer)) func qwenImageEditUploadsItsPicturesAndReportsTheJob() async throws {
        // Two solid pictures: the fake edit returns picture 1 tinted red, its bottom row the picture count / 4.
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var paths: [String] = []
        for (i, color) in [[0.1, 0.1, 0.9], [0.1, 0.9, 0.1]].enumerated() {
            let png = out.appendingPathComponent("pic\(i + 1).png")
            try ImageArray.savePNG(MLXArray.ones([40, 60, 4]) * MLXArray((color + [1.0]).map(Float.init)), to: png)
            paths.append(png.path)
        }
        let records = RecordBox()
        let token = RemoteJobLog.observe { records.add($0) }
        defer { RemoteJobLog.remove(token) }
        let (result, log) = try await run([
            GraphNode(id: "p1", type: "LoadImage", inputs: ["path": .value(.string(paths[0]))]),
            GraphNode(id: "p2", type: "LoadImage", inputs: ["path": .value(.string(paths[1]))]),
            GraphNode(id: "edit", type: "RemoteQwenImageEdit", inputs: [
                "image1": .link(node: "p1", output: "image"), "image2": .link(node: "p2", output: "image"),
                "prompt": .value(.string("picture 1 meets picture 2")), "width": .value(.int(96)),
                "height": .value(.int(64))]),
            GraphNode(id: "show", type: "PreviewImage", inputs: ["image": .link(node: "edit", output: "image")]),
        ])
        #expect(log.statuses.contains("Uploading") && log.statuses.contains { $0.hasPrefix("GPU ") })
        guard case .object(let o)? = result.outputs["show"]?["image"], let image = o as? MLXArray else {
            Issue.record("no image")
            return
        }
        #expect(image.shape[0] == 64 && image.shape[1] == 96)
        let middle = image[32, 48]
        #expect(abs(middle[0].item(Float.self) - 0.55) < 0.05 && abs(middle[2].item(Float.self) - 0.225) < 0.05)
        #expect(abs(image[63, 10][0].item(Float.self) - 0.5) < 0.03)  // two pictures
        let r = try #require(records.all.last)
        #expect(r.task == "qwen-image-edit" && r.width == 96 && r.height == 64 && (r.costUSD ?? 0) > 0)
    }

    @Test(.enabled(if: hasServer)) func fluxUploadsChainedCloudLoRAs() async throws {
        // Two LoRA files and a third LoRA node at strength 0 (skipped): the fake server brightens the image's top
        // row to (LoRAs it got) / 4.
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let a = out.appendingPathComponent("a.safetensors"), b = out.appendingPathComponent("b.safetensors")
        try Data("lora a".utf8).write(to: a)
        try Data(repeating: 7, count: 3 << 20).write(to: b)
        let (result, log) = try await run([
            GraphNode(id: "l1", type: "RemoteLoRA", inputs: ["path": .value(.string(a.path))]),
            GraphNode(id: "l2", type: "RemoteLoRA", inputs: [
                "loras": .link(node: "l1", output: "loras"), "path": .value(.string(b.path)),
                "strength": .value(.double(0.6))]),
            GraphNode(id: "l3", type: "RemoteLoRA", inputs: [
                "loras": .link(node: "l2", output: "loras"), "strength": .value(.double(0))]),
            GraphNode(id: "flux", type: "RemoteFlux", inputs: [
                "loras": .link(node: "l3", output: "loras"), "width": .value(.int(256)), "height": .value(.int(256)),
                "steps": .value(.int(2))]),
            GraphNode(id: "show", type: "PreviewImage", inputs: ["image": .link(node: "flux", output: "image")]),
        ])
        #expect(log.statuses.contains("Uploading 3.1 / 3.1 MB (b.safetensors)"))
        #expect(log.statuses.contains("Uploading a.safetensors"))
        guard case .object(let o)? = result.outputs["show"]?["image"], let image = o as? MLXArray else {
            Issue.record("no image")
            return
        }
        let rgb = image[0..., 0..., ..<3]  // [H, W, 4]
        #expect(abs(rgb[0] - 0.5).max().item(Float.self) < 0.01)  // the top row
        #expect(abs(rgb[1] - 0.5).max().item(Float.self) > 0.01)
    }

    @Test func cloudLoRAsNeedAFileAndStopAtFour() async throws {
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let f = out.appendingPathComponent("x.safetensors")
        try Data("x".utf8).write(to: f)
        RemoteEndpoints.install([])
        let executor = GraphExecutor(registry: RemoteNodes.makeRegistry())
        do {
            _ = try await executor.run(Graph(nodes: [GraphNode(id: "l", type: "RemoteLoRA")]), targets: ["l"]) { _ in }
            Issue.record("expected an error")
        } catch {
            #expect(String(describing: error).contains("Pick a LoRA"))
        }
        var nodes: [GraphNode] = []
        for i in 1...5 {
            nodes.append(GraphNode(id: "l\(i)", type: "RemoteLoRA", inputs: [
                "path": .value(.string(f.path))].merging(i > 1 ? ["loras": .link(node: "l\(i - 1)", output: "loras")] : [:]) { a, _ in a }))
        }
        nodes.append(GraphNode(id: "f", type: "RemoteFlux", inputs: ["loras": .link(node: "l5", output: "loras")]))
        do {
            _ = try await executor.run(Graph(nodes: nodes), targets: ["f"]) { _ in }
            Issue.record("expected an error")
        } catch {
            #expect(String(describing: error).contains("up to 4"))
        }
    }

    @Test(.enabled(if: hasServer)) func cancellingTheRunCancelsTheServerJob() async throws {
        server.install()
        let executor = GraphExecutor(registry: RemoteNodes.makeRegistry())
        let graph = Graph(nodes: [GraphNode(id: "wan", type: "RemoteWanTextToVideo", inputs: [
            "width": .value(.int(256)), "height": .value(.int(256)), "length": .value(.int(5)),
            "steps": .value(.int(60))])])
        let started = Date()
        let task = Task { try await executor.run(graph, targets: ["wan"]) { _ in } }
        try await Task.sleep(nanoseconds: 600_000_000)
        task.cancel()
        await #expect(throws: (any Error).self) { _ = try await task.value }
        #expect(Date().timeIntervalSince(started) < 5)  // 60 fake steps take ~1.2 s more; it stopped early
        // The server shows the job cancelled.
        let client = GPUJobClient(baseURL: server.baseURL, token: server.token)
        let jobs: [String: Int] = (try await client.http.getJSON("v1/info", as: InfoJobs.self)).jobs
        #expect(jobs["running"] == 0)
    }

    struct InfoJobs: Decodable { var jobs: [String: Int] }

    func savedAudio(_ result: RunResult, _ node: String = "save") async throws -> AudioClip {
        let url = try #require({ () -> URL? in
            if case .object(let u)? = result.outputs[node]?["audio_url"] { return u as? URL }; return nil
        }())
        return try await MediaIO.readAudio(url)
    }

    @Test(.enabled(if: hasServer)) func audioTasksDescribeTheirChoices() async throws {
        let info = try await GPUJobClient(baseURL: server.baseURL, token: server.token).info()
        let speech = try #require(info.task("qwen3-tts"))
        #expect(speech.kind == "audio" && speech.optionalInputs == ["reference"])
        #expect(speech.params?["voice"]?.choices == RemoteNodes.speechVoices)
        #expect(speech.params?["language"]?.choices == RemoteNodes.speechLanguages)
        let music = try #require(info.task("ace-step-xl"))
        #expect(music.params?["key"]?.choices == MusicKeys.all)
        #expect(music.params?["language"]?.choices == MusicKeys.languages)
        // Every remote node names a task the server has.
        for type in RemoteNodes.types { #expect(info.task(RemoteNodes.taskID(forNodeType: type) ?? "") != nil, "\(type)") }
    }

    @Test(.enabled(if: hasServer)) func musicAndSoundComeBackAsAudioClips() async throws {
        for (type, seconds, rate, channels) in [("RemoteTextToMusic", 10.0, 48000, 2), ("RemoteTextToSound", 2.5, 44100, 2)] {
            let (result, log) = try await run([
                GraphNode(id: "gen", type: type, inputs: ["seconds": .value(.double(seconds)), "seed": .value(.int(2))]),
                GraphNode(id: "save", type: "SaveAudio", inputs: [
                    "audio": .link(node: "gen", output: "audio"), "directory": .value(.string(out.path))]),
            ])
            let clip = try await savedAudio(result)
            #expect(clip.sampleRate == rate && clip.channels == channels, "\(type)")
            #expect(abs(clip.duration - seconds) < 0.01, "\(type)")
            #expect(log.previews.isEmpty && !log.steps.isEmpty && log.statuses.last?.hasPrefix("GPU ") == true)
        }
    }

    @Test(.enabled(if: hasServer)) func speechUploadsTheReferenceClipAsMonoWAV() async throws {
        // A stereo 44.1 kHz reference: sent as mono (the fake server plays it back first when the rates match, so
        // use its 24 kHz to check the upload arrived).
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let ref = out.appendingPathComponent("ref.wav")
        let tone = MLX.sin(MLXArray(0..<12000).asType(.float32) * 0.05) * 0.5
        try await MediaIO.writeAudio(AudioClip(samples: MLX.stacked([tone, tone]), sampleRate: 24000), to: ref)
        let (result, log) = try await run([
            GraphNode(id: "ref", type: "LoadAudio", inputs: ["path": .value(.string(ref.path))]),
            GraphNode(id: "tts", type: "RemoteTextToSpeech", inputs: [
                "reference": .link(node: "ref", output: "audio"), "text": .value(.string("Hello there."))]),
            GraphNode(id: "save", type: "SaveAudio", inputs: [
                "audio": .link(node: "tts", output: "audio"), "directory": .value(.string(out.path))]),
        ])
        #expect(log.statuses.contains("Uploading"))
        let clip = try await savedAudio(result)
        #expect(clip.sampleRate == 24000 && clip.channels == 1)
        // The reference (0.5 s) came first, then at least 1 s of the fake voice.
        #expect(clip.duration >= 1.5 - 0.01)
        let head = clip.samples[0, ..<12000]
        #expect(abs(head - tone).max().item(Float.self) < 0.001)
    }

    @Test func referenceClipsAreMonoAndCapped() {
        let long = AudioClip(samples: MLXArray.zeros([2, 48000 * 40]), sampleRate: 48000)
        let ref = RemoteTextToSpeechNode.referenceClip(long)
        #expect(ref.channels == 1 && ref.duration == RemoteTextToSpeechNode.maxReferenceSeconds)
    }

    @Test func noServerIsAClearError() async throws {
        RemoteEndpoints.install([])
        let executor = GraphExecutor(registry: RemoteNodes.makeRegistry())
        do {
            _ = try await executor.run(Graph(nodes: [GraphNode(id: "f", type: "RemoteFlux")]), targets: ["f"]) { _ in }
            Issue.record("expected an error")
        } catch {
            #expect(String(describing: error).contains("Servers panel"))
        }
    }

    @Test func endpointsResolveByIDOrFirst() {
        let a = RemoteEndpoints.Endpoint(id: "a", name: "A", baseURL: URL(string: "https://a")!, token: "t")
        let b = RemoteEndpoints.Endpoint(id: "b", name: "B", baseURL: URL(string: "https://b")!, token: "t")
        RemoteEndpoints.install([a, b])
        #expect(RemoteEndpoints.resolve("") == a && RemoteEndpoints.resolve("b") == b && RemoteEndpoints.resolve("c") == nil)
    }
}

@Test func costLineFormatting() {
    #expect(costLine(gpuSeconds: 12.34, loadSeconds: 0, cost: 0.0136) == "GPU 12.3 s · ≈ $0.014")
    #expect(costLine(gpuSeconds: 3, loadSeconds: 48, cost: 0.0029) == "GPU 3.0 s + 48 s loading · ≈ $0.0029")
    #expect(costLine(gpuSeconds: 1, loadSeconds: 0, cost: nil) == "GPU 1.0 s")
}
