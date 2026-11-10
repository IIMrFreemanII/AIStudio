import CryptoKit
import Foundation
import GraphRuntime
import NodesStandard
import Testing
@testable import AIStudioProtocol

struct GraphTransportTests {
    let registry = StandardNodes.makeRegistry()

    func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("transport-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    func graph(image: String, mask: String, lora: String = "") -> Graph {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string("/Users/me/models/qwen-turbo-mlx"))]),
            GraphNode(id: "lora", type: "LoRA", inputs: ["model": link("model", "model"), "path": .value(.string(lora))]),
            GraphNode(id: "img", type: "LoadImage", inputs: ["path": .value(.string(image))]),
            GraphNode(id: "img2", type: "LoadImage", inputs: ["path": .value(.string(mask))]),
            GraphNode(id: "save", type: "SaveImage", inputs: ["image": link("img", "image"),
                                                               "directory": .value(.string("~/Pictures/AIStudio/Fox"))]),
            GraphNode(id: "save2", type: "SaveImage", inputs: ["image": link("img2", "image")]),
        ])
    }

    @Test func rewritesFilesModelsAndOutputs() throws {
        let dir = try tempDir()
        let a = dir.appendingPathComponent("a.PNG")
        try Data("same bytes".utf8).write(to: a)
        let b = dir.appendingPathComponent("b.png")
        try Data("same bytes".utf8).write(to: b)
        var hashed: [URL] = []
        let g = graph(image: a.path, mask: "b.png")
        let p = try GraphTransport.prepare(g, schema: registry.schema,
                                           resolveInput: { $0.hasPrefix("/") ? URL(fileURLWithPath: $0) : dir.appendingPathComponent($0) },
                                           hash: { hashed.append($0); return try FileHasher.hashFile($0) })
        let sha = FileHasher.hex(SHA256.hash(data: Data("same bytes".utf8)))
        // Identical content uploads once; the extension is kept (lowercased) for decoders that need it.
        #expect(p.uploads.count == 1)
        #expect(p.uploads[0].name == "\(sha).png")
        #expect(p.uploads[0].bytes == 10)
        func value(_ n: String, _ i: String) -> String? {
            if case .value(.string(let s))? = p.graph.nodes.first(where: { $0.id == n })?.inputs[i] { return s }
            return nil
        }
        #expect(value("img", "path") == "\(sha).png" && value("img2", "path") == "\(sha).png")
        #expect(value("model", "path") == "qwen-turbo-mlx" && p.models == ["qwen-turbo-mlx"])
        #expect(value("lora", "path") == "")  // empty optional file stays empty
        #expect(value("save", "directory") == "outputs")
        #expect(p.outputDirectories["save"] == "~/Pictures/AIStudio/Fox")
        #expect(p.outputDirectories["save2"] == "outputs")  // the schema default
        #expect(p.original == g)
        #expect(hashed.count == 2)
    }

    @Test func missingInputFileFailsBeforeAnythingIsSent() throws {
        let g = graph(image: "/nonexistent/x.png", mask: "/nonexistent/y.png")
        #expect(throws: GraphTransport.TransportError.missingFile(node: "img", path: "/nonexistent/x.png")) {
            _ = try GraphTransport.prepare(g, schema: registry.schema, resolveInput: { URL(fileURLWithPath: $0) },
                                           hash: { _ in "" })
        }
    }

    @Test func serverNamesAreStrict() {
        let h = String(repeating: "a", count: 64)
        #expect(GraphTransport.isServerName(h))
        #expect(GraphTransport.isServerName(h + ".png"))
        #expect(GraphTransport.isServerName(h + ".mp4"))
        #expect(!GraphTransport.isServerName(h.uppercased()))
        #expect(!GraphTransport.isServerName(h + ".png.exe"))
        #expect(!GraphTransport.isServerName("../" + h))
        #expect(!GraphTransport.isServerName(String(h.dropLast()) + ".png"))
        #expect(!GraphTransport.isServerName(h + "."))
        #expect(GraphTransport.serverName(sha256: h, for: URL(fileURLWithPath: "/x/clip.MOV")) == h + ".mov")
    }

    @Test func hasherStreamsAndCachesByMtime() throws {
        let dir = try tempDir()
        let f = dir.appendingPathComponent("big.bin")
        let bytes = Data((0..<(FileHasher.chunkSize + 12345)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        try bytes.write(to: f)
        let expected = FileHasher.hex(SHA256.hash(data: bytes))
        let cache = dir.appendingPathComponent("cache.json")
        let h = FileHasher(cacheURL: cache)
        #expect(try h.sha256(of: f) == expected)
        h.save()
        // A new hasher reads the cache: same answer without reading the file (proved by making it unreadable).
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: f.path)
        #expect(try FileHasher(cacheURL: cache).sha256(of: f) == expected)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: f.path)
        // Changed content (new size/mtime) is hashed again.
        try Data("changed".utf8).write(to: f)
        #expect(try FileHasher(cacheURL: cache).sha256(of: f) == FileHasher.hex(SHA256.hash(data: Data("changed".utf8))))
    }
}

@Suite(.serialized) struct PathSandboxTests {
    @Test func confinesLoadSaveAndModelPaths() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sandbox-\(UUID().uuidString)")
        let inputs = root.appendingPathComponent("inputs"), models = root.appendingPathComponent("models")
        let run = root.appendingPathComponent("runs/1")
        for d in [inputs, models.appendingPathComponent("qwen"), run] {
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        }
        try Data("x".utf8).write(to: inputs.appendingPathComponent("a.png"))
        try Data("secret".utf8).write(to: root.appendingPathComponent("secret.txt"))
        // A symlink inside the input folder pointing outside it must not be followed out.
        try FileManager.default.createSymbolicLink(at: inputs.appendingPathComponent("link.png"),
                                                   withDestinationURL: root.appendingPathComponent("secret.txt"))
        PathSandbox.roots = .init(inputs: inputs, models: models)
        defer { PathSandbox.roots = nil }

        #expect(try StandardNodes.inputFile("a.png").lastPathComponent == "a.png")
        for bad in ["../secret.txt", "/etc/passwd", "~/x", "sub/../../secret.txt", "link.png", "missing.png", ""] {
            #expect(throws: (any Error).self) { try StandardNodes.inputFile(bad) }
        }
        #expect(try StandardNodes.modelFolder("qwen").lastPathComponent == "qwen")
        #expect(throws: (any Error).self) { try StandardNodes.modelFolder("/Users/x/models/qwen") }
        #expect(throws: (any Error).self) { try StandardNodes.modelFolder("../inputs") }
        #expect(throws: (any Error).self) { try StandardNodes.outputDirectory("/tmp/anywhere") }  // no run folder
        let (out, preview) = try PathSandbox.$runFolder.withValue(run) { () throws -> (URL, URL) in
            (try StandardNodes.outputDirectory("/etc"), try StandardNodes.previewFolder())
        }
        #expect(out == run && preview == run)
    }

    @Test func offMeansTheOldBehaviour() throws {
        PathSandbox.roots = nil
        #expect(try StandardNodes.inputFile("/tmp/x.png").path == "/tmp/x.png")
        #expect(try StandardNodes.outputDirectory("/tmp/out").path == "/tmp/out")
    }
}
