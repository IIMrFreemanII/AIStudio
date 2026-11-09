import Foundation
import GraphRuntime
import NodesStandard
import Testing
@testable import MLXComfyLibrary

private func tempDir() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxcomfy-lib-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private let sample = Graph(nodes: [GraphNode(id: "n", type: "Note", inputs: ["text": .value(.string("hi"))])])

@Suite struct WorkflowStoreTests {
    @Test func sanitizesNames() {
        #expect(WorkflowStore.sanitize("  My/Flow: v2  ") == "My-Flow- v2")
        #expect(WorkflowStore.sanitize("..hidden") == "hidden")
        #expect(WorkflowStore.sanitize("   ") == "Untitled")
        #expect(WorkflowStore.sanitize("fox.json") == "fox")
    }

    @Test func createMakesNamesUnique() throws {
        let store = WorkflowStore(root: try tempDir(), useTrash: false)
        let a = try store.create(sample, name: "Fox")
        let b = try store.create(sample, name: "Fox")
        let c = try store.create(sample, name: "Fox")
        #expect(a.lastPathComponent == "Fox.json")
        #expect(b.lastPathComponent == "Fox 2.json")
        #expect(c.lastPathComponent == "Fox 3.json")
        #expect(try Graph.load(b) == sample)
    }

    @Test func listsATreeSortedByName() throws {
        let root = try tempDir()
        let store = WorkflowStore(root: root, useTrash: false)
        let video = try store.createFolder(named: "Video")
        try store.create(sample, name: "b")
        try store.create(sample, name: "a10")
        try store.create(sample, name: "a2")
        try store.create(sample, name: "clip", in: video)
        try "not a graph".write(to: root.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)
        let tree = store.tree()
        #expect(tree.workflows.map(\.name) == ["a2", "a10", "b"])
        #expect(tree.folders.map(\.name) == ["Video"])
        #expect(tree.folders[0].workflows.map(\.name) == ["clip"])
        #expect(store.all().count == 4)
    }

    @Test func renamesDuplicatesMovesAndDeletes() throws {
        let store = WorkflowStore(root: try tempDir(), useTrash: false)
        let a = try store.create(sample, name: "Draft")
        try store.create(sample, name: "Final")
        // Renaming onto a taken name picks the next free one; renaming to itself is a no-op.
        let renamed = try store.rename(a, to: "Final")
        #expect(renamed.lastPathComponent == "Final 2.json")
        #expect(try store.rename(renamed, to: "Final 2") == renamed)
        let copy = try store.duplicate(renamed)
        #expect(copy.lastPathComponent == "Final 3.json")
        let folder = try store.createFolder(named: "Archive")
        let moved = try store.move(copy, to: folder)
        #expect(moved.deletingLastPathComponent().lastPathComponent == "Archive")
        #expect(throws: (any Error).self) { try store.move(folder, to: folder.appendingPathComponent("inner")) }
        try store.delete(moved)
        #expect(!FileManager.default.fileExists(atPath: moved.path))
        #expect(store.all().map(\.name).sorted() == ["Final", "Final 2"])
    }
}

@Suite struct SessionStoreTests {
    @Test func roundTripsTabsAndDrafts() throws {
        let store = SessionStore(directory: try tempDir())
        #expect(store.load().manifest.tabs.isEmpty)
        let a = UUID(), b = UUID(), gone = UUID()
        let manifest = SessionManifest(tabs: [
            .init(id: a, name: "Fox", fileURL: URL(fileURLWithPath: "/tmp/Fox.json"), isDirty: true),
            .init(id: b, name: "Untitled", presetID: "qwen-txt2img"),
        ], activeTab: b)
        try store.save(manifest)
        try store.writeDraft(sample, id: a)
        try store.writeDraft(sample, id: gone)
        store.prune(keeping: [a, b])
        let (loaded, drafts) = store.load()
        #expect(loaded == manifest)
        #expect(drafts[a] == sample)
        #expect(drafts[b] == nil)
        #expect(!FileManager.default.fileExists(atPath: store.draftURL(gone).path))
    }

    @Test func customLibraryRootKeepsItsOwnSession() {
        let root = URL(fileURLWithPath: "/tmp/lib")
        #expect(LibraryPaths(root: root).sessionDirectory.path == "/tmp/lib/.session")
        #expect(LibraryPaths().sessionDirectory == LibraryPaths.defaultSessionDirectory)
    }
}

@Suite struct PresetTests {
    static let portableGraph = Graph(nodes: [
        GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string("flux-1-dev"))]),
        GraphNode(id: "save", type: "SaveImage", inputs: ["directory": .value(.string(Preset.outputPlaceholder)),
                                                          "filename_prefix": .value(.string("flux"))]),
    ])

    @Test func instantiatesAndBack() {
        let p = Preset(id: "t", name: "T", category: .image, models: ["flux-1-dev"], modelName: "FLUX", summary: "",
                       symbol: "photo", graph: Self.portableGraph)
        let g = p.instantiate(resolveModel: { $0 == "flux-1-dev" ? "/models/flux-1-dev" : nil },
                              outputDirectory: "/out")
        #expect(g["model"]?.inputs["path"] == .value(.string("/models/flux-1-dev")))
        #expect(g["save"]?.inputs["directory"] == .value(.string("/out")))
        #expect(g["save"]?.inputs["filename_prefix"] == .value(.string("flux")))
        let back = Preset.portable(g, folderName: { $0 == "/models/flux-1-dev" ? "flux-1-dev" : nil },
                                   outputDirectory: "/out")
        #expect(back == Self.portableGraph)
        #expect(p.isAvailable(installed: ["flux-1-dev", "x"]))
        #expect(!p.isAvailable(installed: ["qwen-image-2.1-turbo"]))
    }

    @Test func savesAndLoadsUserPresets() throws {
        let dir = try tempDir()
        let p = Preset(id: "", name: "My Flux", category: .image, models: ["flux-1-dev"], modelName: "FLUX.1 [dev]",
                       summary: "mine", symbol: "photo", graph: Self.portableGraph)
        try PresetCatalog.save(p, thumbnailPNG: Data([1, 2, 3]), in: dir)
        try PresetCatalog.save(p, thumbnailPNG: nil, in: dir)
        let loaded = PresetCatalog.user(in: dir)
        #expect(loaded.map(\.id) == ["user.My Flux", "user.My Flux 2"])
        #expect(loaded.allSatisfy { !$0.isBuiltIn && $0.graph == Self.portableGraph })
        let thumb = try #require(PresetCatalog.thumbnailURL(loaded[0]))
        #expect(try Data(contentsOf: thumb) == Data([1, 2, 3]))
        try PresetCatalog.delete(loaded[0])
        #expect(PresetCatalog.user(in: dir).count == 1)
        #expect(!FileManager.default.fileExists(atPath: thumb.path))
    }

    @Test func builtInPresetsAreValidGraphs() throws {
        let presets = PresetCatalog.builtIn()
        #expect(presets.count >= 15)
        #expect(Set(presets.map(\.id)).count == presets.count)
        let executor = GraphExecutor(registry: StandardNodes.makeRegistry())
        for p in presets {
            #expect(p.isBuiltIn)
            #expect(!p.models.isEmpty, "\(p.id) names no model")
            do { _ = try executor.validate(p.graph) } catch { Issue.record("\(p.id): \(error)") }
            // Portable: no absolute model paths or output folders baked in.
            for n in p.graph.nodes {
                if n.type == "LoadModel", case .value(.string(let path))? = n.inputs["path"] {
                    #expect(!path.contains("/"), "\(p.id): model path \(path)")
                }
                if case .value(.string(let d))? = n.inputs["directory"] {
                    #expect(d == Preset.outputPlaceholder, "\(p.id): directory \(d)")
                }
            }
        }
    }
}
