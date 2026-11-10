import Foundation
import Testing
@testable import MLXDiffusionCore

/// Data written before the rename (MLXComfyUI → AIStudio) is found under the new names.
@Suite struct LegacyNamesTests {
    @Test func modelFolderFilesAreRenamedOnceAndNeverOverwritten() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("legacy-model-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: dir) }
        let sub = dir.appendingPathComponent("transformer")
        try fm.createDirectory(at: dir.appendingPathComponent(".mlxcomfy-cache"), withIntermediateDirectories: true)
        try fm.createDirectory(at: sub, withIntermediateDirectories: true)
        try Data(#"{"class":"WanPipeline"}"#.utf8).write(to: dir.appendingPathComponent("mlxcomfy.json"))
        try Data("q8".utf8).write(to: dir.appendingPathComponent(".mlxcomfy-cache/transformer-q8.safetensors"))
        try Data("link".utf8).write(to: sub.appendingPathComponent("mlxcomfy-link.json"))
        // A new-name file that already exists wins over the old one.
        try Data("new".utf8).write(to: sub.appendingPathComponent("aistudio.json"))
        try Data("old".utf8).write(to: sub.appendingPathComponent("mlxcomfy.json"))

        #expect(ModelRegistry.className(at: dir) == "WanPipeline")
        #expect(!fm.fileExists(atPath: dir.appendingPathComponent("mlxcomfy.json").path))
        #expect(fm.fileExists(atPath: dir.appendingPathComponent(".aistudio-cache/transformer-q8.safetensors").path))
        #expect(fm.fileExists(atPath: sub.appendingPathComponent("aistudio-link.json").path))
        #expect(try String(contentsOf: sub.appendingPathComponent("aistudio.json"), encoding: .utf8) == "new")
        #expect(fm.fileExists(atPath: sub.appendingPathComponent("mlxcomfy.json").path))
    }

    @Test func pathsUnderTheOldFoldersPointAtTheNewOnes() throws {
        let (old, new) = try #require(LegacyNames.folderMoves.first)
        #expect(LegacyNames.upgradedPath(old.path + "/models/flux") == new.path + "/models/flux")
        #expect(LegacyNames.upgradedPath(old.path) == new.path)
        #expect(LegacyNames.upgradedPath(old.path + "-other/x") == old.path + "-other/x")
        #expect(LegacyNames.upgradedPath("/tmp/models") == "/tmp/models")
    }
}
