import Foundation
import Testing
@testable import LLMClient

/// The GPU server stores uploads by content (`<sha256>.<ext>`): a `ui_record` recording gets such a name.
struct MediaUploadNameTests {
    private func file(_ name: String, _ text: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("upload-name-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try Data(text.utf8).write(to: url)
        return url
    }

    @Test func aRecordingIsNamedByItsHash() throws {
        let url = try file("recording-20261114-151520.MP4", "abc")
        // sha256("abc")
        #expect(try LLMServerClient.contentName(url)
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad.mp4")
    }

    @Test func aContentNameIsKept() throws {
        let name = String(repeating: "0", count: 64) + ".mp4"
        #expect(try LLMServerClient.contentName(try file(name, "not the hash")) == name)
    }
}
