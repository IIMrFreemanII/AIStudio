import Foundation
import Testing
@testable import Agent

/// Commands start with default signal handling, whatever this process ignores: an app the agent launches must stop
/// on a plain SIGTERM.
struct ShellRunnerSignalTests {
    @Test func aCommandDoesNotInheritAnIgnoredSigterm() async throws {
        let previous = signal(SIGTERM, SIG_IGN)
        defer { signal(SIGTERM, previous) }
        let dir = FileManager.default.temporaryDirectory
        let r = await ShellRunner().run("kill -TERM $$; echo alive", cwd: dir, timeout: 20)
        #expect(!r.output.contains("alive"), "the shell ignored SIGTERM: \(r.output)")
    }
}
