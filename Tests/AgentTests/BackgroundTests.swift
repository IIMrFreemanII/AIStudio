import Foundation
import Testing
@testable import Agent

private func waitUntil(_ seconds: Double = 3, _ cond: () -> Bool) -> Bool {
    let end = Date().addingTimeInterval(seconds)
    while Date() < end {
        if cond() { return true }
        Thread.sleep(forTimeInterval: 0.05)
    }
    return cond()
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func bump() { lock.withLock { n += 1 } }
    var value: Int { lock.withLock { n } }
}

struct BackgroundTests {
    let cwd = FileManager.default.temporaryDirectory

    @Test func echoThenSleepRunsUntilStopped() throws {
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        let job = try jobs.start("echo hi; sleep 30", cwd: cwd)
        #expect(waitUntil { job.output().contains("hi") })
        #expect(job.readNew().contains("hi"))
        #expect(job.isRunning)
        #expect(job.status == nil)
        let t = Date()
        job.stop()
        job.stop()
        #expect(waitUntil { !job.isRunning })
        #expect(Date().timeIntervalSince(t) < 3)
        #expect(job.status != nil)
    }

    @Test func exitStatus() throws {
        let jobs = BackgroundJobs()
        let job = try jobs.start("exit 3", cwd: cwd)
        #expect(waitUntil { !job.isRunning })
        #expect(job.status == 3)
    }

    @Test func readNewReturnsOnlyNewText() throws {
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        let job = try jobs.start("echo one; sleep 1; echo two; sleep 30", cwd: cwd)
        #expect(waitUntil { job.output().contains("one") })
        let first = job.readNew()
        #expect(first.contains("one") && !first.contains("two"))
        #expect(waitUntil { job.output().contains("two") })
        let second = job.readNew()
        #expect(second.contains("two") && !second.contains("one"))
        #expect(job.readNew() == "")
        #expect(job.output().contains("one") && job.output().contains("two"))
    }

    @Test func stopEndsChildren() throws {
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        let job = try jobs.start("sleep 100 & echo $!; wait", cwd: cwd)
        #expect(waitUntil { job.output().contains("\n") })
        let pid = try #require(Int32(job.output().trimmingCharacters(in: .whitespacesAndNewlines)))
        #expect(kill(pid, 0) == 0)
        job.stop()
        #expect(waitUntil { !job.isRunning })
        #expect(waitUntil { kill(pid, 0) != 0 })
    }

    @Test func idsStopAllAndOnChange() throws {
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        let changes = Counter()
        jobs.onChange = { changes.bump() }
        let a = try jobs.start("sleep 30", cwd: cwd)
        let b = try jobs.start("sleep 30", cwd: cwd)
        #expect(a.id == "job1" && b.id == "job2")
        #expect(jobs.all.map(\.id) == ["job1", "job2"])
        #expect(jobs.job("job2") === b)
        #expect(jobs.job("job9") == nil)
        #expect(changes.value >= 2)
        #expect(jobs.running.count == 2)
        jobs.stopAll()
        #expect(waitUntil { jobs.running.isEmpty })
        #expect(waitUntil { changes.value >= 4 })
    }

    @Test func runCommandInBackground() async throws {
        let ctx = ToolContext(root: try tempProject())
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        ctx.jobs = jobs
        let args = try ToolArgs(json: #"{"command":"echo early; sleep 30","background":true}"#)
        let tool = RunCommandTool()
        #expect(try tool.preview(args, ctx).title == "Run in background: echo early; sleep 30")
        let out = try await tool.run(args, ctx)
        #expect(!out.isError)
        #expect(out.text.contains("Started job1 in the background."))
        #expect(out.text.contains("early"))
        #expect(out.text.contains("command_output"))
        #expect(jobs.job("job1")?.isRunning == true)

        let quick = try await tool.run(try ToolArgs(json: #"{"command":"echo x; exit 2","background":true}"#), ctx)
        #expect(quick.isError)
        #expect(quick.text.contains("already ended with exit status 2"))
    }

    @Test func runCommandBackgroundWithoutJobs() async throws {
        let ctx = ToolContext(root: try tempProject())
        let out = try await RunCommandTool().run(try ToolArgs(json: #"{"command":"echo hi","background":true}"#), ctx)
        #expect(out.isError)
        #expect(out.text.contains("aren't available"))
    }

    @Test func commandOutputWaitsAndStopCommand() async throws {
        let ctx = ToolContext(root: try tempProject())
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        ctx.jobs = jobs
        _ = try jobs.start("sleep 1; echo later; sleep 30", cwd: ctx.root)
        let read = CommandOutputTool()
        let none = try await read.run(try ToolArgs(json: #"{"id":"job1"}"#), ctx)
        #expect(none.text.contains("job1: running") && none.text.contains("(no new output)"))
        let waited = try await read.run(try ToolArgs(json: #"{"id":"job1","wait":5}"#), ctx)
        #expect(waited.text.contains("later"))

        let stopped = try await StopCommandTool().run(try ToolArgs(json: #"{"id":"job1"}"#), ctx)
        #expect(stopped.text.hasPrefix("Stopped job1."))
        #expect(jobs.job("job1")?.isRunning == false)
        let ended = try await read.run(try ToolArgs(json: #"{"id":"job1"}"#), ctx)
        #expect(ended.text.contains("job1: ended with exit status"))
    }

    @Test func unknownIds() async throws {
        let ctx = ToolContext(root: try tempProject())
        let none = try await CommandOutputTool().run(try ToolArgs(json: #"{"id":"job7"}"#), ctx)
        #expect(none.isError)
        #expect(none.text == "There is no background command job7. None are running.")
        let jobs = BackgroundJobs()
        defer { jobs.stopAll() }
        ctx.jobs = jobs
        _ = try jobs.start("sleep 30", cwd: ctx.root)
        let stop = try await StopCommandTool().run(try ToolArgs(json: #"{"id":"nope"}"#), ctx)
        #expect(stop.isError)
        #expect(stop.text == "There is no background command nope. Running: job1.")
    }
}
