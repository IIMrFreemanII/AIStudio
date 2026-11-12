import Darwin
import Foundation

/// Runs a shell command for the agent: `/bin/zsh -c` in the project folder, in its own process group (so a timeout
/// or a cancel ends everything it started), stdin closed, output streamed as it comes.
public struct ShellRunner: Sendable {
    public struct Result: Sendable, Equatable {
        public var output: String
        public var status: Int32
        public var timedOut: Bool
        public var cancelled: Bool
    }

    public var shell = "/bin/zsh"
    /// PATH for commands: the login shell's, read once (an app launched from Finder has a minimal one).
    public static let loginPath: String = {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-l", "-c", "printf %s \"$PATH\""]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        p.standardInput = FileHandle.nullDevice
        // Not waitUntilExit(): it spins the run loop, and on the main thread (a SwiftUI body reading this) that can
        // re-enter the body and this initializer, which traps. Reading to EOF and a semaphore block without it.
        let done = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in done.signal() }
        do { try p.run() } catch { return ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        done.wait()
        let s = String(decoding: data, as: UTF8.self)
        return s.isEmpty ? (ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin") : s
    }()

    public init() {}

    /// Runs `command` in `cwd`. `onOutput` gets output chunks as they arrive (on a background thread). Cancelling the
    /// calling task ends the process group.
    public func run(_ command: String, cwd: URL, timeout: TimeInterval = 120,
                    onOutput: (@Sendable (String) -> Void)? = nil) async -> Result {
        let box = RunBox()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Result, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    cont.resume(returning: self.runBlocking(command, cwd: cwd, timeout: timeout, box: box,
                                                            onOutput: onOutput))
                }
            }
        } onCancel: {
            box.cancel()
        }
    }

    func runBlocking(_ command: String, cwd: URL, timeout: TimeInterval, box: RunBox,
                     onOutput: (@Sendable (String) -> Void)?) -> Result {
        let pid: pid_t
        let readFD: Int32
        switch spawnShell(command, cwd: cwd) {
        case .pipeFailed:
            return Result(output: "couldn't create a pipe", status: -1, timedOut: false, cancelled: false)
        case .failed(let err):
            return Result(output: "couldn't start the shell (\(String(cString: strerror(err))))", status: -1,
                          timedOut: false, cancelled: false)
        case .started(let p, let fd):
            pid = p
            readFD = fd
        }
        box.set(pid: pid)

        // A timer ends the group at the timeout: SIGTERM, then SIGKILL two seconds later.
        let timer = DispatchSource.makeTimerSource(queue: .global())
        let timedOut = AtomicFlag()
        timer.schedule(deadline: .now() + timeout)
        timer.setEventHandler {
            timedOut.set()
            box.terminate()
        }
        timer.resume()

        var data = Data()
        var buf = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = read(readFD, &buf, buf.count)
            if n <= 0 { break }
            let chunk = Data(buf[0..<n])
            data.append(chunk)
            onOutput?(String(decoding: chunk, as: UTF8.self))
        }
        close(readFD)
        var status: Int32 = 0
        waitpid(pid, &status, 0)
        timer.cancel()
        // Children that kept running in the background go too.
        kill(-pid, SIGKILL)
        let code: Int32 = (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
        return Result(output: String(decoding: data, as: UTF8.self), status: code, timedOut: timedOut.isSet,
                      cancelled: box.isCancelled)
    }

    enum SpawnOutcome {
        case started(pid_t, Int32)
        case pipeFailed
        case failed(Int32)
    }

    /// Spawns `shell -c command` in `cwd` in its own process group, stdin closed, stdout and stderr into one pipe.
    /// Returns the pid and the pipe's read end.
    func spawnShell(_ command: String, cwd: URL) -> SpawnOutcome {
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else { return .pipeFailed }
        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, fds[1], 1)
        posix_spawn_file_actions_adddup2(&actions, fds[1], 2)
        posix_spawn_file_actions_addclose(&actions, fds[0])
        posix_spawn_file_actions_addclose(&actions, fds[1])
        posix_spawn_file_actions_addchdir_np(&actions, cwd.path)
        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
        posix_spawnattr_setpgroup(&attr, 0)

        var env = ProcessInfo.processInfo.environment
        env["PATH"] = Self.loginPath
        env["PAGER"] = "cat"
        env["GIT_PAGER"] = "cat"
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        env["CLICOLOR"] = "0"
        let argv: [String] = [shell, "-c", command]
        let envList = env.map { "\($0.key)=\($0.value)" }
        var pid: pid_t = 0
        let spawned = withCStrings(argv) { cArgv in
            withCStrings(envList) { cEnv in
                posix_spawn(&pid, shell, &actions, &attr, cArgv, cEnv)
            }
        }
        close(fds[1])
        guard spawned == 0 else {
            close(fds[0])
            return .failed(spawned)
        }
        return .started(pid, fds[0])
    }
}

final class RunBox: @unchecked Sendable {
    private let lock = NSLock()
    private var pid: pid_t = 0
    private var cancelled = false

    func set(pid: pid_t) {
        lock.lock()
        self.pid = pid
        let c = cancelled
        lock.unlock()
        if c { terminate() }
    }

    var isCancelled: Bool { lock.withLock { cancelled } }

    func cancel() {
        lock.withLock { cancelled = true }
        terminate()
    }

    func terminate() {
        let p = lock.withLock { pid }
        guard p > 0 else { return }
        kill(-p, SIGTERM)
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { kill(-p, SIGKILL) }
    }
}

final class AtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set() { lock.withLock { value = true } }
    var isSet: Bool { lock.withLock { value } }
}

private func withCStrings<R>(_ strings: [String], _ body: ([UnsafeMutablePointer<CChar>?]) -> R) -> R {
    var ptrs: [UnsafeMutablePointer<CChar>?] = strings.map { strdup($0) }
    ptrs.append(nil)
    defer { for p in ptrs { free(p) } }
    return body(ptrs)
}

/// A command left running in the background (a dev server, a watcher): its output so far and how it ended.
public final class BackgroundJob: @unchecked Sendable {
    public let id: String
    public let command: String
    public let started: Date

    static let keepLimit = 200_000
    static let readLimit = 30_000

    private let lock = NSLock()
    private let pid: pid_t
    private var buffer = ""
    /// Characters dropped from the front of `buffer` so far.
    private var dropped = 0
    /// Absolute offset (dropped + buffer position) up to which `readNew()` has returned.
    private var readOffset = 0
    private var exitStatus: Int32?
    /// The leader has been reaped: its process group id must not be signalled any more.
    private var exited = false
    private let readerDone = DispatchSemaphore(value: 0)

    /// Spawns the command; nil with the reason when it could not start.
    init(id: String, command: String, cwd: URL, onEnd: (@Sendable () -> Void)?) throws {
        self.id = id
        self.command = command
        started = Date()
        let fd: Int32
        switch ShellRunner().spawnShell(command, cwd: cwd) {
        case .pipeFailed: throw ToolError("couldn't start the command: couldn't create a pipe")
        case .failed(let err): throw ToolError("couldn't start the command: \(String(cString: strerror(err)))")
        case .started(let p, let f):
            pid = p
            fd = f
        }
        DispatchQueue.global().async { [self] in readLoop(fd) }
        let pid = self.pid
        Thread.detachNewThread { [self] in
            var st: Int32 = 0
            while waitpid(pid, &st, 0) < 0 && errno == EINTR {}
            lock.withLock { exited = true }
            // Children that kept running go too, so the reader gets EOF.
            kill(-pid, SIGKILL)
            // Let the reader take the last output (a child in another group may hold the pipe open).
            _ = readerDone.wait(timeout: .now() + 1)
            let code: Int32 = (st & 0x7f) == 0 ? (st >> 8) & 0xff : 128 + (st & 0x7f)
            lock.withLock { exitStatus = code }
            onEnd?()
        }
    }

    private func readLoop(_ fd: Int32) {
        var buf = [UInt8](repeating: 0, count: 65536)
        var pending = Data()
        while true {
            let n = read(fd, &buf, buf.count)
            if n <= 0 { break }
            pending.append(contentsOf: buf[0..<n])
            // Keep an incomplete UTF-8 sequence at the end for the next chunk.
            var cut = pending.count
            var back = 0
            while back < 4, cut > 0, pending[pending.startIndex + cut - 1] & 0xC0 == 0x80 { cut -= 1; back += 1 }
            if cut > 0, pending[pending.startIndex + cut - 1] >= 0xC0 {
                let lead = pending[pending.startIndex + cut - 1]
                let need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2
                if back + 1 < need { cut -= 1 } else { cut = pending.count }
            } else {
                cut = pending.count
            }
            let text = String(decoding: pending.prefix(cut), as: UTF8.self)
            pending = Data(pending.dropFirst(cut))
            append(text)
        }
        if !pending.isEmpty { append(String(decoding: pending, as: UTF8.self)) }
        close(fd)
        readerDone.signal()
    }

    private func append(_ text: String) {
        lock.withLock {
            buffer += text
            let extra = buffer.count - Self.keepLimit
            if extra > 0 {
                buffer.removeFirst(extra)
                dropped += extra
            }
        }
    }

    /// Still running.
    public var isRunning: Bool { lock.withLock { exitStatus == nil } }
    /// The exit status once it has ended.
    public var status: Int32? { lock.withLock { exitStatus } }

    /// The output since the last `readNew()` (clipped to the last 30,000 characters).
    public func readNew() -> String {
        lock.withLock {
            let start = max(readOffset, dropped)
            let new = String(buffer.dropFirst(start - dropped))
            readOffset = dropped + buffer.count
            guard new.count > Self.readLimit else { return new }
            return "… [\(new.count - Self.readLimit) characters left out] …\n" + String(new.suffix(Self.readLimit))
        }
    }

    /// All the output kept (at most the last 200,000 characters).
    public func output() -> String { lock.withLock { buffer } }

    /// Ends it and everything it started: SIGTERM to its process group, SIGKILL two seconds later.
    public func stop() {
        let alive = lock.withLock { !exited }
        guard alive else { return }
        kill(-pid, SIGTERM)
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { [self] in
            if lock.withLock({ !exited }) { kill(-pid, SIGKILL) }
        }
    }
}

/// A chat's background commands. `onChange` is called (on any thread) when one starts, ends or is stopped.
public final class BackgroundJobs: @unchecked Sendable {
    private let lock = NSLock()
    private var _onChange: (@Sendable () -> Void)?
    private var jobs: [BackgroundJob] = []
    private var counter = 0

    public var onChange: (@Sendable () -> Void)? {
        get { lock.withLock { _onChange } }
        set { lock.withLock { _onChange = newValue } }
    }

    public init() {}

    /// Starts `command` (`/bin/zsh -c`, in `cwd`, own process group, stdin closed) and returns it at once.
    public func start(_ command: String, cwd: URL) throws -> BackgroundJob {
        let id = lock.withLock { () -> String in
            counter += 1
            return "job\(counter)"
        }
        let job = try BackgroundJob(id: id, command: command, cwd: cwd) { [weak self] in self?.onChange?() }
        lock.withLock { jobs.append(job) }
        onChange?()
        return job
    }

    public func job(_ id: String) -> BackgroundJob? { lock.withLock { jobs.first { $0.id == id } } }
    /// Every job started, oldest first.
    public var all: [BackgroundJob] { lock.withLock { jobs } }
    public var running: [BackgroundJob] { all.filter(\.isRunning) }
    /// Stops every running job (the chat closes, the app quits).
    public func stopAll() { for j in all where j.isRunning { j.stop() } }
}
