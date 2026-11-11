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
        do { try p.run() } catch { return ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin" }
        p.waitUntilExit()
        let s = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
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
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else { return Result(output: "couldn't create a pipe", status: -1, timedOut: false, cancelled: false) }
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
            return Result(output: "couldn't start the shell (\(String(cString: strerror(spawned))))", status: -1,
                          timedOut: false, cancelled: false)
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
            let n = read(fds[0], &buf, buf.count)
            if n <= 0 { break }
            let chunk = Data(buf[0..<n])
            data.append(chunk)
            onOutput?(String(decoding: chunk, as: UTF8.self))
        }
        close(fds[0])
        var status: Int32 = 0
        waitpid(pid, &status, 0)
        timer.cancel()
        // Children that kept running in the background go too.
        kill(-pid, SIGKILL)
        let code: Int32 = (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
        return Result(output: String(decoding: data, as: UTF8.self), status: code, timedOut: timedOut.isSet,
                      cancelled: box.isCancelled)
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
