import Foundation

/// Calls `onChange` (on the main actor, coalesced) when the file at `url` is written, replaced or removed. A file
/// replaced atomically (a new inode, as the coding agent writes) is followed: the watch is reopened on the path, and
/// retried while the path is missing so a recreated file is seen.
final class FileWatcher: @unchecked Sendable {
    private let url: URL
    private let onChange: @MainActor () -> Void
    private let queue = DispatchQueue(label: "FileWatcher")
    // All state below is touched on `queue` only.
    private var source: DispatchSourceFileSystemObject?
    private var stopped = false
    private var pending: DispatchWorkItem?
    private var missingSince: Date?

    init(url: URL, onChange: @escaping @MainActor () -> Void) {
        self.url = url
        self.onChange = onChange
        queue.async { [self] in open(initial: true) }
    }

    func stop() {
        queue.async { [self] in
            stopped = true
            pending?.cancel()
            source?.cancel()
            source = nil
        }
    }

    private func open(initial: Bool = false) {
        guard !stopped else { return }
        let fd = Darwin.open(url.path, O_EVTONLY)
        if fd < 0 {
            if missingSince == nil { missingSince = Date() }
            // 50 ms for the first 2 s, then every second.
            let delay: Double = Date().timeIntervalSince(missingSince!) < 2 ? 0.05 : 1
            queue.asyncAfter(deadline: .now() + delay) { self.open() }
            return
        }
        let reopened = missingSince != nil
        missingSince = nil
        let src = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd, eventMask: [.write, .extend, .delete, .rename, .attrib, .link], queue: queue)
        src.setEventHandler { [weak self, weak src] in
            guard let self, let src, !self.stopped else { return }
            let event = src.data
            self.fire()
            if !event.isDisjoint(with: [.delete, .rename]) {
                src.cancel()
                if self.source === src { self.source = nil }
                self.queue.asyncAfter(deadline: .now() + 0.05) { self.open() }
            }
        }
        src.setCancelHandler { Darwin.close(fd) }
        source = src
        src.resume()
        if reopened { fire() }
    }

    private func fire() {
        pending?.cancel()
        let work = DispatchWorkItem { [onChange] in
            Task { @MainActor in onChange() }
        }
        pending = work
        queue.asyncAfter(deadline: .now() + 0.1, execute: work)
    }
}
