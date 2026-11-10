import Darwin
import Foundation
import IOKit
import Metal

/// Host facts and live gauges for `/v1/info` and the dashboard.
enum SystemInfo {
    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    static let chip: String = sysctlString("machdep.cpu.brand_string") ?? "Unknown"
    static let gpu: String = MTLCreateSystemDefaultDevice()?.name ?? "No Metal device"
    static let hostName: String = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
    static let osVersion: String = ProcessInfo.processInfo.operatingSystemVersionString
    static let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)

    static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// GPU busy percentage from the accelerator's `PerformanceStatistics` (no privileges needed), or nil.
    static func gpuUtilization() -> Double? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS
        else { return nil }
        defer { IOObjectRelease(iterator) }
        var best: Double?
        while case let entry = IOIteratorNext(iterator), entry != 0 {
            defer { IOObjectRelease(entry) }
            guard let cf = IORegistryEntryCreateCFProperty(entry, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0),
                  let stats = cf.takeRetainedValue() as? [String: Any] else { continue }
            if let v = (stats["Device Utilization %"] ?? stats["GPU Activity(%)"]) as? NSNumber {
                best = max(best ?? 0, v.doubleValue)
            }
        }
        return best
    }

    /// Free and total bytes of the volume holding `url`.
    static func disk(_ url: URL) -> DiskSpace? {
        guard let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeTotalCapacityKey])
        else { return nil }
        return DiskSpace(free: Int(v.volumeAvailableCapacityForImportantUsage ?? 0), total: v.volumeTotalCapacity ?? 0)
    }

    struct DiskSpace { var free: Int; var total: Int }

    /// Process CPU time (user + system), seconds.
    static func cpuSeconds() -> Double {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        func s(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }
        return s(u.ru_utime) + s(u.ru_stime)
    }

    /// Recursive size of a folder's regular files.
    static func folderBytes(_ url: URL) -> Int {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total = 0
        for case let f as URL in e {
            let v = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true { total += v?.fileSize ?? 0 }
        }
        return total
    }
}
