import Foundation
import MLX
import MLXDiffusionCore

/// The installed base model's transformer as its headers describe it: what an import must match, and how it is
/// quantized (the import quantizes exactly the weights the base has quantized, with the same bits and group size).
struct BaseTransformer {
    var shapes: [String: [Int]] = [:]
    /// `<layer>.weight` keys that the base stores quantized.
    var quantized: Set<String> = []
    var bits = 8
    var groupSize = 64

    var isQuantized: Bool { !quantized.isEmpty }

    init(headers: [SafetensorsHeader], bitsHint: Int?) {
        var meta: [String: String] = [:]
        for h in headers {
            for (k, e) in h.tensors { shapes[k] = e.shape }
            meta.merge(h.metadata) { _, new in new }
        }
        quantized = Set(shapes.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(".scales".count)) + ".weight" })
        bits = bitsHint ?? meta["quantization_level"].flatMap { Int($0) } ?? 8
        if let w = quantized.sorted().first, let ws = shapes[w], let ss = shapes[String(w.dropLast(".weight".count)) + ".scales"],
           ws.count == 2, ss.count == 2, ss[1] > 0 {
            groupSize = ws[1] * 32 / bits / ss[1]
        }
    }
}

/// What a converted source is written from: the planned tensors (lazy) and a summary for the result and manifest.
struct Prepared {
    struct Summary {
        var schema: KeySchema
        var dtype: String
        var prefix: String
        var dropped: [String]
        var bits: Int?
        var groupSize: Int
    }

    var tensors: [PlannedTensor]
    var metadata: [String: String]
    var bytes: Int
    var summary: Summary
}

extension ModelImporter {
    /// Opens the source, maps its names, plans the quantization like the base's and checks the result against it.
    ///
    /// The source checkpoint object is released on return: only the lazy output arrays reference its tensors, so
    /// each source tensor is freed once the outputs computed from it are written.
    static func prepare(files: [URL], expect: Architecture, base: BaseTransformer, defaultBits: Int?,
                        familyName: String, baseName: String) throws -> Prepared {
        let src = try SourceCheckpoint(files: files)
        guard src.layout.architecture == expect else {
            throw ModelImportError.wrongFamily(expected: familyName, found: src.layout.architecture.rawValue)
        }
        let named: [Convert.Named]
        switch expect {
        case .flux1: named = try Convert.flux(src)
        case .qwenImage21: named = try Convert.qwen(src)
        case .wan22: named = try Convert.wan(src)
        }
        let tensors: [PlannedTensor]
        let details: [String]
        let bits: Int?, groupSize: Int
        if base.isQuantized {
            (bits, groupSize) = (base.bits, base.groupSize)
            tensors = Convert.quantize(named, bits: bits, groupSize: groupSize) { name, _ in base.quantized.contains(name) }
            details = Convert.mismatches(tensors.map { ($0.name, $0.array.shape) }, expected: base.shapes)
        } else {
            (bits, groupSize) = (defaultBits, 64)
            details = Convert.mismatches(named.map { ($0.name, $0.array.shape) }, expected: base.shapes)
            tensors = Convert.quantize(named, bits: bits, groupSize: groupSize,
                                       should: Convert.defaultShouldQuantize(groupSize: groupSize))
        }
        if !details.isEmpty {
            throw ModelImportError.architectureMismatch(family: familyName, base: baseName, details: details,
                                                        hint: mismatchHint(expect, named: named, base: base))
        }
        var metadata = ["format": "mlx", "mlxcomfy_import": "1"]
        if let bits { metadata["quantization_level"] = "\(bits)" }
        return Prepared(tensors: tensors, metadata: metadata, bytes: tensors.reduce(0) { $0 + $1.byteCount },
                        summary: .init(schema: src.layout.schema, dtype: src.weightDtype, prefix: src.layout.prefix,
                                       dropped: src.layout.droppedComponents, bits: bits, groupSize: groupSize))
    }

    /// A guess at what the checkpoint is when it doesn't fit the base.
    static func mismatchHint(_ arch: Architecture, named: [Convert.Named], base: BaseTransformer) -> String? {
        let have = Dictionary(named.map { ($0.name, $0.array.shape) }, uniquingKeysWith: { a, _ in a })
        switch arch {
        case .flux1:
            let g = "time_text_embed.guidance_embedder.linear_1.weight"
            if base.shapes[g] != nil, have[g] == nil {
                return "it has no guidance embedding: probably FLUX.1 schnell or a de-distilled model; "
                    + "only FLUX.1 dev fine-tunes can use this base"
            }
        case .wan22:
            let k = "patch_embedding_proj.weight"
            if let h = have[k], let b = base.shapes[k], h.count == 2, b.count == 2, h[0] == b[0], h[1] != b[1] {
                let kind = { (w: Int) in w > 64 ? "image-to-video (I2V)" : "text-to-video (T2V)" }
                return "the checkpoint is \(kind(h[1])) but the base model is \(kind(b[1]))"
            }
            if let h = have[k], let b = base.shapes[k], h.first != b.first {
                return "a different model size (for example Wan 2.2 TI2V-5B); only A14B fine-tunes can use this base"
            }
        case .qwenImage21:
            break
        }
        return nil
    }

    // MARK: sources

    static func architecture(family: String?, base: URL) throws -> Architecture {
        let fromBase: Architecture? = switch ModelRegistry.className(at: base) {
        case "FluxPipeline": .flux1
        case "QwenImage21Pipeline": .qwenImage21
        case "WanPipeline": .wan22
        default: nil
        }
        var asked: Architecture?
        if let family {
            let f = family.lowercased()
            asked = f.contains("flux") ? .flux1 : f.contains("qwen") ? .qwenImage21 : f.contains("wan") ? .wan22 : nil
            if asked == nil { throw ModelImportError.unsupportedVariant("\(family) models can't be imported") }
        }
        guard let fromBase else {
            throw ModelImportError.badBase(
                "\(base.lastPathComponent) is not an installed FLUX.1, Qwen-Image 2.1 or Wan 2.2 model folder")
        }
        if let asked, asked != fromBase {
            throw ModelImportError.badBase("\(base.lastPathComponent) is a \(fromBase.rawValue) model, not \(asked.rawValue)")
        }
        return fromBase
    }

    /// The `.safetensors` of one checkpoint: the file itself, or a folder's shards (directly or in `transformer/`).
    static func sourceFiles(_ url: URL) throws -> [URL] {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        guard isDir.boolValue else { return [url] }
        let root = try safetensorsFiles(in: url)
        if !root.isEmpty { return root }
        let t = url.appendingPathComponent("transformer")
        if let files = try? safetensorsFiles(in: t), !files.isEmpty { return files }
        throw ModelImportError.noTensors(url.lastPathComponent)
    }

    /// Wan experts in a source: `high_noise_model/` + `low_noise_model/` (official), `transformer/` +
    /// `transformer_2/` (diffusers: high, then low), `high_noise_model*.safetensors` files, or one expert (`expert`
    /// or guessed from the name). With `expert` given, only that one is taken from a two-expert folder.
    static func wanSources(_ url: URL, expert: WanExpert?) throws -> [WanExpert: [URL]] {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        var found: [WanExpert: [URL]] = [:]
        if isDir.boolValue {
            for (h, l) in [("high_noise_model", "low_noise_model"), ("transformer", "transformer_2")] {
                for (e, name) in [(WanExpert.high, h), (.low, l)] {
                    if let files = try? safetensorsFiles(in: url.appendingPathComponent(name)), !files.isEmpty {
                        found[e] = files
                    }
                }
                if !found.isEmpty { break }
            }
            if found.isEmpty {
                for f in try safetensorsFiles(in: url) {
                    for e in WanExpert.allCases where f.lastPathComponent.hasPrefix("\(e.rawValue)_noise_model") {
                        found[e, default: []].append(f)
                    }
                }
            }
        }
        if !found.isEmpty {
            if let expert {
                guard let f = found[expert] else { throw ModelImportError.wanExpertUnknown(url.lastPathComponent) }
                return [expert: f]
            }
            return found
        }
        guard let e = expert ?? WanExpert.guess(url.deletingPathExtension().lastPathComponent) else {
            throw ModelImportError.wanExpertUnknown(url.lastPathComponent)
        }
        return [e: try sourceFiles(url)]
    }

    // MARK: assembling

    static func checkSpace(_ bytes: Int, at destination: URL) throws {
        let parent = destination.deletingLastPathComponent()
        guard let free = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage else { return }
        let needed = Int64(bytes) + 512 << 20
        if needed > free { throw ModelImportError.notEnoughSpace(needed: needed, available: free) }
    }

    /// Runs `body` on a hidden sibling folder of `destination` and renames it into place, or removes it on failure.
    static func assemble<T>(_ destination: URL, _ body: (URL) throws -> T) throws -> T {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) || (try? fm.destinationOfSymbolicLink(atPath: destination.path)) != nil {
            throw ModelImportError.destinationExists(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let tmp = parent.appendingPathComponent(".\(destination.lastPathComponent).partial-\(UUID().uuidString.prefix(8))")
        try fm.createDirectory(at: tmp, withIntermediateDirectories: false)
        do {
            let result = try body(tmp)
            try fm.moveItem(at: tmp, to: destination)
            return result
        } catch {
            try? fm.removeItem(at: tmp)
            throw error
        }
    }

    struct Entry {
        var name: String
        /// With symbolic links resolved (a base that is itself an import links to the real parts).
        var url: URL
        var isDirectory: Bool
    }

    /// The visible entries of a folder (hidden caches like `.mlxcomfy-cache` are skipped).
    static func entries(of dir: URL) throws -> [Entry] {
        try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted().compactMap { name in
            guard !name.hasPrefix(".") else { return nil }
            let real = dir.appendingPathComponent(name).resolvingSymlinksInPath()
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: real.path, isDirectory: &isDir) else { return nil }
            return Entry(name: name, url: real, isDirectory: isDir.boolValue)
        }
    }

    /// A symbolic link at `link` to `target`, relative to the link's final folder `finalDir` when they share a parent
    /// path below the root (so `../flux-1-dev/vae`), else absolute.
    static func link(_ link: URL, to target: URL, finalDir: URL) throws {
        let t = target.resolvingSymlinksInPath().standardizedFileURL.pathComponents
        let from = (finalDir.deletingLastPathComponent().resolvingSymlinksInPath()
            .appendingPathComponent(finalDir.lastPathComponent)).standardizedFileURL.pathComponents
        var common = 0
        while common < min(t.count, from.count), t[common] == from[common] { common += 1 }
        let dest = common <= 1 ? "/" + t.dropFirst().joined(separator: "/")
            : (Array(repeating: "..", count: from.count - common) + t[common...]).joined(separator: "/")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: dest)
    }

    /// `dir` mirrored at `link` (whose final place is `final`): real folders, a symbolic link per file.
    static func mirror(_ dir: URL, at link: URL, final: URL) throws {
        try FileManager.default.createDirectory(at: link, withIntermediateDirectories: false)
        for e in try entries(of: dir) {
            if e.isDirectory {
                try mirror(e.url, at: link.appendingPathComponent(e.name), final: final.appendingPathComponent(e.name))
            } else {
                try Self.link(link.appendingPathComponent(e.name), to: e.url, finalDir: final)
            }
        }
    }

    static func writeJSON(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object,
                                              options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: url)
    }

    static func writeManifest(_ r: ImportResult, base: URL, sourceName: String, sourceInfo: [String: String],
                              prefix: String, groupSize: Int, into dir: URL) throws {
        var source: [String: Any] = ["file": sourceName]
        for (k, v) in sourceInfo { source[k] = v }
        var conversion: [String: Any] = [
            "schema": r.schema.rawValue, "prefix": prefix, "sourceDtype": r.sourceDtype,
            "bits": r.bits.map { $0 as Any } ?? NSNull(), "groupSize": groupSize, "bytes": r.bytesWritten,
            "droppedComponents": r.droppedComponents,
        ]
        if !r.importedExperts.isEmpty { conversion["importedExperts"] = r.importedExperts.map(\.rawValue) }
        let manifest: [String: Any] = [
            "kind": "fine-tune", "importerVersion": 1, "family": r.family,
            "base": base.lastPathComponent, "basePath": base.resolvingSymlinksInPath().path,
            "created": ISO8601DateFormatter().string(from: Date()),
            "source": source, "conversion": conversion,
        ]
        try writeJSON(manifest, to: dir.appendingPathComponent(linkManifestName))
    }
}
