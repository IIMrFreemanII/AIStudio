import Foundation
import MLX
import MLXDiffusionCore

/// What ``ModelImporter/inspect(_:)`` finds in a checkpoint, before anything is converted.
public struct CheckpointInfo: Sendable {
    /// The `.safetensors` files read (one, or the shards of a diffusers folder; for a Wan folder with both experts,
    /// the high-noise expert's).
    public var files: [URL]
    public var architecture: Architecture
    public var schema: KeySchema
    /// The prefix the transformer sits under (`model.diffusion_model.` in ComfyUI checkpoints, else empty).
    public var prefix: String
    /// The stored dtype of most weights (`BF16`, `F16`, `F32`, `F8_E4M3`).
    public var weightDtype: String
    public var transformerTensors: Int
    public var transformerBytes: Int
    /// Top-level names of the other parts of a full checkpoint (`vae`, `text_encoders`…), which are not imported.
    public var droppedComponents: [String]
    /// Wan only: the experts found (both for a folder with `high_noise_model` + `low_noise_model` or diffusers'
    /// `transformer` + `transformer_2`; one when the file or folder name says which; empty when it can't tell).
    public var wanExperts: [WanExpert]
}

/// Progress of an import, reported as each tensor is written.
public struct ImportProgress: Sendable {
    public enum Stage: String, Sendable { case checking, writing, linking, done }
    public var stage: Stage
    /// Overall fraction done, 0…1 (by bytes written).
    public var fraction: Double
    public var message: String
}

/// The outcome of an import.
public struct ImportResult: Sendable {
    /// The new model folder.
    public var destination: URL
    /// The family as the app names it (`flux.1`, `qwen-image-2.1`, `wan-2.2-t2v`, `wan-2.2-i2v`).
    public var family: String
    public var schema: KeySchema
    public var sourceDtype: String
    /// Bits of the stored transformer (nil when kept dense, as for a bf16 Wan base).
    public var bits: Int?
    public var bytesWritten: Int
    public var droppedComponents: [String]
    /// Wan only: which experts were imported (the others link to the base model's).
    public var importedExperts: [WanExpert]
}

/// Installs a fine-tuned transformer as a model folder the app loads like the base model: the converted transformer
/// (already quantized) plus links to the installed base model's text encoders, tokenizers, VAE and scheduler.
///
/// Layouts written into `<models>/<id>/`:
/// - FLUX.1 dev: `transformer/{diffusion_pytorch_model.safetensors (8-bit), config.json}`, a copy of
///   `model_index.json`, and linked `text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2`, `vae`, `scheduler`;
/// - Qwen-Image 2.1: `transformer/{model.safetensors, config.json}` quantized like the base's, `model_index.json`,
///   linked `text_encoder`, `processor`, `vae`, `scheduler`;
/// - Wan 2.2: `config.json` (the base's, so the same quantization bits), `high_noise_model.safetensors` and
///   `low_noise_model.safetensors` (an expert not imported links to the base's), `aistudio.json`, linked `common`,
///   `lora` and the base's other weight files.
/// plus `aistudio-link.json` recording the base, the source and the conversion. A linked folder is a real folder
/// mirroring the base's, with a symbolic link per file: `FileManager.contentsOfDirectory` (which the loaders list
/// folders with) does not follow a link to a directory. Links are relative when the base sits in the same folder, so
/// the models folder can be moved.
///
/// Conversion reads the source one tensor at a time and writes each output tensor as it is computed, so memory stays
/// around the largest tensor rather than the model. Everything is written to a hidden `.<id>.partial-…` folder that is
/// renamed at the end, so a failed or cancelled import (`Task` cancellation is checked between tensors) leaves
/// nothing behind.
public enum ModelImporter {
    public static let linkManifestName = "aistudio-link.json"

    // MARK: inspect

    /// Reads a checkpoint's headers and says what it is (no tensor data is read).
    ///
    /// Throws for GGUF, NF4/bitsandbytes, already-quantized and unrecognised files, with a message for the user.
    public static func inspect(_ url: URL) throws -> CheckpointInfo {
        let wan = try? wanSources(url, expert: nil)
        let files = try wan.flatMap { $0[.high] ?? $0[.low] } ?? sourceFiles(url)
        let src = try SourceCheckpoint(files: files)
        var experts: [WanExpert] = []
        if src.layout.architecture == .wan22 {
            experts = wan.map { WanExpert.allCases.filter($0.keys.contains) } ?? []
        }
        return CheckpointInfo(files: files, architecture: src.layout.architecture, schema: src.layout.schema,
                              prefix: src.layout.prefix, weightDtype: src.weightDtype,
                              transformerTensors: src.keys.count, transformerBytes: src.transformerBytes,
                              droppedComponents: src.layout.droppedComponents, wanExperts: experts)
    }

    // MARK: import

    /// Converts the fine-tuned transformer at `source` and assembles it with `base` into `destination`.
    ///
    /// - Parameters:
    ///   - source: a `.safetensors` file (Civitai style, transformer-only or a full checkpoint with VAE/text encoders,
    ///     which are ignored) or a diffusers folder (the model folder or its `transformer/`). For Wan, a folder with
    ///     both experts imports both; a single file imports one expert (`wanExpert`, else guessed from the name).
    ///   - family: `flux.1`, `qwen-image-2.1`, `wan-2.2` (or the pipeline class name); `nil` takes the base's.
    ///   - base: the installed base model folder (e.g. `<models>/flux-1-dev`).
    ///   - destination: the new model folder `<models>/<id>`; must not exist.
    ///   - sourceInfo: free-form strings stored in `aistudio-link.json` under `"source"` (page URL, version id…).
    @discardableResult
    public static func importTransformer(from source: URL, family: String? = nil, base: URL, into destination: URL,
                                         wanExpert: WanExpert? = nil, sourceInfo: [String: String] = [:],
                                         progress: (ImportProgress) -> Void = { _ in }) throws -> ImportResult {
        let arch = try architecture(family: family, base: base)
        if arch == .wan22 {
            let experts = try wanSources(source, expert: wanExpert)
            return try importWan(experts, base: base, into: destination, sourceName: source.lastPathComponent,
                                 sourceInfo: sourceInfo, progress: progress)
        }
        return try importImage(arch, files: sourceFiles(source), base: base, into: destination,
                               sourceName: source.lastPathComponent, sourceInfo: sourceInfo, progress: progress)
    }

    /// Wan 2.2 with each expert from its own file or folder; a `nil` expert links to the base model's.
    @discardableResult
    public static func importWan(high: URL?, low: URL?, base: URL, into destination: URL,
                                 sourceInfo: [String: String] = [:],
                                 progress: (ImportProgress) -> Void = { _ in }) throws -> ImportResult {
        guard try architecture(family: nil, base: base) == .wan22 else {
            throw ModelImportError.badBase("\(base.lastPathComponent) is not a Wan 2.2 model folder")
        }
        var experts: [WanExpert: [URL]] = [:]
        if let high { experts[.high] = try sourceFiles(high) }
        if let low { experts[.low] = try sourceFiles(low) }
        guard !experts.isEmpty else { throw ModelImportError.wanExpertUnknown("no source") }
        let name = [high, low].compactMap { $0?.lastPathComponent }.joined(separator: " + ")
        return try importWan(experts, base: base, into: destination, sourceName: name, sourceInfo: sourceInfo,
                             progress: progress)
    }

    // MARK: FLUX / Qwen

    static func importImage(_ arch: Architecture, files: [URL], base: URL, into destination: URL, sourceName: String,
                            sourceInfo: [String: String], progress: (ImportProgress) -> Void) throws -> ImportResult {
        progress(ImportProgress(stage: .checking, fraction: 0, message: "reading \(sourceName)"))
        let baseT = base.appendingPathComponent("transformer")
        guard let baseHeaders = try? SafetensorsHeader.directory(baseT), !baseHeaders.isEmpty else {
            throw ModelImportError.badBase("\(base.lastPathComponent) has no transformer weights to check against")
        }
        let reference = BaseTransformer(headers: baseHeaders, bitsHint: nil)
        var prepared = try prepare(files: files, expect: arch, base: reference, defaultBits: 8,
                                   familyName: arch.rawValue, baseName: base.lastPathComponent)
        try checkSpace(prepared.bytes, at: destination)
        let summary = prepared.summary

        let fm = FileManager.default
        return try assemble(destination) { tmp in
            let tDir = tmp.appendingPathComponent("transformer")
            try fm.createDirectory(at: tDir, withIntermediateDirectories: true)
            let file = tDir.appendingPathComponent(arch == .flux1 ? "diffusion_pytorch_model.safetensors"
                : "model.safetensors")
            let total = prepared.bytes
            try SafetensorsWriter.write(&prepared.tensors, metadata: prepared.metadata, to: file) { done in
                try Task.checkCancellation()
                progress(ImportProgress(stage: .writing, fraction: Double(done) / Double(max(total, 1)) * 0.99,
                                        message: "converting the transformer"))
            }
            var config = (try? JSONSerialization.jsonObject(with: Data(contentsOf: baseT.appendingPathComponent(
                "config.json"))) as? [String: Any]) ?? [:]
            if let bits = summary.bits, arch == .flux1 {
                config["quantization"] = ["bits": bits, "group_size": summary.groupSize]
            }
            try writeJSON(config, to: tDir.appendingPathComponent("config.json"))

            progress(ImportProgress(stage: .linking, fraction: 0.99, message: "linking the base model's parts"))
            for entry in try entries(of: base) {
                if entry.name == "transformer" || entry.name == linkManifestName { continue }
                if entry.name == "model_index.json" || entry.name == "aistudio.json" {
                    try fm.copyItem(at: entry.url, to: tmp.appendingPathComponent(entry.name))
                } else if entry.isDirectory {
                    try mirror(entry.url, at: tmp.appendingPathComponent(entry.name),
                               final: destination.appendingPathComponent(entry.name))
                }
            }
            let result = ImportResult(destination: destination, family: arch.rawValue, schema: summary.schema,
                                      sourceDtype: summary.dtype, bits: summary.bits, bytesWritten: total,
                                      droppedComponents: summary.dropped, importedExperts: [])
            try writeManifest(result, base: base, sourceName: sourceName, sourceInfo: sourceInfo,
                              prefix: summary.prefix, groupSize: summary.groupSize, into: tmp)
            progress(ImportProgress(stage: .done, fraction: 1, message: "imported"))
            return result
        }
    }

    // MARK: Wan

    static func importWan(_ experts: [WanExpert: [URL]], base: URL, into destination: URL, sourceName: String,
                          sourceInfo: [String: String], progress: (ImportProgress) -> Void) throws -> ImportResult {
        progress(ImportProgress(stage: .checking, fraction: 0, message: "reading \(sourceName)"))
        let fm = FileManager.default
        guard let modelDir = ["", "common", "q4", "q8", "bf16"].map({ $0.isEmpty ? base : base.appendingPathComponent($0) })
            .first(where: { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }) else {
            throw ModelImportError.badBase("\(base.lastPathComponent) has no Wan config.json")
        }
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = (try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]) ?? [:]
        let quant = config["quantization"] as? [String: Any]
        let family = (config["in_dim"] as? Int ?? 16) > 16 ? "wan-2.2-i2v" : "wan-2.2-t2v"

        var plans: [WanExpert: Prepared] = [:]
        for (expert, files) in experts.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            let baseFile = modelDir.appendingPathComponent(expert.fileName)
            guard let header = try? SafetensorsHeader(url: baseFile) else {
                throw ModelImportError.badBase("\(base.lastPathComponent) has no \(expert.fileName)")
            }
            let reference = BaseTransformer(headers: [header], bitsHint: quant?["bits"] as? Int)
            plans[expert] = try prepare(files: files, expect: .wan22, base: reference, defaultBits: nil,
                                        familyName: family, baseName: base.lastPathComponent)
        }
        let total = plans.values.reduce(0) { $0 + $1.bytes }
        let summary = plans.sorted { $0.key.rawValue < $1.key.rawValue }[0].value.summary
        try checkSpace(total, at: destination)

        return try assemble(destination) { tmp in
            var done = 0
            for expert in WanExpert.allCases {
                let out = tmp.appendingPathComponent(expert.fileName)
                guard var p = plans.removeValue(forKey: expert) else {
                    try link(out, to: modelDir.appendingPathComponent(expert.fileName), finalDir: destination)
                    continue
                }
                let before = done
                try SafetensorsWriter.write(&p.tensors, metadata: p.metadata, to: out) { n in
                    try Task.checkCancellation()
                    done = before + n
                    progress(ImportProgress(stage: .writing, fraction: Double(done) / Double(max(total, 1)) * 0.99,
                                            message: "converting the \(expert.rawValue)-noise expert"))
                }
            }
            try fm.copyItem(at: configURL, to: tmp.appendingPathComponent("config.json"))

            progress(ImportProgress(stage: .linking, fraction: 0.99, message: "linking the base model's parts"))
            for entry in try entries(of: base) {
                let n = entry.name
                if n == linkManifestName || n == "config.json" || WanExpert.allCases.map(\.fileName).contains(n) {
                    continue
                }
                if entry.url.standardizedFileURL == modelDir.standardizedFileURL { continue }  // q4/ etc.
                if n == "aistudio.json" {
                    try fm.copyItem(at: entry.url, to: tmp.appendingPathComponent(n))
                } else if entry.isDirectory {
                    try mirror(entry.url, at: tmp.appendingPathComponent(n), final: destination.appendingPathComponent(n))
                } else if ["safetensors", "json"].contains(entry.url.pathExtension) {
                    try link(tmp.appendingPathComponent(n), to: entry.url, finalDir: destination)
                }
            }
            if !fm.fileExists(atPath: tmp.appendingPathComponent("aistudio.json").path) {
                try writeJSON(["class": "WanPipeline"], to: tmp.appendingPathComponent("aistudio.json"))
            }
            let result = ImportResult(destination: destination, family: family, schema: summary.schema,
                                      sourceDtype: summary.dtype, bits: summary.bits, bytesWritten: total,
                                      droppedComponents: summary.dropped,
                                      importedExperts: WanExpert.allCases.filter { experts[$0] != nil })
            try writeManifest(result, base: base, sourceName: sourceName, sourceInfo: sourceInfo,
                              prefix: summary.prefix, groupSize: summary.groupSize, into: tmp)
            progress(ImportProgress(stage: .done, fraction: 1, message: "imported"))
            return result
        }
    }
}
