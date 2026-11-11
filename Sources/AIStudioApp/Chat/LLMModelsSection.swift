import LLMClient
import SwiftUI

/// Asks for the Add a Model sheet (optionally on a given server).
struct LLMAddRequest: Identifiable {
    let id = UUID()
    var serverID: String?
}

/// Asks for an installed model's settings sheet.
struct LLMSettingsRequest: Identifiable {
    let id = UUID()
    var key: LLMKey
    /// Snapshots: the plan to show instead of asking the server.
    var preview: LLMPlan? = nil
}

/// The user's logins for model links, read from the Keychain only when they're saved.
@MainActor
enum LinkLogins {
    static var hf: String? { HFToken.isSaved ? HFToken.read() : nil }
    static var civitai: String? { KeychainToken.civitai.isSaved ? KeychainToken.civitai.read() : nil }
}

/// Servers panel section: the language models on the GPU servers, installs in progress, and Add from Link.
struct LLMModelsSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let llm = model.llm
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                MGListHeader(title: "Language Models")
                Button { model.llmAddRequest = LLMAddRequest() } label: { Image(systemName: "plus") }
                    .buttonStyle(.mg(.icon))
                    .disabled(llm.servers.isEmpty)
                    .mgHelp("Add a model from a Hugging Face link").accessibilityLabel("Add a model")
                    .padding(.top, 8)
            }
            ForEach(llm.servers) { s in
                if let error = llm.errors[s.id] {
                    Text("\(s.name): \(error)").font(.mgFootnote).foregroundStyle(MG.destructive)
                        .fixedSize(horizontal: false, vertical: true).padding(.horizontal, 8)
                }
                ForEach((llm.listings[s.id]?.installs ?? [:]).sorted { $0.key < $1.key }, id: \.key) { id, state in
                    InstallRow(id: id, state: state)
                }
                ForEach(llm.listings[s.id]?.models ?? []) { m in
                    let key = LLMKey(serverID: s.id, modelID: m.id)
                    LLMModelRow(model: m, status: llm.status(key), server: llm.servers.count > 1 ? s.name : nil)
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) { model.newChat(model: key) }
                        .mgContextMenu {
                            MGMenuItem.button("New Chat", icon: "square.and.pencil") { model.newChat(model: key) }
                            if llm.status(key)?.isActive == true {
                                MGMenuItem.button("Shut Down GPU", icon: "power") { Task { await llm.stop(key) } }
                            } else {
                                MGMenuItem.button("Start", icon: "play") { Task { await llm.warm(key) } }
                            }
                            MGMenuItem.button("GPUs and Context…", icon: "slider.horizontal.3") {
                                model.llmSettingsRequest = LLMSettingsRequest(key: key)
                            }
                            MGMenuItem.divider
                            MGMenuItem.button("Remove…", destructive: true) { confirmRemove(key, m) }
                        }
                }
            }
            if !llm.servers.isEmpty && llm.models.isEmpty && llm.listings.values.allSatisfy({ $0.installs.isEmpty }) {
                Text("Add any model by its Hugging Face link (safetensors: full, FP8, AWQ or GPTQ). The server downloads it and picks the cheapest GPUs it fits on.")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .fixedSize(horizontal: false, vertical: true).padding(.horizontal, 8)
            }
        }
        .task { await llm.refreshAll(); llm.start() }
    }

    func confirmRemove(_ key: LLMKey, _ m: LLMModel) {
        model.presenter.alert("Remove \(m.name) from the server?",
                              message: "Its weights are deleted from the GPU server's volume; adding it again downloads them again.",
                              actions: [.destructive("Remove") { Task { await model.llm.remove(key) } }, .cancel()])
    }
}

struct LLMModelRow: View {
    let model: LLMModel
    let status: LLMStatus?
    var server: String?

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "text.bubble").foregroundStyle(MG.secondaryLabel)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.name).font(.mgCallout.weight(.medium)).lineLimit(1)
                Text([server, model.gpuLabel, "\(model.context / 1024)k context", model.engine == "llamacpp" ? "GGUF" : nil,
                      model.source == "civitai" ? "Civitai" : nil, model.vision == true ? "vision" : nil]
                        .compactMap { $0 }.joined(separator: " · "))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                EngineChip(status: status, model: model)
            }
            Spacer(minLength: 0)
        }
        .padding(8)
    }
}

struct InstallRow: View {
    let id: String
    let state: LLMInstallState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(state.name ?? id).font(.mgCallout.weight(.medium)).lineLimit(1)
                Spacer()
                Text(label).font(.mgFootnote).foregroundStyle(state.state == "failed" ? MG.destructive : MG.secondaryLabel)
            }
            if ["downloading", "starting", "unpacking"].contains(state.state) {
                MGProgress(value: state.total.map { t in Double(state.bytes ?? 0) / Double(max(t, 1)) }).frame(height: 4)
            }
            if let e = state.error {
                Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(3)
            }
        }
        .padding(8)
    }

    var label: String {
        switch state.state {
        case "downloading":
            guard let t = state.total, t > 0 else { return "Downloading" }
            return String(format: "%.0f of %.0f GB", Double(state.bytes ?? 0) / 1e9, Double(t) / 1e9)
        case "starting": return "Starting"
        case "unpacking": return "Unpacking"
        case "failed": return "Failed"
        case "cancelled": return "Cancelled"
        default: return state.state.capitalized
        }
    }
}

/// Add a model from a link: look it up on the server (size, context, which GPUs it fits), pick the GPUs and context,
/// install.
struct LLMAddSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let request: LLMAddRequest
    @State private var serverID = ""
    @State private var link = ""
    @State private var busy = false
    @State private var error: String?
    @State private var found: LLMInspection?
    @State private var option: GPUOption?
    @State private var context = 0
    @State private var name = ""
    @State private var modelID = ""
    @State private var quant: String?

    var body: some View {
        let servers = model.llm.servers
        VStack(alignment: .leading, spacing: 14) {
            Text("Add a Model from a Link").font(.mgHeadline)
            Text("A Hugging Face model (owner/name or its URL) in safetensors (full precision, FP8, AWQ, GPTQ) or GGUF, or a Civitai model with a GGUF file or a zipped model folder. It downloads to the GPU server; nothing comes to this Mac.")
                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            MGSection {
                if servers.count > 1 {
                    MGFormRow("Server") {
                        MGPicker(selection: $serverID, options: servers.map { ($0.id, $0.name) })
                    }
                }
                MGFormRow("Link") {
                    HStack {
                        TextField("Link", text: $link, prompt: Text("Qwen/Qwen3.8-Flash-Next-FP8 or a Civitai link"))
                            .frame(width: 300)
                            .onSubmit { Task { await lookUp() } }
                            .onChange(of: link) { _, _ in quant = nil }
                        Button("Look Up") { Task { await lookUp() } }
                            .disabled(link.trimmingCharacters(in: .whitespaces).isEmpty || busy)
                    }
                }
            }
            if let f = found {
                details(f)
                MGSection {
                    if let opts = f.facts.ggufOptions, opts.count > 1 {
                        MGFormRow("Quantization") {
                            MGPicker(selection: Binding(get: { quant ?? f.facts.quant?["type"]?.string ?? opts[0].quant },
                                                        set: { q in quant = q; Task { await lookUp() } }),
                                     options: opts.map { ($0.quant, String(format: "%@ · %.0f GB", $0.quant, Double($0.bytes) / 1e9)) })
                        }
                    }
                    MGFormRow("GPUs") {
                        MGPicker(selection: Binding(get: { option ?? f.plan.recommended ?? f.plan.options[0] },
                                                    set: { option = $0; clampContext() }),
                                 options: f.plan.options.filter(\.fits).map { ($0, optionTitle($0, recommended: $0 == f.plan.recommended)) })
                    }
                    MGFormRow("Context") {
                        MGPicker(selection: $context, options: contextChoices(f).map { ($0, "\($0 / 1024)k tokens") })
                    }
                    MGFormRow("Name") { TextField("Name", text: $name).frame(width: 300) }
                    MGFormRow("ID") { TextField("ID", text: $modelID).frame(width: 300) }
                }
            }
            if let error {
                Label(error, systemImage: "xmark.octagon").font(.mgSubheadline).foregroundStyle(MG.destructive)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                if busy { MGSpinner().controlSize(.small) }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Install") { Task { await install() } }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                    .disabled(found == nil || busy || found?.plan.recommended == nil)
            }
        }
        .padding(20)
        .frame(width: 560)
        .onAppear { serverID = request.serverID ?? servers.first?.id ?? "" }
    }

    @ViewBuilder
    func details(_ f: LLMInspection) -> some View {
        let gb = Double(f.facts.weightBytes) / 1e9
        let quant = f.facts.engine == "llamacpp" ? "GGUF \(f.facts.quant?["type"]?.string ?? "")"
            : f.facts.quant?["method"]?.string.map { $0.uppercased() } ?? "full precision"
        VStack(alignment: .leading, spacing: 4) {
            Text(f.facts.repo).font(.mgCallout.weight(.semibold))
            Text("\(f.facts.modelType) · \(String(format: "%.0f GB", gb)) · \(quant) · \(f.facts.nativeContext / 1024)k context"
                 + (f.facts.vision ? " · vision" : "") + ((f.facts.experts ?? 0) > 0 ? " · MoE" : ""))
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            Text("Tool calls: \(f.profile.toolParser ?? "none") · thinking: \(f.profile.reasoningParser ?? "none")")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            ForEach(f.facts.notes ?? [], id: \.self) { n in
                Text(n).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            if f.plan.recommended == nil {
                Text("It doesn't fit on 8 GPUs of any kind. Try a quantized version (FP8, AWQ).")
                    .font(.mgFootnote).foregroundStyle(MG.destructive)
            }
        }
    }

    func optionTitle(_ o: GPUOption, recommended: Bool) -> String {
        String(format: "%@ · $%.2f/h · up to %dk", o.label, o.usdPerHour, o.maxContext / 1024)
            + (recommended ? " (cheapest)" : "") + ((o.note ?? "").isEmpty ? "" : " · \(o.note!)")
    }

    func contextChoices(_ f: LLMInspection) -> [Int] {
        let top = min((option ?? f.plan.recommended)?.maxContext ?? f.plan.context, f.facts.nativeContext)
        var out = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1_048_576].filter { $0 < top }
        out.append(top)
        return out
    }

    func clampContext() {
        guard let f = found else { return }
        let choices = contextChoices(f)
        if !choices.contains(context) { context = choices.last { $0 <= max(context, choices[0]) } ?? choices[0] }
    }

    func lookUp() async {
        guard let client = model.llm.client(serverID) else { error = "Choose a GPU server"; return }
        busy = true
        error = nil
        defer { busy = false }
        do {
            let f = try await client.inspect(url: link.trimmingCharacters(in: .whitespaces), hfToken: LinkLogins.hf,
                                             civitaiToken: LinkLogins.civitai, quant: quant)
            found = f
            option = f.plan.recommended
            context = min(f.plan.context, f.plan.recommended?.maxContext ?? f.plan.context)
            name = f.name
            modelID = f.suggestedId
            clampContext()
        } catch {
            found = nil
            self.error = error.localizedDescription
        }
    }

    func install() async {
        guard let client = model.llm.client(serverID), let o = option ?? found?.plan.recommended else { return }
        busy = true
        error = nil
        defer { busy = false }
        do {
            _ = try await client.install(LLMInstallRequest(
                url: link.trimmingCharacters(in: .whitespaces), id: modelID.isEmpty ? nil : modelID,
                name: name.isEmpty ? nil : name, gpu: o.gpu, count: o.count, context: context,
                quant: found?.facts.engine == "llamacpp" ? found?.facts.quant?["type"]?.string : nil,
                hfToken: LinkLogins.hf, civitaiToken: LinkLogins.civitai))
            await model.llm.refresh(serverID)
            dismiss()
        } catch {
            self.error = error.localizedDescription
        }
    }
}

/// An installed model's GPUs and context: the options it fits, from the facts the server kept. Takes effect the next
/// time it starts (a running one can restart now).
struct LLMSettingsSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let request: LLMSettingsRequest
    var preview: LLMPlan? { request.preview }
    @State private var plan: LLMPlan?
    @State private var option: GPUOption?
    @State private var context = 0
    /// How the engine reads the weights: "auto", "runai" or "processed" (fast start).
    @State private var loadFormat = "auto"
    @State private var snapshot = false
    /// Faster replies with the model's MTP layers (shown only for models that have them).
    @State private var mtp = true
    @State private var error: String?
    @State private var busy = false

    /// Fast start needs one GPU: on several it shows (and saves) Standard.
    var fastStartAllowed: Bool { (option?.count ?? model.llm.model(request.key)?.count ?? 1) == 1 }
    var loadFormatShown: String { loadFormat == "processed" && !fastStartAllowed ? "auto" : loadFormat }

    var loadFormatHelp: String {
        switch loadFormatShown {
        case "processed":
            return "Fast start: the first start loads the model as usual, then saves a copy of its weights in the GPU's own layout "
                + "(about as big as the model, on the server's volume) and takes about 2 minutes longer. Later starts load that "
                + "copy: Qwen3.8-Flash-Next went from 111 s to 60 s of loading. A new copy is made by itself after a change of GPU "
                + "or engine; switching back to Standard deletes it. One GPU only."
        case "runai":
            return "Reads the weights with Run:ai's streamer (many reads at once). Made for object storage: on Modal's volume it "
                + "was slower than the default, so try it only on other servers."
        default:
            return fastStartAllowed ? "vLLM's own loader. Fast start loads big mixture-of-experts models about twice as fast."
                : "vLLM's own loader. (Fast start works on one GPU.)"
        }
    }

    /// Snapshots need vLLM on one GPU (Modal's GPU snapshots don't handle several).
    var snapshotAllowed: Bool {
        model.llm.model(request.key)?.engine != "llamacpp" && (option?.count ?? 1) == 1
    }

    var body: some View {
        let m = model.llm.model(request.key)
        let st = model.llm.status(request.key)
        let running = st?.isActive == true
        let gguf = m?.engine == "llamacpp"
        VStack(alignment: .leading, spacing: 14) {
            Text("\(m?.name ?? request.key.modelID): GPUs, Context and Start-up").font(.mgHeadline)
            if let plan {
                MGSection {
                    MGFormRow("GPUs") {
                        MGPicker(selection: Binding(get: { option ?? plan.options[0] }, set: { option = $0; clamp() }),
                                 options: plan.options.filter(\.fits).map {
                                     ($0, String(format: "%@ · $%.2f/h · up to %dk", $0.label, $0.usdPerHour, $0.maxContext / 1024))
                                 })
                    }
                    MGFormRow("Context") {
                        MGPicker(selection: $context, options: choices.map { ($0, "\($0 / 1024)k tokens") })
                    }
                }
                MGSection("Start-up") {
                    MGFormRow("Weight loading") {
                        MGPicker(selection: Binding(get: { gguf ? "auto" : loadFormatShown }, set: { loadFormat = $0 }),
                                 options: [("auto", "Standard"), ("processed", "Fast start"), ("runai", "Run:ai streamer")])
                            .disabled(gguf)
                    }
                    .mgHelp(gguf ? "GGUF files are read by llama.cpp." : loadFormatHelp)
                    MGFormRow("GPU snapshot (experimental)") {
                        Toggle("", isOn: Binding(get: { snapshot && snapshotAllowed }, set: { snapshot = $0 }))
                            .labelsHidden().disabled(!snapshotAllowed)
                    }
                    .mgHelp(snapshotAllowed
                            ? "The first start saves the warmed-up engine as a GPU memory snapshot; later starts restore it instead of loading and compiling. Experimental: turn it off if the model fails to start."
                            : "Only for vLLM models on one GPU: GPU snapshots don't work across several GPUs.")
                }
                if m?.hasMTP == true {
                    MGSection("Replies") {
                        MGFormRow("Faster replies (MTP)") {
                            Toggle("", isOn: $mtp).labelsHidden()
                        }
                        .mgHelp("The model's own multi-token-prediction layer guesses the next 2 tokens and the model checks "
                                + "them in one step: the same replies, faster. Qwen3.8-Flash-Next on a B200 went from 191 to "
                                + "298 tokens/s. Each start takes about 40 s longer, and the KV cache is about a quarter smaller.")
                    }
                }
                if let cs = st?.coldStart {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Starts take about \(ColdStartRecord.duration(cs.typicalSeconds))"
                             + (cs.count > 1 ? " (typical of the last \(min(cs.count, 5)))" : "") + ".")
                        Text("Last start: \(ColdStartRecord.duration(cs.last.seconds))\(cs.last.how == "snapshot" ? " from a snapshot" : ""). \(cs.last.breakdown)")
                            .foregroundStyle(MG.secondaryLabel)
                    }
                    .font(.mgSubheadline)
                    .fixedSize(horizontal: false, vertical: true)
                }
                if running {
                    Text("The model is running: the change applies after it restarts.")
                        .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                }
            } else if error == nil {
                MGSpinner().controlSize(.small)
            }
            if let error {
                Label(error, systemImage: "xmark.octagon").font(.mgSubheadline).foregroundStyle(MG.destructive)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                if busy { MGSpinner().controlSize(.small) }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                if running {
                    Button("Save and Restart") { Task { await save(restart: true) } }.disabled(plan == nil || busy)
                }
                Button("Save") { Task { await save(restart: false) } }
                    .buttonStyle(.mg(.prominent)).keyboardShortcut(.defaultAction).disabled(plan == nil || busy)
            }
        }
        .padding(20)
        .frame(width: 520)
        .task { await load() }
    }

    var choices: [Int] {
        guard let o = option ?? plan?.options.first else { return [] }
        var out = [8192, 16384, 32768, 65536, 131072, 262144, 524288].filter { $0 < o.maxContext }
        out.append(o.maxContext)
        return out
    }

    func clamp() {
        if !choices.contains(context) { context = choices.last { $0 <= max(context, choices.first ?? 0) } ?? choices.first ?? 0 }
    }

    func load() async {
        let client = model.llm.client(request.key.serverID)
        guard client != nil || preview != nil else { return }
        do {
            let p: LLMPlan
            if let preview { p = preview } else { p = try await client!.plan(request.key.modelID) }
            plan = p
            option = p.options.first { $0.gpu == p.current.gpu && $0.count == p.current.count && $0.fits } ?? p.recommended
            context = p.current.context
            loadFormat = model.llm.model(request.key)?.loadFormat ?? "auto"
            snapshot = model.llm.model(request.key)?.snapshot == true
            mtp = model.llm.model(request.key)?.mtp != false
            if !choices.contains(context) { context = choices.last { $0 <= context } ?? choices.first ?? context }
        } catch {
            self.error = error.localizedDescription
        }
    }

    func save(restart: Bool) async {
        guard let client = model.llm.client(request.key.serverID), let o = option else { return }
        busy = true
        defer { busy = false }
        do {
            var changes: [String: JSONValue] = ["gpu": .string(o.gpu), "count": .number(Double(o.count)),
                                                 "context": .number(Double(context)), "snapshot": .bool(snapshot && snapshotAllowed)]
            if model.llm.model(request.key)?.engine != "llamacpp" { changes["loadFormat"] = .string(loadFormatShown) }
            if model.llm.model(request.key)?.hasMTP == true { changes["mtp"] = .bool(mtp) }
            _ = try await client.update(request.key.modelID, changes)
            if restart { await model.llm.stop(request.key) }
            await model.llm.refresh(request.key.serverID)
            dismiss()
        } catch {
            self.error = error.localizedDescription
        }
    }
}
