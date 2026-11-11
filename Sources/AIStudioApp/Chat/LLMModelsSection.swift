import LLMClient
import SwiftUI

/// Asks for the Add a Model sheet (optionally on a given server).
struct LLMAddRequest: Identifiable {
    let id = UUID()
    var serverID: String?
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
                Text([server, model.gpuLabel, "\(model.context / 1024)k context", model.vision == true ? "vision" : nil]
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
            if state.state == "downloading" || state.state == "starting" {
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

    var body: some View {
        let servers = model.llm.servers
        VStack(alignment: .leading, spacing: 14) {
            Text("Add a Model from a Link").font(.mgHeadline)
            Text("A Hugging Face model (owner/name or its URL), in safetensors: full precision, FP8, AWQ or GPTQ. It downloads to the GPU server; nothing comes to this Mac.")
                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            MGSection {
                if servers.count > 1 {
                    MGFormRow("Server") {
                        MGPicker(selection: $serverID, options: servers.map { ($0.id, $0.name) })
                    }
                }
                MGFormRow("Link") {
                    HStack {
                        TextField("Link", text: $link, prompt: Text("Qwen/Qwen3.8-Flash-Next-FP8"))
                            .frame(width: 300)
                            .onSubmit { Task { await lookUp() } }
                        Button("Look Up") { Task { await lookUp() } }
                            .disabled(link.trimmingCharacters(in: .whitespaces).isEmpty || busy)
                    }
                }
            }
            if let f = found {
                details(f)
                MGSection {
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
        let quant = f.facts.quant?["method"]?.string.map { $0.uppercased() } ?? "full precision"
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
            let f = try await client.inspect(url: link.trimmingCharacters(in: .whitespaces),
                                             hfToken: HFToken.isSaved ? HFToken.read() : nil)
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
                hfToken: HFToken.isSaved ? HFToken.read() : nil))
            await model.llm.refresh(serverID)
            dismiss()
        } catch {
            self.error = error.localizedDescription
        }
    }
}
