import AppKit
import AIStudioProtocol
import NodesRemote
import SwiftUI

/// Rail panel: the servers this app can send work to.
struct ServersPanel: View {
    @Environment(AppModel.self) private var model
    @State private var editing: ServerConnection?

    var body: some View {
        let store = model.connections
        VStack(spacing: 0) {
            PanelHeader(title: "Servers") {
                let refreshing = store.refreshingAll || !model.llm.refreshing.isEmpty
                Button {
                    Task {
                        await store.refreshAll()
                        await model.llm.refreshAll()  // and the language models (new installs, removals)
                    }
                } label: {
                    if refreshing { MGSpinner().controlSize(.small) } else { Image(systemName: "arrow.clockwise") }
                }
                    .disabled(refreshing)
                    .mgHelp(refreshing ? "Checking the servers…" : "Check every server").accessibilityLabel("Check every server")
                    .mgTestAnchor("servers-refresh")
                Button { editing = ServerConnection(name: "", baseURL: "http://") } label: { Image(systemName: "plus") }
                    .mgHelp("Add a server").accessibilityLabel("Add a server")
            }
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ServerRow(name: "This Mac", detail: "Runs here with MLX", status: nil, isTarget: model.active.backendID == BackendID.local)
                        .onTapGesture { model.active.backendID = BackendID.local }
                    ForEach(store.macServers) { c in
                        ServerRow(name: c.name, detail: c.baseURL, status: store.status[c.id] ?? .unknown,
                                  isTarget: model.active.backendID == c.id, insecure: c.isInsecure)
                            .contentShape(Rectangle())
                            .onTapGesture { model.active.backendID = c.id }
                            .mgContextMenu {
                                MGMenuItem.button("Run This Tab Here") { model.active.backendID = c.id }
                                MGMenuItem.button(checking(c.id) ? "Checking…" : "Check Connection", enabled: !checking(c.id)) {
                                    Task { await store.test(c.id) }
                                }
                                MGMenuItem.button("Open Dashboard", enabled: c.url != nil) {
                                    if let u = c.url { NSWorkspace.shared.open(u) }
                                }
                                MGMenuItem.button("Edit…") { editing = c }
                                MGMenuItem.divider
                                MGMenuItem.button("Remove", destructive: true) {
                                    for doc in model.tabs where doc.backendID == c.id { doc.backendID = BackendID.local }
                                    store.remove(c.id)
                                    model.llm.forget(server: c.id)
                                }
                            }
                    }
                    if store.macServers.isEmpty {
                        Text("Add a Mac running `aistudio-cli serve` to run a tab's graphs there.")
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.horizontal, 8)
                    }
                    MGListHeader(title: "GPU Servers").padding(.top, 6)
                    ForEach(store.gpuServers) { c in
                        ServerRow(name: c.name, detail: c.baseURL, status: store.status[c.id] ?? .unknown, isTarget: false,
                                  insecure: c.isInsecure, gpu: true)
                            .contentShape(Rectangle())
                            .onTapGesture(count: 2) { editing = c }
                            .mgContextMenu {
                                MGMenuItem.button(checking(c.id) ? "Checking…" : "Check Connection", enabled: !checking(c.id)) {
                                    Task { await store.test(c.id) }
                                }
                                MGMenuItem.button("Edit…") { editing = c }
                                MGMenuItem.divider
                                MGMenuItem.button("Remove", destructive: true) {
                                    store.remove(c.id)
                                    model.llm.forget(server: c.id)  // its models and anything they were doing
                                }
                            }
                    }
                    if !store.gpuServers.isEmpty {
                        LLMModelsSection().padding(.top, 6)
                    }
                    if store.gpuServers.isEmpty {
                        Text("Add a GPU server (Modal, or a GPU box) for the Cloud GPU nodes: Wan 2.2 video and FLUX.1 dev on NVIDIA, billed per second on Modal. See docs/remote-server.md.")
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.horizontal, 8)
                    }
                }
                .padding(12)
            }
            .mgScrollIndicators()
        }
        .mgSheet("server-editor", item: $editing) { c in ServerEditor(connection: c).environment(model) }
        .task { if store.checksOnAppear { await store.refreshAll() } }
    }
}

extension ServersPanel {
    func checking(_ id: String) -> Bool {
        if case .checking? = model.connections.status[id] { return true }
        return false
    }
}

struct ServerRow: View {
    let name: String
    let detail: String
    let status: ConnectionStore.Status?
    let isTarget: Bool
    var insecure = false
    /// A GPU server: not a run target (its nodes pick it).
    var gpu = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if gpu {
                Image(systemName: "cpu").foregroundStyle(MG.secondaryLabel)
            } else {
                Image(systemName: isTarget ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isTarget ? MG.accent : MG.secondaryLabel)
                    .mgHelp(isTarget ? "This tab runs here" : "Click to run this tab here")
            }
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(name).font(.mgCallout.weight(.medium)).lineLimit(1)
                    if let status { StatusDot(status: status) }
                }
                Text(detail).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
                switch status {
                case .online(let info)?:
                    Text("\(info.chip) · \(info.memory.total >> 30) GB · \(info.models.count) models")
                        .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                case .gpuOnline(let info)?:
                    GPUTasksSummary(info: info)
                case .offline(let message)?:
                    Text(message).font(.mgFootnote).foregroundStyle(MG.destructive).fixedSize(horizontal: false, vertical: true)
                default: EmptyView()
                }
                if insecure {
                    Label("Plain http to a public address: the token is sent unencrypted. Use Tailscale, an SSH tunnel or https.",
                          systemImage: "exclamationmark.triangle")
                        .font(.mgFootnote).foregroundStyle(MG.warning).fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
            .fill(isTarget ? MG.selectionInactive : .clear))
    }
}

struct StatusDot: View {
    let status: ConnectionStore.Status
    var body: some View {
        switch status {
        case .checking: MGSpinner().controlSize(.mini)
        case .online, .gpuOnline:
            Circle().fill(MG.success).frame(width: 7, height: 7).mgHelp("Online").accessibilityLabel("Online")
        case .offline: Circle().fill(MG.destructive).frame(width: 7, height: 7).mgHelp("Offline").accessibilityLabel("Offline")
        case .unknown: Circle().fill(MG.tertiaryLabel).frame(width: 7, height: 7).mgHelp("Not checked")
            .accessibilityLabel("Not checked")
        }
    }
}

/// A GPU server's tasks: which models are installed, on which GPU, at what price.
struct GPUTasksSummary: View {
    let info: GPUServerInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            ForEach(info.tasks.filter { $0.gpu != "CPU" }) { t in
                HStack(spacing: 4) {
                    Image(systemName: t.installed ? "checkmark" : "arrow.down.circle")
                        .foregroundStyle(t.installed ? MG.success : MG.secondaryLabel)
                        .mgHelp(t.installed ? "Installed" : "Not downloaded on the server yet")
                    Text(t.title).lineLimit(1)
                    Spacer(minLength: 4)
                    Text(t.gpu + (t.pricePerSecond.map { String(format: " · $%.2f/h", $0 * 3600) } ?? ""))
                        .foregroundStyle(MG.secondaryLabel)
                }
            }
            Text(info.runner == "modal" ? "Modal: billed per second, idle after \(info.idleSeconds ?? 120) s"
                 : "GPU box (\(info.runner))")
                .foregroundStyle(MG.tertiaryLabel)
        }
        .font(.mgFootnote)
    }
}

/// Add / edit a server: kind, name, URL and token, with a connection test.
struct ServerEditor: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    @State var connection: ServerConnection
    @State private var token = ""
    @State private var testing = false
    @State private var result: String?
    @State private var ok = false

    var isNew: Bool { model.connections.connection(connection.id) == nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(isNew ? "Add Server" : "Edit Server").font(.mgHeadline)
            if isNew {
                MGSegmentedPicker(selection: $connection.kind,
                                  options: ServerConnection.Kind.allCases.map { ($0, $0.label) })
                Text(connection.kind == .gpu
                     ? "Runs the Cloud GPU nodes (Wan 2.2, FLUX.1 dev) on NVIDIA: on Modal, billed per second, or on a GPU box."
                     : "Another Mac running `aistudio-cli serve`: a tab's whole graph runs there.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            }
            MGSection {
                MGFormRow("Name") {
                    TextField("Name", text: $connection.name,
                              prompt: Text(connection.kind == .gpu ? "Modal" : "Studio Mac")).frame(width: 260)
                }
                MGFormRow("URL") {
                    TextField("URL", text: $connection.baseURL,
                              prompt: Text(connection.kind == .gpu ? "https://you--aistudio-gpu-api.modal.run"
                                           : "https://studio.tail1234.ts.net")).frame(width: 260)
                }
                MGFormRow("Token") {
                    SecureField("Token", text: $token,
                                prompt: Text(!isNew ? "unchanged" : connection.kind == .gpu ? "wk-….ws-… (Modal proxy token)"
                                             : "from aistudio-cli serve"))
                        .frame(width: 260)
                }
            }
            if connection.isInsecure {
                Label("This is plain http to a public address: anyone on the path can read the token. Prefer Tailscale, an SSH tunnel (http://127.0.0.1:…), or https.",
                      systemImage: "exclamationmark.triangle")
                    .font(.mgSubheadline).foregroundStyle(MG.warning).fixedSize(horizontal: false, vertical: true)
            }
            if let result {
                Label(result, systemImage: ok ? "checkmark.circle" : "xmark.octagon")
                    .font(.mgSubheadline).foregroundStyle(ok ? MG.success : MG.destructive)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Test") { Task { await test() } }.disabled(connection.url == nil || testing)
                if testing { MGSpinner().controlSize(.small) }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save") {
                    if connection.name.trimmingCharacters(in: .whitespaces).isEmpty {
                        connection.name = connection.url?.host ?? "Server"
                    }
                    model.connections.save(connection, token: token.isEmpty ? nil : token)
                    Task { await model.connections.test(connection.id) }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.mg(.prominent))
                .disabled(connection.url == nil || (isNew && token.isEmpty))
            }
        }
        .padding(20)
        .frame(width: 460)
    }

    func test() async {
        guard let url = connection.url else { return }
        let t = token.isEmpty ? (ConnectionToken.read(connection.id) ?? "") : token
        testing = true
        defer { testing = false }
        do {
            switch try await ConnectionStore.check(kind: connection.kind, url: url, token: t) {
            case .online(let info):
                result = "Connected: \(info.hostName), \(info.chip), \(info.memory.total >> 30) GB, \(info.models.count) models, server \(info.serverVersion)"
            case .gpuOnline(let info):
                let tasks = info.tasks.filter { $0.gpu != "CPU" }
                let ready = tasks.filter(\.installed).map(\.title)
                result = "Connected (\(info.runner == "modal" ? "Modal" : info.runner), server \(info.version)): "
                    + (ready.isEmpty ? "no models downloaded yet — run the download step"
                       : "\(ready.joined(separator: ", ")) ready") + ". No GPU was started."
            default: result = nil
            }
            ok = true
        } catch {
            ok = false
            result = ConnectionStore.describe(error)
        }
    }
}

/// Toolbar menu: where the active tab's runs go.
struct RunTargetPicker: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let current = model.active.backendID
        MGMenuButton(kind: .bordered) {
            MGMenuItem.header("Run On")
            MGMenuItem.button("This Mac", icon: "laptopcomputer", checked: current == BackendID.local) {
                model.active.backendID = BackendID.local
            }
            for c in model.connections.macServers {
                MGMenuItem.button(c.name, icon: "server.rack", checked: current == c.id) { model.active.backendID = c.id }
            }
            MGMenuItem.divider
            MGMenuItem.button("Manage Servers…") { model.showServersPanel = true }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: current == BackendID.local ? "laptopcomputer" : "server.rack")
                Text(model.backendName(current)).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
            }
        }
        .mgHelp("Where this tab's graphs run")
        .accessibilityLabel("Run on \(model.backendName(current))")
    }
}
