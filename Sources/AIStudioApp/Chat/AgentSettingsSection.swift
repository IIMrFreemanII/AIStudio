import Agent
import SwiftUI

/// Settings for the coding agent: the web search key and the skills.
struct AgentSettingsSection: View {
    @Environment(AppModel.self) private var model
    @State private var key = ""
    @State private var keySaved = false
    @State private var skills: [Skill] = []
    @State private var disabled: Set<String> = []
    @AppStorage("notifySessions") private var notifySessions = true
    @AppStorage("worktreeBranchPrefix") private var branchPrefix = "aistudio/"
    @AppStorage("gitCoAuthor") private var gitCoAuthor = true
    @AppStorage("codeGraph") private var codeGraph = true
    @AppStorage("flowConcurrency") private var flowConcurrency = 0
    @AppStorage("flowMaxAgents") private var flowMaxAgents = 30
    @State private var graphifyPath: String?
    @State private var installing = false
    @State private var installError: String?

    var body: some View {
        MGSection("Coding Agent") {
            MGFormRow("Web search") {
                HStack(spacing: 8) {
                    SecureField("Brave Search API key", text: $key,
                                prompt: Text(keySaved ? "Saved — type to replace" : "Brave Search API key"))
                        .frame(maxWidth: 260)
                    Button("Save") {
                        model.setSearchAPIKey(key)
                        key = ""
                        keySaved = model.searchAPIKey != nil
                    }
                    .buttonStyle(.mg(.bordered))
                    .disabled(key.isEmpty && !keySaved)
                    .mgHelp(key.isEmpty ? "Remove the saved key" : "Save the key")
                    if keySaved { Text("Key saved").font(.mgFootnote).foregroundStyle(MG.secondaryLabel) }
                }
            }
            MGFormRow("Worktree branch prefix") {
                TextField("aistudio/", text: $branchPrefix).frame(maxWidth: 260)
            }
            VStack(alignment: .leading, spacing: 3) {
                Toggle("Add a co-author line to commits", isOn: $gitCoAuthor)
                Text("Co-Authored-By: AIStudio (<model>) <noreply@aistudio.local> — used by the Commit button and asked of the agent.")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            VStack(alignment: .leading, spacing: 3) {
                Toggle("Project map for the agent (graphify)", isOn: $codeGraph)
                Text("The agent asks a graph of the project where code lives and how it connects, then reads only those lines — fewer tokens than searching and reading whole files. Built locally, kept in the app's cache.")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                HStack(spacing: 8) {
                    if let graphifyPath {
                        Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(MG.success)
                            .mgHelp(graphifyPath)
                    } else {
                        Label("graphify isn't installed", systemImage: "exclamationmark.circle").foregroundStyle(MG.warning)
                        if CodeGraphInstall.uvPath != nil {
                            Button(installing ? "Installing…" : "Install") { confirmInstall() }
                                .buttonStyle(.mg(.bordered)).disabled(installing)
                        } else {
                            Text("Install with: uv tool install graphifyy").textSelection(.enabled)
                        }
                    }
                }
                .font(.mgFootnote)
                if let installError {
                    Text(installError).font(.mgFootnote).foregroundStyle(MG.destructive).textSelection(.enabled).lineLimit(4)
                }
            }
            VStack(alignment: .leading, spacing: 3) {
                MGFormRow("Flows: agents at a time") {
                    MGPicker(selection: $flowConcurrency,
                             options: [(0, "Auto")] + [2, 3, 4, 6, 8, 12, 16].map { ($0, "\($0)") })
                }
                Text("Auto fits the GPU's memory (KV cache).").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            MGFormRow("Max agents per flow") {
                HStack(spacing: 8) {
                    Text("\(flowMaxAgents)").foregroundStyle(MG.secondaryLabel).monospacedDigit()
                    MGStepper(value: $flowMaxAgents, in: 5...100, step: 5)
                }
            }
            Toggle("Notify when a background session finishes or needs you", isOn: $notifySessions)
            VStack(alignment: .leading, spacing: 6) {
                Text("Skills").font(.mgCallout)
                if skills.isEmpty {
                    Text("No personal skills yet.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
                ForEach(skills, id: \.name) { s in
                    Toggle(isOn: Binding(get: { !disabled.contains(s.name) }, set: { on in
                        if on { disabled.remove(s.name) } else { disabled.insert(s.name) }
                        model.disabledSkills = disabled
                    })) {
                        Text("\(s.name) — \(s.description)").font(.mgFootnote).lineLimit(2)
                    }
                }
                HStack(spacing: 8) {
                    Button("Show Folder") { model.showPersonalSkillsFolder() }.buttonStyle(.mg(.bordered))
                    Button("New Skill…") {
                        model.newPersonalSkill()
                        reload()
                    }
                    .buttonStyle(.mg(.bordered))
                }
            }
        } footer: {
            Text("Skills are folders with a SKILL.md. Personal skills live here; a project's .claude/skills and .aistudio/skills are loaded in its coding chats. web_search is offered to the agent only when a key is saved.")
        }
        .onAppear {
            keySaved = model.searchAPIKey != nil
            graphifyPath = ProjectGraphs.findBinary()
            reload()
        }
    }

    func confirmInstall() {
        model.presenter.alert("Install graphify?", message: "Runs “uv tool install graphifyy”, which downloads graphify from PyPI into ~/.local.",
                              actions: [
            MGAlertAction(label: "Install") {
                installing = true
                installError = nil
                Task {
                    installError = await CodeGraphInstall.install()
                    graphifyPath = ProjectGraphs.findBinary()
                    installing = false
                }
            },
            .cancel(),
        ])
    }

    func reload() {
        disabled = model.disabledSkills
        skills = SkillStore.load(root: nil)
    }
}
