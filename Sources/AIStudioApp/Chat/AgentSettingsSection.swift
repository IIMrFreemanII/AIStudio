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

    var body: some View {
        MGSection("Coding Agent") {
            MGFormRow("Web search") {
                HStack(spacing: 8) {
                    SecureField("Brave Search API key", text: $key,
                                prompt: Text(keySaved ? "Saved in the Keychain — type to replace" : "Brave Search API key"))
                        .frame(maxWidth: 260)
                    Button("Save") {
                        model.setSearchAPIKey(key)
                        key = ""
                        keySaved = model.searchAPIKey != nil
                    }
                    .buttonStyle(.mg(.bordered))
                    .disabled(key.isEmpty && !keySaved)
                    .mgHelp(key.isEmpty ? "Remove the saved key" : "Save the key in the Keychain")
                    if keySaved { Text("Key saved").font(.mgFootnote).foregroundStyle(MG.secondaryLabel) }
                }
            }
            MGFormRow("Worktree branch prefix") {
                TextField("aistudio/", text: $branchPrefix).frame(maxWidth: 260)
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
            reload()
        }
    }

    func reload() {
        disabled = model.disabledSkills
        skills = SkillStore.load(root: nil)
    }
}
