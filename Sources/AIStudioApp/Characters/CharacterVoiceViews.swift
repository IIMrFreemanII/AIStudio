import CharacterCore
import MovieCore
import SwiftUI

private func describe(_ error: Error) -> String {
    (error as? CustomStringConvertible)?.description ?? error.localizedDescription
}

/// The character's voice: its description, the speech model, and the voice samples designed from it (the chosen one
/// is cloned for every voice message and video message). Like the Look section, the sample buttons write the stored
/// character, so only the sample fields of the edited copy are refreshed here.
struct CharacterVoiceSection: View {
    @Environment(AppModel.self) private var model
    @Binding var c: AppModel.Companion
    let isNew: Bool
    @State private var running = false
    @State private var problem: String?

    private var server: String? { model.connections.gpuServers.first?.id }
    private static let typicalLine = "Hey you, I was just thinking about you. Call me when you're free tonight?"

    var body: some View {
        MGSection("Voice") {
            MGFormRow("Voice description") { TextField("soft, warm and bright", text: $c.voice).frame(maxWidth: 300) }
            MGFormRow("Speech model") {
                MGSegmentedPicker(selection: $c.voiceEngine,
                                  options: VoiceEngine.allCases.map { ($0, $0 == .breeze ? "Breeze 2" : "Qwen3-TTS") })
                    .frame(width: 220)
                    .mgHelp("Qwen3-TTS speaks 10 languages; Breeze TTS 2 speaks English and Chinese more expressively (non-commercial use only)")
            }
            if !isNew {
                samples
                HStack(spacing: 8) {
                    Button(c.voiceSamples.isEmpty ? "Make a Voice Sample" : "Make Another") { make() }
                        .buttonStyle(.mg(.bordered)).disabled(server == nil || running)
                        .mgHelp(server == nil ? "Add a GPU server in Servers first"
                                : "Design a voice from the description, saying a sample sentence")
                    if running {
                        MGSpinner().controlSize(.small)
                        Text("Designing the voice…").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    }
                    Spacer(minLength: 0)
                }
            }
        } footer: {
            VStack(alignment: .leading, spacing: 2) {
                Text(footer)
                if let problem { Text(problem).foregroundStyle(MG.destructive) }
            }
        }
    }

    private var footer: String {
        if isNew { return "Save the character first, then design their voice." }
        var s = "Voice messages and video messages clone the chosen sample. With none, the first voice message makes one."
        if let server, let each = model.mediaEstimate(.voice, character: c, text: Self.typicalLine, server: server) {
            s += " A voice message costs about \(CostEstimate.format(each))\(c.voiceSamples.isEmpty ? ", the first with its sample" : "")."
        }
        if c.voiceEngine == .breeze { s += " Breeze TTS 2 is for research and non-commercial use only." }
        return s
    }

    @ViewBuilder private var samples: some View {
        ForEach(c.voiceSamples, id: \.self) { name in
            HStack(spacing: 8) {
                ChatAudioPlayer(url: model.characterStore.voiceFolder(c.id).appendingPathComponent(name))
                if c.selectedVoiceSample == name {
                    Text("Chosen").font(.mgFootnote).foregroundStyle(MG.accentForeground)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(MG.accent))
                } else {
                    Button("Use") { model.selectVoiceSample(name, for: c.id); refresh() }
                        .buttonStyle(.mg(.bordered)).controlSize(.small)
                        .mgHelp("Clone this voice for their messages")
                }
                Spacer(minLength: 0)
                Button { confirmDelete(name) } label: { Image(systemName: "trash") }
                    .accessibilityLabel("Delete Voice Sample")
                    .buttonStyle(.mg(.icon)).controlSize(.small)
                    .mgHelp("Delete this voice sample")
            }
        }
    }

    private func refresh() {
        guard let fresh = model.characterStore.load(c.id) else { return }
        if c.voiceSamples != fresh.voiceSamples { c.voiceSamples = fresh.voiceSamples }
        if c.selectedVoiceSample != fresh.selectedVoiceSample { c.selectedVoiceSample = fresh.selectedVoiceSample }
    }

    private func make() {
        guard !running else { return }
        running = true
        problem = nil
        // The sample is designed from the description and model as edited (saved first, keeping the stored
        // portraits and samples, which the edited copy may not have yet).
        if var stored = model.characterStore.load(c.id) {
            stored.voice = c.voice
            stored.voiceEngine = c.voiceEngine
            model.saveCharacter(stored)
        }
        let id = c.id
        Task { @MainActor in
            do { try await model.makeVoiceSample(for: id) } catch { problem = describe(error) }
            refresh()
            running = false
        }
    }

    private func confirmDelete(_ name: String) {
        let id = c.id
        model.presenter.alert("Delete this voice sample?", message: "The recording moves to the Trash.", actions: [
            .destructive("Delete") {
                model.deleteVoiceSample(name, for: id)
                refresh()
            },
            .cancel(),
        ])
    }
}

/// Which kinds of media the character may send.
struct CharacterMediaKindsRow: View {
    @Binding var c: AppModel.Companion

    var body: some View {
        MGFormRow("Sends") {
            HStack(spacing: 12) {
                ForEach(CharacterCore.MediaKind.allCases, id: \.self) { k in
                    Toggle(CharacterDisplay.label(k).name + "s", isOn: Binding(
                        get: { c.media.contains(k) },
                        set: { on in
                            if on { c.media = CharacterCore.MediaKind.allCases.filter { c.media.contains($0) || $0 == k } }
                            else { c.media.removeAll { $0 == k } }
                        }))
                        .toggleStyle(MGCheckboxStyle())
                }
            }
            .mgHelp("What they can send in chats: videos and video messages always wait for your OK")
        }
    }
}
