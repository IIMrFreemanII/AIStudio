import CharacterCore
import MovieCore
import SwiftUI

private func describe(_ error: Error) -> String {
    (error as? CustomStringConvertible)?.description ?? error.localizedDescription
}

/// The character's voice: its options and description, the speech model, and the voice samples designed from them
/// (the chosen one is cloned for every voice message and video message). Like the Look section it works before the
/// character is saved: samples go to its folder and into the edited copy (`c`), and into the stored one if any.
struct CharacterVoiceSection: View {
    @Environment(AppModel.self) private var model
    @Binding var c: AppModel.Companion
    /// The voice option chips above the description (the editor; the creator shows them itself).
    var chips = true
    @State private var running = false
    @State private var problem: String?

    private var server: String? { model.connections.gpuServers.first?.id }
    private static let typicalLine = "Hey you, I was just thinking about you. Call me when you're free tonight?"

    var body: some View {
        MGSection("Voice") {
            if chips { TraitGroupView(group: .voice, c: $c) }
            MGFormRow(chips ? "More about the voice" : "Voice description") {
                TextField("soft, with a smile in it", text: $c.voice).frame(maxWidth: 300)
            }
            MGFormRow("Speech model") {
                MGSegmentedPicker(selection: $c.voiceEngine,
                                  options: VoiceEngine.allCases.map { ($0, $0 == .breeze ? "Breeze 2" : "Qwen3-TTS") })
                    .frame(width: 220)
                    .mgHelp("Qwen3-TTS speaks 10 languages; Breeze TTS 2 speaks English and Chinese more expressively (non-commercial use only)")
            }
            samples
            HStack(spacing: 8) {
                Button(c.voiceSamples.isEmpty ? "Make a Voice Sample" : "Make Another") { make() }
                    .buttonStyle(.mg(c.voiceSamples.isEmpty ? .prominent : .bordered)).disabled(server == nil || running)
                    .mgHelp(server == nil ? "Add a GPU server in Servers first"
                            : "Design a voice from the options and description, saying a sample sentence")
                if running {
                    MGSpinner().controlSize(.small)
                    Text("Designing the voice…").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
                Spacer(minLength: 0)
            }
        } footer: {
            VStack(alignment: .leading, spacing: 2) {
                Text(footer)
                if let problem { Text(problem).foregroundStyle(MG.destructive) }
            }
        }
    }

    private var footer: String {
        if server == nil { return "Add a GPU server in Servers to hear their voice." }
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
                    Button("Use") {
                        c.selectedVoiceSample = name
                        model.selectVoiceSample(name, for: c.id)
                    }
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

    private func make() {
        guard !running else { return }
        running = true
        problem = nil
        let snapshot = c
        Task { @MainActor in
            do {
                let made = try await model.makeVoiceSample(for: snapshot)
                if !c.voiceSamples.contains(made.name) { c.voiceSamples.append(made.name) }
                if c.selectedVoiceSample == nil { c.selectedVoiceSample = made.name }
            } catch { problem = describe(error) }
            running = false
        }
    }

    private func confirmDelete(_ name: String) {
        let id = c.id
        model.presenter.alert("Delete this voice sample?", message: "The recording moves to the Trash.", actions: [
            .destructive("Delete") {
                model.deleteVoiceSample(name, for: id)
                c.voiceSamples.removeAll { $0 == name }
                if c.selectedVoiceSample == name { c.selectedVoiceSample = c.voiceSamples.last }
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
                ForEach(CharacterCore.MediaKind.toggled, id: \.self) { k in
                    Toggle(CharacterDisplay.label(k).name + "s", isOn: Binding(
                        get: { c.media.contains(k) },
                        set: { on in
                            if on { c.media = CharacterCore.MediaKind.toggled.filter { c.media.contains($0) || $0 == k } }
                            else { c.media.removeAll { $0 == k } }
                        }))
                        .toggleStyle(MGCheckboxStyle())
                }
            }
            .mgHelp("What they can send in chats: videos and video messages always wait for your OK")
        }
    }
}
