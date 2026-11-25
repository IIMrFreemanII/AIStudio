import AVFoundation
import Speech
import SwiftUI

/// The composer's microphone: what you say is typed into the draft as you speak (Apple's speech recognition, on this
/// Mac when the language supports it, so nothing goes to a server). Click again to stop; the words stay to edit or send.
@MainActor @Observable
final class ChatDictation {
    static let shared = ChatDictation()

    /// The chat being dictated into (one at a time).
    private(set) var chatID: UUID?
    /// Why dictation couldn't start, and in which chat (shown under its composer until the next try).
    private(set) var problem: String?
    private var problemChatID: UUID?
    private var engine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    /// The draft before dictation started: the words are added after it.
    private var before = ""

    func isOn(_ doc: ChatDocument) -> Bool { chatID == doc.id }
    func problem(in doc: ChatDocument) -> String? { problemChatID == doc.id ? problem : nil }

    func toggle(_ doc: ChatDocument) {
        if isOn(doc) { stop() } else { Task { await start(doc) } }
    }

    func start(_ doc: ChatDocument) async {
        stop()
        problem = nil
        problemChatID = doc.id
        guard await Self.allowed() else {
            problem = "Allow Speech Recognition and the Microphone for AIStudio in System Settings › Privacy & Security."
            return
        }
        guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else {
            problem = "Speech recognition isn't available for your language right now."
            return
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition { request.requiresOnDeviceRecognition = true }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            problem = "No microphone found."
            return
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in request.append(buffer) }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            problem = "Couldn't start the microphone: \(error.localizedDescription)"
            return
        }
        before = doc.draft
        chatID = doc.id
        self.engine = engine
        self.request = request
        task = recognizer.recognitionTask(with: request) { [weak self, weak doc] result, error in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal == true
            Task { @MainActor in
                guard let self, let doc, self.chatID == doc.id else { return }
                if let text {
                    let lead = self.before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : self.before + " "
                    doc.draft = lead + text
                }
                if error != nil || final { self.stop() }
            }
        }
    }

    func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        request?.endAudio()
        task?.finish()
        engine = nil
        request = nil
        task = nil
        chatID = nil
    }

    /// Asks for speech recognition and the microphone the first time (macOS shows its own prompts).
    private static func allowed() async -> Bool {
        let speech: Bool = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        guard speech else { return false }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }
}

/// The composer's microphone button.
struct DictationButton: View {
    @Bindable var doc: ChatDocument
    private var dictation: ChatDictation { .shared }

    var body: some View {
        let on = dictation.isOn(doc)
        Button { dictation.toggle(doc) } label: {
            Image(systemName: on ? "mic.fill" : "mic").foregroundStyle(on ? MG.destructive : MG.label)
        }
        .accessibilityLabel(on ? "Stop Dictation" : "Dictate")
        .buttonStyle(.mg(.icon))
        .controlSize(.large)
        .mgHelp(on ? "Stop listening (the words stay in the message)" : "Say your message: it's typed in as you speak")
        .onDisappear { if dictation.isOn(doc) { dictation.stop() } }
    }
}
