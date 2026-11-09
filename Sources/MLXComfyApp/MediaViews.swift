import AVFoundation
import AVKit
import MLXDiffusionCore
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

/// What the Simple panel makes.
enum SimpleMode: String, CaseIterable, Identifiable {
    case image = "Image"
    case video = "Video"
    case music = "Music"
    case speech = "Speech"
    case sound = "Sound"
    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .music: return "music.note"
        case .speech: return "waveform"
        case .sound: return "speaker.wave.2"
        }
    }

    var catalogKind: CatalogEntry.Kind {
        switch self {
        case .image: return .image
        case .video: return .video
        case .music: return .music
        case .speech: return .speech
        case .sound: return .sound
        }
    }
}

/// Plays a video or audio result in the canvas.
struct MediaCanvas: View {
    let url: URL
    let kind: MediaKind

    var body: some View {
        switch kind {
        case .video: VideoResultView(url: url)
        case .audio: AudioResultView(url: url)
        case .image: EmptyView()
        }
    }
}

struct VideoResultView: View {
    let url: URL
    @State private var player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .shadow(color: .black.opacity(0.25), radius: 12, y: 4)
            .padding(24)
            .task(id: url) {
                let p = AVPlayer(url: url)
                p.actionAtItemEnd = .none
                player = p
            }
            .onDisappear { player?.pause() }
    }
}

/// Waveform with a playhead; click to play or pause, click elsewhere on the waveform to seek.
struct AudioResultView: View {
    let url: URL
    @State private var peaks: [Float] = []
    @State private var player: AVAudioPlayer?
    @State private var position: Double = 0
    @State private var playing = false
    @State private var duration: Double = 0
    private let timer = Timer.publish(every: 1.0 / 30, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            GeometryReader { geo in
                Waveform(peaks: peaks, progress: duration > 0 ? position / duration : 0)
                    .contentShape(Rectangle())
                    .onTapGesture { loc in seek(to: loc.x / geo.size.width) }
            }
            .frame(height: 160)
            .padding(.horizontal, 32)
            HStack(spacing: 16) {
                Button { toggle() } label: {
                    Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill").font(.system(size: 44))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.space, modifiers: [])
                Text("\(format(position)) / \(format(duration))").font(.title3.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(url.lastPathComponent).font(.caption).foregroundStyle(.tertiary)
            Spacer()
        }
        .task(id: url) {
            player?.stop()
            playing = false
            position = 0
            player = try? AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            duration = player?.duration ?? 0
            // Previews (voice samples, PreviewAudio nodes) play right away.
            if url.standardizedFileURL.path.hasPrefix(previewDirectory.standardizedFileURL.path) {
                player?.play()
                playing = player != nil
            }
            peaks = await MediaIO.waveform(url, buckets: 240)
        }
        .onReceive(timer) { _ in
            guard let player else { return }
            position = player.currentTime
            if playing && !player.isPlaying { playing = false }
        }
        .onDisappear { player?.stop() }
    }

    func toggle() {
        guard let player else { return }
        if player.isPlaying { player.pause(); playing = false } else { player.play(); playing = true }
    }

    func seek(to fraction: CGFloat) {
        guard let player else { return }
        player.currentTime = Double(min(max(fraction, 0), 1)) * player.duration
        position = player.currentTime
    }

    func format(_ s: Double) -> String { String(format: "%d:%04.1f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60)) }
}

struct Waveform: View {
    let peaks: [Float]
    var progress: Double = 0

    var body: some View {
        Canvas { ctx, size in
            guard !peaks.isEmpty else { return }
            let w = size.width / CGFloat(peaks.count)
            for (i, p) in peaks.enumerated() {
                let h = max(2, CGFloat(min(p, 1)) * size.height)
                let rect = CGRect(x: CGFloat(i) * w + w * 0.15, y: (size.height - h) / 2, width: w * 0.7, height: h)
                let played = Double(i) / Double(peaks.count) < progress
                ctx.fill(Path(roundedRect: rect, cornerRadius: w * 0.3),
                         with: .color(played ? Color.accentColor : Color.secondary.opacity(0.5)))
            }
        }
    }
}

/// A small waveform thumbnail for history rows.
struct AudioThumb: View {
    let url: URL
    @State private var peaks: [Float] = []

    var body: some View {
        Rectangle().fill(.quaternary)
            .overlay { Waveform(peaks: peaks).padding(6) }
            .task(id: url) { peaks = await MediaIO.waveform(url, buckets: 24) }
    }
}

/// The Simple panel for video, music, speech and sound: which model the mode needs and whether it's installed.
/// Generation controls arrive with each model (M10–M13).
struct MediaModePanel: View {
    @Environment(ModelStore.self) private var store
    @Environment(\.openWindow) private var openWindow
    let mode: SimpleMode

    var body: some View {
        let entries = ModelCatalog.entries.filter { $0.kind == mode.catalogKind }
        Form {
            Section {
                ForEach(entries) { e in
                    VStack(alignment: .leading, spacing: 4) {
                        Label(e.name, systemImage: mode.symbol).fontWeight(.medium)
                        Text(e.notes).font(.caption).foregroundStyle(.secondary)
                        if e.supported, case .installed? = store.status[e.id] {
                            Label("Installed", systemImage: "checkmark.circle.fill").font(.caption).foregroundStyle(.green)
                        } else if e.supported {
                            Text("Not installed").font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("This app version can't run it yet; support is coming in a later update.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 2)
                }
                Button("Get Model…") { openWindow(id: "models") }
            } header: {
                Text("\(mode.rawValue) Model")
            }
            Section {
                Text("Meanwhile, the node graph already handles video and audio: Test Pattern, Test Tone, Load Video, Load Audio, Mix Audio, Add Audio to Video, Save Video and Save Audio. See examples/media-test.json.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

/// The Simple panel's Video mode: edits the video graph's prompt, first frame, size, length and seed.
struct VideoPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        @Bindable var model = model
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let choices = store.videoChoices
        let installed = store.installed(.video)
        let loader = model.firstNode(ofType: "LoadModel")
        let prompt = model.firstNode(ofType: "TextEncode")
        let latent = model.firstNode(ofType: "EmptyVideoLatent")
        let sampler = model.firstNode(ofType: "VideoSampler")
        let loaderPath: String = {
            if let loader, case .value(.string(let p))? = loader.inputs["path"] { return p }
            return ""
        }()
        let wan = VideoFamily.of(path: loaderPath) == .wan
        let hasWan = choices.wanT2V != nil || choices.wanI2V != nil

        if installed == nil {
            MediaModePanel(mode: .video)
        } else {
            Form {
                if hasWan && choices.ltx != nil {
                    Section {
                        Picker("Model", selection: Binding(
                            get: { wan ? VideoFamily.wan : .ltx },
                            set: { f in
                                let path = f == .wan
                                    ? (model.firstFramePath != nil ? choices.wanI2V ?? choices.wanT2V : choices.wanT2V ?? choices.wanI2V)
                                    : choices.ltx
                                if let path { model.setVideoModel(path: path.path, undoManager: undoManager) }
                            })) {
                            Text("LTX-2.3 · with sound").tag(VideoFamily.ltx)
                            Text("Wan 2.2 A14B · silent").tag(VideoFamily.wan)
                        }
                    }
                }
                Section("Prompt") {
                    if let prompt {
                        TextEditor(text: b.string(prompt.id, "prompt"))
                            .font(.body)
                            .frame(minHeight: 120)
                            .scrollContentBackground(.hidden)
                        Text(wan ? "Describe the subject, the motion and the camera. Wan makes silent clips."
                                 : "Describe the action, camera and sound: the model also generates the soundtrack.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("First Frame") {
                    if let path = model.firstFramePath {
                        HStack(spacing: 10) {
                            Thumbnail(path: path, maskPath: nil).frame(width: 64, height: 64)
                            VStack(alignment: .leading, spacing: 3) {
                                Text((path as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                                Text("The clip starts from this image.").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Remove", role: .destructive) {
                                model.clearFirstFrame(undoManager: undoManager)
                                if wan, let t2v = choices.wanT2V { model.setVideoModel(path: t2v.path, undoManager: undoManager) }
                            }
                        }
                    } else if wan && choices.wanI2V == nil {
                        Text("Download Wan 2.2 I2V in the Models window to animate an image.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Button("Choose Image…") { chooseFirstFrame(wanI2V: wan ? choices.wanI2V : nil) }
                        Text(wan && choices.wanT2V == nil ? "Required: Wan 2.2 I2V animates a still image."
                                                          : "Optional: animate a still image instead of starting from text alone.")
                            .font(.caption).foregroundStyle(wan && choices.wanT2V == nil ? .orange : .secondary)
                    }
                }
                if let latent {
                    Section("Clip") {
                        let presets = wan ? GraphTemplates.wanSizePresets : GraphTemplates.videoSizePresets
                        let w = b.int(latent.id, "width"), h = b.int(latent.id, "height")
                        Picker("Size", selection: Binding(
                            get: { presets.firstIndex { $0.width == w.wrappedValue && $0.height == h.wrappedValue } ?? -1 },
                            set: { i in
                                guard i >= 0 else { return }
                                let p = presets[i]
                                model.mutate("Change Size", undoManager: undoManager) { g in
                                    try? g.setValue(.int(p.width), node: latent.id, input: "width")
                                    try? g.setValue(.int(p.height), node: latent.id, input: "height")
                                }
                            })) {
                            ForEach(presets.indices, id: \.self) { i in
                                Text("\(presets[i].name) · \(presets[i].width)×\(presets[i].height)").tag(i)
                            }
                            Text("Custom").tag(-1)
                        }
                        let secs = b.double(latent.id, "seconds")
                        LabeledContent("Length") {
                            HStack {
                                Slider(value: secs, in: wan ? 1...5 : 1...10, step: wan ? 0.25 : 0.5)
                                Text(String(format: "%.1f s", secs.wrappedValue)).monospacedDigit().frame(width: 44, alignment: .trailing)
                            }
                        }
                        if wan {
                            LabeledContent("Frame rate", value: "16 fps")
                        } else {
                            Picker("Frame rate", selection: b.double(latent.id, "fps")) {
                                ForEach([24.0, 25, 30], id: \.self) { Text("\(Int($0)) fps").tag($0) }
                            }
                        }
                    }
                }
                if let sampler {
                    Section("Sampling") {
                        HStack {
                            TextField("Seed", value: b.int(sampler.id, "seed"), format: .number.grouping(.never))
                            Button { b.int(sampler.id, "seed").wrappedValue = Int.random(in: 0...Int(Int32.max)) } label: {
                                Image(systemName: "dice")
                            }
                            .buttonStyle(.borderless)
                            .help("New random seed")
                        }
                        Toggle("Randomize seed each run", isOn: $model.randomizeSeed)
                        if wan {
                            let quality = b.string(sampler.id, "quality")
                            Picker("Quality", selection: Binding(
                                get: { quality.wrappedValue.isEmpty ? "fast" : quality.wrappedValue },
                                set: { quality.wrappedValue = $0; b.int(sampler.id, "steps").wrappedValue = 0 })) {
                                Text("Fast · 4 steps").tag("fast")
                                Text("Full · 40 steps").tag("full")
                            }
                            Text(quality.wrappedValue == "full"
                                 ? "The official sampler with guidance: about 20× slower than Fast."
                                 : "Lightning distillation: about 5–10 minutes for 5 s at 832×480.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Stepper(value: b.int(sampler.id, "steps"), in: 0...8) {
                                let s = b.int(sampler.id, "steps").wrappedValue
                                LabeledContent("Steps", value: s == 0 ? "Model default (8 + 3)" : "\(s) + 3")
                            }
                        }
                    }
                }
                Section("Model") {
                    if let loader {
                        let path = b.string(loader.id, "path")
                        LabeledContent("Folder") {
                            Text((path.wrappedValue as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        }
                        if !FileManager.default.fileExists(atPath: ModelPaths.resolve(path.wrappedValue).path),
                           let fallback = wan ? (choices.wanT2V ?? choices.wanI2V) : (choices.ltx ?? installed) {
                            Button("Use \(fallback.lastPathComponent)") { path.wrappedValue = fallback.path }
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .safeAreaInset(edge: .bottom) {
                Button { model.enqueue(undoManager: undoManager) } label: {
                    Label("Generate Video", systemImage: "film").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .padding(12)
                .background(.bar)
            }
        }
    }

    /// Picks the first frame; with Wan selected, switches the graph to the I2V model.
    func chooseFirstFrame(wanI2V: URL?) {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]
        guard p.runModal() == .OK, let u = p.url else { return }
        model.setFirstFrame(u, undoManager: undoManager)
        if let wanI2V { model.setVideoModel(path: wanI2V.path, undoManager: undoManager) }
    }
}

/// The Simple panel's Sound mode: prompt, length and seed of the Text to Audio node.
struct SoundPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        @Bindable var model = model
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let installed = store.installed(.sound)
        let loader = model.firstNode(ofType: "LoadModel")
        let gen = model.firstNode(ofType: "TextToAudio")

        if installed == nil {
            MediaModePanel(mode: .sound)
        } else {
            Form {
                if let gen {
                    Section("Prompt") {
                        TextEditor(text: b.string(gen.id, "prompt"))
                            .font(.body)
                            .frame(minHeight: 120)
                            .scrollContentBackground(.hidden)
                        Text("Describe the sound: sources, setting, texture, tempo for loops (e.g. \"128 BPM tech house drum loop\").")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Sound") {
                        let secs = b.double(gen.id, "seconds")
                        LabeledContent("Length") {
                            HStack {
                                Slider(value: secs, in: 1...11, step: 0.5)
                                Text(String(format: "%.1f s", secs.wrappedValue)).monospacedDigit().frame(width: 44, alignment: .trailing)
                            }
                        }
                        HStack {
                            TextField("Seed", value: b.int(gen.id, "seed"), format: .number.grouping(.never))
                            Button { b.int(gen.id, "seed").wrappedValue = Int.random(in: 0...Int(Int32.max)) } label: {
                                Image(systemName: "dice")
                            }
                            .buttonStyle(.borderless)
                            .help("New random seed")
                        }
                        Toggle("Randomize seed each run", isOn: $model.randomizeSeed)
                        Stepper(value: b.int(gen.id, "steps"), in: 0...50) {
                            let s = b.int(gen.id, "steps").wrappedValue
                            LabeledContent("Steps", value: s == 0 ? "Model default (8)" : "\(s)")
                        }
                        Stepper(value: $model.batchCount, in: 1...16) {
                            LabeledContent("Batch", value: "\(model.batchCount) clip\(model.batchCount == 1 ? "" : "s")")
                        }
                    }
                }
                Section("Model") {
                    if let installed, let loader {
                        let path = b.string(loader.id, "path")
                        LabeledContent("Folder") {
                            Text((path.wrappedValue as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        }
                        if URL(fileURLWithPath: path.wrappedValue).standardizedFileURL != installed.standardizedFileURL {
                            Button("Use \(installed.lastPathComponent)") { path.wrappedValue = installed.path }
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .safeAreaInset(edge: .bottom) {
                Button { model.enqueue(undoManager: undoManager) } label: {
                    Label(model.batchCount > 1 ? "Generate \(model.batchCount)" : "Generate Sound", systemImage: "waveform")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .padding(12)
                .background(.bar)
            }
        }
    }
}

/// The Simple panel's Music mode: style, lyrics and song settings of the Text to Music node.
struct MusicPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        @Bindable var model = model
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let installed = store.installed(.music)
        let loader = model.firstNode(ofType: "LoadModel")
        let gen = model.firstNode(ofType: "TextToMusic")

        if installed == nil {
            MediaModePanel(mode: .music)
        } else {
            Form {
                if let gen {
                    Section("Style") {
                        TextEditor(text: b.string(gen.id, "tags"))
                            .font(.body)
                            .frame(minHeight: 80)
                            .scrollContentBackground(.hidden)
                        Text("Genre, instruments, mood, vocals (e.g. \"dreamy indie pop, female vocals, jangly guitars, 90 BPM\").")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Lyrics") {
                        let lyrics = b.string(gen.id, "lyrics")
                        let instrumental = Binding(
                            get: { lyrics.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines) == "[Instrumental]" },
                            set: { lyrics.wrappedValue = $0 ? "[Instrumental]" : "[Verse]\n\n[Chorus]\n" })
                        Toggle("Instrumental", isOn: instrumental)
                        if !instrumental.wrappedValue {
                            TextEditor(text: lyrics)
                                .font(.body.monospaced())
                                .frame(minHeight: 160)
                                .scrollContentBackground(.hidden)
                            Text("Mark sections with [Verse], [Chorus], [Bridge]… One line per sung phrase.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Section("Song") {
                        let secs = b.double(gen.id, "seconds")
                        LabeledContent("Length") {
                            HStack {
                                // Rounded in the binding rather than with `step:` (which draws 46 tick marks).
                                Slider(value: Binding(get: { secs.wrappedValue },
                                                      set: { secs.wrappedValue = ($0 / 5).rounded() * 5 }),
                                       in: 10...240)
                                Text(Self.duration(secs.wrappedValue)).monospacedDigit().frame(width: 44, alignment: .trailing)
                            }
                        }
                        Toggle(isOn: b.bool(gen.id, "planner")) {
                            Text("Plan the song")
                            Text("A language model writes the arrangement first: better structure and vocals, slower.")
                        }
                        let bpm = b.int(gen.id, "bpm")
                        LabeledContent("Tempo") {
                            HStack {
                                TextField("Tempo", value: Binding(get: { bpm.wrappedValue > 0 ? bpm.wrappedValue : nil },
                                                                  set: { bpm.wrappedValue = min(300, max(0, $0 ?? 0)) }),
                                          format: .number.grouping(.never), prompt: Text("Auto"))
                                    .labelsHidden()
                                    .multilineTextAlignment(.trailing)
                                    .frame(width: 64)
                                Text("BPM").foregroundStyle(.secondary)
                            }
                        }
                        Picker("Key", selection: b.string(gen.id, "key")) {
                            ForEach(MusicKeys.all, id: \.self) { Text($0 == "auto" ? "Auto" : $0).tag($0) }
                        }
                        Picker("Time signature", selection: b.string(gen.id, "time_signature")) {
                            ForEach(MusicKeys.timeSignatures, id: \.self) { Text($0 == "auto" ? "Auto" : "\($0)/\($0 == "6" ? "8" : "4")").tag($0) }
                        }
                        Picker("Vocal language", selection: b.string(gen.id, "language")) {
                            ForEach(MusicKeys.languages, id: \.self) { Text(MusicChoices.languageName($0)).tag($0) }
                        }
                    }
                    Section("Generation") {
                        HStack {
                            TextField("Seed", value: b.int(gen.id, "seed"), format: .number.grouping(.never))
                            Button { b.int(gen.id, "seed").wrappedValue = Int.random(in: 0...Int(Int32.max)) } label: {
                                Image(systemName: "dice")
                            }
                            .buttonStyle(.borderless)
                            .help("New random seed")
                        }
                        Toggle("Randomize seed each run", isOn: $model.randomizeSeed)
                        Stepper(value: b.int(gen.id, "steps"), in: 0...20) {
                            let s = b.int(gen.id, "steps").wrappedValue
                            LabeledContent("Steps", value: s == 0 ? "Model default (8)" : "\(s)")
                        }
                        Stepper(value: $model.batchCount, in: 1...16) {
                            LabeledContent("Batch", value: "\(model.batchCount) song\(model.batchCount == 1 ? "" : "s")")
                        }
                    }
                }
                Section("Model") {
                    if let installed, let loader {
                        let path = b.string(loader.id, "path")
                        LabeledContent("Folder") {
                            Text((path.wrappedValue as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        }
                        if URL(fileURLWithPath: path.wrappedValue).standardizedFileURL != installed.standardizedFileURL {
                            Button("Use \(installed.lastPathComponent)") { path.wrappedValue = installed.path }
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .safeAreaInset(edge: .bottom) {
                Button { model.enqueue(undoManager: undoManager) } label: {
                    Label(model.batchCount > 1 ? "Generate \(model.batchCount)" : "Generate Song", systemImage: "music.note")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .padding(12)
                .background(.bar)
            }
        }
    }

    static func duration(_ s: Double) -> String {
        let t = Int(s.rounded())
        return t >= 60 ? String(format: "%d:%02d", t / 60, t % 60) : "\(t) s"
    }
}

/// Names for the Music panel's language choices (the codes come from the Text to Music node).
enum MusicChoices {
    static func languageName(_ code: String) -> String {
        if code == "auto" { return "Auto" }
        if code == "yue" { return "Cantonese" }
        return Locale.current.localizedString(forLanguageCode: code) ?? code
    }
}

/// The Simple panel's Speech mode: text, voice (with a spoken preview) and speed of the Text to Speech node.
struct SpeechPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        @Bindable var model = model
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let installed = store.installed(.speech)
        let loader = model.firstNode(ofType: "LoadModel")
        let gen = model.firstNode(ofType: "TextToSpeech")

        if installed == nil {
            MediaModePanel(mode: .speech)
        } else {
            Form {
                if let gen {
                    Section("Text") {
                        TextEditor(text: b.string(gen.id, "text"))
                            .font(.body)
                            .frame(minHeight: 200)
                            .scrollContentBackground(.hidden)
                        // Verbatim: the bracket syntax would otherwise render as Markdown links.
                        Text(verbatim: "Any length: long text is read in sentence-sized chunks. Force a pronunciation with [word](/phonemes/), e.g. [Kokoro](/kˈOkəɹO/).")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Voice") {
                        let voice = b.string(gen.id, "voice")
                        Picker("Voice", selection: voice) {
                            ForEach(["American", "British"], id: \.self) { accent in
                                Section(accent) {
                                    ForEach(SpeechVoices.all.filter { $0.accent == accent }, id: \.id) { v in
                                        Text("\(v.name) · \(v.gender) · \(v.grade)").tag(v.id)
                                    }
                                }
                            }
                        }
                        let speed = b.double(gen.id, "speed")
                        LabeledContent("Speed") {
                            HStack {
                                Slider(value: Binding(get: { speed.wrappedValue },
                                                      set: { speed.wrappedValue = ($0 * 20).rounded() / 20 }),
                                       in: 0.5...2)
                                Text(String(format: "%.2f×", speed.wrappedValue)).monospacedDigit().frame(width: 48, alignment: .trailing)
                            }
                        }
                        if let loader {
                            Button {
                                model.previewVoice(voice.wrappedValue, speed: speed.wrappedValue,
                                                   modelPath: b.string(loader.id, "path").wrappedValue)
                            } label: {
                                Label("Preview Voice", systemImage: "play.circle")
                            }
                            .disabled(model.isRunning)
                            Text("Grades (A to F) are from the model's authors: how well each voice was trained.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Section("Generation") {
                        Stepper(value: $model.batchCount, in: 1...16) {
                            LabeledContent("Batch", value: "\(model.batchCount) take\(model.batchCount == 1 ? "" : "s")")
                        }
                    }
                }
                Section("Model") {
                    if let installed, let loader {
                        let path = b.string(loader.id, "path")
                        LabeledContent("Folder") {
                            Text((path.wrappedValue as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        }
                        if URL(fileURLWithPath: path.wrappedValue).standardizedFileURL != installed.standardizedFileURL {
                            Button("Use \(installed.lastPathComponent)") { path.wrappedValue = installed.path }
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .safeAreaInset(edge: .bottom) {
                Button { model.enqueue(undoManager: undoManager) } label: {
                    Label(model.batchCount > 1 ? "Generate \(model.batchCount)" : "Generate Speech", systemImage: "waveform")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .padding(12)
                .background(.bar)
            }
        }
    }
}
