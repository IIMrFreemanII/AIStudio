import AVFoundation
import AVKit
import MLXDiffusionCore
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
        let installed = store.installed(.video)
        let loader = model.firstNode(ofType: "LoadModel")
        let prompt = model.firstNode(ofType: "TextEncode")
        let latent = model.firstNode(ofType: "EmptyVideoLatent")
        let sampler = model.firstNode(ofType: "VideoSampler")

        if installed == nil {
            MediaModePanel(mode: .video)
        } else {
            Form {
                Section("Prompt") {
                    if let prompt {
                        TextEditor(text: b.string(prompt.id, "prompt"))
                            .font(.body)
                            .frame(minHeight: 120)
                            .scrollContentBackground(.hidden)
                        Text("Describe the action, camera and sound: the model also generates the soundtrack.")
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
                            Button("Remove", role: .destructive) { model.clearFirstFrame(undoManager: undoManager) }
                        }
                    } else {
                        Button("Choose Image…") { chooseFirstFrame() }
                        Text("Optional: animate a still image instead of starting from text alone.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if let latent {
                    Section("Clip") {
                        let w = b.int(latent.id, "width"), h = b.int(latent.id, "height")
                        Picker("Size", selection: Binding(
                            get: { GraphTemplates.videoSizePresets.firstIndex { $0.width == w.wrappedValue && $0.height == h.wrappedValue } ?? -1 },
                            set: { i in
                                guard i >= 0 else { return }
                                let p = GraphTemplates.videoSizePresets[i]
                                model.mutate("Change Size", undoManager: undoManager) { g in
                                    try? g.setValue(.int(p.width), node: latent.id, input: "width")
                                    try? g.setValue(.int(p.height), node: latent.id, input: "height")
                                }
                            })) {
                            ForEach(GraphTemplates.videoSizePresets.indices, id: \.self) { i in
                                let p = GraphTemplates.videoSizePresets[i]
                                Text("\(p.name) · \(p.width)×\(p.height)").tag(i)
                            }
                            Text("Custom").tag(-1)
                        }
                        let secs = b.double(latent.id, "seconds")
                        LabeledContent("Length") {
                            HStack {
                                Slider(value: secs, in: 1...10, step: 0.5)
                                Text(String(format: "%.1f s", secs.wrappedValue)).monospacedDigit().frame(width: 44, alignment: .trailing)
                            }
                        }
                        Picker("Frame rate", selection: b.double(latent.id, "fps")) {
                            ForEach([24.0, 25, 30], id: \.self) { Text("\(Int($0)) fps").tag($0) }
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
                        Stepper(value: b.int(sampler.id, "steps"), in: 0...8) {
                            let s = b.int(sampler.id, "steps").wrappedValue
                            LabeledContent("Steps", value: s == 0 ? "Model default (8 + 3)" : "\(s) + 3")
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

    func chooseFirstFrame() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]
        if p.runModal() == .OK, let u = p.url { model.setFirstFrame(u, undoManager: undoManager) }
    }
}
