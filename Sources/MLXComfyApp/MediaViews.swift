import AVFoundation
import AVKit
import MLXDiffusionCore
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

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
                        .foregroundStyle(MG.accent)
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.space, modifiers: [])
                Text("\(format(position)) / \(format(duration))").font(.mgTitle3.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
            }
            Text(url.lastPathComponent).font(.mgSubheadline).foregroundStyle(MG.tertiaryLabel)
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
                         with: .color(played ? MG.accent : MG.tertiaryLabel))
            }
        }
    }
}

/// A small waveform thumbnail for history rows.
struct AudioThumb: View {
    let url: URL
    @State private var peaks: [Float] = []

    var body: some View {
        Rectangle().fill(MG.fill)
            .overlay { Waveform(peaks: peaks).padding(6) }
            .task(id: url) { peaks = await MediaIO.waveform(url, buckets: 24) }
    }
}

extension CatalogEntry.Kind {
    var symbol: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .music: return "music.note"
        case .speech: return "waveform"
        case .sound: return "speaker.wave.2"
        }
    }
}
