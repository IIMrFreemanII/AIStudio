import AVFoundation
import Combine
import MLXDiffusionCore
import SwiftUI

/// A voice message in a chat: play/pause, its waveform (click or drag to seek) and its length.
struct ChatAudioPlayer: View {
    let url: URL
    @State private var peaks: [Float] = []
    @State private var player: AVAudioPlayer?
    @State private var position: Double = 0
    @State private var playing = false
    private let timer = Timer.publish(every: 1.0 / 20, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 10) {
            Button { toggle() } label: {
                Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 28)).foregroundStyle(MG.accent)
            }
            .buttonStyle(.plain)
            .mgHelp(playing ? "Pause" : "Play")
            GeometryReader { geo in
                Waveform(peaks: peaks, progress: duration > 0 ? position / duration : 0)
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0)
                        .onChanged { v in seek(to: v.location.x / max(geo.size.width, 1)) })
            }
            .frame(height: 28)
            Text(format(playing || position > 0 ? position : duration))
                .font(.mgFootnote.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                .frame(width: 36, alignment: .trailing)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .frame(width: 320)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .task(id: url) {
            player?.stop()
            playing = false
            position = 0
            player = try? AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            peaks = await MediaIO.waveform(url, buckets: 60)
        }
        .onReceive(timer) { _ in
            guard let player, playing else { return }
            position = player.currentTime
            if !player.isPlaying { playing = false; position = 0 }
        }
        .onDisappear { player?.stop() }
    }

    private var duration: Double { player?.duration ?? 0 }

    private func toggle() {
        guard let player else { return }
        if player.isPlaying { player.pause(); playing = false } else { player.play(); playing = true }
    }

    private func seek(to fraction: CGFloat) {
        guard let player else { return }
        player.currentTime = Double(min(max(fraction, 0), 1)) * player.duration
        position = player.currentTime
    }

    private func format(_ s: Double) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s.rounded(.down)) % 60) }
}
