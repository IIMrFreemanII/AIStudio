import AppKit
import AVFoundation
import MovieCore
import SwiftUI

// MARK: voices

/// Voices: a voice for each character who speaks (designed from their voice description), then every line spoken in
/// it. Talking shots are animated from these lines next, so their lips follow the words.
struct MovieVoicesPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.movie.approved.contains(.voices)
        let speaking = model.speakingCharacters(doc.movie)
        let lines = doc.movie.lineCount
        let silent = doc.movie.silentLines
        let noVoice = speaking.filter { $0.selectedVoiceSampleFile == nil }.count
        MoviePage(title: "Voices",
                  subtitle: lines == 0 ? "No one speaks in this film: nothing to voice."
                      : "\(lines - silent) of \(lines) lines are spoken. Each character's lines use the voice you choose for them.") {
            Button(noVoice > 0 ? "Make Voices…" : "Retake Voices…") { model.requestVoiceSamples(doc) }
                .buttonStyle(.mg(noVoice > 0 ? .prominent : .bordered))
                .disabled(doc.isBusy || speaking.isEmpty)
                .mgHelp(noVoice > 0 ? "A voice for each speaking character without one (you see the cost first)"
                        : "A new voice take for every speaking character")
                .mgTestAnchor("movie-make-voices")
            Button(silent > 0 ? (silent == lines ? "Speak Lines…" : "Speak Missing Lines…") : "Retake Lines…") {
                model.requestLines(doc)
            }
            .buttonStyle(.mg(noVoice == 0 && silent > 0 ? .prominent : .bordered))
            .disabled(doc.isBusy || lines == 0 || noVoice > 0)
            .mgHelp(noVoice > 0 ? "Choose a voice for every speaking character first"
                    : silent > 0 ? "Speaks each line without a take, in its character's voice (you see the cost first)"
                    : "A new take of every line")
            .mgTestAnchor("movie-speak-lines")
            Button(approved ? "Approved" : "Approve Voices") {
                doc.movie.approve(.voices)
                doc.stage = .video
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy)
            .mgHelp(silent > 0 ? "\(silent) line\(silent == 1 ? "" : "s") without a take will be silent (and not lip-synced)"
                    : "Locks the lines in; talking shots are animated from them next")
            .mgTestAnchor("movie-approve-voices")
        } content: {
            VStack(alignment: .leading, spacing: 22) {
                if speaking.isEmpty {
                    MGEmptyState(symbol: "waveform", title: "No dialogue",
                                 message: "No shot has lines. Add them on the Shot List page, or approve and go on.")
                        .frame(maxWidth: .infinity, minHeight: 200)
                } else {
                    Text("CAST").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                    ForEach(speaking) { c in
                        if let i = doc.movie.characters.firstIndex(where: { $0.id == c.id }) {
                            VoiceCard(doc: doc, character: $doc.movie.characters[i])
                        }
                    }
                    Text("LINES").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                        .padding(.top, 6)
                    ForEach(Array(doc.movie.shots.enumerated()), id: \.element.id) { i, shot in
                        if !shot.dialogue.isEmpty { ShotLinesCard(doc: doc, shot: shot, number: i + 1) }
                    }
                }
            }
        }
    }
}

private struct VoiceCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var character: MovieCharacter

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(character.name).font(.mgHeadline)
                Spacer()
                Button(character.voiceSamples.isEmpty ? "Make Voice…" : "Another Take…") {
                    model.requestVoiceSamples(doc, characters: [character.id])
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small).disabled(doc.isBusy)
                .mgHelp("A voice for \(character.name) from the description (you see the cost first)")
            }
            MovieField(label: "Voice", hint: "edit, then make another take") {
                TextField("Age, pitch, accent, texture (warm low voice, slight rasp)", text: $character.voice,
                          axis: .vertical)
                    .lineLimit(1...3).font(.mgCallout)
            }
            if character.voiceSamples.isEmpty {
                Text("No voice yet").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
            } else {
                SoundTakeStrip(doc: doc, takes: character.voiceSamples, selected: character.selectedVoiceSample,
                               select: { model.selectVoiceSample(doc, character: character.id, take: $0) })
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .mgTestAnchor("movie-voice-card")
    }
}

private struct ShotLinesCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    let shot: Shot
    let number: Int

    var body: some View {
        let needed = MovieVoices.neededDuration(shot)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("\(number)").font(.mg(size: 12, weight: .bold)).monospacedDigit()
                    .frame(minWidth: 22, minHeight: 22)
                    .background(Circle().fill(MG.accent.opacity(0.15)))
                    .foregroundStyle(MG.accent)
                Text([shot.framing, MovieFormat.seconds(shot.duration)].filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                if shot.talking {
                    Label("Lip-sync", systemImage: "mouth").font(.mgFootnote).foregroundStyle(MG.accent)
                        .mgHelp(doc.movie.lipSync ? "The speaker is on screen: this shot is animated from its lines"
                                : "Lip-sync is off for this movie (Shots page)")
                }
                if let needed, needed > shot.duration + 0.05 {
                    Label("Lines need \(MovieFormat.seconds((needed * 10).rounded(.up) / 10))", systemImage: "exclamationmark.triangle")
                        .font(.mgFootnote).foregroundStyle(MG.warning)
                }
                Spacer(minLength: 2)
                Button { model.requestLines(doc, shots: [shot.id]) } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.mg(.icon)).disabled(doc.isBusy)
                    .mgHelp("Speak this shot's lines again (you see the cost first)")
            }
            ForEach(shot.dialogue) { line in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(doc.movie.character(line.character)?.name ?? line.character).font(.mgSubheadline)
                        Text(line.text).font(.mgCallout)
                        if !line.delivery.isEmpty {
                            Text("(\(line.delivery))").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                        }
                    }
                    if !line.takes.isEmpty {
                        SoundTakeStrip(doc: doc, takes: line.takes, selected: line.selectedTake,
                                       select: { model.selectLine(doc, shot: shot.id, line: line.id, take: $0) })
                    }
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .mgTestAnchor("movie-lines-card")
    }
}

// MARK: sound

/// Sound: each shot's effects (from its sound prompt, as long as the shot) and each scene's music, plus how loud
/// each kind plays in the film.
struct MovieSoundPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.movie.approved.contains(.sound)
        let withFX = doc.movie.shots.filter { MovieSound.effectsPrompt($0) != nil }
        let fxMissing = withFX.filter { $0.sounds.isEmpty }.count
        let musicMissing = doc.movie.music.filter { $0.takes.isEmpty }.count
        MoviePage(title: "Sound",
                  subtitle: "Effects for \(withFX.count - fxMissing) of \(withFX.count) shots, music for "
                      + "\(doc.movie.music.count - musicMissing) of \(doc.movie.music.count) scenes. Both are optional.") {
            Button(fxMissing > 0 ? "Make Effects…" : "Retake Effects…") { model.requestEffects(doc) }
                .buttonStyle(.mg(fxMissing > 0 ? .prominent : .bordered))
                .disabled(doc.isBusy || withFX.isEmpty)
                .mgHelp(withFX.isEmpty ? "No shot has a sound prompt (add one below)"
                        : "Sound effects for each shot with a sound prompt and no take (you see the cost first)")
                .mgTestAnchor("movie-make-effects")
            Button(musicMissing > 0 ? "Make Music…" : "Retake Music…") { model.requestMusic(doc) }
                .buttonStyle(.mg(musicMissing > 0 ? .prominent : .bordered))
                .disabled(doc.isBusy || doc.movie.music.isEmpty)
                .mgHelp("A piece of music for each scene without one (you see the cost first)")
                .mgTestAnchor("movie-make-music")
            Button(approved ? "Approved" : "Approve Sound") {
                doc.movie.approve(.sound)
                doc.stage = .edit
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy)
            .mgHelp("Locks the sound in; the edit mixes it with the voices under the picture")
            .mgTestAnchor("movie-approve-sound")
        } content: {
            VStack(alignment: .leading, spacing: 22) {
                MixControls(doc: doc)
                Text("MUSIC").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                ForEach($doc.movie.music) { $cue in
                    MusicCueCard(doc: doc, cue: $cue)
                }
                Text("EFFECTS").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                    .padding(.top, 6)
                ForEach(Array(doc.movie.shots.indices), id: \.self) { i in
                    EffectsCard(doc: doc, shot: $doc.movie.shots[i], number: i + 1)
                }
            }
        }
        .onAppear {
            let cues = MovieSound.cues(for: doc.movie)
            if cues != doc.movie.music { doc.movie.music = cues }
        }
    }
}

/// The mix: how loud dialogue, effects and music play, and how far the music dips under the lines.
struct MixControls: View {
    @Bindable var doc: MovieDocument

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("MIX").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                row("Dialogue", value: $doc.movie.mix.dialogue, help: "How loud the lines play")
                row("Effects", value: $doc.movie.mix.effects, help: "How loud each shot's sound effects play")
                row("Music", value: $doc.movie.mix.music, help: "How loud the music plays when no one speaks")
                row("Under dialogue", value: $doc.movie.mix.duck,
                    help: "The music's level while someone speaks, as a share of its own (it dips and comes back)")
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: doc.movie.mix) { _, _ in
            if doc.movie.approved.contains(.edit) { doc.movie.reopen(.edit) }
        }
    }

    private func row(_ label: String, value: Binding<Double>, help: String) -> some View {
        GridRow {
            Text(label).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
            Slider(value: value, in: 0...1).frame(width: 220)
            Text("\(Int((value.wrappedValue * 100).rounded())) %").font(.mgFootnote).monospacedDigit()
                .foregroundStyle(MG.secondaryLabel).frame(width: 44, alignment: .trailing)
        }
        .mgHelp(help)
    }
}

private struct MusicCueCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var cue: MusicCue

    var body: some View {
        let heading = doc.movie.scenes.first { $0.id == cue.id }?.heading ?? cue.id
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(heading).font(.mgHeadline)
                Text(MovieFormat.seconds(MovieSound.sceneSeconds(cue.id, in: doc.movie))).font(.mgFootnote)
                    .foregroundStyle(MG.secondaryLabel)
                Spacer()
                Button(cue.takes.isEmpty ? "Make Music…" : "Another Take…") {
                    model.requestMusic(doc, scenes: [cue.id])
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small).disabled(doc.isBusy)
                .mgHelp("Music for this scene (you see the cost first)")
            }
            TextField("Style: genre, instruments, mood", text: $cue.prompt, axis: .vertical)
                .lineLimit(1...3).font(.mgCallout)
            if !cue.takes.isEmpty {
                SoundTakeStrip(doc: doc, takes: cue.takes, selected: cue.selected,
                               select: { model.selectMusic(doc, scene: cue.id, take: $0) })
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .mgTestAnchor("movie-music-card")
    }
}

private struct EffectsCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var shot: Shot
    let number: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("\(number)").font(.mg(size: 12, weight: .bold)).monospacedDigit()
                    .frame(minWidth: 22, minHeight: 22)
                    .background(Circle().fill(MG.accent.opacity(0.15)))
                    .foregroundStyle(MG.accent)
                TextField("What it sounds like (rain on a tin roof, distant thunder)", text: $shot.sfx)
                    .font(.mgCallout)
                Text(MovieFormat.seconds(shot.duration)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Button { model.requestEffects(doc, shots: [shot.id]) } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.mg(.icon)).disabled(doc.isBusy || MovieSound.effectsPrompt(shot) == nil)
                    .mgHelp(shot.sounds.isEmpty ? "Make this shot's sound effects (you see the cost first)"
                            : "Another take of this shot's sound effects (you see the cost first)")
            }
            if !shot.sounds.isEmpty {
                SoundTakeStrip(doc: doc, takes: shot.sounds, selected: shot.selectedSound,
                               select: { model.selectSound(doc, shot: shot.id, take: $0) })
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .mgTestAnchor("movie-effects-card")
    }
}

// MARK: shared

/// A row of sound takes: play one, choose one; its menu reveals or deletes it.
struct SoundTakeStrip: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    let takes: [Take]
    let selected: UUID?
    let select: (UUID) -> Void
    @State private var player = MovieSoundPlayer.shared

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(takes.enumerated()), id: \.element.id) { i, take in
                    let url = model.movieStore.url(doc.id, relative: take.file)
                    let chosen = take.id == selected
                    HStack(spacing: 6) {
                        Button { player.toggle(url) } label: {
                            Image(systemName: player.playing == url ? "stop.fill" : "play.fill")
                        }
                        .buttonStyle(.mg(.icon))
                        .mgHelp(player.playing == url ? "Stop" : "Play this take")
                        Button { select(take.id) } label: {
                            HStack(spacing: 4) {
                                Text("Take \(i + 1)").font(.mgFootnote)
                                if let s = take.seconds {
                                    Text(String(format: "%.1f s", s)).font(.mgFootnote).monospacedDigit()
                                        .foregroundStyle(MG.secondaryLabel)
                                }
                                if chosen { Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.accent) }
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(doc.isBusy)
                        .mgHelp(chosen ? "Chosen" + costText(take) : "Choose this take" + costText(take))
                    }
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Capsule().fill(chosen ? MG.accent.opacity(0.12) : MG.fill))
                    .overlay(Capsule().strokeBorder(chosen ? MG.accent : MG.separator, lineWidth: chosen ? 1.5 : 1))
                    .mgContextMenu {
                        MGMenuItem.button("Choose") { select(take.id) }
                        MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        MGMenuItem.divider
                        MGMenuItem.button("Move to Trash", destructive: true) { model.deleteSoundTake(doc, take: take.id) }
                    }
                }
            }
            .padding(2)
        }
    }

    private func costText(_ take: Take) -> String {
        take.costUSD.map { " · " + CostEstimate.format($0) } ?? ""
    }
}

/// Plays one take at a time (starting another stops the first).
@MainActor @Observable
final class MovieSoundPlayer: NSObject, AVAudioPlayerDelegate {
    static let shared = MovieSoundPlayer()
    private(set) var playing: URL?
    @ObservationIgnored private var player: AVAudioPlayer?

    func toggle(_ url: URL) {
        if playing == url { stop(); return }
        stop()
        guard let p = try? AVAudioPlayer(contentsOf: url) else { return }
        p.delegate = self
        p.play()
        player = p
        playing = url
    }

    func stop() {
        player?.stop()
        player = nil
        playing = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stop() }
    }
}
