import AVFoundation
import AVKit
import AppKit
import Combine
import MovieCore
import SwiftUI

// MARK: shots (clips)

/// Shots: every shot animated from its keyframe (a continuing shot from the clip before's last frame); a few takes
/// each, one chosen.
struct MovieClipsPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.isApproved(.video)
        let engine = doc.movie.videoEngine
        let shots = doc.scopeShots
        let index = doc.movie.shotIndex
        let ready = doc.movie.videosAreReady(in: doc.sequence)
        let missing = shots.filter { !$0.hasClip(by: engine) }.count
        let done = shots.count - missing
        MoviePage(title: "Shots" + (doc.scopeTitle.map { " · " + $0 } ?? ""),
                  subtitle: engine == .wan22
                      ? "\(done) of \(shots.count) shots have a clip. Each is animated from its keyframe."
                      : "\(done) of \(shots.count) shots have a MiniMax H3 clip, with its own sound.") {
            Button(missing > 0 ? (done == 0 ? "Make Clips…" : "Make Missing Clips…") : "Retake All…") {
                model.requestClips(doc)
            }
            .buttonStyle(.mg(missing > 0 ? .prominent : .bordered))
            .disabled(doc.isBusy || shots.isEmpty)
            .mgHelp(missing > 0 ? "One clip for each shot without one from this engine (you see the cost first)"
                    : "A new take for every shot; the chosen ones stay until you pick another")
            .mgTestAnchor("movie-make-clips")
            MGPicker(selection: $doc.movie.videoEngine, options: VideoEngine.allCases.map { ($0, $0.title) },
                     maxWidth: 200)
                .disabled(doc.isBusy)
                .mgHelp("Wan 2.2: silent clips; the lines, effects and lip-sync come from the other stages. MiniMax "
                        + "H3: each clip with its own sound (its lines in the characters' voices, effects, ambience), "
                        + "on an H200; its licence excludes the US, EU, UK and Korea without MiniMax's own licence.")
                .mgTestAnchor("movie-video-engine")
                .onChange(of: doc.movie.videoEngine) { doc.movie.reopen(.video) }
            if doc.movie.videoEngine == .wan22 {
            Toggle("Lip-sync", isOn: $doc.movie.lipSync)
                .toggleStyle(.switch).controlSize(.small)
                .disabled(doc.isBusy)
                .mgHelp("Talking shots with spoken lines are animated from their voices, lips in sync (slower and "
                        + "dearer); off: like every other shot")
                .mgTestAnchor("movie-lip-sync")
            if doc.movie.lipSync {
                MGPicker(selection: $doc.movie.lipSyncEngine,
                         options: LipSyncEngine.allCases.map { ($0, $0.title) }, maxWidth: 160)
                    .disabled(doc.isBusy)
                    .mgHelp("Wan 2.2 S2V: from the keyframe, 40 steps. LongCat Avatar: from the keyframe, 8 steps, "
                            + "one or two speakers. InfiniteTalk: makes the shot's ordinary clip, then re-syncs the "
                            + "speaker to the lines (talking shots up to 7.5 s)")
                    .mgTestAnchor("movie-lip-sync-engine")
                    .onChange(of: doc.movie.lipSyncEngine) { model.fitShotsToLines(doc) }
            }
            }
            Button("Edit Template") {
                if doc.movie.videoEngine == .minimaxH3 {
                    model.editMovieTemplate(MovieH3.TemplateName.video, for: doc) { MovieH3.videoTemplate() }
                } else {
                    model.editMovieTemplate(MovieShots.TemplateName.shot, for: doc) { MovieShots.template() }
                }
            }
            .buttonStyle(.mg(.borderless))
            .mgHelp("Opens the graph clips are made with as a workflow; this movie uses your edited copy")
            .mgContextMenu {
                MGMenuItem.button("Edit Talking-Shot Template") {
                    if doc.movie.videoEngine == .minimaxH3 {
                        model.editMovieTemplate(MovieH3.TemplateName.references, for: doc) { MovieH3.referencesTemplate() }
                        return
                    }
                    let engine = doc.movie.lipSyncEngine
                    model.editMovieTemplate(MovieShots.TemplateName.talking(engine), for: doc) {
                        MovieShots.talkingTemplate(for: engine)
                    }
                }
            }
            Button(approved ? "Approved" : "Approve Shots") {
                doc.approve(.video)
                doc.stage = .sound
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || !ready)
            .mgHelp(ready ? "Locks the clips in; their sound effects and music come next"
                    : "Every shot needs a chosen clip first")
            .mgTestAnchor("movie-approve-clips")
        } content: {
            LazyVStack(alignment: .leading, spacing: 26) {
                ForEach(doc.scopeScenes) { scene in
                    let ids = shots.filter { $0.scene == scene.id }.map(\.id)
                    if !ids.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text(scene.heading).font(.mgHeadline)
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 420), spacing: 12, alignment: .top)],
                                      alignment: .leading, spacing: 12) {
                                ForEach(ids, id: \.self) { id in
                                    if let i = index[id] {
                                        ClipCard(doc: doc, shot: $doc.movie.shots[i], number: i + 1)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private struct ClipCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var shot: Shot
    let number: Int

    var body: some View {
        let ends = MovieShots.ends(for: shot, in: doc.movie)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("\(number)").font(.mg(size: 12, weight: .bold)).monospacedDigit()
                    .frame(minWidth: 22, minHeight: 22)
                    .background(Circle().fill(MG.accent.opacity(0.15)))
                    .foregroundStyle(MG.accent)
                Text([shot.framing, shot.camera, MovieFormat.seconds(shot.duration)].filter { !$0.isEmpty }
                    .joined(separator: " · "))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                Spacer(minLength: 2)
                Button { model.requestClips(doc, shots: [shot.id]) } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.mg(.icon)).disabled(doc.isBusy)
                    .mgHelp(shot.videos.isEmpty ? "Make this shot's clip (you see the cost first)"
                            : "Another take of this clip (you see the cost first)")
            }
            if let take = shot.selectedVideoTake {
                MovieClipPlayer(url: model.movieStore.url(doc.id, relative: take.file),
                                poster: take.poster.map { model.movieStore.url(doc.id, relative: $0) },
                                open: { model.viewShotClips(doc, shot: shot.id) })
                    .aspectRatio(doc.movie.aspect.videoRatio, contentMode: .fit)
            } else if let file = shot.selectedKeyframeFile {
                MovieStill(url: model.movieStore.url(doc.id, relative: file), maxPixels: 840)
                    .aspectRatio(doc.movie.aspect.videoRatio, contentMode: .fill)
                    .clipped()
                    .overlay(alignment: .bottomLeading) {
                        Label("No clip yet", systemImage: "film").font(.mgFootnote)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(Capsule().fill(.black.opacity(0.55))).foregroundStyle(.white).padding(8)
                    }
            }
            if shot.videos.count > 1 {
                TakeStrip(doc: doc, takes: shot.videos, selected: shot.selectedVideo, height: 54,
                          select: { model.selectClip(doc, shot: shot.id, take: $0) },
                          delete: { model.deleteTake(doc, shot: shot.id, take: $0) },
                          open: { model.viewShotClips(doc, shot: shot.id, take: $0) })
            }
            TextField("What moves: people, camera, the place", text: $shot.motionPrompt, axis: .vertical)
                .lineLimit(2...5).font(.mgCallout)
            HStack(spacing: 6) {
                Image(systemName: startsFromClip(ends) ? "film.stack" : "photo")
                    .foregroundStyle(MG.tertiaryLabel)
                Text(startsFromClip(ends) ? "From shot \(number - 1)'s last frame to its keyframe" : "From its keyframe")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                if shot.talking {
                    let synced = MovieShots.usesLipSync(shot, in: doc.movie)
                    Text(synced ? "lip-synced" : "no lip-sync").font(.mg(size: 9.5, weight: .medium))
                        .foregroundStyle(MG.tertiaryLabel)
                        .mgHelp(synced ? "A talking shot: animated from its lines by Wan 2.2 S2V, the lips in sync"
                                : doc.movie.lipSync ? "A talking shot with no spoken lines yet (Voices): animated like the others"
                                : "Lip-sync is off on this page: animated like the others")
                }
            }
            .mgHelp(startsFromClip(ends) ? "Continues the shot before without a cut, ending where this shot was planned"
                    : "Starts on the chosen keyframe")
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: shot.motionPrompt) { _, _ in
            if doc.movie.isApproved(.video, in: doc.movie.sequence(of: shot)), !doc.isBusy {
                doc.reopen(.video, shot: shot.id)
            }
        }
        .mgTestAnchor("movie-clip-card")
        .mgScrollTarget("movie-clip:\(shot.id.uuidString)")
    }

    private func startsFromClip(_ ends: MovieShots.Ends?) -> Bool {
        if case .lastFrame? = ends?.start { return true }
        return false
    }
}

/// A clip's poster with a play button; plays inline once clicked. Its corner button opens it in the viewer (and it
/// pauses while the viewer is open).
struct MovieClipPlayer: View {
    @Environment(AppModel.self) private var model
    let url: URL
    let poster: URL?
    var open: (() -> Void)? = nil
    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            if let player {
                VideoPlayer(player: player)
            } else {
                if let poster { MovieStill(url: poster, maxPixels: 840) } else { Rectangle().fill(.black) }
                Button {
                    let p = AVPlayer(url: url)
                    player = p
                    p.play()
                } label: {
                    Image(systemName: "play.circle.fill").font(.system(size: 34)).foregroundStyle(.white)
                        .shadow(radius: 6)
                }
                .buttonStyle(.plain)
                .mgHelp("Play this clip")
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous))
        .overlay(alignment: .topTrailing) {
            if let open { MovieExpandButton { player?.pause(); open() }.padding(6) }
        }
        .onChange(of: model.chatViewer != nil) { _, viewing in if viewing { player?.pause() } }
        .onChange(of: url) { _, _ in player?.pause(); player = nil }
        .onDisappear { player?.pause() }
    }
}

extension MovieAspect {
    var videoRatio: CGFloat {
        let s = videoSize
        return CGFloat(s.width) / CGFloat(s.height)
    }
}

// MARK: edit

/// Edit: the clips cut together on a timeline (reorder, trim, transitions), played live, and exported as an MP4.
struct MovieEditPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument
    @State private var timeline: MovieTimeline?
    @State private var player = AVPlayer()
    @State private var position: Double = 0
    @State private var playing = false
    @State private var selected: UUID?
    @State private var scale: Double = 90  // points per second
    @State private var codec: MovieComposer.Codec = .h264
    @State private var buildError: String?
    private let clock = Timer.publish(every: 1.0 / 30, on: .main, in: .common).autoconnect()

    /// What the composition depends on: rebuilt when it changes.
    private var editKey: [String] {
        let m = doc.movie.mix
        return doc.movie.shots.map { s in
            "\(s.id)|\(s.selectedVideo?.uuidString ?? "-")|\(s.trimStart)|\(s.trimEnd)|\(s.transition.rawValue)|"
                + "\(s.selectedSound?.uuidString ?? "-")|" + s.dialogue.map { $0.selectedTake?.uuidString ?? "-" }.joined(separator: ",")
        } + doc.movie.music.map { "\($0.id)|\($0.selected?.uuidString ?? "-")" }
            + ["mix|\(m.dialogue)|\(m.effects)|\(m.music)|\(m.duck)"]
    }

    /// The line being spoken at the playhead (the subtitles are in order: binary search).
    private var subtitle: String? {
        guard let subs = timeline?.subtitles, !subs.isEmpty else { return nil }
        var lo = 0, hi = subs.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if subs[mid].start <= position { lo = mid + 1 } else { hi = mid }
        }
        var i = lo - 1
        while i >= 0, i >= lo - 3 {
            if position < subs[i].end { return subs[i].shown }
            i -= 1
        }
        return nil
    }

    var body: some View {
        let approved = doc.isApproved(.edit)
        MoviePage(title: "Edit",
                  subtitle: "Drag clips to reorder, drag their edges to trim, choose how each starts. Then export the film.") {
            MGSegmentedPicker(selection: $codec, options: MovieComposer.Codec.allCases.map { ($0, $0.title) })
                .frame(width: 140)
                .mgHelp("H.264 plays everywhere; HEVC is smaller")
            if doc.movie.sequences.count > 1 {
                MGMenuButton("Export Sequence") {
                    MGMenuItem.header("One sequence on its own")
                    for (i, q) in doc.movie.sequences.enumerated() {
                        MGMenuItem.button(q.chapterTitle(number: i + 1) + "…",
                                          enabled: !doc.isBusy && doc.movie.shots(inSequence: q.id).contains { $0.selectedVideoTake != nil }) {
                            model.exportMovieAs(doc, codec: codec, sequence: q.id)
                        }
                    }
                }
                .disabled(doc.isBusy)
                .mgHelp("Writes one sequence as its own MP4, to review it before the rest is done")
                .mgTestAnchor("movie-export-sequence")
            }
            Button("Export…") { model.exportMovieAs(doc, codec: codec) }
                .buttonStyle(.mg(.prominent))
                .disabled(doc.isBusy || (timeline?.clips.isEmpty ?? true))
                .mgHelp(doc.movie.sequences.count > 1 ? "Writes the film as an MP4 with a chapter per sequence (on this Mac, free)"
                        : "Writes the film as an MP4 (on this Mac, free)")
                .mgTestAnchor("movie-export")
            if let last = doc.lastExport {
                Button { NSWorkspace.shared.activateFileViewerSelecting([last]) } label: { Image(systemName: "folder") }
                    .buttonStyle(.mg(.icon))
                    .mgHelp("Show \(last.lastPathComponent) in Finder")
            }
            Button(approved ? "Approved" : "Approve Edit") { doc.approve(.edit) }
                .buttonStyle(.mg(.bordered))
                .disabled(approved || doc.isBusy || (timeline?.clips.isEmpty ?? true) || !doc.movie.editIsApprovable)
                .mgHelp(doc.movie.editIsApprovable ? "Marks the cut as final"
                        : "Every sequence's sound must be approved first")
        } content: {
            VStack(alignment: .leading, spacing: 14) {
                VideoPlayer(player: player)
                    .aspectRatio(doc.movie.aspect.videoRatio, contentMode: .fit)
                    .frame(maxHeight: 460)
                    .frame(maxWidth: .infinity)
                    .background(.black)
                    .overlay(alignment: .bottom) {
                        if let subtitle {
                            Text(subtitle).font(.mg(size: 15, weight: .semibold)).foregroundStyle(.white)
                                .multilineTextAlignment(.center)
                                .shadow(color: .black, radius: 3)
                                .padding(.horizontal, 24).padding(.bottom, 18)
                                .allowsHitTesting(false)
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
                    .mgTestAnchor("movie-edit-player")
                if let buildError {
                    Label(buildError, systemImage: "exclamationmark.triangle").font(.mgCallout).foregroundStyle(MG.warning)
                }
                if let timeline {
                    HStack(spacing: 10) {
                        Button { togglePlay() } label: {
                            Image(systemName: playing ? "pause.fill" : "play.fill")
                        }
                        .buttonStyle(.mg(.icon))
                        .mgHelp("Play or pause the film")
                        Text("\(clockText(position)) / \(clockText(timeline.duration))")
                            .font(.mgFootnote.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                        if !timeline.missing.isEmpty {
                            Text("\(timeline.missing.count) shots have no clip").font(.mgFootnote).foregroundStyle(MG.warning)
                        }
                        Spacer()
                        // How the export writes the subtitles.
                        Toggle("Burn In Subtitles", isOn: $doc.movie.mix.burnSubtitles)
                            .toggleStyle(.checkbox).font(.mgFootnote).fixedSize()
                            .mgHelp("Draws the lines into the exported picture, two rows at most; an .srt file is written beside the MP4 either way")
                            .mgTestAnchor("movie-burn-subtitles")
                        Toggle("Speaker Names in .srt", isOn: $doc.movie.mix.speakerNames)
                            .toggleStyle(.checkbox).font(.mgFootnote).fixedSize()
                            .mgHelp("Starts each line in the .srt with who says it (MARA: …)")
                            .mgTestAnchor("movie-speaker-names")
                            .padding(.trailing, 12)
                        Image(systemName: "minus.magnifyingglass").foregroundStyle(MG.tertiaryLabel)
                        MGSlider(value: $scale, in: 30...240).frame(width: 140)
                            .mgHelp("Timeline zoom")
                    }
                    MovieTimelineView(doc: doc, timeline: timeline, scale: scale, position: position,
                                      selected: $selected, seek: seek)
                    if let id = selected, let i = doc.movie.shotIndex[id],
                       let clip = timeline.clips.first(where: { $0.shot == id }) {
                        ClipInspector(doc: doc, shot: $doc.movie.shots[i], number: i + 1, clip: clip)
                    }
                } else {
                    MGSpinner()
                }
            }
        }
        .task(id: editKey) { await rebuild() }
        .onReceive(clock) { _ in
            let t = player.currentTime().seconds
            if t.isFinite, t != position { position = t }
            let now = player.timeControlStatus == .playing
            if now != playing { playing = now }
        }
        .onChange(of: model.chatViewer != nil) { _, viewing in if viewing { player.pause() } }
        .onDisappear { player.pause() }
    }

    private func rebuild() async {
        let t = await model.movieTimeline(doc)
        timeline = t
        guard !t.clips.isEmpty else {
            player.replaceCurrentItem(with: nil)
            buildError = "No shot has a chosen clip yet (Shots)."
            return
        }
        do {
            let c = try await MovieComposer.build(t, folder: model.movieStore.folder(doc.id), size: doc.movie.videoSize)
            let item = AVPlayerItem(asset: c.asset)
            item.videoComposition = c.video
            item.audioMix = c.audioMix
            let at = min(position, max(0, t.duration - 0.05))
            player.replaceCurrentItem(with: item)
            await player.seek(to: CMTime(seconds: at, preferredTimescale: 9600), toleranceBefore: .zero, toleranceAfter: .zero)
            buildError = nil
        } catch {
            buildError = "\(error)"
        }
    }

    private func seek(_ t: Double) {
        position = t
        player.seek(to: CMTime(seconds: t, preferredTimescale: 9600), toleranceBefore: .zero, toleranceAfter: .zero)
    }

    private func togglePlay() {
        if player.timeControlStatus == .playing { player.pause(); return }
        if let d = timeline?.duration, position >= d - 0.05 { seek(0) }
        player.play()
    }

    private func clockText(_ s: Double) -> String {
        let sec = s.truncatingRemainder(dividingBy: 60)
        if s >= 3600 { return String(format: "%d:%02d:%04.1f", Int(s) / 3600, Int(s) / 60 % 60, sec) }
        return String(format: "%d:%04.1f", Int(s) / 60, sec)
    }
}

/// The clips on one lane, as long as they play (crossfades overlap), with a ruler and the playhead.
private struct MovieTimelineView: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    let timeline: MovieTimeline
    let scale: Double
    let position: Double
    @Binding var selected: UUID?
    let seek: (Double) -> Void

    private let laneHeight: CGFloat = 64
    private let rulerHeight: CGFloat = 18
    private let soundHeight: CGFloat = 18
    /// The scrolled-to part of the timeline (points): only what's in it (and a screen either side) is drawn, so a
    /// long film's hundreds of clips don't all lay out on every playhead tick.
    @State private var viewport: ClosedRange<CGFloat> = 0...2000

    /// The sound lanes under the picture: dialogue, effects, music, each as many rows as it has overlaps.
    private var soundRows: [(kind: MovieTimeline.SoundKind, track: Int)] {
        MovieTimeline.SoundKind.allCases.flatMap { kind -> [(kind: MovieTimeline.SoundKind, track: Int)] in
            let n = (timeline.sounds.filter { $0.kind == kind }.map(\.track).max() ?? -1) + 1
            return (0..<n).map { (kind: kind, track: $0) }
        }
    }

    private var soundsHeight: CGFloat { soundRows.isEmpty ? 0 : CGFloat(soundRows.count) * (soundHeight + 4) + 6 }

    var body: some View {
        let width = max(CGFloat(timeline.duration * scale) + 80, 300)
        let rows = soundRows
        let index = doc.movie.shotIndex
        let span = viewport.upperBound - viewport.lowerBound
        let lo = Double(max(0, viewport.lowerBound - span)) / scale
        let hi = Double(viewport.upperBound + span) / scale
        ScrollView(.horizontal, showsIndicators: true) {
            ZStack(alignment: .topLeading) {
                Color.clear.frame(width: width, height: rulerHeight)
                    .contentShape(Rectangle())
                    .onTapGesture { loc in seek(max(0, min(timeline.duration, loc.x / scale))) }
                ruler(from: lo, to: min(hi, timeline.duration + 1))
                    .offset(x: CGFloat(lo * scale))
                    .allowsHitTesting(false)
                ForEach(Array(timeline.clips.enumerated()).filter { $0.element.end >= lo && $0.element.start <= hi },
                        id: \.element.id) { i, clip in
                    if let n = index[clip.shot] {
                        TimelineClipView(doc: doc, clip: clip, number: n + 1, scale: scale, height: laneHeight,
                                         selected: selected == clip.shot,
                                         open: { model.viewShotClips(doc, shot: clip.shot) }) {
                            selected = clip.shot
                            seek(clip.start)
                        }
                        .offset(x: CGFloat(clip.start * scale), y: rulerHeight + 6)
                        .zIndex(Double(i))
                    }
                }
                // Drop past the last clip: move to the end.
                Color.clear.frame(width: 80, height: laneHeight)
                    .offset(x: CGFloat(timeline.duration * scale), y: rulerHeight + 6)
                    .dropDestination(for: String.self) { items, _ in
                        guard let id = items.first.flatMap(UUID.init(uuidString:)) else { return false }
                        model.moveShot(doc, id, before: nil)
                        return true
                    }
                ForEach(timeline.sounds.filter { $0.end >= lo && $0.start <= hi }) { sound in
                    if let row = rows.firstIndex(where: { $0.kind == sound.kind && $0.track == sound.track }) {
                        TimelineSoundView(sound: sound, scale: scale, height: soundHeight) {
                            model.openMovieViewer(doc, files: [sound.file], titles: [sound.label], index: 0)
                        }
                            .offset(x: CGFloat(sound.start * scale),
                                    y: rulerHeight + 6 + laneHeight + 10 + CGFloat(row) * (soundHeight + 4))
                    }
                }
                Rectangle().fill(MG.accent).frame(width: 2, height: laneHeight + rulerHeight + 12 + soundsHeight)
                    .offset(x: CGFloat(position * scale) - 1)
                    .allowsHitTesting(false)
                    .zIndex(1000)
            }
            .frame(width: width, height: laneHeight + rulerHeight + 14 + soundsHeight, alignment: .topLeading)
            .padding(.bottom, 6)
            .background(GeometryReader { g in
                Color.clear.preference(key: TimelineOffsetKey.self, value: -g.frame(in: .named("movie-timeline")).minX)
            })
        }
        .coordinateSpace(name: "movie-timeline")
        .background(GeometryReader { g in
            Color.clear.onAppear { viewport = viewport.lowerBound...(viewport.lowerBound + g.size.width) }
                .onChange(of: g.size.width) { _, w in viewport = viewport.lowerBound...(viewport.lowerBound + w) }
        })
        .onPreferenceChange(TimelineOffsetKey.self) { x in
            let w = viewport.upperBound - viewport.lowerBound
            // Redrawn only when the view moved a good part of a screen.
            if abs(x - viewport.lowerBound) > w / 4 { viewport = max(0, x)...(max(0, x) + w) }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .mgTestAnchor("movie-timeline")
    }

    /// The ruler's ticks from `lo` to `hi` seconds (drawn at its own offset; a 20-minute film is too wide for one
    /// canvas).
    private func ruler(from lo: Double, to hi: Double) -> some View {
        let step: Double = scale >= 120 ? 1 : scale >= 50 ? 2 : 5
        return Canvas { ctx, _ in
            var t = (lo / step).rounded(.down) * step
            while t <= hi {
                let x = CGFloat((t - lo) * scale)
                ctx.fill(Path(CGRect(x: x, y: rulerHeight - 6, width: 1, height: 6)), with: .color(MG.tertiaryLabel))
                let label = t >= 3600 ? String(format: "%d:%02d:%02d", Int(t) / 3600, Int(t) / 60 % 60, Int(t) % 60)
                    : String(format: "%d:%02d", Int(t) / 60, Int(t) % 60)
                ctx.draw(Text(label).font(.mg(size: 9)).foregroundStyle(MG.tertiaryLabel),
                         at: CGPoint(x: x + 3, y: 6), anchor: .leading)
                t += step
            }
        }
        .frame(width: max(1, CGFloat((hi - lo) * scale)), height: laneHeight + rulerHeight + 12)
    }
}

private struct TimelineOffsetKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

/// A sound on the timeline: a clip's own sound (H3), a line (with its speaker), a shot's effects or a scene's music.
private struct TimelineSoundView: View {
    let sound: MovieTimeline.Sound
    let scale: Double
    let height: CGFloat
    /// Opens its take in the viewer (double-click).
    let open: () -> Void

    var body: some View {
        let tint: Color = switch sound.kind {
        case .clip: MG.success
        case .dialogue: MG.accent
        case .effects: MG.warning
        case .music: .purple
        }
        let symbol = switch sound.kind {
        case .clip: "film"
        case .dialogue: "person.wave.2"
        case .effects: "speaker.wave.2"
        case .music: "music.note"
        }
        HStack(spacing: 4) {
            Image(systemName: symbol).font(.mg(size: 9))
            Text(sound.label).font(.mg(size: 10)).lineLimit(1)
        }
        .padding(.horizontal, 5)
        .frame(width: max(CGFloat(sound.duration * scale), 6), height: height, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 4, style: .continuous).fill(tint.opacity(0.22)))
        .overlay(RoundedRectangle(cornerRadius: 4, style: .continuous).strokeBorder(tint.opacity(0.6)))
        .foregroundStyle(tint)
        .clipped()
        .contentShape(Rectangle())
        .onTapGesture(count: 2, perform: open)
        .mgContextMenu { MGMenuItem.button("Open in Viewer", action: open) }
        .mgHelp("\(sound.kind == .clip ? "The clip's own sound" : sound.kind == .dialogue ? "Line" : sound.kind == .effects ? "Sound effects" : "Music"): \(sound.label) · "
                + String(format: "%.1f s", sound.duration) + " · double-click to open it")
    }
}

/// One clip on the timeline: its poster, number and length, how it starts, and trim handles on both edges.
private struct TimelineClipView: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    let clip: MovieTimeline.Clip
    let number: Int
    let scale: Double
    let height: CGFloat
    let selected: Bool
    /// Opens the shot's clips in the viewer (double-click).
    let open: () -> Void
    let tap: () -> Void
    /// Seconds trimmed while a handle is dragged (committed on release).
    @State private var dragStart: Double = 0
    @State private var dragEnd: Double = 0

    var body: some View {
        let width = max(CGFloat((clip.duration - dragStart - dragEnd) * scale), 12)
        ZStack(alignment: .topLeading) {
            Group {
                if let poster = clip.poster {
                    MovieStill(url: model.movieStore.url(doc.id, relative: poster), maxPixels: 240)
                        .aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(MG.accent.opacity(0.25))
                }
            }
            .frame(width: width, height: height)
            .clipped()
            LinearGradient(colors: [.black.opacity(0.55), .clear], startPoint: .top, endPoint: .center)
            HStack(spacing: 4) {
                if clip.transition != .cut {
                    Image(systemName: transitionSymbol).font(.mg(size: 9, weight: .bold))
                }
                Text("\(number)").font(.mg(size: 11, weight: .bold))
                Text(MovieFormat.seconds((clip.duration - dragStart - dragEnd).rounded(toPlaces: 1)))
                    .font(.mg(size: 9.5))
            }
            .foregroundStyle(.white).padding(.horizontal, 5).padding(.top, 3)
            .frame(width: width, alignment: .leading).clipped()
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous)
            .strokeBorder(selected ? MG.accent : .white.opacity(0.25), lineWidth: selected ? 2 : 1))
        .overlay(alignment: .leading) { handle(leading: true) }
        .overlay(alignment: .trailing) { handle(leading: false) }
        .offset(x: CGFloat(dragStart * scale))
        .contentShape(Rectangle())
        .onTapGesture(count: 2, perform: open)
        .onTapGesture(perform: tap)
        .draggable(clip.shot.uuidString) {
            Text("Shot \(number)").font(.mgFootnote).padding(6).background(MG.card)
        }
        .dropDestination(for: String.self) { items, _ in
            guard let id = items.first.flatMap(UUID.init(uuidString:)), id != clip.shot else { return false }
            model.moveShot(doc, id, before: clip.shot)
            return true
        }
        .mgContextMenu {
            MGMenuItem.button("Open in Viewer", action: open)
            MGMenuItem.divider
            for t in ShotTransition.allCases {
                MGMenuItem.button(t.title, checked: clip.transition == t) { setTransition(t) }
            }
            MGMenuItem.divider
            MGMenuItem.button("Reset Trim") { setTrim(start: 0, end: 0) }
        }
        .mgHelp("Shot \(number): click to select, double-click to open, drag to move, drag an edge to trim")
    }

    private var transitionSymbol: String {
        switch clip.transition {
        case .cut: "scissors"
        case .crossfade: "square.on.square"
        case .fadeFromBlack: "circle.lefthalf.filled"
        case .fadeToBlack: "circle.righthalf.filled"
        }
    }

    private func handle(leading: Bool) -> some View {
        Rectangle().fill(.white.opacity(selected ? 0.9 : 0.4)).frame(width: 5, height: height * 0.5)
            .clipShape(Capsule())
            .padding(.horizontal, 2)
            .contentShape(Rectangle().inset(by: -4))
            .gesture(DragGesture(minimumDistance: 1).onChanged { g in
                let dt = g.translation.width / scale
                let room = clip.duration - MovieTimeline.minimumClip
                if leading {
                    dragStart = max(-clip.trimStart, min(room, dt))
                } else {
                    dragEnd = max(-(clip.sourceDuration - clip.trimStart - clip.duration), min(room, -dt))
                }
            }.onEnded { _ in
                guard let shot = doc.movie.shots.first(where: { $0.id == clip.shot }) else { return }
                let s = shot.trimStart + dragStart, e = shot.trimEnd + dragEnd
                dragStart = 0
                dragEnd = 0
                setTrim(start: max(0, s), end: max(0, e))
            })
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
    }

    private func setTransition(_ t: ShotTransition) {
        guard let i = doc.movie.shots.firstIndex(where: { $0.id == clip.shot }) else { return }
        doc.movie.shots[i].transition = t
        doc.movie.reopen(.edit)
    }

    private func setTrim(start: Double, end: Double) {
        guard let i = doc.movie.shots.firstIndex(where: { $0.id == clip.shot }) else { return }
        doc.movie.shots[i].trimStart = (start * 100).rounded() / 100
        doc.movie.shots[i].trimEnd = (end * 100).rounded() / 100
        doc.movie.reopen(.edit)
    }
}

/// The selected clip's settings under the timeline: how it starts and its trims, as fields.
private struct ClipInspector: View {
    let doc: MovieDocument
    @Binding var shot: Shot
    let number: Int
    let clip: MovieTimeline.Clip

    var body: some View {
        HStack(spacing: 16) {
            Text("Shot \(number)").font(.mgHeadline)
            MovieField(label: "Starts with") {
                MGPicker(selection: $shot.transition, options: ShotTransition.allCases.map { ($0, $0.title) }, maxWidth: 130)
            }
            MovieField(label: "Trim start", hint: "s") {
                MGSlider(value: $shot.trimStart, in: 0...max(0.01, clip.sourceDuration - MovieTimeline.minimumClip - shot.trimEnd),
                         step: 0.05)
                    .frame(width: 140)
            }
            MovieField(label: "Trim end", hint: "s") {
                MGSlider(value: $shot.trimEnd, in: 0...max(0.01, clip.sourceDuration - MovieTimeline.minimumClip - shot.trimStart),
                         step: 0.05)
                    .frame(width: 140)
            }
            Text("plays \(MovieFormat.seconds(clip.duration.rounded(toPlaces: 2))) of \(MovieFormat.seconds(clip.sourceDuration.rounded(toPlaces: 2)))")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            Spacer()
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: shot.transition) { _, _ in doc.movie.reopen(.edit) }
        .onChange(of: shot.trimStart) { _, _ in doc.movie.reopen(.edit) }
        .onChange(of: shot.trimEnd) { _, _ in doc.movie.reopen(.edit) }
        .mgTestAnchor("movie-clip-inspector")
    }
}

private extension Double {
    func rounded(toPlaces n: Int) -> Double {
        let f = Foundation.pow(10, Double(n))
        return (self * f).rounded() / f
    }
}
