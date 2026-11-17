import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO

/// MiniMax H3 shots: each clip comes with its own sound, so its lines are spoken inside the clip (in the characters'
/// voices) rather than by the Voices stage, and its effects come with it.
///
/// - A shot without spoken lines (or whose speakers have no voice yet) is `h3-video` from its start picture (the
///   keyframe, or the clip before's last frame when it continues it, ending on its own keyframe), its prompt written
///   in the structure MiniMax's prompting guide describes, its dialogue as `<d>[English] …</d>`.
/// - A talking shot is `h3-references`: the keyframe as `<Picture 1>`, then views cut from the speakers' character
///   sheets (the face close-up, and with one speaker the front full-body view too: a whole three-view sheet was
///   sometimes copied into the clip), and their voice samples as `<Audio n>`, the voice each speaks in.
///
/// H3 makes 5–15 s; a shorter shot plays the start of its clip (the edit trims it to the shot). Line times are
/// estimated from the words (there are no line takes to measure), for the prompt's order and to make room for the
/// lines. H3 picks its own times, so the subtitles go where the clip's voice is (`MovieSpeech`, `heardTimes`).
public enum MovieH3 {
    public static let fps = 24.0
    public static let minSeconds = 5.0
    public static let maxSeconds = 15.0
    public static let maxPictures = 4
    public static let maxVoices = 3
    /// Spoken pace for the estimate, and the quiet before, between and after lines.
    public static let wordsPerSecond = 2.6
    public static let leadIn = 0.4, gap = 0.3, tail = 0.5

    public enum TemplateName {
        public static let video = "shot-h3"
        public static let references = "shot-h3-references"
    }

    // MARK: lines

    public struct Placed: Equatable, Sendable {
        public var line: UUID
        /// Seconds from the shot's start.
        public var start: Double
        public var seconds: Double
    }

    /// How long a line takes to say, from its words (at least 0.8 s).
    public static func estimatedSeconds(_ text: String) -> Double {
        let words = text.split { $0.isWhitespace }.count
        return max(0.8, Double(words) / wordsPerSecond)
    }

    /// The shot's lines with text, in order, at estimated times.
    public static func lineTimes(_ shot: Shot) -> [Placed] {
        var t = leadIn
        var out: [Placed] = []
        for line in shot.dialogue where line.mode == .onScreen {
            let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let s = estimatedSeconds(text)
            out.append(Placed(line: line.id, start: t, seconds: s))
            t += s + gap
        }
        return out
    }

    /// The shot's lines where its clip speaks them (`voice`, from `MovieSpeech`), in order: with one line, from the
    /// first voice to the last; with several, the voice split at its widest pauses, one stretch per line. Short
    /// blips beyond one per line are left out first. A line
    /// shows for at least 0.8 s (but not past the next). The estimate when nothing is heard, or fewer stretches than
    /// lines.
    public static func heardTimes(_ shot: Shot, voice: [VoiceSpan]) -> [Placed] {
        let estimate = lineTimes(shot)
        var spans = voice.sorted { $0.start < $1.start }
        // Blips (under 0.2 s) beyond one per line go first: a creak or a breath shouldn't stretch a line.
        while spans.count > estimate.count, let i = spans.indices.min(by: { spans[$0].length < spans[$1].length }),
              spans[i].length < 0.2 {
            spans.remove(at: i)
        }
        guard !estimate.isEmpty, spans.count >= estimate.count else { return estimate }
        // The n − 1 widest pauses end the groups.
        let cuts = Set(spans.indices.dropLast()
            .sorted { spans[$0 + 1].start - spans[$0].end > spans[$1 + 1].start - spans[$1].end }
            .prefix(estimate.count - 1))
        var groups: [VoiceSpan] = []
        var open: VoiceSpan?
        for (i, s) in spans.enumerated() {
            open = open.map { VoiceSpan(start: $0.start, end: max($0.end, s.end)) } ?? s
            if cuts.contains(i) || i == spans.count - 1, let g = open {
                groups.append(g)
                open = nil
            }
        }
        return estimate.indices.map { i in
            let g = groups[i]
            var seconds = max(0.8, g.end - g.start)
            if i + 1 < groups.count { seconds = min(seconds, max(g.end - g.start, groups[i + 1].start - g.start)) }
            return Placed(line: estimate[i].line, start: g.start, seconds: seconds)
        }
    }

    /// How long the shot must be for its lines (nil: none).
    public static func neededDuration(_ shot: Shot) -> Double? {
        lineTimes(shot).last.map { $0.start + $0.seconds + tail }
    }

    /// The clip to ask for: the shot's length, at least H3's 5 s, at most 15 s.
    public static func clipSeconds(_ shot: Shot) -> Double {
        min(maxSeconds, max(minSeconds, (shot.duration * 2).rounded(.up) / 2))
    }

    /// Makes talking shots long enough for their lines (at most 15 s); returns those that still don't fit.
    @discardableResult
    public static func fitDurations(_ movie: inout MovieProject) -> [UUID] {
        var unfit: [UUID] = []
        for i in movie.shots.indices {
            guard let need = neededDuration(movie.shots[i]) else { continue }
            if need > maxSeconds + 1e-9 { unfit.append(movie.shots[i].id) }
            if movie.shots[i].duration < need { movie.shots[i].duration = min(maxSeconds, (need * 10).rounded(.up) / 10) }
        }
        return unfit
    }

    /// The characters who speak on screen in the shot (in the clip), in the order they first speak. Off-screen
    /// lines and narration are spoken by the voice engine and laid over the clip.
    public static func speakers(_ shot: Shot) -> [String] {
        var out: [String] = []
        for line in shot.dialogue where line.mode == .onScreen
            && !line.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if !out.contains(line.character) { out.append(line.character) }
        }
        return out
    }

    /// A character's chosen voice sample (relative file).
    public static func voiceFile(_ c: MovieCharacter) -> String? {
        c.voiceSamples.first { $0.id == c.selectedVoiceSample }?.file
    }

    /// Whether the shot is made from references: it has lines, and a speaker has a voice to speak in.
    public static func usesReferences(_ shot: Shot, in movie: MovieProject) -> Bool {
        speakers(shot).contains { movie.character($0).flatMap(voiceFile) != nil }
    }

    // MARK: prompts

    private static func sentence(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return "" }
        return t.hasSuffix(".") || t.hasSuffix("!") || t.hasSuffix("?") ? t : t + "."
    }

    private static func soundscape(_ shot: Shot) -> String {
        let sfx = shot.sfx.trimmingCharacters(in: .whitespacesAndNewlines)
        return sfx.isEmpty ? "The natural ambience of the scene, with the sounds of its action." : sentence(sfx.prefix(1).uppercased() + sfx.dropFirst())
    }

    /// The shot's lines as the guide writes them: who speaks (name, voice, speaker ID, delivery) outside `<d>`, the
    /// words inside, verbatim. `who` names the speaker (a label such as "<Subject 1>", or nil: their name).
    private static func dialogue(_ shot: Shot, in movie: MovieProject, who: (String) -> String?) -> String {
        let ids = speakers(shot)
        return shot.dialogue.compactMap { line -> String? in
            let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.mode == .onScreen, !text.isEmpty, let k = ids.firstIndex(of: line.character) else { return nil }
            let c = movie.character(line.character)
            var speaker = who(line.character) ?? c?.name ?? line.character
            let voice = c?.voice.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !voice.isEmpty, who(line.character) == nil { speaker += ", with a \(voice) voice," }
            let delivery = line.delivery.trimmingCharacters(in: .whitespacesAndNewlines)
            return "\(speaker) (S\(k + 1)) says\(delivery.isEmpty ? "" : " (\(delivery))"): <d>[English] \(text)</d>"
        }.joined(separator: " ")
    }

    /// An `h3-video` prompt: `start` / `end` say whether the clip has a first and a last picture (`<Picture 1>` is
    /// the first one there is).
    public static func prompt(for shot: Shot, in movie: MovieProject, start: Bool, end: Bool) -> String {
        let framing = shot.framing.trimmingCharacters(in: .whitespacesAndNewlines)
        var body = "[Shot 1] \(MovieStyle.named(movie.style).name), a \(framing.isEmpty ? "medium" : framing) shot."
        if start { body += " It begins exactly on <Picture 1>, keeping its subjects, clothing, setting and light." }
        body += " " + sentence(MovieShots.motionPrompt(for: shot, in: movie))
        let lines = dialogue(shot, in: movie) { _ in nil }
        if !lines.isEmpty { body += " " + lines }
        if end { body += " The shot ends on the composition of <Picture \(start ? 2 : 1)>." }
        var head = ""
        if start { head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n" }
        return head + "integrated_multimodal_description: \(body)\noverall_soundscape: \(soundscape(shot))\nnon_diegetic_music: none"
    }

    /// A reference picture of a character: one view cut from their sheet, or the whole sheet when it couldn't be cut.
    public struct SubjectPicture: Equatable, Sendable {
        /// `place`: the scene's place, empty (its plate); `id` is then the place's.
        public enum View: String, Sendable { case face, body, place }
        public var id: String
        public var view: View?
        public init(id: String, view: View? = nil) { self.id = id; self.view = view }
    }

    /// An `h3-references` prompt: `<Picture 1>` the keyframe, then `pictures` (`<Picture 2>`…, each of a character;
    /// each character is a `<Subject n>`, in order), and `<Audio n>` the voice of the n-th character in `voices`.
    public static func referencePrompt(for shot: Shot, in movie: MovieProject, pictures: [SubjectPicture],
                                       voices: [String]) -> String {
        let ids = speakers(shot)
        func name(_ id: String) -> String { movie.character(id)?.name ?? id }
        var subjects: [String] = []
        for p in pictures where p.view != .place && !subjects.contains(p.id) { subjects.append(p.id) }
        func subject(_ id: String) -> String? { subjects.firstIndex(of: id).map { "<Subject \($0 + 1)>" } }
        func place(_ id: String) -> String { movie.locations.first { $0.id == id }?.name ?? id }
        func sid(_ id: String) -> String { ids.firstIndex(of: id).map { " (S\($0 + 1))" } ?? "" }
        func label(_ i: Int) -> String { "<Picture \(i + 2)>" }
        var defs = ["<Picture 1> is the planned opening composition of the shot: its setting, framing and light."]
        for (k, id) in subjects.enumerated() {
            let shown = pictures.enumerated().filter { $0.element.id == id }.map { i, p -> String in
                switch p.view {
                case .face: "their face and hair in \(label(i))"
                case .body: "their whole figure and clothing in \(label(i))"
                case .place, nil: "their face, hair and clothing in \(label(i))"
                }
            }
            defs.append("<Subject \(k + 1)> is \(name(id)): \(shown.joined(separator: ", and ")).")
        }
        for (i, p) in pictures.enumerated() where p.view == .place {
            defs.append("\(label(i)) is the place, \(place(p.id)), empty: its architecture, materials and light.")
        }
        for (i, id) in voices.enumerated() {
            defs.append("<Audio \(i + 1)> is the voice-timbre reference for \(subject(id) ?? name(id))\(sid(id)).")
        }
        let framing = shot.framing.trimmingCharacters(in: .whitespacesAndNewlines)
        let names = ids.map(name)
        let who = names.count <= 1 ? (names.first ?? "a character") : names.dropLast().joined(separator: ", ") + " and " + names.last!
        var retention = ["<Picture 1>: reference - the setting, composition and lighting are kept."]
        for (i, p) in pictures.enumerated() {
            let kept = switch p.view {
            case .face: "\(name(p.id))'s face and hair are kept"
            case .body: "\(name(p.id))'s clothing and build are kept"
            case .place: "the place's architecture and light are kept"
            case nil: "\(name(p.id))'s appearance and clothing are kept"
            }
            retention.append("\(label(i)): reference - \(kept), never shown as a picture of its own.")
        }
        for (i, id) in voices.enumerated() {
            retention.append("<Audio \(i + 1)>: reference - \(name(id)) speaks in this voice timbre without copying the recording.")
        }
        var body = "[Shot 1] \(MovieStyle.named(movie.style).name), a \(framing.isEmpty ? "medium" : framing) shot in the "
            + "setting of <Picture 1>, one continuous take. " + sentence(MovieShots.motionPrompt(for: shot, in: movie))
        let lines = dialogue(shot, in: movie, who: subject)
        if !lines.isEmpty { body += " " + lines }
        return """
            subject_definitions: \(defs.joined(separator: " "))
            summary: One \(framing.isEmpty ? "medium" : framing) shot in the setting of <Picture 1> in which \(who) \(names.count > 1 ? "speak" : "speaks").
            retention_analysis: \(retention.joined(separator: " "))
            detailed_description: \(body)
            overall_soundscape: \(soundscape(shot))
            non_diegetic_music: none
            """
    }

    /// A character sheet's views as separate pictures: the front full-body view (its left third) and the face
    /// close-up (its right third), written once beside it as `<name>-body.png` and `<name>-face.png`. Empty when the
    /// sheet can't be read or written.
    public static func sheetPanels(_ sheet: URL) -> [SubjectPicture.View: URL] {
        let base = sheet.deletingPathExtension().lastPathComponent
        let dir = sheet.deletingLastPathComponent()
        let urls: [SubjectPicture.View: URL] = [.body: dir.appendingPathComponent("\(base)-body.png"),
                                                 .face: dir.appendingPathComponent("\(base)-face.png")]
        if urls.values.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) { return urls }
        guard let source = CGImageSourceCreateWithURL(sheet as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return [:] }
        let w = Double(image.width), h = Double(image.height), inset = w * 0.015  // clear of the seams between views
        let third = w / 3
        let rects: [SubjectPicture.View: CGRect] = [
            .body: CGRect(x: inset, y: 0, width: third - 2 * inset, height: h),
            .face: CGRect(x: 2 * third + inset, y: 0, width: third - 2 * inset, height: h),
        ]
        for (view, rect) in rects {
            guard let crop = image.cropping(to: rect.integral), (try? MovieComposer.writePNG(crop, to: urls[view]!)) != nil
            else { return [:] }
        }
        return urls
    }

    /// The pictures of a talking shot's speakers, at most `maxPictures - 1`: with one speaker their face and their
    /// whole figure, with more each one's face; a sheet that can't be cut goes whole. `paths` holds their files.
    public static func subjectPictures(for shot: Shot, in movie: MovieProject, folder: URL)
        -> (pictures: [SubjectPicture], paths: [String])
    {
        let sheets = references(for: shot, in: movie).sheets
        var pictures: [SubjectPicture] = [], paths: [String] = []
        for id in sheets {
            guard let file = movie.character(id)?.selectedSheetFile else { continue }
            let sheet = folder.appendingPathComponent(file)
            let panels = sheetPanels(sheet)
            if let face = panels[.face] {
                pictures.append(.init(id: id, view: .face)); paths.append(face.path)
                if sheets.count == 1, let body = panels[.body] {
                    pictures.append(.init(id: id, view: .body)); paths.append(body.path)
                }
            } else {
                pictures.append(.init(id: id)); paths.append(sheet.path)
            }
        }
        var n = min(pictures.count, maxPictures - 1)
        // Room left: the place's plate, so the place stays the one the other shots show.
        if n < maxPictures - 1, let place = MovieArt.location(of: shot, in: movie), let file = place.selectedPlateFile {
            pictures.insert(.init(id: place.id, view: .place), at: n)
            paths.insert(folder.appendingPathComponent(file).path, at: n)
            n += 1
        }
        return (Array(pictures.prefix(n)), Array(paths.prefix(n)))
    }

    // MARK: templates

    public static let videoSlots = ["prompt", "seed", "seconds", "aspect", "server"]

    /// Load Image (start) and Load Image (end) into MiniMax H3 Video with Sound into a Preview Video.
    public static func videoTemplate(name: String = TemplateName.video) -> MovieTemplate {
        let start = GraphNode(id: "start", type: "LoadImage", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 60))
        let end = GraphNode(id: "end", type: "LoadImage", inputs: ["path": .value(.string(""))],
                            position: GraphPoint(x: 40, y: 220))
        let h3 = GraphNode(id: "video", type: "RemoteH3Video", inputs: [
            "image": .link(node: "start", output: "image"), "last_image": .link(node: "end", output: "image"),
            "prompt": .value(.string("")), "seed": .value(.int(0)), "seconds": .value(.double(5)),
            "size": .value(.string("768p")), "aspect": .value(.string("16:9")), "fast": .value(.bool(true)),
            "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "video", output: "video")],
                            position: GraphPoint(x: 720, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: videoSlots.map { ($0, MovieTemplate.Slot(node: "video", input: $0)) })
        slots["start"] = .init(node: "start", input: "path")
        slots["end"] = .init(node: "end", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [start, end, h3, out]), slots: slots)
    }

    /// A video clip's values: `startImage` absolute (the keyframe, or the clip before's last frame); the end picture
    /// is left out unless the shot has one.
    public static func videoRender(_ shot: Shot, movie: MovieProject, folder: URL, startImage: URL,
                                   ends: MovieShots.Ends, seed: Int, server: String) -> MovieArt.Render {
        var values: [String: JSONValue] = [
            "prompt": .string(prompt(for: shot, in: movie, start: true, end: ends.end != nil)), "seed": .int(seed),
            "seconds": .double(clipSeconds(shot)), "aspect": .string(movie.aspect.h3Aspect),
            "server": .string(server), "start": .string(startImage.path),
        ]
        if let end = ends.end { values["end"] = .string(folder.appendingPathComponent(end).path) }
        return MovieArt.Render(values: values, unusedPictures: ends.end == nil ? ["end"] : [])
    }

    /// Load Image ×4 (the keyframe, then character sheets) and Load Audio ×3 (voices) into MiniMax H3 from
    /// References into a Preview Video.
    public static func referencesTemplate(name: String = TemplateName.references) -> MovieTemplate {
        var nodes: [GraphNode] = []
        var inputs: [String: NodeInput] = [
            "prompt": .value(.string("")), "seed": .value(.int(0)), "seconds": .value(.double(5)),
            "size": .value(.string("768p")), "aspect": .value(.string("16:9")), "fast": .value(.bool(true)),
            "server": .value(.string("")),
        ]
        var slots = Dictionary(uniqueKeysWithValues: videoSlots.map { ($0, MovieTemplate.Slot(node: "video", input: $0)) })
        for i in 1...maxPictures {
            nodes.append(GraphNode(id: "picture\(i)", type: "LoadImage", inputs: ["path": .value(.string(""))],
                                   position: GraphPoint(x: 40, y: Double(60 + (i - 1) * 160))))
            inputs["image\(i)"] = .link(node: "picture\(i)", output: "image")
            slots["picture\(i)"] = .init(node: "picture\(i)", input: "path")
        }
        for i in 1...maxVoices {
            nodes.append(GraphNode(id: "voice\(i)", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                   position: GraphPoint(x: 40, y: Double(700 + (i - 1) * 120))))
            inputs["audio\(i)"] = .link(node: "voice\(i)", output: "audio")
            slots["voice\(i)"] = .init(node: "voice\(i)", input: "path")
        }
        nodes.append(GraphNode(id: "video", type: "RemoteH3References", inputs: inputs, position: GraphPoint(x: 360, y: 60)))
        nodes.append(GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "video", output: "video")],
                               position: GraphPoint(x: 740, y: 80)))
        return MovieTemplate(name: name, graph: Graph(nodes: nodes), slots: slots)
    }

    /// The characters whose sheets and voices a talking shot uses: speakers with a chosen sheet (up to three, after
    /// the keyframe) and speakers with a chosen voice (up to three).
    public static func references(for shot: Shot, in movie: MovieProject) -> (sheets: [String], voices: [String]) {
        let ids = speakers(shot)
        let sheets = ids.filter { movie.character($0)?.selectedSheetFile != nil }.prefix(maxPictures - 1)
        let voices = ids.filter { movie.character($0).flatMap(voiceFile) != nil }.prefix(maxVoices)
        return (Array(sheets), Array(voices))
    }

    /// A talking clip's values: `keyframe` absolute; the speakers' sheets and voice samples from the movie's folder.
    public static func referencesRender(_ shot: Shot, movie: MovieProject, folder: URL, keyframe: URL, seed: Int,
                                        server: String) -> MovieArt.Render {
        let refs = references(for: shot, in: movie)
        let subjects = subjectPictures(for: shot, in: movie, folder: folder)
        var values: [String: JSONValue] = [
            "prompt": .string(referencePrompt(for: shot, in: movie, pictures: subjects.pictures, voices: refs.voices)),
            "seed": .int(seed), "seconds": .double(clipSeconds(shot)), "aspect": .string(movie.aspect.h3Aspect),
            "server": .string(server), "picture1": .string(keyframe.path),
        ]
        var unused = Set<String>()
        for i in 2...maxPictures {
            if i - 2 < subjects.paths.count {
                values["picture\(i)"] = .string(subjects.paths[i - 2])
            } else {
                unused.insert("picture\(i)")
            }
        }
        for i in 1...maxVoices {
            let id = i - 1 < refs.voices.count ? refs.voices[i - 1] : nil
            if let file = id.flatMap({ movie.character($0).flatMap(voiceFile) }) {
                values["voice\(i)"] = .string(folder.appendingPathComponent(file).path)
            } else {
                unused.insert("voice\(i)")
            }
        }
        return MovieArt.Render(values: values, unusedPictures: unused)
    }

    /// GPU work of a clip for the learned timings: megapixels × frames, as `MovieShots.work`.
    public static func work(_ shot: Shot, in movie: MovieProject) -> Double {
        let size = movie.aspect.h3Size
        return MovieShots.work(width: size.width, height: size.height, frames: Int((clipSeconds(shot) * fps).rounded(.up)))
    }
}
