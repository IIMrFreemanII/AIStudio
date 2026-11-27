import CharacterCore
import Foundation
import LLMClient
import MovieCore
import NodesRemote

/// The media in a character chat: the photos, voice messages, videos and video messages a reply's tags ask for, made
/// on the chat's Cloud GPU server. A photo is SDXL or Chroma keeping the face of the chosen portrait; a voice message
/// clones the character's voice sample; a video animates a fresh photo (Wan 2.2); a video message lip-syncs a fresh
/// photo to a voice message (LongCat Avatar). Photos and voice run at once when they cost less than the limit;
/// videos and video messages always wait for the user's OK.
extension AppModel {
    /// What a photo or voice message may cost without asking (UserDefaults; 5 cents when unset). The key keeps its
    /// first name. Automated runs keep it in memory: they must not change the user's setting.
    var characterAutoMediaUSD: Double {
        get {
            if Self.isAutomatedRun { return Self.automatedAutoMediaUSD }
            return UserDefaults.standard.object(forKey: "characterAutoPhotoUSD") as? Double ?? 0.05
        }
        set {
            if Self.isAutomatedRun { Self.automatedAutoMediaUSD = newValue }
            else { UserDefaults.standard.set(newValue, forKey: "characterAutoPhotoUSD") }
        }
    }

    private static var automatedAutoMediaUSD = 0.05

    /// The server a chat's media is made on: the chat's own when it is a GPU server, else the first one.
    func photoServer(for doc: ChatDocument) -> String? {
        if let id = doc.transcript.serverID, connections.connection(id)?.kind == .gpu { return id }
        return connections.gpuServers.first?.id
    }

    /// The kinds the character may send: theirs whose tasks the server runs (all of theirs while its tasks aren't
    /// known; none without a server).
    func mediaKinds(_ c: Companion, server: String?) -> [CharacterCore.MediaKind] {
        guard let server else { return [] }
        guard case .gpuOnline(let info)? = connections.status[server] else { return c.sends }
        return c.sends.filter { k in
            CharacterMedia.jobs(k, character: c, text: "", hasVoiceSample: false).allSatisfy { info.task($0.task) != nil }
        }
    }

    /// The chosen voice sample's file, when it exists.
    func voiceSampleURL(_ c: Companion) -> URL? {
        guard let name = c.selectedVoiceSample else { return nil }
        let url = characterStore.voiceFolder(c.id).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// What one more photo of `family` would cost on `server` (its job, plus loading the model when it's cold); nil
    /// when the price isn't known.
    func photoEstimate(server: String, family: ArtModel.Family = .sdxl) -> Double? {
        var c = Companion(name: "", age: 18, gender: "")
        c.artModel.family = family
        return mediaEstimate(.photo, character: c, text: "", server: server)
    }

    /// What a request would cost on `server`: its jobs, plus loading each model that's cold (a voice sample is made
    /// first when the character has none). nil when a price isn't known.
    func mediaEstimate(_ kind: CharacterCore.MediaKind, character c: Companion, text: String, server: String) -> Double? {
        guard case .gpuOnline(let info)? = connections.status[server] else { return nil }
        let timings = GPUTimings.load(gpuTimingsURL)
        let idle = Double(info.idleSeconds ?? 120)
        return CharacterMedia.estimate(kind, character: c, text: text, hasVoiceSample: voiceSampleURL(c) != nil,
                                       timings: timings, price: { info.task($0)?.pricePerSecond },
                                       warm: { timings.isWarm(task: $0, idleSeconds: idle) })
    }

    // MARK: a reply's media

    /// Makes a `ChatMedia` for each tag of the turn's text and starts the ones that run at once (the rest wait for
    /// the user). Media the user didn't ask for counts against the character's daily initiative, and is skipped once
    /// that is used up.
    /// The user's message a reply answers: its parent, or in a group the user's message before the other characters'
    /// replies of the same round. "" when it follows the character's own reply (a check-in): media in it is unasked.
    func askingText(_ doc: ChatDocument, before turn: ChatTurn) -> String {
        var id = turn.parent
        while let p = id.flatMap({ doc.transcript.turn($0) }) {
            if p.role == .user { return p.text }
            guard p.role == .assistant, let who = p.speaker, who != turn.speaker else { return "" }
            id = p.parent
        }
        return ""
    }

    /// The video the user's message before `turn` carries (its last one), for a motion clip.
    func askingVideo(_ doc: ChatDocument, before turn: ChatTurn) -> String? {
        var id = turn.parent
        while let p = id.flatMap({ doc.transcript.turn($0) }) {
            if p.role == .user { return p.images.last { Self.isClip($0) } }
            guard p.role == .assistant, let who = p.speaker, who != turn.speaker else { return nil }
            id = p.parent
        }
        return nil
    }

    static func isClip(_ name: String) -> Bool { ["mp4", "mov", "m4v"].contains((name as NSString).pathExtension.lowercased()) }

    func startCharacterMedia(_ doc: ChatDocument, turnID: UUID, character c: Companion) {
        guard let turn = doc.transcript.turn(turnID), turn.error == nil, turn.speaker == c.id, turn.media == nil else { return }
        let requests = MediaTags.parse(turn.text).requests
        guard !requests.isEmpty else { return }
        let server = photoServer(for: doc)
        let kinds = mediaKinds(c, server: server)
        let asking = askingText(doc, before: turn)
        let clip = askingVideo(doc, before: turn)
        var left = CharacterCost.initiativeLeft(c, memory: characterStore.memory(c.id))
        var own = 0
        let media = requests.map { request -> ChatMedia in
            // A video after the user sent a clip copies its moves (when the server can); a motion tag without a clip
            // is an ordinary video.
            var r = request
            var source: String?
            if r.kind == .video || r.kind == .motion {
                if let clip, kinds.contains(.motion) { r.kind = .motion; source = clip } else { r.kind = .video }
            }
            let kind = r.kind.rawValue
            guard let server else {
                return ChatMedia(kind: kind, prompt: r.text, status: .failed, error: "No GPU server for media")
            }
            guard kinds.contains(r.kind) else {
                return ChatMedia(kind: kind, prompt: r.text, status: .skipped,
                                 error: c.sends.contains(r.kind) ? "The GPU server can't make these" : "Turned off for \(c.name)")
            }
            let asked = MediaAsk.asked(r.kind, in: asking)
            let estimate = mediaEstimate(r.kind, character: c, text: r.text, server: server)
            switch CharacterCost.decide(r.kind, estimate: estimate, autoLimit: characterAutoMediaUSD, asked: asked,
                                        initiativeLeft: left) {
            case .skip(let why):
                return ChatMedia(kind: kind, prompt: r.text, status: .skipped, error: why)
            case let decision:
                if !asked { left -= 1; own += 1 }
                return ChatMedia(kind: kind, prompt: r.text, status: decision == .run ? .running : .waiting,
                                 estimateUSD: estimate, source: source)
            }
        }
        if own > 0 {
            var m = characterStore.memory(c.id)
            for _ in 0..<own { m.noteOwnMedia() }
            try? characterStore.saveMemory(m, for: c.id)
        }
        doc.transcript.update(turnID) { $0.media = media }
        saveChat(doc)
        let running = media.filter { $0.status == .running }.map(\.id)
        guard !running.isEmpty else { return }
        Task { @MainActor [weak self, weak doc] in
            for id in running {
                guard let self, let doc else { return }
                await self.makeMedia(doc, turnID: turnID, mediaID: id, character: c)
            }
        }
    }

    /// The character who can copy a clip's moves in this chat: a one-character chat whose character sends motion
    /// videos and whose GPU server makes them (nil otherwise).
    func motionCharacter(_ doc: ChatDocument) -> Companion? {
        let people = characters(of: doc)
        guard people.count == 1, let c = people.first else { return nil }
        return mediaKinds(c, server: photoServer(for: doc)).contains(.motion) ? c : nil
    }

    /// "Have them do this" on a clip the user sent: the character's turn with a motion video of that clip, waiting for
    /// the user's OK (its cost shows on it, as every video's does).
    func requestMotion(_ doc: ChatDocument, clip: String, character c: Companion) {
        guard !doc.isStreaming, let server = photoServer(for: doc) else { return }
        let estimate = mediaEstimate(.motion, character: c, text: "", server: server)
        var t = ChatTurn(role: .assistant, text: "<video>the moves from your clip</video>", model: doc.transcript.modelID)
        t.speaker = c.id
        t.media = [ChatMedia(kind: CharacterCore.MediaKind.motion.rawValue, prompt: "", status: .waiting,
                             estimateUSD: estimate, source: clip)]
        doc.transcript.append(t)
        saveChat(doc)
    }

    /// The user said yes to waiting media.
    func approveMedia(_ doc: ChatDocument, turnID: UUID, mediaID: UUID) {
        guard let c = doc.transcript.turn(turnID)?.speaker.flatMap({ character($0) }) ?? characters(of: doc).first,
              doc.transcript.turn(turnID)?.media?.first(where: { $0.id == mediaID })?.status == .waiting else { return }
        setMedia(doc, turnID, mediaID) { $0.status = .running; $0.error = nil }
        Task { @MainActor [weak self, weak doc] in
            guard let self, let doc else { return }
            await self.makeMedia(doc, turnID: turnID, mediaID: mediaID, character: c)
        }
    }

    /// The user said no to waiting media.
    func skipMedia(_ doc: ChatDocument, turnID: UUID, mediaID: UUID) {
        setMedia(doc, turnID, mediaID) { $0.status = .skipped }
    }

    private func setMedia(_ doc: ChatDocument, _ turnID: UUID, _ mediaID: UUID, _ change: (inout ChatMedia) -> Void) {
        guard doc.transcript.turn(turnID)?.media?.contains(where: { $0.id == mediaID }) == true else { return }
        doc.transcript.update(turnID) { t in
            guard let i = t.media?.firstIndex(where: { $0.id == mediaID }) else { return }
            change(&t.media![i])
        }
        saveChat(doc)
    }

    /// Makes one request and puts it in the chat and the album. Every GPU job is charged to the character as it
    /// finishes, so a later step that goes wrong doesn't lose what the earlier ones cost.
    private func makeMedia(_ doc: ChatDocument, turnID: UUID, mediaID: UUID, character c: Companion) async {
        guard let media = doc.transcript.turn(turnID)?.media?.first(where: { $0.id == mediaID }),
              let kind = CharacterCore.MediaKind(rawValue: media.kind) else { return }
        var cost: Double?
        // Each step's progress shows on the media's line while it's made (cold starts can take minutes).
        func phase(_ step: String) -> @MainActor (String) -> Void {
            { [weak doc] text in doc?.mediaPhase[mediaID] = step.isEmpty ? text : "\(step): \(text)" }
        }
        defer { doc.mediaPhase[mediaID] = nil }
        func charge(_ c2: Double?) {
            guard let c2 else { return }
            cost = (cost ?? 0) + c2
            var m = characterStore.memory(c.id)
            m.spentUSD += c2
            try? characterStore.saveMemory(m, for: c.id)
        }
        do {
            guard let server = photoServer(for: doc) else { throw CharacterMediaError.noServer }
            if kind != .voice { try CharacterArt.check(media.prompt) }
            let scene = scene(of: doc)
            let base = "\(kind.rawValue)-\(mediaID.uuidString.prefix(8).lowercased())"
            let folder = chatStore.filesFolder(doc.id)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let file: URL
            switch kind {
            case .photo:
                file = folder.appendingPathComponent("\(base).png")
                charge(try await drawPhoto(c, prompt: CharacterArt.photoPrompt(c, request: media.prompt, scene: scene, context: artContext(c)),
                                           server: server, to: file, title: "\(c.name)'s photo", progress: phase("")))
            case .voice:
                let (audio, spent) = try await speak(c, text: media.prompt, server: server, to: folder, base: base,
                                                     phase: phase)
                charge(spent)
                file = audio
            case .video:
                let still = folder.appendingPathComponent("\(base)-start.png")
                charge(try await drawPhoto(c, prompt: CharacterMedia.videoStillPrompt(c, request: media.prompt, scene: scene, context: artContext(c)),
                                           server: server, to: still, title: "\(c.name)'s video (picture)",
                                           progress: phase("picture")))
                let render = CharacterMedia.videoRender(c, prompt: CharacterMedia.motionPrompt(c, request: media.prompt),
                                                        start: still, seed: Self.mediaSeed(), server: server)
                let (outputs, jobs) = try await runHeadlessGraph(template: CharacterMedia.videoTemplate(), render: render,
                                                                 title: "\(c.name)'s video", progress: phase("clip"))
                let s = CharacterMedia.videoSize
                charge(recordGPUJobs(jobs, work: MovieShots.work(width: s.width, height: s.height,
                                                                 frames: MovieShots.frames(for: CharacterMedia.videoSeconds))))
                file = try copyOutput(outputs, .video, to: folder, base: base)
            case .motion:
                guard let source = media.source, FileManager.default.fileExists(atPath: folder.appendingPathComponent(source).path)
                else { throw CharacterMediaError.noClip }
                let clip = folder.appendingPathComponent(source)
                let seconds = min(await MovieComposer.duration(of: clip) ?? CharacterMedia.maxMotionSeconds,
                                  CharacterMedia.maxMotionSeconds)
                let still = folder.appendingPathComponent("\(base)-start.png")
                charge(try await drawPhoto(c, prompt: CharacterMedia.motionStillPrompt(c, request: media.prompt, scene: scene, context: artContext(c)),
                                           server: server, to: still, title: "\(c.name)'s video (picture)",
                                           progress: phase("picture")))
                let render = CharacterMedia.motionRender(c, still: still, clip: clip, seconds: seconds,
                                                         scene: CharacterMedia.motionScene(media.prompt),
                                                         seed: Self.mediaSeed(), server: server)
                let (outputs, jobs) = try await runHeadlessGraph(template: CharacterMedia.motionTemplate(), render: render,
                                                                 title: "\(c.name)'s video", progress: phase("moves"))
                charge(recordGPUJobs(jobs, work: CharacterMedia.motionWork(seconds: seconds)))
                file = try copyOutput(outputs, .video, to: folder, base: base)
            case .talk:
                let still = folder.appendingPathComponent("\(base)-start.png")
                charge(try await drawPhoto(c, prompt: CharacterMedia.talkStillPrompt(c, scene: scene, context: artContext(c)), server: server,
                                           to: still, title: "\(c.name)'s video message (picture)", progress: phase("picture")))
                let (voice, spent) = try await speak(c, text: media.prompt, server: server, to: folder, base: "\(base)-voice",
                                                     phase: phase)
                charge(spent)
                let seconds = await MovieComposer.duration(of: voice) ?? CharacterMedia.speechSeconds(media.prompt)
                guard seconds <= CharacterMedia.maxTalkSeconds + 0.05 else { throw CharacterMediaError.tooLong(seconds) }
                let render = CharacterMedia.talkRender(c, prompt: CharacterMedia.talkPrompt(c), start: still, voice: voice,
                                                       seed: Self.mediaSeed(), server: server)
                let (outputs, jobs) = try await runHeadlessGraph(template: CharacterMedia.talkTemplate(), render: render,
                                                                 title: "\(c.name)'s video message", progress: phase("clip"))
                let s = CharacterMedia.videoSize
                charge(recordGPUJobs(jobs, work: MovieShots.talkingWork(width: s.width, height: s.height, seconds: seconds)))
                file = try copyOutput(outputs, .video, to: folder, base: base)
            }
            guard doc.transcript.turn(turnID) != nil else { return }
            let album = characterStore.albumFolder(c.id)
            try FileManager.default.createDirectory(at: album, withIntermediateDirectories: true)
            let copy = album.appendingPathComponent(file.lastPathComponent)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: file, to: copy)
            var m = characterStore.memory(c.id)
            m.album.append(AlbumItem(file: file.lastPathComponent, kind: kind, prompt: media.prompt, scene: scene?.title,
                                     chatID: doc.id))
            try? characterStore.saveMemory(m, for: c.id)
            setMedia(doc, turnID, mediaID) { $0.status = .done; $0.file = file.lastPathComponent; $0.costUSD = cost; $0.error = nil }
        } catch {
            let text = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
            setMedia(doc, turnID, mediaID) { $0.status = .failed; $0.error = text; $0.costUSD = cost }
        }
    }

    private static func mediaSeed() -> Int { Int.random(in: 0..<Int(Int32.max)) }

    /// Draws one picture of the character (with the chosen portrait's face, cropped to it unless `wholePortrait`;
    /// `face` instead, when given) into `file` (PNG); returns its cost.
    func drawPhoto(_ c: Companion, prompt: String, server: String, to file: URL, title: String, seed: Int? = nil,
                   wholePortrait: Bool = false, extraNegative: String = "", face given: URL? = nil, keepHair: Bool = true,
                   faceStrength: Double = CharacterArt.faceStrength,
                   progress: @escaping @MainActor (String) -> Void) async throws -> Double? {
        let face: URL?
        if let given { face = given } else { face = wholePortrait ? portraitURL(c) : await faceReference(c) }
        let render = CharacterArt.render(c, prompt: prompt, seed: seed ?? Self.mediaSeed(), server: server, face: face,
                                         keepHair: keepHair, extraNegative: extraNegative, context: artContext(c),
                                         faceStrength: faceStrength)
        let (outputs, jobs) = try await runHeadlessGraph(template: CharacterArt.template(c.artModel.family, loras: c.loras.count),
                                                         render: render, title: title, progress: progress)
        // The jobs cost money even if the picture then can't be saved.
        let cost = recordGPUJobs(jobs)
        guard let image = outputs.first(where: { $0.kind == .image })?.image else { throw MovieArtError.noImage }
        try MovieComposer.writePNG(image, to: file)
        return cost
    }

    /// Says `text` in the character's voice (cloned from their chosen sample, made first when they have none) into
    /// `folder` as `base` plus the file's own extension; returns the file and the cost of its jobs. `phase` makes each
    /// step's progress callback.
    private func speak(_ c: Companion, text: String, server: String, to folder: URL, base: String,
                       phase: (String) -> @MainActor (String) -> Void) async throws -> (URL, Double?) {
        var cost: Double?
        var sample = voiceSampleURL(c)
        var who = c
        if sample == nil {
            let made = try await makeVoiceSample(for: c.id, server: server, progress: phase("voice sample"))
            cost = made.cost
            who = character(c.id) ?? c
            sample = voiceSampleURL(who)
        }
        guard let sample else { throw CharacterMediaError.noVoice }
        let render = CharacterMedia.voiceLineRender(who, text: text, sample: sample, seed: Self.mediaSeed(), server: server)
        let (outputs, jobs) = try await runHeadlessGraph(template: CharacterMedia.voiceLineTemplate(who), render: render,
                                                         title: "\(c.name)'s voice message", progress: phase("voice"))
        let file = try copyOutput(outputs, .audio, to: folder, base: base)
        let seconds = await MovieComposer.duration(of: file) ?? CharacterMedia.speechSeconds(text)
        if let c2 = recordGPUJobs(jobs, work: MovieVoices.work(seconds: seconds)) { cost = (cost ?? 0) + c2 }
        return (file, cost)
    }

    /// Copies the graph's first output of `kind` (a file) to `folder` as `base` plus its extension.
    private func copyOutput(_ outputs: [EngineOutput], _ kind: MediaKind, to folder: URL, base: String) throws -> URL {
        guard let made = outputs.first(where: { $0.kind == kind })?.url else { throw CharacterMediaError.noOutput(kind.rawValue) }
        let ext = made.pathExtension.isEmpty ? (kind == .audio ? "wav" : "mp4") : made.pathExtension
        let file = folder.appendingPathComponent("\(base).\(ext)")
        try? FileManager.default.removeItem(at: file)
        try FileManager.default.copyItem(at: made, to: file)
        return file
    }

    // MARK: voice samples

    /// Designs a voice from the character's description saying the sample sentence, on `server` (the first GPU
    /// server when nil); it's added to their samples and chosen when they have none chosen. Returns its file name
    /// and cost (also added to what the character has spent).
    @discardableResult
    func makeVoiceSample(for id: UUID, server: String? = nil,
                         progress: @escaping @MainActor (String) -> Void = { _ in }) async throws -> (name: String, cost: Double?) {
        guard let c = character(id) else { throw PortraitError.noCharacter }
        return try await makeVoiceSample(for: c, server: server, progress: progress)
    }

    /// The same for `c` as given, saved or not yet (the file goes to its voice folder either way; the stored
    /// character, when there is one, gets it added).
    @discardableResult
    func makeVoiceSample(for c: Companion, server: String? = nil,
                         progress: @escaping @MainActor (String) -> Void = { _ in }) async throws -> (name: String, cost: Double?) {
        let id = c.id
        guard let server = server ?? connections.gpuServers.first?.id else { throw PortraitError.noServer }
        let folder = characterStore.voiceFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let render = CharacterMedia.voiceSampleRender(c, seed: Self.mediaSeed(), server: server)
        let (outputs, jobs) = try await runHeadlessGraph(template: CharacterMedia.voiceSampleTemplate(c), render: render,
                                                         title: "\(c.name)'s voice", progress: progress)
        let file = try copyOutput(outputs, .audio, to: folder, base: "voice-\(UUID().uuidString.prefix(8).lowercased())")
        let seconds = await MovieComposer.duration(of: file) ?? CharacterMedia.speechSeconds(MovieVoices.sampleText)
        let cost = recordGPUJobs(jobs, work: MovieVoices.work(seconds: seconds))
        var m = characterStore.memory(id)
        m.spentUSD += cost ?? 0
        try? characterStore.saveMemory(m, for: id)
        try dropIfDiscarded(id)
        if var stored = characterStore.load(id) {
            stored.voiceSamples.append(file.lastPathComponent)
            if voiceSampleURL(stored) == nil { stored.selectedVoiceSample = file.lastPathComponent }
            saveCharacter(stored)
        }
        return (file.lastPathComponent, cost)
    }

    func selectVoiceSample(_ name: String, for id: UUID) {
        guard var c = characterStore.load(id), c.voiceSamples.contains(name), c.selectedVoiceSample != name else { return }
        c.selectedVoiceSample = name
        saveCharacter(c)
    }

    /// Takes the sample out of the character and its file to the Trash.
    /// Also for a character not saved yet (only the file goes).
    func deleteVoiceSample(_ name: String, for id: UUID) {
        if var c = characterStore.load(id), c.voiceSamples.contains(name) {
            c.voiceSamples.removeAll { $0 == name }
            if c.selectedVoiceSample == name { c.selectedVoiceSample = c.voiceSamples.last }
            saveCharacter(c)
        }
        let url = characterStore.voiceFolder(id).appendingPathComponent(name)
        if Self.isAutomatedRun { try? FileManager.default.removeItem(at: url) }
        else { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
    }
}

enum CharacterMediaError: Error, CustomStringConvertible {
    case noServer, noVoice, noOutput(String), tooLong(Double), noClip
    var description: String {
        switch self {
        case .noServer: "No GPU server for media"
        case .noClip: "The clip to copy is gone from this chat"
        case .noVoice: "They have no voice sample yet"
        case .noOutput(let kind): "The server sent back no \(kind)"
        case .tooLong(let s):
            "The words take \(String(format: "%.0f", s)) s to say; a video message holds \(Int(CharacterMedia.maxTalkSeconds)) s at most"
        }
    }
}
