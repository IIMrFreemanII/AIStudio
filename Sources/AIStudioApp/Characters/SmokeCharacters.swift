import AppKit
import CharacterCore
import Foundation
import LLMClient
import MovieCore

extension SmokeTest {
    /// `--smoke-characters <url>`: against a GPU server run with `AISTUDIO_FAKE_LLM=1`, checks the age guard, saving a
    /// character, a chat with it (greeting, speaker, the character's own prompt, photo tags), the memory pass after six
    /// replies, scenes, a generated portrait and the selfie it takes, and the saved chat. The server must also run with
    /// `AISTUDIO_FAKE_MODELS=1`. Everything it makes is deleted at the end.
    static func runCharacters(_ model: AppModel, url: String) async -> Bool {
        var ok = true
        func say(_ s: String) { print("[smoke-characters] \(s)"); fflush(stdout) }
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }

        model.characterAutoMediaUSD = 1.0  // fake prices never gate the photo (in memory only in smoke runs)
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else {
            say("FAIL server not reachable"); return false
        }
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("FAIL no client"); return false }
        await llm.refresh(conn.id)
        let link = "fake/tiny"
        var mid = llm.listings[conn.id]?.models.first { $0.id == link || $0.repo == link }?.id
        if mid == nil {
            do {
                let found = try await client.inspect(url: link)
                let st = try await client.install(LLMInstallRequest(url: link))
                mid = st.id ?? found.suggestedId
            } catch {
                say("FAIL install: \(error.localizedDescription)"); return false
            }
            let deadline = Date().addingTimeInterval(30)
            while Date() < deadline {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        guard let mid, llm.model(LLMKey(serverID: conn.id, modelID: mid))?.installed == true else {
            say("FAIL model didn't install"); return false
        }
        let key = LLMKey(serverID: conn.id, modelID: mid)

        func send(_ doc: ChatDocument, _ text: String) async -> ChatTurn? {
            model.sendText(doc, text)
            let start = Date()
            while doc.isStreaming, Date().timeIntervalSince(start) < 30 { try? await Task.sleep(nanoseconds: 50_000_000) }
            guard let reply = doc.transcript.turns.last, reply.role == .assistant else { return nil }
            return reply
        }

        // a. A character under 18 isn't saved.
        var young = CharacterTemplates.make(CharacterTemplates.all[0])
        young.age = 17
        let problems = model.saveCharacter(young)
        check(!problems.isEmpty && model.characterStore.load(young.id) == nil && !model.characters.contains { $0.id == young.id },
              "an age of 17 is refused and nothing is saved")

        // b. A template copy is saved.
        let c = CharacterTemplates.make(CharacterTemplates.all[0])
        let saved = model.saveCharacter(c)
        check(saved.isEmpty && model.characters.contains { $0.id == c.id } && model.characterStore.load(c.id)?.id == c.id,
              "\(c.name) is saved and listed")

        // c. A chat with it.
        let doc = model.newCharacterChat([c.id], model: key)
        let first = doc.transcript.turns.first
        check(doc.isCharacter && doc.title == c.name && doc.transcript.sceneID == Scenario.defaultID
              && first?.role == .assistant && first?.text == c.greeting && first?.speaker == c.id,
              "a new chat with \(c.name) starts with her greeting")
        check(model.characters(of: doc).map(\.id) == [c.id], "the chat's characters")

        // d. Her own prompt: the fake engine reads her name from it.
        let who = await send(doc, "who are you")
        check(who?.error == nil && who?.text == "I'm \(c.name)." && who?.speaker == c.id,
              "the reply comes from her prompt (\(who?.text ?? who?.error ?? "none"))")

        // e. A portrait, then a photo tag in a reply that becomes a picture.
        var portraits: [String] = []
        do { portraits = try await model.generatePortraits(for: c.id, count: 1) } catch { say("portrait failed: \(error)") }
        let portraitFile = portraits.first.map { model.characterStore.portraitsFolder(c.id).appendingPathComponent($0) }
        check(portraits.count == 1 && portraitFile.map { FileManager.default.fileExists(atPath: $0.path) } == true
              && model.character(c.id)?.selectedPortrait == portraits.first, "a portrait is made and selected")
        let selfie = await send(doc, "send a selfie")
        let requests = MediaTags.parse(selfie?.text ?? "").requests
        check(selfie?.error == nil && requests.count == 1 && requests.first?.kind == .photo, "a selfie reply has one photo tag")
        let photoDeadline = Date().addingTimeInterval(30)
        var photoTurn = selfie
        while Date() < photoDeadline {
            photoTurn = selfie.flatMap { doc.transcript.turn($0.id) }
            if let st = photoTurn?.media?.first?.status, st == .done || st == .failed { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let photo = photoTurn?.media?.first
        check(photo?.status == .done, "the photo is done (\(photo?.status.rawValue ?? "none") \(photo?.error ?? ""))")
        if let file = photo?.file {
            let inChat = model.chatStore.filesFolder(doc.id).appendingPathComponent(file)
            let inAlbum = model.characterStore.albumFolder(c.id).appendingPathComponent(file)
            check(FileManager.default.fileExists(atPath: inChat.path), "the photo is in the chat's files")
            check(FileManager.default.fileExists(atPath: inAlbum.path), "the photo is in the album folder")
            let album = model.characterStore.memory(c.id).album
            check(album.count == 1 && album.first?.kind == .photo && album.first?.file == file, "the album lists the photo")
        } else {
            check(false, "the photo has a file")
        }
        check(photoTurn.map { CharacterDisplay.markdown($0).contains("![Photo](") } == true, "the reply shows the picture")

        // f. Six replies: the memory pass runs.
        for text in ["I went hiking today", "it was lovely", "what about you", "tell me more"] {
            let r = await send(doc, text)
            if r?.error != nil || r?.text != "You said: \(text)" { check(false, "reply to \"\(text)\" (\(r?.text ?? r?.error ?? "none"))") }
        }
        check(doc.transcript.turns.filter { $0.role == .assistant && $0.speaker == c.id }.count == 7,
              "every reply is written by \(c.name)")
        let deadline = Date().addingTimeInterval(10)
        var memory = model.characterStore.memory(c.id)
        while Date() < deadline, !memory.facts.contains(where: { $0.text.contains("Miso") }) {
            try? await Task.sleep(nanoseconds: 100_000_000)
            memory = model.characterStore.memory(c.id)
        }
        check(memory.facts.contains { $0.text.contains("Miso") }, "the memory pass remembered Miso")
        check(memory.relationship.stage == .flirting && memory.relationship.affection == 23,
              "stage \(memory.relationship.stage.rawValue), affection \(memory.relationship.affection)")
        check(memory.turnsSinceUpdate == 0, "the counter is back to 0 (\(memory.turnsSinceUpdate))")

        // g. A scene.
        model.setScene("dinner", for: doc)
        check(model.scene(of: doc)?.id == "dinner", "the scene is the dinner date")
        let after = await send(doc, "how is the food")
        check(after?.error == nil && after?.text == "You said: how is the food", "a reply in the new scene")

        // g2. The same character drawn by Chroma: the selfie runs the server's `chroma` task (full Chroma, with the
        // portrait as the face).
        if var chroma = model.character(c.id) {
            chroma.artModel.family = .chroma
            check(model.saveCharacter(chroma).isEmpty, "\(c.name) now draws with Chroma")
        }
        let chromaJobsBefore = GPUTimings.load(model.gpuTimingsURL).lastJob["chroma"]
        let chromaSelfie = await send(doc, "send a selfie")
        var chromaTurn = chromaSelfie
        let chromaDeadline = Date().addingTimeInterval(30)
        while Date() < chromaDeadline {
            chromaTurn = chromaSelfie.flatMap { doc.transcript.turn($0.id) }
            if let st = chromaTurn?.media?.first?.status, st == .done || st == .failed { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let chromaPhoto = chromaTurn?.media?.first
        let chromaJob = GPUTimings.load(model.gpuTimingsURL).lastJob["chroma"]
        check(chromaPhoto?.status == .done && chromaJob != nil && chromaJob != chromaJobsBefore,
              "the Chroma photo is done by the chroma task (\(chromaPhoto?.status.rawValue ?? "none") \(chromaPhoto?.error ?? ""))")
        check(model.characterStore.memory(c.id).album.count == 2, "the album lists both photos")

        // g2b. And by MiniMax H3 Image: the selfie runs `h3-image` with the face crop as picture 1. Back to Chroma after.
        if var minimax = model.character(c.id) {
            minimax.artModel.family = .minimax
            check(model.saveCharacter(minimax).isEmpty, "\(c.name) now draws with MiniMax")
        }
        let h3JobsBefore = GPUTimings.load(model.gpuTimingsURL).lastJob["h3-image"]
        let h3Selfie = await send(doc, "send a selfie")
        var h3Turn = h3Selfie
        let h3Deadline = Date().addingTimeInterval(30)
        while Date() < h3Deadline {
            h3Turn = h3Selfie.flatMap { doc.transcript.turn($0.id) }
            if let st = h3Turn?.media?.first?.status, st == .done || st == .failed { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let h3Photo = h3Turn?.media?.first
        let h3Job = GPUTimings.load(model.gpuTimingsURL).lastJob["h3-image"]
        check(h3Photo?.status == .done && h3Job != nil && h3Job != h3JobsBefore,
              "the MiniMax photo is done by the h3-image task (\(h3Photo?.status.rawValue ?? "none") \(h3Photo?.error ?? ""))")
        if var back = model.character(c.id) {
            back.artModel.family = .chroma
            _ = model.saveCharacter(back)
        }

        // g3. Voice, video and video messages.
        func settled(_ turn: ChatTurn?, _ waitFor: Set<ChatMedia.Status> = [.done, .failed, .skipped, .waiting])
            async -> ChatMedia?
        {
            guard let id = turn?.id else { return nil }
            let deadline = Date().addingTimeInterval(30)
            while Date() < deadline {
                if let m = doc.transcript.turn(id)?.media?.first, waitFor.contains(m.status) { return m }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return doc.transcript.turn(id)?.media?.first
        }
        func describe(_ m: ChatMedia?) -> String { "\(m?.status.rawValue ?? "none") \(m?.error ?? "")" }
        do {
            let made = try await model.makeVoiceSample(for: c.id)
            let sample = model.characterStore.voiceFolder(c.id).appendingPathComponent(made.name)
            check(FileManager.default.fileExists(atPath: sample.path) && model.character(c.id)?.selectedVoiceSample == made.name,
                  "a voice sample is made and chosen")
        } catch {
            check(false, "the voice sample (\(error))")
        }
        let voiceTurn = await send(doc, "send a voice message")
        let voice = await settled(voiceTurn, [.done, .failed, .skipped])
        check(voice?.kind == "voice" && voice?.status == .done && voice?.file?.hasSuffix(".wav") == true,
              "the voice message is made at once (\(describe(voice)))")
        check(voiceTurn.flatMap { doc.transcript.turn($0.id) }.map { CharacterDisplay.markdown($0).contains("I miss you already") } == true,
              "the voice message shows its words")
        let videoTurn = await send(doc, "send a video")
        let waiting = await settled(videoTurn)
        check(waiting?.kind == "video" && waiting?.status == .waiting && waiting?.estimateUSD != nil,
              "a video waits for the OK with its price (\(describe(waiting)))")
        if let t = videoTurn, let m = waiting { model.approveMedia(doc, turnID: t.id, mediaID: m.id) }
        let video = await settled(videoTurn, [.done, .failed])
        check(video?.status == .done && video?.file?.hasSuffix(".mp4") == true, "the approved video is made (\(describe(video)))")
        let talkTurn = await send(doc, "say it on camera")
        let talkWaiting = await settled(talkTurn)
        check(talkWaiting?.kind == "talk" && talkWaiting?.status == .waiting, "a video message waits for the OK (\(describe(talkWaiting)))")
        if let t = talkTurn, let m = talkWaiting { model.approveMedia(doc, turnID: t.id, mediaID: m.id) }
        let talk = await settled(talkTurn, [.done, .failed])
        check(talk?.status == .done && talk?.file?.hasSuffix(".mp4") == true, "the video message is made (\(describe(talk)))")
        let skippedTurn = await send(doc, "send a video")
        if let t = skippedTurn, let m = await settled(skippedTurn) { model.skipMedia(doc, turnID: t.id, mediaID: m.id) }
        check(skippedTurn.flatMap { doc.transcript.turn($0.id) }?.media?.first?.status == .skipped, "a skipped video stays skipped")
        // g3b. Motion: "Have her do this" on a clip, and a video asked for after sending one, copy its moves.
        if let clip = video?.file, let her = model.character(c.id) {
            check(model.motionCharacter(doc) != nil, "she can copy a clip's moves")
            model.requestMotion(doc, clip: clip, character: her)
            let motionTurn = doc.transcript.turns.last
            let motionWaiting = await settled(motionTurn)
            check(motionWaiting?.kind == "motion" && motionWaiting?.status == .waiting && motionWaiting?.source == clip
                  && motionWaiting?.estimateUSD != nil, "the clip's motion video waits for the OK with its price (\(describe(motionWaiting)))")
            if let t = motionTurn, let m = motionWaiting { model.approveMedia(doc, turnID: t.id, mediaID: m.id) }
            let motion = await settled(motionTurn, [.done, .failed])
            check(motion?.status == .done && motion?.file?.hasSuffix(".mp4") == true, "the motion video is made (\(describe(motion)))")
            doc.pending = [.image(clip)]
            let askedTurn = await send(doc, "send a video")
            let asked = await settled(askedTurn)
            check(asked?.kind == "motion" && asked?.source == clip, "a video asked for after a clip copies it (\(describe(asked)))")
            if let t = askedTurn, let m = asked { model.skipMedia(doc, turnID: t.id, mediaID: m.id) }
        } else {
            check(false, "the motion step needs the approved video")
        }
        let kinds = model.characterStore.memory(c.id).album.map(\.kind)
        check(kinds == [.photo, .photo, .photo, .voice, .video, .talk, .motion],
              "the album lists the voice message, both videos and the motion video (\(kinds))")
        check(model.characterStore.memory(c.id).ownMedia.isEmpty, "asked-for media uses none of her initiative")

        // g4. She texts first after time away (her chat's last message 9 hours ago), once.
        if let last = doc.transcript.turns.last {
            let past = Date().addingTimeInterval(-9 * 3600)
            doc.transcript.update(last.id) { $0.date = past; $0.ended = past }
        }
        let beforeCheckIn = doc.transcript.turns.count
        model.characterCheckInIfDue(doc)
        let checkInStart = Date()
        while doc.isStreaming, Date().timeIntervalSince(checkInStart) < 30 { try? await Task.sleep(nanoseconds: 50_000_000) }
        let checkIn = doc.transcript.turns.last
        check(doc.transcript.turns.count == beforeCheckIn + 1 && checkIn?.speaker == c.id && checkIn?.error == nil
              && checkIn?.text.contains("About 9 hours since your last messages") == true,
              "after 9 hours away she writes first, from the check-in note (\(checkIn?.text.prefix(60) ?? "none"))")
        model.characterCheckInIfDue(doc)
        check(!doc.isStreaming && doc.transcript.turns.count == beforeCheckIn + 1, "she doesn't write first twice in a row")

        // g5. A group chat: everyone answers in turn, each from their own prompt, seeing the others' lines; a message
        // that names one of them gets only that one.
        let c2 = CharacterTemplates.make(CharacterTemplates.all[1])
        check(model.saveCharacter(c2).isEmpty, "\(c2.name) is saved")
        let group = model.newCharacterChat([c.id, c2.id], model: key)
        check(group.title == "\(c.name) and \(c2.name)" && group.transcript.turns.isEmpty, "a group chat is titled with both names (\(group.title)) and has no greeting")
        /// Sends and waits for the whole round (the next one answers once the previous reply wraps up).
        func round(_ doc: ChatDocument, _ text: String, replies: Int) async -> [ChatTurn] {
            model.sendText(doc, text)
            let start = Date()
            func answers() -> [ChatTurn] {
                let turns = doc.transcript.turns
                guard let u = turns.lastIndex(where: { $0.role == .user }) else { return [] }
                return Array(turns[(u + 1)...].filter { $0.role == .assistant })
            }
            // Done once nothing streams and the answers are in (or nothing has streamed for 2 s).
            var quietSince: Date?
            while Date().timeIntervalSince(start) < 30 {
                if doc.isStreaming {
                    quietSince = nil
                } else {
                    if quietSince == nil { quietSince = Date() }
                    if answers().count >= replies || Date().timeIntervalSince(quietSince!) > 2 { break }
                }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            try? await Task.sleep(nanoseconds: 300_000_000)  // nobody else starts
            return answers()
        }
        let both = await round(group, "who are you", replies: 2)
        check(both.map(\.speaker) == [c.id, c2.id] && both.allSatisfy { $0.error == nil },
              "both answer, \(c.name) first (\(both.map { $0.speaker.flatMap { model.character($0)?.name } ?? "?" }))")
        check(both.first?.text == "I'm \(c.name)." && both.last?.text == "I'm \(c2.name).",
              "each answers from their own prompt (\(both.map(\.text)))")
        let echo = await round(group, "hello there", replies: 2)
        check(echo.map(\.speaker) == [c2.id, c.id], "the next round starts with the other one")
        check(echo.last?.text.contains("\(c2.name): You said: hello there") == true,
              "the second one sees the first one's line after the user's message (\(echo.last?.text ?? "none"))")
        let named = await round(group, "\(c2.name), who are you", replies: 1)
        check(named.map(\.speaker) == [c2.id], "naming \(c2.name) gets only \(c2.name)'s answer")

        // g6. An album photo's look comes back as the scene of your chat with her (the group, in front).
        if let photo = model.characterStore.memory(c.id).album.first(where: { $0.kind == .photo }) {
            model.reuseAlbumLook(photo, characterID: c.id)
            let id = "album-\(photo.id.uuidString)"
            let scene = model.characterStore.scenarios().first { $0.id == id }
            check(scene?.outfit == photo.prompt.trimmingCharacters(in: .whitespacesAndNewlines) && group.transcript.sceneID == id,
                  "the photo's look is a saved scene, set on the chat in front")
        } else {
            check(false, "an album photo to reuse")
        }
        model.deleteChat(group.id)
        model.deleteCharacter(c2.id)

        // g7. The creator: a character picked from options gets a photo and a voice before it's saved; a cancelled
        // one leaves nothing behind.
        var draft = CharacterTemplates.make(CharacterTemplates.all[0])
        draft.name = "Draft Mia"
        for (cat, opt) in [("hairColor", "red"), ("hairStyle", "curly"), ("habits", "yoga"), ("voiceTone", "husky")] {
            draft.traits = CharacterOptions.toggle(draft.traits, category: cat, option: opt)
        }
        check(CharacterArt.portraitPrompt(draft).contains("red hair") && CharacterArt.portraitPrompt(draft).contains("curly hair"),
              "the portrait prompt has the picked look")
        check(draft.fullVoice.contains("husky") && CharacterPrompt.system(draft, memory: CharacterMemory(), scene: nil)
              .contains("Habits: "), "the voice and the prompt have the picked options")
        var arrived: [String] = []
        let drafted = (try? await model.generatePortraits(for: draft, count: 1, added: { arrived.append($0) })) ?? []
        let draftSample = try? await model.makeVoiceSample(for: draft)
        check(drafted.count == 1 && arrived == drafted && draftSample != nil
              && FileManager.default.fileExists(atPath: model.characterStore.portraitsFolder(draft.id)
                  .appendingPathComponent(drafted[0]).path)
              && !model.characters.contains { $0.id == draft.id },
              "an unsaved character gets a portrait and a voice sample, and isn't listed yet")
        draft.portraits = drafted
        draft.selectedPortrait = drafted.first
        if let s = draftSample { draft.voiceSamples = [s.name]; draft.selectedVoiceSample = s.name }
        check(model.saveCharacter(draft).isEmpty && model.character(draft.id)?.selectedPortrait == drafted.first
              && model.portraitURL(draft) != nil && model.voiceSampleURL(draft) != nil,
              "saved, its portrait is the profile photo and its sample the voice")
        model.discardDraft(draft.id)
        check(model.characterStore.load(draft.id) != nil, "a saved character isn't discarded as a draft")
        model.deleteCharacter(draft.id)
        let cancelled = CharacterTemplates.make(CharacterTemplates.all[1])
        _ = try? await model.generatePortraits(for: cancelled, count: 1)
        model.discardDraft(cancelled.id)
        check(!FileManager.default.fileExists(atPath: model.characterStore.folder(cancelled.id).path),
              "a cancelled draft leaves no folder")
        // Cancelled while its portrait is still being drawn: the portrait that arrives after is removed with the folder.
        let midway = CharacterTemplates.make(CharacterTemplates.all[2])
        let drawing = Task { @MainActor in try await model.generatePortraits(for: midway, count: 2) }
        try? await Task.sleep(nanoseconds: 200_000_000)
        model.discardDraft(midway.id)
        let midResult = await drawing.result
        try? await Task.sleep(nanoseconds: 300_000_000)
        check((try? midResult.get()) == nil
              && !FileManager.default.fileExists(atPath: model.characterStore.folder(midway.id).path),
              "a draft cancelled mid-portrait leaves no folder")
        // g8. Surprise Me: a random character, then its details written by the chat model.
        let rolled = CharacterSurprise.roll(CharacterTemplates.make(CharacterTemplates.all[0]))
        let core = CharacterOptions.all.filter {
            $0.group != .intimate && !CharacterOptions.detailCategories.contains($0.id)
                && !CharacterOptions.options($0, gender: rolled.gender).isEmpty
        }
        check(core.allSatisfy { !(rolled.traits[$0.id] ?? []).isEmpty } && rolled.greeting.isEmpty && !rolled.name.isEmpty
              && CharacterOptions.intimateCategories.allSatisfy { rolled.traits[$0] == nil },
              "Surprise Me picks every core option, no 18+ ones, and a name (\(rolled.name), \(rolled.age))")
        do {
            let (details, _) = try await model.writeCharacterDetails(rolled, newName: true)
            let written = details.apply(to: rolled, name: true)
            check(written.name == "Juno" && written.greeting.hasPrefix("Hey!") && written.likes.contains("vinyl")
                  && AgeGuard.problems(written).isEmpty, "the chat model writes their name, story and first message")
        } catch {
            check(false, "the written details (\(error))")
        }
        // The face reference: the fake portraits have no face, so the whole portrait is used.
        if let stored = model.character(c.id), let face = await model.faceReference(stored) {
            check(face == model.portraitURL(stored), "with no face found, photos use the whole portrait")
        } else {
            check(false, "a face reference for \(c.name)")
        }
        // g9. Option pictures: the example maker on the fake server (two files), and the bundled examples.
        let examplesOut = FileManager.default.temporaryDirectory.appendingPathComponent("option-examples-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: examplesOut) }
        let (examples, _) = await OptionImageMaker.make(model, server: conn.id, out: examplesOut, categories: ["hairStyle"],
                                                        limit: 2, say: say)
        let sizes = examples.compactMap { NSImage(contentsOf: $0)?.representations.first }
            .map { ($0.pixelsWide, $0.pixelsHigh) }
        check(examples.count == 2 && sizes.allSatisfy { max($0.0, $0.1) == OptionImageMaker.maxPixelSize },
              "the example maker writes small JPEGs (\(sizes))")
        let bundled = CharacterOptions.examples.filter {
            OptionImages.url(category: $0.category, option: $0.option, gender: $0.gender) != nil
        }
        check(bundled.count == CharacterOptions.examples.count,
              "every option example is bundled (\(bundled.count) of \(CharacterOptions.examples.count))")
        check(OptionImages.image(category: "hairStyle", option: "bob", gender: "non-binary") != nil,
              "another gender gets the woman's example")

        // g10. The Prompt Builder's preview: from the character (their own family, face and fine-tune), then plain
        // SDXL with the same seed, which the strip keeps newest first; video models draw no preview.
        if let stored = model.character(c.id) {
            var d = PromptBuilderDraft()
            let context = model.artContext(stored)
            d.apply(character: stored, model: CharacterArt.promptModel(stored, context: context),
                    triggerWords: context.triggerWords)
            d.action = "reading by the window"
            do {
                let before = model.promptPreviews().count
                let a = try await model.previewPrompt(d)
                check(model.previewCharacter(d)?.id == stored.id && a.model == d.model
                      && FileManager.default.fileExists(atPath: model.promptPreviewURL(a).path)
                      && model.promptPreviews().count == before + 1,
                      "the builder previews \(stored.name) on their own model (\(a.model.title))")
                var plain = d
                plain.detachCharacter()
                plain.model = .sdxl
                let b = try await model.previewPrompt(plain)
                let changes = b.draft.changes(from: a.draft)
                check(b.seed == a.seed && model.promptPreviews().prefix(2).map(\.id) == [b.id, a.id]
                      && changes.contains("Model: SDXL"), "a plain SDXL preview keeps the seed (\(changes))")
                var video = plain
                video.model = .wan
                do {
                    _ = try await model.previewPrompt(video)
                    check(false, "a video model draws no preview")
                } catch {
                    check(error is PromptPreviewError, "a video model draws no preview (\(error))")
                }
                model.deletePromptPreview(a)
                model.deletePromptPreview(b)
            } catch {
                check(false, "the builder preview (\(error))")
            }
        }

        // h. Saved.
        model.saveChat(doc)
        check(model.chatStore.list().contains { $0.id == doc.id && $0.character }, "the chat is listed as a character chat")
        let disk = model.chatStore.load(doc.id)
        check(disk?.characterIDs == [c.id] && disk?.sceneID == "dinner", "the saved chat keeps its character and scene")

        // i. Clean up.
        model.deleteChat(doc.id)
        model.deleteCharacter(c.id)
        check(!model.characters.contains { $0.id == c.id } && model.characterStore.load(c.id) == nil
              && model.chatStore.load(doc.id) == nil, "the chat and the character are deleted")
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }
}
