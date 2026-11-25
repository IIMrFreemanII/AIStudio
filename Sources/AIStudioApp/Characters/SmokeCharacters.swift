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
        let kinds = model.characterStore.memory(c.id).album.map(\.kind)
        check(kinds == [.photo, .photo, .voice, .video, .talk], "the album lists the voice message and both videos (\(kinds))")
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
            while Date().timeIntervalSince(start) < 30,
                  doc.isStreaming || answers().count < replies || answers().contains(where: { $0.ended == nil && $0.error == nil }) {
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
