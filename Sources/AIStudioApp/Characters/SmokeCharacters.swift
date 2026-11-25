import CharacterCore
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-characters <url>`: against a GPU server run with `AISTUDIO_FAKE_LLM=1`, checks the age guard, saving a
    /// character, a chat with it (greeting, speaker, the character's own prompt, photo tags), the memory pass after six
    /// replies, scenes, and the saved chat. Everything it makes is deleted at the end.
    static func runCharacters(_ model: AppModel, url: String) async -> Bool {
        var ok = true
        func say(_ s: String) { print("[smoke-characters] \(s)"); fflush(stdout) }
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }

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

        // e. A photo tag in a reply.
        let selfie = await send(doc, "send a selfie")
        let requests = MediaTags.parse(selfie?.text ?? "").requests
        check(selfie?.error == nil && requests.count == 1 && requests.first?.kind == .photo, "a selfie reply has one photo tag")

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
