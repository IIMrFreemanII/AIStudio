import CharacterCore
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-characters-real <model id>`: a short SFW chat with a template character on a real model, on your
    /// first saved GPU server (read from your connections, kept in memory only). It prints every reply, checks each
    /// one came back in character, waits for the memory pass and prints what it remembered. Costs GPU time; the
    /// character and the chat live in the run's throwaway library.
    static func runCharactersReal(_ model: AppModel, modelID: String) async -> Bool {
        var ok = true
        func say(_ s: String) { print("[smoke-characters-real] \(s)"); fflush(stdout) }
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }

        guard let data = try? Data(contentsOf: ConnectionStore.defaultFileURL),
              let saved = try? JSONDecoder().decode([ServerConnection].self, from: data),
              let conn = saved.first(where: { $0.kind == .gpu }),
              let token = ConnectionToken.read(conn.id) else {
            say("FAIL no saved GPU server with a token"); return false
        }
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else { say("FAIL \(conn.name) not reachable"); return false }
        let llm = model.llm
        await llm.refresh(conn.id)
        let key = LLMKey(serverID: conn.id, modelID: modelID)
        guard llm.model(key)?.installed == true else { say("FAIL \(modelID) isn't installed on \(conn.name)"); return false }

        func send(_ doc: ChatDocument, _ text: String) async -> ChatTurn? {
            model.sendText(doc, text)
            let start = Date()
            // The first reply waits for a cold start.
            while doc.isStreaming, Date().timeIntervalSince(start) < 420 { try? await Task.sleep(nanoseconds: 200_000_000) }
            guard let reply = doc.transcript.turns.last, reply.role == .assistant else { return nil }
            return reply
        }

        let c = CharacterTemplates.make(CharacterTemplates.all[0])
        check(model.saveCharacter(c).isEmpty, "\(c.name) is saved")
        let doc = model.newCharacterChat([c.id], model: key)
        say("greeting: \(doc.transcript.turns.first?.text ?? "")")

        let lines = [
            "Hi \(c.name)! I just got back from a hike with my dog Miso. How was your day?",
            "Who are you, in a few words? What do you like doing on weekends?",
            "Could you send me a selfie from where you are right now?",
            "I work as a sound designer, mostly for indie games.",
            "What would a perfect first date look like for you?",
            "That sounds lovely. Let's plan it for Saturday.",
        ]
        let banned = ["as an ai", "language model", "i'm an assistant", "i am an assistant", "<|", "[inst]"]
        for (i, text) in lines.enumerated() {
            let start = Date()
            let r = await send(doc, text)
            let secs = Int(Date().timeIntervalSince(start))
            say("you: \(text)")
            say("\(c.name) (\(secs)s): \(r?.text ?? r?.error ?? "none")")
            let reply = r?.text ?? ""
            let lower = reply.lowercased()
            check(r?.error == nil && !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && r?.speaker == c.id,
                  "reply \(i + 1) came back from \(c.name)")
            check(!banned.contains { lower.contains($0) }, "reply \(i + 1) stays in character")
            let tags = MediaTags.parse(reply).requests
            if i == 2 {
                // Media is off until photos land: no tag, and no pretend attachment.
                let pretends = ["attach", "here's a pic", "here is a pic", "sending you"].contains { lower.contains($0) }
                check(tags.isEmpty && !pretends, "the selfie ask gets no tag and no pretend photo")
            }
        }

        let deadline = Date().addingTimeInterval(180)
        var memory = model.characterStore.memory(c.id)
        while Date() < deadline, memory.turnsSinceUpdate != 0 || memory.facts.isEmpty {
            try? await Task.sleep(nanoseconds: 500_000_000)
            memory = model.characterStore.memory(c.id)
        }
        check(memory.turnsSinceUpdate == 0 && !memory.facts.isEmpty, "the memory pass ran")
        for f in memory.facts { say("fact: \(f.text)") }
        say("relationship: \(memory.relationship.stage.rawValue), affection \(memory.relationship.affection), mood \(memory.relationship.mood)")
        say("summary: \(memory.summary)")
        check(memory.facts.contains { $0.text.localizedCaseInsensitiveContains("Miso") }, "it remembered Miso")

        model.deleteChat(doc.id)
        model.deleteCharacter(c.id)
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }
}
