import CharacterCore
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-characters-real <model id>` (+ `--smoke-real-out <folder>`, `--smoke-real-portraits <n>`, default 4):
    /// a short SFW date with a template character on a real model, on your first saved GPU server (read from your
    /// connections, kept in memory only). It draws the portraits (SDXL), chats (printing every reply and checking each
    /// came back in character), takes the two selfies the chat asks for with the chosen portrait's face, then a voice
    /// message, a video and a video message (approving the ones that wait), waits for the memory pass and prints what
    /// it remembered and what it all cost. The pictures, recordings and clips are copied into the out folder.
    /// Costs GPU time; the character and the chat live in the run's throwaway library.
    static func runCharactersReal(_ model: AppModel, modelID: String) async -> Bool {
        var ok = true
        func say(_ s: String) { print("[smoke-characters-real] \(s)"); fflush(stdout) }
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        let out = argument("--smoke-real-out").map { URL(fileURLWithPath: $0, isDirectory: true) }
        let portraitCount = argument("--smoke-real-portraits").flatMap(Int.init) ?? 4
        // `--smoke-real-lines <n>`: only the first n messages (a cheap cold-start check); the memory checks need all.
        let lineLimit = argument("--smoke-real-lines").flatMap(Int.init)
        let fm = FileManager.default
        func keep(_ file: URL, as name: String) {
            guard let out else { return }
            try? fm.createDirectory(at: out, withIntermediateDirectories: true)
            let dest = out.appendingPathComponent(name)
            try? fm.removeItem(at: dest)
            try? fm.copyItem(at: file, to: dest)
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
        // Photos and voice run without asking (the run's setting only, kept in memory); videos are approved below.
        model.characterAutoMediaUSD = 1.0

        var c = CharacterTemplates.make(CharacterTemplates.all[0])
        c.initiative = 0  // no media on her own: the test checks she sends only what she's asked for
        check(model.saveCharacter(c).isEmpty, "\(c.name) is saved")

        // a. Portraits: the first becomes the face of the photos.
        if portraitCount > 0 {
            let start = Date()
            var portraits: [String] = []
            do {
                portraits = try await model.generatePortraits(for: c.id, count: portraitCount) { say("portrait: \($0)") }
            } catch {
                say("portrait failed: \(error)")
            }
            say("portraits took \(Int(Date().timeIntervalSince(start))) s, spent $\(String(format: "%.3f", model.characterStore.memory(c.id).spentUSD))")
            check(portraits.count == portraitCount && model.character(c.id)?.selectedPortrait == portraits.first,
                  "\(portraitCount) portraits are made and the first is chosen")
            for (i, name) in portraits.enumerated() {
                keep(model.characterStore.portraitsFolder(c.id).appendingPathComponent(name), as: "portrait-\(i + 1).png")
            }
        }

        func send(_ doc: ChatDocument, _ text: String) async -> ChatTurn? {
            model.sendText(doc, text)
            let start = Date()
            // The first reply waits for a cold start (a busy region can take 10+ minutes to find a GPU). A message sent
            // while a reply still runs would be dropped, so a reply that never ends stops the run.
            var phase: String?
            while doc.isStreaming, Date().timeIntervalSince(start) < 1200 {
                let now = doc.engine.map { "\($0.state) \($0.phase ?? "")" }
                if let now, now != phase { say("  engine at \(Int(Date().timeIntervalSince(start))) s: \(now)") }
                phase = now ?? phase
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if doc.isStreaming { model.stopReply(doc); return nil }
            guard let reply = doc.transcript.turns.last, reply.role == .assistant else { return nil }
            if let s = reply.stats {
                say("  stats: \(String(format: "%.1f", s.seconds)) s, wait \(s.waitSeconds.map { String(format: "%.1f", $0) } ?? "?") s, "
                    + "tokens \(s.promptTokens.map(String.init) ?? "?")/\(s.completionTokens.map(String.init) ?? "?")")
            } else {
                say("  stats: none (the server's usage event never came)")
            }
            return reply
        }
        /// Waits for the turn's media to finish, approving what waits for an OK; returns the turn as it ends up. A
        /// video message can need three cold models (picture, speech, LongCat), so this waits up to 20 minutes.
        func media(_ doc: ChatDocument, _ turnID: UUID) async -> ChatTurn? {
            let deadline = Date().addingTimeInterval(1200)
            while Date() < deadline {
                let t = doc.transcript.turn(turnID)
                for m in t?.media ?? [] where m.status == .waiting {
                    say("approving the \(m.kind) (about \(m.estimateUSD.map { String(format: "$%.3f", $0) } ?? "?"))")
                    model.approveMedia(doc, turnID: turnID, mediaID: m.id)
                }
                if let media = t?.media, media.allSatisfy({ ![.running, .waiting].contains($0.status) }) { return t }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            return doc.transcript.turn(turnID)
        }

        let doc = model.newCharacterChat([c.id], model: key)
        say("greeting: \(doc.transcript.turns.first?.text ?? "")")

        let allLines = [
            "Hi \(c.name)! I just got back from a hike with my dog Miso. How was your day?",
            "Who are you, in a few words? What do you like doing on weekends?",
            "Could you send me a selfie from where you are right now?",
            "I work as a sound designer, mostly for indie games.",
            "What would a perfect first date look like for you?",
            "That sounds lovely. Let's plan it for Saturday. Send me a photo of the outfit you'd wear?",
            "I'd love to hear your voice. Could you send me a voice message saying goodnight?",
            "Send me a short video of you waving from your window?",
            "Could you send me a video message saying hi to Miso?",
        ]
        let lines = lineLimit.map { Array(allLines.prefix($0)) } ?? allLines
        let asks: [Int: CharacterCore.MediaKind] = [2: .photo, 5: .photo, 6: .voice, 7: .video, 8: .talk]
        let banned = ["as an ai", "language model", "i'm an assistant", "i am an assistant", "<|", "[inst]"]
        var kept: [String: Int] = [:]
        for (i, text) in lines.enumerated() {
            let start = Date()
            let r = await send(doc, text)
            let secs = Int(Date().timeIntervalSince(start))
            say("you: \(text)")
            say("\(c.name) (\(secs)s): \(r?.text ?? "none")\(r?.error.map { " [error: \($0)]" } ?? "")")
            guard r != nil else { check(false, "reply \(i + 1) came back within 20 minutes"); break }
            let reply = r?.text ?? ""
            let lower = reply.lowercased()
            check(r?.error == nil && !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && r?.speaker == c.id,
                  "reply \(i + 1) came back from \(c.name)")
            check(!banned.contains { lower.contains($0) }, "reply \(i + 1) stays in character")
            let tags = MediaTags.parse(reply).requests
            guard let wanted = asks[i] else {
                // Her own-initiative media is off (initiative 0): only asked-for ones.
                check(tags.isEmpty, "reply \(i + 1) sends nothing nobody asked for")
                continue
            }
            check(tags.count == 1 && tags.first?.kind == wanted,
                  "reply \(i + 1) sends one \(wanted.rawValue) (\(tags.map(\.kind.rawValue)))")
            guard let r, !tags.isEmpty else { continue }
            let mediaStart = Date()
            let t = await media(doc, r.id)
            for m in t?.media ?? [] {
                say("\(m.kind) \"\(m.prompt)\": \(m.status.rawValue) in \(Int(Date().timeIntervalSince(mediaStart))) s, "
                    + "estimate \(m.estimateUSD.map { String(format: "$%.3f", $0) } ?? "?"), "
                    + "cost \(m.costUSD.map { String(format: "$%.3f", $0) } ?? "?")\(m.error.map { ", \($0)" } ?? "")")
                check(m.status == .done && m.file != nil, "the \(m.kind) of reply \(i + 1) is done")
                if let file = m.file {
                    let inChat = model.chatStore.filesFolder(doc.id).appendingPathComponent(file)
                    check(fm.fileExists(atPath: inChat.path)
                          && fm.fileExists(atPath: model.characterStore.albumFolder(c.id).appendingPathComponent(file).path),
                          "the \(m.kind) is in the chat's files and the album")
                    kept[m.kind, default: 0] += 1
                    let stem = m.kind == "photo" ? "selfie" : m.kind
                    keep(inChat, as: "\(stem)-\(kept[m.kind]!).\(inChat.pathExtension)")
                }
            }
        }

        // Any other media the chat started (some nobody asked for, say) lands in the album too.
        for t in doc.transcript.turns where t.media != nil { _ = await media(doc, t.id) }
        let made = doc.transcript.turns.flatMap { $0.media ?? [] }.filter { $0.status == .done }.count
        guard lines.count == allLines.count else {
            say("spent on \(c.name): $\(String(format: "%.3f", model.characterStore.memory(c.id).spentUSD))")
            model.deleteChat(doc.id)
            model.deleteCharacter(c.id)
            say(ok ? "ok all passed" : "FAIL some checks failed")
            return ok
        }
        let deadline = Date().addingTimeInterval(180)
        var memory = model.characterStore.memory(c.id)
        // The pass runs after every MemoryPass.every replies (a reply landing while it runs may be counted or not).
        while Date() < deadline, memory.facts.isEmpty || memory.summary.isEmpty {
            try? await Task.sleep(nanoseconds: 500_000_000)
            memory = model.characterStore.memory(c.id)
        }
        check(!memory.facts.isEmpty && !memory.summary.isEmpty && memory.turnsSinceUpdate < MemoryPass.every,
              "the memory pass ran")
        for f in memory.facts { say("fact: \(f.text)") }
        say("relationship: \(memory.relationship.stage.rawValue), affection \(memory.relationship.affection), mood \(memory.relationship.mood)")
        say("summary: \(memory.summary)")
        check(memory.facts.contains { $0.text.localizedCaseInsensitiveContains("Miso") }, "it remembered Miso")
        check(memory.album.count == made, "the album lists all \(made) media")
        say("spent on \(c.name): $\(String(format: "%.3f", memory.spentUSD)) (portraits, media and memory updates)")
        if let out { say("files in \(out.path)") }

        model.deleteChat(doc.id)
        model.deleteCharacter(c.id)
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }
}
