import CharacterCore
import Foundation
import ImageIO
import LLMClient
import MovieCore

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

        // The builder's dry run writes prompts only: no server.
        if CommandLine.arguments.contains("--smoke-real-builder"), CommandLine.arguments.contains("--smoke-real-dry") {
            return await realBuilderPreviews(model, out: out, say: say)
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
        if CommandLine.arguments.contains("--smoke-real-outfit") {
            return await realOutfit(model, server: conn.id, keep: keep, say: say)
        }
        if CommandLine.arguments.contains("--smoke-real-option-images") {
            return await realOptionImages(model, server: conn.id, out: out, say: say)
        }
        if CommandLine.arguments.contains("--smoke-real-builder") {
            return await realBuilderPreviews(model, out: out, say: say)
        }
        if let family = argument("--smoke-real-train").flatMap(ArtModel.Family.init(rawValue:)) {
            return await realTraining(model, family: family, keep: keep, say: say)
        }
        if CommandLine.arguments.contains("--smoke-real-creator") {
            return await realCreator(model, count: portraitCount, keep: keep, say: say)
        }
        let llm = model.llm
        await llm.refresh(conn.id)
        let key = LLMKey(serverID: conn.id, modelID: modelID)
        guard llm.model(key)?.installed == true else { say("FAIL \(modelID) isn't installed on \(conn.name)"); return false }
        if CommandLine.arguments.contains("--smoke-real-details") { return await realDetails(model, key: key, say: say) }
        // Photos and voice run without asking (the run's setting only, kept in memory); videos are approved below.
        model.characterAutoMediaUSD = 1.0
        if CommandLine.arguments.contains("--smoke-real-group") { return await realGroup(model, key: key, say: say) }

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

    /// `--smoke-characters-real <any> --smoke-real-creator` (+ `--smoke-real-portraits <n>`, `--smoke-real-out`): no
    /// chat. A character made only from options, as the creator does: `n` portraits and one voice sample while it's
    /// still unsaved, then saved with the first portrait as its profile photo, then deleted. Prints the prompts and
    /// the cost; the pictures and the recording go to the out folder.
    static func realCreator(_ model: AppModel, count: Int, keep: (URL, String) -> Void,
                            say: @escaping (String) -> Void) async -> Bool {
        var ok = true
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        var c = AppModel.Companion(name: "Nora", age: 29, gender: "woman")
        for (cat, opt) in [("occupation", "photographer"), ("hairColor", "black"), ("hairStyle", "curly"), ("eyes", "hazel"),
                           ("skin", "olive"), ("body", "athletic"), ("clothes", "streetwear"), ("extras", "glasses"),
                           ("extras", "tattoos"), ("traits", "playful"), ("traits", "confident"), ("voiceTone", "husky"),
                           ("voiceTone", "warm"), ("voicePitch", "low"), ("voicePace", "relaxed"), ("accent", "british")] {
            c.traits = CharacterOptions.toggle(c.traits, category: cat, option: opt)
        }
        say("portrait prompt: \(CharacterArt.portraitPrompt(c))")
        say("voice: \(c.fullVoice)")
        let start = Date()
        var portraits: [String] = []
        do {
            portraits = try await model.generatePortraits(for: c, count: count, progress: { say("portrait: \($0)") })
        } catch { say("portrait failed: \(error)") }
        say("portraits took \(Int(Date().timeIntervalSince(start))) s")
        check(portraits.count == count && model.characterStore.load(c.id) == nil, "\(count) portraits of the unsaved character")
        for (i, name) in portraits.enumerated() {
            keep(model.characterStore.portraitsFolder(c.id).appendingPathComponent(name), "creator-portrait-\(i + 1).png")
        }
        let voiceStart = Date()
        var sample: String?
        do { sample = try await model.makeVoiceSample(for: c, progress: { say("voice: \($0)") }).name } catch { say("voice failed: \(error)") }
        say("voice sample took \(Int(Date().timeIntervalSince(voiceStart))) s")
        check(sample != nil, "a voice sample of the unsaved character")
        if let sample { keep(model.characterStore.voiceFolder(c.id).appendingPathComponent(sample), "creator-voice.wav") }
        c.portraits = portraits
        c.selectedPortrait = portraits.first
        c.voiceSamples = sample.map { [$0] } ?? []
        c.selectedVoiceSample = sample
        say("voice description: \(CharacterMedia.voiceDescription(c))")
        check(model.saveCharacter(c).isEmpty && (count == 0 || model.portraitURL(c) != nil) && model.voiceSampleURL(c) != nil,
              "saved with its profile photo and voice")
        say("spent $\(String(format: "%.3f", model.characterStore.memory(c.id).spentUSD))")
        model.deleteCharacter(c.id)
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }

    /// `--smoke-characters-real <model> --smoke-real-details`: no media. Two Surprise Me rolls (a woman and a man),
    /// each with its details written by the model; prints the picks and what it wrote, and the time and cost.
    static func realDetails(_ model: AppModel, key: LLMKey, say: @escaping (String) -> Void) async -> Bool {
        var ok = true
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        var total = 0.0
        for gender in ["woman", "man"] {
            let rolled = CharacterSurprise.roll(AppModel.Companion(name: "", age: 25, gender: gender))
            say("rolled \(rolled.name), \(rolled.age), \(rolled.gender)")
            say("  look: \(rolled.fullLook)")
            say("  personality: \(rolled.fullPersonality) | texting: \(rolled.fullSpeakingStyle)")
            say("  backstory: \(rolled.fullBackstory) | habits: \(rolled.habits) | likes: \(rolled.fullLikes)")
            let start = Date()
            do {
                let (d, cost) = try await model.writeCharacterDetails(rolled, newName: true, model: key) { say("  model: \($0)") }
                total += cost
                let c = d.apply(to: rolled, name: true)
                say("  took \(Int(Date().timeIntervalSince(start))) s, $\(String(format: "%.3f", cost))")
                for (label, text) in [("name", c.name), ("look detail", c.look), ("personality", c.personality),
                                      ("backstory", c.backstory), ("texting", c.speakingStyle), ("likes", c.likes),
                                      ("first message", c.greeting)] {
                    say("  \(label): \(text)")
                }
                check(![d.personality, d.backstory, d.likes, d.greeting].contains { ($0 ?? "").isEmpty },
                      "every field is written for the \(gender)")
                let guardProblems = AgeGuard.problems(c)
                check(guardProblems.isEmpty, "nothing in it trips the age guard \(guardProblems)")
            } catch {
                check(false, "details for the \(gender): \(error)")
            }
        }
        say("spent $\(String(format: "%.3f", total))")
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }

    /// `--smoke-characters-real <any> --smoke-real-option-images --smoke-real-out <folder>` (+ `--smoke-real-only
    /// <category,…>`, `--smoke-real-files <file pattern,…>` such as `body-*-woman.jpg`, `--smoke-real-art minimax`, `--smoke-real-faces scripts/option-models`): no chat. Draws the creator's option example photos (SDXL) into the folder, skipping files already
    /// there, so a run can be resumed or a bad file deleted and remade. Copy the result into Resources/OptionImages.
    static func realOptionImages(_ model: AppModel, server: String, out: URL?, say: @escaping (String) -> Void) async -> Bool {
        guard let out else { say("FAIL --smoke-real-out <folder> is needed"); return false }
        let only = argument("--smoke-real-only").map { Set($0.split(separator: ",").map(String.init)) }
        let files = argument("--smoke-real-files").map { $0.split(separator: ",").map(String.init) }
        let family = argument("--smoke-real-art").flatMap(ArtModel.Family.init(rawValue:)) ?? .sdxl
        var faces: [String: URL] = [:]  // --smoke-real-faces <dir>: woman.png / man.png (scripts/option-models)
        if let dir = argument("--smoke-real-faces") {
            for g in ["woman", "man"] {
                let url = URL(fileURLWithPath: dir).appendingPathComponent("\(g).png")
                if FileManager.default.fileExists(atPath: url.path) { faces[g] = url }
            }
            guard !faces.isEmpty else { say("FAIL no woman.png or man.png in \(dir)"); return false }
        }
        let (made, cost) = await OptionImageMaker.make(model, server: server, out: out, categories: only, files: files,
                                                       family: family, faces: faces, say: say)
        let wanted = CharacterOptions.examples.filter { ex in
            (only?.contains(ex.category) ?? true) && (files.map { patterns in
                let name = CharacterOptions.exampleFile(category: ex.category, option: ex.option, gender: ex.gender)
                return patterns.contains { NSPredicate(format: "SELF LIKE %@", $0).evaluate(with: name) }
            } ?? true)
        }
        let missing = wanted.filter {
            !FileManager.default.fileExists(atPath: out.appendingPathComponent(
                CharacterOptions.exampleFile(category: $0.category, option: $0.option, gender: $0.gender)).path)
        }
        say("made \(made.count), spent $\(String(format: "%.3f", cost)); \(missing.count) of \(wanted.count) missing")
        say(missing.isEmpty ? "ok all passed" : "FAIL some examples are missing")
        return missing.isEmpty
    }

    /// `--smoke-characters-real <any> --smoke-real-builder` (+ `--smoke-real-out`, `--smoke-real-models
    /// model[:fine-tune id],…`, `--smoke-real-steps plain,face-body,heritage`, `--smoke-real-dry`): no chat, no window
    /// (it runs headless, see `SmokeTest.headless`). The Prompt Builder's preview, SFW, one seed per model: a plain
    /// woman, then the same with face and body options changed (and, when asked, with a heritage), to see that the
    /// options show. The default models are every image model the preview draws. A model that fails is reported and the
    /// run goes on. Each picture goes to the out folder as `<model>-<step>.png`, with `<model>-<step>.txt` holding its
    /// prompt, negative, seed and cost. `--smoke-real-dry` writes and checks the prompts only: no GPU, no server.
    static func realBuilderPreviews(_ model: AppModel, out: URL?, say: @escaping (String) -> Void) async -> Bool {
        let defaultModels = "sdxl,sdxlIllustrious:ilustmix-v11-1,sd35,flux,chroma,anima,qwenImage,minimaxImage"
        var models: [(PromptModel, String)] = []
        for item in (argument("--smoke-real-models") ?? defaultModels).split(separator: ",") {
            let parts = item.split(separator: ":", maxSplits: 1).map(String.init)
            guard let m = PromptModel(rawValue: parts[0]), PromptPreview.nodeType(m) != nil else {
                say("FAIL \(item) isn't an image model the preview draws"); return false
            }
            models.append((m, parts.count > 1 ? parts[1] : ""))
        }
        let dry = CommandLine.arguments.contains("--smoke-real-dry")
        // `--smoke-real-loras id[:strength],…`: library LoRAs every preview draws with (their trigger words join the
        // prompt), for the models whose family they fit.
        let loraPicks: [PromptBuilderLoRA] = (argument("--smoke-real-loras") ?? "").split(separator: ",").compactMap { item in
            let parts = item.split(separator: ":", maxSplits: 1).map(String.init)
            guard let e = model.store?.library.item(parts[0]) else { say("FAIL no library LoRA \(parts[0])"); return nil }
            return PromptBuilderLoRA(id: e.id, strength: parts.count > 1 ? Double(parts[1]) ?? 0.8 : 0.8,
                                     triggerWords: e.triggerWords)
        }
        var base = PromptBuilderDraft()
        base.gender = "woman"
        base.age = 28
        base.traits = ["hairColor": ["brown"], "hairStyle": ["long-wavy"], "eyes": ["green"], "body": ["average"],
                       "clothes": ["casual"]]
        base.action = "standing in a park, smiling at the camera"
        base.picks = ["medium": ["candid-phone"], "shot": ["cowboy"], "lighting": ["natural"]]
        base.seed = 424_242
        var changed = base
        changed.traits.merge(["faceShape": ["heart"], "lips": ["full"], "makeup": ["soft-glam"], "body": ["athletic"],
                              "muscle": ["defined"]]) { $1 }
        var heritage = base
        heritage.traits["heritage"] = ["japanese"]
        // `--smoke-real-steps plain,heritage`: only those (default plain and face-body).
        let steps = Set((argument("--smoke-real-steps") ?? "plain,face-body").split(separator: ",").map(String.init))
        let fm = FileManager.default
        if let out { try? fm.createDirectory(at: out, withIntermediateDirectories: true) }
        var failures: [String] = [], total = 0.0, costs: [String] = []
        func fail(_ what: String) { say("FAIL \(what)"); failures.append(what) }
        for (m, fineTune) in models {
            let name = fineTune.isEmpty ? m.rawValue : "\(m.rawValue)-\(fineTune)"
            var spent = 0.0
            for (step, d) in [("plain", base), ("face-body", changed), ("heritage", heritage)] where steps.contains(step) {
                var draft = d
                draft.model = m
                draft.variant = fineTune
                draft.loras = loraPicks
                draft.keepLoRAs(fitting: PromptBuilderDraft.loraFamily(m), library: model.store?.library)
                // The written prompt: not empty, allowed, with a negative where the model takes one.
                guard let wp = draft.written, !wp.prompt.isEmpty else { fail("\(name) \(step): no prompt written"); continue }
                if let problem = draft.problem { fail("\(name) \(step): refused: \(problem)"); continue }
                do { try CharacterArt.check(wp.prompt) } catch { fail("\(name) \(step): \(error)"); continue }
                if m.takesNegative && wp.negative.isEmpty { fail("\(name) \(step): no negative"); continue }
                if dry {
                    say("ok \(name) \(step) written: \(wp.prompt)\(wp.negative.isEmpty ? "" : " | negative: \(wp.negative)")")
                    if !draft.loras.isEmpty {  // the LoRA nodes the job would send
                        let job = try? await model.previewJob(draft, server: "dry")
                        let g = job.flatMap { try? $0.template.bind($0.render.values, optional: Set($0.render.values.keys),
                                                                    remove: $0.render.unusedPictures) }
                        let nodes = g?.nodes.filter { $0.type == "RemoteLoRA" }.map { n -> String in
                            if case .value(.string(let id))? = n.inputs["lora"] { return id }; return "?"
                        } ?? []
                        let wired = g?[ "image"]?.inputs["loras"] != nil
                        say("  loras picked \(draft.loras.map(\.id)) → graph \(nodes), wired \(wired)")
                        if nodes.count != draft.loras.count || !wired { fail("\(name) \(step): the LoRAs aren't in the graph") }
                    }
                    continue
                }
                do {
                    let start = Date()
                    let shot = try await model.previewPrompt(draft) { _ in }
                    let cost = shot.costUSD ?? 0
                    spent += cost
                    let file = model.promptPreviewURL(shot)
                    guard let spread = Self.brightnessSpread(file), spread > 4 else {
                        fail("\(name) \(step): the picture is blank (a safety filter or a broken render?)"); continue
                    }
                    say("ok \(name) \(step) in \(Int(Date().timeIntervalSince(start))) s, $\(String(format: "%.4f", cost)): \(shot.prompt)")
                    if let out {
                        let png = out.appendingPathComponent("\(name)-\(step).png")
                        try? fm.removeItem(at: png)
                        try? fm.copyItem(at: file, to: png)
                        let text = "model: \(m.title)\(fineTune.isEmpty ? "" : " (\(fineTune))")\nseed: \(shot.seed)\n"
                            + "cost: $\(String(format: "%.4f", cost))\n\nprompt:\n\(shot.prompt)\n\nnegative:\n\(shot.negative)\n"
                        try? text.write(to: out.appendingPathComponent("\(name)-\(step).txt"), atomically: true, encoding: .utf8)
                    }
                } catch {
                    fail("\(name) \(step): \(error)")
                }
            }
            total += spent
            if !dry { costs.append("\(name) $\(String(format: "%.3f", spent))") }
        }
        if !dry { say("spent $\(String(format: "%.3f", total)) (\(costs.joined(separator: ", ")))") }
        if failures.isEmpty { say("ok all passed") } else {
            say("FAIL \(failures.count) failed: \(failures.joined(separator: "; "))")
        }
        return failures.isEmpty
    }

    /// The spread (standard deviation, 0–255) of a 32×32 grey copy's brightness: near 0 for a black or flat picture.
    static func brightnessSpread(_ file: URL) -> Double? {
        guard let src = CGImageSourceCreateWithURL(file as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let side = 32
        var pixels = [UInt8](repeating: 0, count: side * side)
        let drawn: Bool = pixels.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        guard drawn else { return nil }
        let values = pixels.map(Double.init)
        let mean = values.reduce(0, +) / Double(values.count)
        return (values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count)).squareRoot()
    }

    /// `--smoke-characters-real <any> --smoke-real-train sdxl|minimax` (+ `--smoke-real-out`, `--smoke-real-train-pictures
    /// <n>` default 20, `--smoke-real-train-steps <n>` default the server's): no chat. A template character on that
    /// model gets one portrait, n training pictures with its face, a trained LoRA (added to the library and to the
    /// character), then the same four photos in new outfits and places before (face only) and after (face + LoRA).
    /// Everything goes to the out folder; the costs are printed. Costs GPU time.
    static func realTraining(_ model: AppModel, family: ArtModel.Family, keep: (URL, String) -> Void,
                             say: @escaping (String) -> Void) async -> Bool {
        var ok = true
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        guard let server = model.connections.gpuServers.first?.id else { say("FAIL no GPU server"); return false }
        let pictures = argument("--smoke-real-train-pictures").flatMap(Int.init) ?? 20
        let steps = argument("--smoke-real-train-steps").flatMap(Int.init) ?? 0
        var c = CharacterTemplates.make(CharacterTemplates.all[0])
        c.artModel = ArtModel(family: family)
        check(model.saveCharacter(c).isEmpty, "\(c.name) is saved on \(family.title)")
        let start = Date()
        do {
            let portraits = try await model.generatePortraits(for: c.id, count: 1) { say("portrait: \($0)") }
            c = model.character(c.id) ?? c
            check(portraits.count == 1 && model.portraitURL(c) != nil, "a portrait is made and chosen")
            if let p = model.portraitURL(c) { keep(p, "portrait.png") }

            try await model.makeTrainingPictures(c, count: pictures) { say("  \($0)") }
            let set = model.trainingSet(c.id)
            check(set.pictures.count == pictures && set.status == .ready, "\(pictures) training pictures are drawn")
            let folder = model.characterStore.trainingFolder(c.id)
            for (i, p) in set.pictures.prefix(8).enumerated() {
                keep(folder.appendingPathComponent(p.file), String(format: "train-%02d.png", i + 1))
            }
            say("pictures spent $\(String(format: "%.3f", set.spentUSD)) in \(Int(Date().timeIntervalSince(start))) s")

            let before = c
            let t0 = Date()
            let id = try await model.trainLoRA(c, steps: steps) { say("  \($0)") }
            c = model.character(c.id) ?? c
            let trained = model.trainingSet(c.id)
            check(c.loras.contains { $0.lora == id } && trained.status == .done, "the LoRA \(id) is trained and on \(c.name)")
            check(model.store?.library.item(id)?.triggerWords == [trained.trigger ?? ""], "its trigger word is in the library")
            say("training took \(Int(Date().timeIntervalSince(t0))) s; all spent $\(String(format: "%.3f", trained.spentUSD))")

            let asks = ["wearing a red evening dress at a rooftop party at night",
                        "hiking on a mountain trail in a yellow rain jacket",
                        "reading in a library, wearing glasses and a grey cardigan",
                        "cooking in a sunny kitchen, wearing an apron"]
            var spent = 0.0
            for (i, ask) in asks.enumerated() {
                for (label, who) in [("before", before), ("after", c)] {
                    let file = FileManager.default.temporaryDirectory.appendingPathComponent("train-\(label)-\(i).png")
                    let prompt = CharacterArt.photoPrompt(who, request: ask, scene: nil, context: model.artContext(who))
                    spent += try await model.drawPhoto(who, prompt: prompt, server: server, to: file, title: "Training check",
                                                       seed: 1000 + i, progress: { _ in }) ?? 0
                    keep(file, "photo-\(i + 1)-\(label).png")
                }
            }
            say("photos spent $\(String(format: "%.3f", spent))")
            // The trained LoRA went into the real Models library: the out folder keeps the photos, the library not it.
            model.store?.library.remove(id)
        } catch {
            say("FAIL \(error)")
            ok = false
        }
        say(ok ? "ok all passed" : "FAIL training check")
        return ok
    }

    /// `--smoke-characters-real <any> --smoke-real-outfit` (+ `--smoke-real-out`): no chat. One portrait of Mia, then
    /// the same outfit ask with the same seed twice: the face from the whole portrait (as before) and from its face
    /// crop. Both photos, the portrait and the crop go to the out folder to compare which one wears the asked outfit.
    static func realOutfit(_ model: AppModel, server: String, keep: (URL, String) -> Void,
                           say: @escaping (String) -> Void) async -> Bool {
        var ok = true
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        var c = CharacterTemplates.make(CharacterTemplates.all[0])
        check(model.saveCharacter(c).isEmpty, "\(c.name) is saved")
        let portraits = (try? await model.generatePortraits(for: c, count: 1, progress: { say("portrait: \($0)") })) ?? []
        c = model.character(c.id) ?? c
        guard let portrait = model.portraitURL(c) else { say("FAIL no portrait"); return false }
        keep(portrait, "outfit-portrait.png")
        let face = await model.faceReference(c)
        check(face != nil && face != portrait, "Vision found the face and cropped it")
        if let face { keep(face, "outfit-face-crop.png") }
        let ask = "wearing a red evening dress at a rooftop party at night, city lights behind her"
        let prompt = CharacterArt.photoPrompt(c, request: ask, scene: nil)
        say("prompt: \(prompt)")
        let seed = 424_242
        for whole in [true, false] {
            let file = model.characterStore.albumFolder(c.id).appendingPathComponent(whole ? "whole.png" : "crop.png")
            do {
                let cost = try await model.drawPhoto(c, prompt: prompt, server: server, to: file,
                                                     title: "outfit test", seed: seed, wholePortrait: whole) { _ in }
                say("\(whole ? "whole portrait" : "face crop"): $\(String(format: "%.3f", cost ?? 0))")
                keep(file, whole ? "outfit-from-whole-portrait.png" : "outfit-from-face-crop.png")
            } catch { check(false, "photo (\(whole ? "whole" : "crop")): \(error)") }
        }
        check(portraits.count == 1, "a portrait was made")
        say("spent $\(String(format: "%.3f", model.characterStore.memory(c.id).spentUSD)) on portraits")
        model.deleteCharacter(c.id)
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }

    /// `--smoke-characters-real <model> --smoke-real-group`: text only (no portraits or media). A group chat with two
    /// template characters for three rounds (both answer, one is named, both again), checking each reply is from the
    /// right character, in character, without a name in front or lines for the other; then a one-to-one chat whose
    /// last message was 9 hours ago, where she writes first.
    static func realGroup(_ model: AppModel, key: LLMKey, say: @escaping (String) -> Void) async -> Bool {
        var ok = true
        func check(_ good: Bool, _ what: String) {
            if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false }
        }
        func make(_ i: Int) -> AppModel.Companion {
            var c = CharacterTemplates.make(CharacterTemplates.all[i])
            c.media = []
            c.initiative = 0
            return c
        }
        let a = make(0), b = make(2)
        check(model.saveCharacter(a).isEmpty && model.saveCharacter(b).isEmpty, "\(a.name) and \(b.name) are saved")
        let banned = ["as an ai", "language model", "<|", "[inst]"]
        func wait(_ doc: ChatDocument, replies: Int, since: Int) async -> [ChatTurn] {
            let start = Date()
            func answers() -> [ChatTurn] { doc.transcript.turns.dropFirst(since).filter { $0.role == .assistant } }
            var phase: String?
            // Done once nothing streams and the answers are in; or nothing has streamed for 3 s (the next one
            // starts within a moment of the last ending, so a quiet spell means the round is over).
            var quietSince: Date?
            while Date().timeIntervalSince(start) < 1200 {
                let now = doc.engine.map { "\($0.state) \($0.phase ?? "")" }
                if let now, now != phase { say("  engine at \(Int(Date().timeIntervalSince(start))) s: \(now)") }
                phase = now ?? phase
                if doc.isStreaming {
                    quietSince = nil
                } else {
                    if quietSince == nil { quietSince = Date() }
                    if answers().count >= replies || answers().contains(where: { $0.error != nil })
                        || Date().timeIntervalSince(quietSince!) > 3 { break }
                }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)  // nobody else starts
            return answers()
        }
        func name(_ id: UUID?) -> String { id.flatMap { model.character($0)?.name } ?? "?" }

        let group = model.newCharacterChat([a.id, b.id], model: key)
        let rounds: [(String, [UUID])] = [
            ("Hi you two! I'm so glad we could all chat. What are you both up to this evening?", [a.id, b.id]),
            ("\(b.name), what do you do for work? I don't think you've told me.", [b.id]),
            ("What should the three of us do together this weekend?", [a.id, b.id]),
        ]
        for (text, expected) in rounds {
            let since = group.transcript.turns.count + 1
            let start = Date()
            model.sendText(group, text)
            let replies = await wait(group, replies: expected.count, since: since)
            say("you: \(text)")
            for r in replies {
                say("\(name(r.speaker)) (\(Int(Date().timeIntervalSince(start)))s): \(r.text)\(r.error.map { " [error: \($0)]" } ?? "")")
            }
            check(replies.map(\.speaker) == expected, "the round goes to \(expected.map(name)) (\(replies.map { name($0.speaker) }))")
            for r in replies {
                let me = name(r.speaker), other = r.speaker == a.id ? b.name : a.name
                let lower = r.text.lowercased()
                check(r.error == nil && !r.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "\(me)'s reply came back")
                check(!banned.contains { lower.contains($0) }, "\(me) stays in character")
                check(!lower.hasPrefix(me.lowercased() + ":") && !lower.hasPrefix("**" + me.lowercased()),
                      "\(me)'s reply has no name in front")
                check(!r.text.contains("\(other):") && !r.text.contains("\n\(other) "), "\(me) doesn't write \(other)'s lines")
            }
        }

        // Writing first, one to one.
        let solo = model.newCharacterChat([a.id], model: key)
        let since = solo.transcript.turns.count + 1
        model.sendText(solo, "I'm off to bed, talk tomorrow? I have a big presentation in the morning.")
        let night = await wait(solo, replies: 1, since: since)
        say("\(a.name): \(night.first?.text ?? "none")")
        if let last = solo.transcript.turns.last {
            let past = Date().addingTimeInterval(-9 * 3600)
            solo.transcript.update(last.id) { $0.date = past; $0.ended = past }
        }
        let before = solo.transcript.turns.count
        model.characterCheckInIfDue(solo)
        let first = await wait(solo, replies: 1, since: before)
        say("\(a.name) writes first: \(first.first?.text ?? "none")\(first.first?.error.map { " [error: \($0)]" } ?? "")")
        check(first.count == 1 && first.first?.speaker == a.id && first.first?.error == nil
              && !(first.first?.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "she writes first after 9 hours")
        check(!(first.first?.text ?? "").contains("note") && !(first.first?.text ?? "").contains("["),
              "her message doesn't mention the note")

        let spent = model.characterStore.memory(a.id).spentUSD + model.characterStore.memory(b.id).spentUSD
        say("memory passes spent $\(String(format: "%.3f", spent)); chat replies: see the server's costs")
        let cost = (group.transcript.costUSD + solo.transcript.costUSD)
        say("the chats' GPU time cost $\(String(format: "%.3f", cost))")
        model.deleteChat(group.id)
        model.deleteChat(solo.id)
        model.deleteCharacter(a.id)
        model.deleteCharacter(b.id)
        say(ok ? "ok all passed" : "FAIL some checks failed")
        return ok
    }
}
