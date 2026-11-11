import AVFoundation
import Agent
import AppKit
import Foundation
import LLMClient
import SwiftUI

extension SmokeTest {
    /// `--smoke-chat <url> [--smoke-token-file <file>] [--llm <link or id>] [--prompt <text>]`: adds the GPU server,
    /// installs the model if it isn't there (default `fake/tiny`, for a server run with `AISTUDIO_FAKE_LLM=1`), opens a
    /// chat, sends two messages and checks the replies, their stats, the saved transcript, and the engine's state.
    static func runChat(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-chat] \(s)"); fflush(stdout) }
        var token = "smoke"  // a local server without auth takes any token
        if let path = argument("--smoke-token-file") {
            token = (try? String(contentsOfFile: path, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        }
        let link = argument("--llm") ?? "fake/tiny"
        let prompt = argument("--prompt") ?? "hello there"
        let fake = link.hasPrefix("fake/")
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else {
            say("server not reachable: \(model.connections.status[conn.id].map { "\($0)" } ?? "?")"); return false
        }
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        if let e = llm.errors[conn.id] { say("listing failed: \(e)"); return false }

        // The model: installed already (by id), else installed from the link.
        var mid = llm.listings[conn.id]?.models.first { $0.id == link || $0.repo == link }?.id
        if mid == nil {
            do {
                let found = try await client.inspect(url: link)
                say("\(found.facts.repo): \(found.facts.modelType), \(found.facts.weightBytes / 1_000_000_000) GB, "
                    + "recommended \(found.plan.recommended?.label ?? "none")")
                let st = try await client.install(LLMInstallRequest(url: link))
                mid = st.id ?? found.suggestedId
            } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            let deadline = Date().addingTimeInterval(fake ? 30 : 6 * 3600)
            while Date() < deadline {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                if let st = llm.listings[conn.id]?.installs[mid!], st.state == "failed" {
                    say("install failed: \(st.error ?? "?")"); return false
                }
                try? await Task.sleep(nanoseconds: fake ? 200_000_000 : 10_000_000_000)
            }
        }
        guard let mid, llm.model(LLMKey(serverID: conn.id, modelID: mid))?.installed == true else {
            say("model didn't install"); return false
        }
        let key = LLMKey(serverID: conn.id, modelID: mid)
        say("model \(mid) on \(llm.model(key)!.gpuLabel)")

        // Two messages: the second checks the history goes along.
        let doc = model.newChat(model: key)
        var ok = true
        // A real model may ask a question card at any step: answered with each question's first option (and said),
        // except in the live run's own question step.
        let answerer = fake ? nil : Task { @MainActor in
            while !Task.isCancelled {
                if let q = doc.question, !liveOwnsQuestions {
                    say("(it asked: \(q.questions.map { "\($0.label): \($0.options.map(\.label).joined(separator: " | "))" }.joined(separator: "; ")) — answering the first options)")
                    doc.answerQuestion(q.questions.map { UserAnswer(picked: [$0.options[0].label]) })
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
        defer { answerer?.cancel() }
        for (n, text) in [prompt, "and once more"].enumerated() {
            doc.draft = text
            model.send(doc)
            let start = Date()
            while doc.isStreaming, Date().timeIntervalSince(start) < (fake ? 30 : 3600) {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            guard let reply = doc.transcript.turns.last, reply.role == .assistant else { say("no reply"); return false }
            if let e = reply.error { say("reply \(n + 1) failed: \(e)"); ok = false; continue }
            let s = reply.stats
            say("reply \(n + 1) in \(String(format: "%.1f", Date().timeIntervalSince(start))) s: \(reply.text.prefix(200))")
            say("  thinking \(reply.reasoning?.count ?? 0) chars, \(s?.completionTokens ?? 0) tokens, "
                + "\(s?.tokPerSec.map { String(format: "%.0f tok/s", $0) } ?? "? tok/s"), "
                + "first token \(s?.ttftMs.map { String(format: "%.1f s", $0 / 1000) } ?? "?"), "
                + "\(s?.costUSD.map { String(format: "$%.4f", $0) } ?? "$?")")
            let tools = doc.transcript.turns.suffix(from: max(0, doc.transcript.turns.lastIndex { $0.role == .user } ?? 0))
                .flatMap(\.toolCalls).map(\.name)
            if !tools.isEmpty { say("  tools: \(tools.joined(separator: ", "))") }
            if reply.text.isEmpty { say("empty reply"); ok = false }
            if s == nil { say("no stats"); ok = false }
            if fake {
                if reply.text != "You said: \(text)" { say("unexpected reply"); ok = false }
                if reply.reasoning != "Thinking about it." { say("thinking not split out"); ok = false }
            }
        }
        if doc.title != ChatTranscript.title(from: prompt) { say("title is \(doc.title)"); ok = false }
        guard let saved = model.chatStore.load(doc.id), saved.turns.count == 4 else {
            say("transcript not saved"); return false
        }
        if !model.chatSummaries.contains(where: { $0.id == doc.id }) { say("not in the Chats panel"); ok = false }
        await llm.refreshStatus(key)
        say("engine: \(llm.status(key)?.state ?? "?") \(llm.status(key)?.gpu ?? "") uptime "
            + "\(Int(llm.status(key)?.uptime ?? 0)) s, \(String(format: "$%.3f", llm.status(key)?.costUSD ?? 0))")
        if llm.status(key)?.state != "ready" { say("engine not ready after the replies"); ok = false }
        // M28: the start was timed, phase by phase.
        if let cs = llm.status(key)?.coldStart {
            say("cold start: \(ColdStartRecord.duration(cs.last.seconds)), typical \(ColdStartRecord.duration(cs.typicalSeconds)): \(cs.last.breakdown)")
            if cs.last.phases.isEmpty || cs.last.seconds <= 0 { say("the cold start has no phases"); ok = false }
        } else {
            say("no cold start recorded"); ok = false
        }
        // The start-up settings round-trip (fast loading; a snapshot only on one GPU).
        if fake, let c = llm.client(key.serverID), let m = llm.model(key) {
            do {
                let one = m.count == 1
                let updated = try await c.update(key.modelID, ["loadFormat": .string("runai"), "snapshot": .bool(one)])
                if updated.loadFormat != "runai" || updated.snapshot != one { say("start-up settings not saved"); ok = false }
                _ = try await c.update(key.modelID, ["loadFormat": .string("auto"), "snapshot": .bool(false)])
                if !one {
                    do {
                        _ = try await c.update(key.modelID, ["snapshot": .bool(true)])
                        say("a snapshot on \(m.count) GPUs was accepted"); ok = false
                    } catch {}
                }
            } catch {
                say("start-up settings: \(error.localizedDescription)"); ok = false
            }
        }

        // Branches: another first reply, then back; an edited first question; attachments.
        func waitReply() async -> ChatTurn? {
            let start = Date()
            while doc.isStreaming, Date().timeIntervalSince(start) < (fake ? 30 : 3600) {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return doc.transcript.turns.last
        }
        let firstReply = doc.transcript.turns[1]
        model.regenerate(doc, firstReply.id)
        let retry = await waitReply()
        if doc.transcript.turns.count != 2 || doc.transcript.siblings(of: firstReply.id).count != 2 || retry?.error != nil {
            say("regenerate didn't make a second version"); ok = false
        }
        doc.transcript.select(firstReply.id)
        if doc.transcript.turns.count != 4 { say("going back to the first version lost the rest"); ok = false }
        model.resend(doc, editing: doc.transcript.turns[0].id, text: "an edited question")
        let edited = await waitReply()
        if fake, edited?.text != "You said: an edited question" { say("edit: \(edited?.text ?? "no reply")"); ok = false }
        if doc.transcript.siblings(of: doc.transcript.turns[0].id).count != 2 { say("edit didn't branch"); ok = false }
        let tmp = FileManager.default.temporaryDirectory
        let note = tmp.appendingPathComponent("smoke-note.txt")
        try? "the note says 42".write(to: note, atomically: true, encoding: .utf8)
        let png = tmp.appendingPathComponent("smoke-attach.png")
        if let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2400, pixelsHigh: 1200, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0) {
            try? rep.representation(using: .png, properties: [:])?.write(to: png)
        }
        if let why = model.attach([note, png], to: doc) { say("attach: \(why)"); ok = false }
        if case .image(let name)? = doc.pending.first(where: { if case .image = $0 { return true }; return false }),
           let img = NSImage(contentsOf: model.chatStore.filesFolder(doc.id).appendingPathComponent(name)),
           let rep = img.representations.first, max(rep.pixelsWide, rep.pixelsHigh) <= 1536 {
        } else { say("the image wasn't saved at most 1536 px"); ok = false }
        doc.draft = "with attachments"
        model.send(doc)
        let withFiles = await waitReply()
        if fake, withFiles?.text.contains("the note says 42") != true { say("the file didn't reach the model"); ok = false }
        if doc.transcript.turns.dropLast().last?.images.count != 1 { say("the image isn't on the message"); ok = false }
        // A video (models with vision): made smaller on this Mac, uploaded, and read by the engine from the uploads.
        if llm.model(key)?.vision == true {
            let mov = tmp.appendingPathComponent("smoke-clip.mov")
            if await writeTestVideo(to: mov) {
                model.attach([mov], to: doc)
                for _ in 0..<600 where doc.preparingVideos > 0 { try? await Task.sleep(for: .milliseconds(50)) }
                let video = doc.pending.compactMap { if case .image(let n) = $0, ChatTurn.isVideo(n) { return n }; return nil }.first
                if let video {
                    let file = model.chatStore.filesFolder(doc.id).appendingPathComponent(video)
                    if !FileManager.default.fileExists(atPath: AppModel.posterURL(file).path) { say("no poster frame"); ok = false }
                    doc.draft = "what happens in it?"
                    model.send(doc)
                    let watched = await waitReply()
                    if !fake { say("video reply: \((watched?.text.isEmpty == false ? watched?.text : watched?.error) ?? "none")") }
                    if fake, watched?.text.hasSuffix("[1 video]") != true {
                        say("the video didn't reach the engine: \(watched?.text ?? watched?.error ?? "no reply")"); ok = false
                    }
                    say("video: \(video.prefix(12))… \((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { "\($0 / 1024) KB" } ?? "?")")
                } else {
                    say("video not attached: \(doc.attachNote ?? "?")"); ok = false
                }
            } else {
                say("couldn't write a test video"); ok = false
            }
        }
        if fake { ok = await runInteractive(model, doc: doc, waitReply: waitReply, say: say) && ok }
        else if CommandLine.arguments.contains("--interactive") { ok = await runLive(model, doc: doc, waitReply: waitReply, say: say) && ok }
        let shown = doc.transcript.turns.count
        say("branches and attachments: \(doc.transcript.nodes.count) turns, \(shown) shown")

        // Closing and reopening the tab brings the conversation back, on the same branch.
        model.closeChat(doc.id)
        model.openChat(doc.id)
        if model.activeChat?.transcript.turns.count != shown { say("reopened chat lost turns"); ok = false }
        model.closeChat(doc.id)

        if fake {
            await llm.stop(key)
            if llm.status(key)?.state != "cold" { say("stop didn't stop it"); ok = false }
            model.deleteChat(doc.id)
            await llm.remove(key)
            if llm.model(key) != nil { say("model not removed"); ok = false }
        } else {
            say("left \(mid) running: it stops by itself after the idle time (or Shut Down GPU in Servers)")
        }
        say(ok ? "OK" : "FAILED")
        return ok
    }
}

extension SmokeTest {
    /// `--smoke-agent <url>` (a server with `AISTUDIO_FAKE_LLM=1`): a coding chat in a fresh git repository. The fake
    /// model writes hello.txt, then runs `cat hello.txt`: the write is allowed once, the command "always"; the file and
    /// the reply are checked. Then plan mode: the same request is refused without asking, and nothing is written.
    static func runAgent(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-agent] \(s)"); fflush(stdout) }
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/agent" }?.id
        if mid == nil {
            do { mid = try await client.install(LLMInstallRequest(url: "fake/agent")).id } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            for _ in 0..<100 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        guard let mid else { say("no model"); return false }
        let key = LLMKey(serverID: conn.id, modelID: mid)

        let repo = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-agent-\(UUID().uuidString.prefix(8))")
        try? FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        _ = await ShellRunner().run("git init -q && echo '# Smoke' > README.md", cwd: repo)
        let doc = model.newCodingChat(project: repo, model: key)
        var ok = true
        var asked: [String] = []

        // Answers questions as they come: allow the write, always allow the command.
        let answerer = Task { @MainActor in
            while !Task.isCancelled {
                if let r = doc.permission {
                    asked.append(r.call.name)
                    doc.answer(r.call.name == "write_file" ? .allow : .allowAlways)
                }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        func send(_ text: String) async -> ChatTurn? {
            doc.draft = text
            model.send(doc)
            let start = Date()
            while doc.isStreaming, Date().timeIntervalSince(start) < 60 { try? await Task.sleep(nanoseconds: 100_000_000) }
            return doc.transcript.turns.last
        }

        let reply = await send("create hello.txt and show it")
        let written = try? String(contentsOf: repo.appendingPathComponent("hello.txt"), encoding: .utf8)
        say("reply: \(reply?.text ?? reply?.error ?? "none")")
        if written != "hi from the agent\n" { say("hello.txt is \(written.map { "\"\($0)\"" } ?? "missing")"); ok = false }
        if asked != ["write_file", "run_command"] { say("asked for \(asked)"); ok = false }
        if reply?.text.contains("hi from the agent") != true { say("the command's output didn't come back"); ok = false }
        let tools = doc.transcript.turns.filter { $0.role == .tool }
        if tools.map(\.toolStatus) != ["done", "done"] { say("tool turns: \(tools.map { "\($0.toolTitle ?? "?"): \($0.toolStatus ?? "?")" })"); ok = false }
        if tools.first?.toolDetail?.contains("+hi from the agent") != true { say("no diff on the write card"); ok = false }
        if doc.transcript.allowCommands != ["cat"] { say("always-allow not kept: \(doc.transcript.allowCommands ?? [])"); ok = false }

        // Rewind to the first message: hello.txt (which the agent created) goes away, the message goes back in the
        // field and the chat is shown from before it.
        if let first = doc.transcript.turns.first(where: { $0.role == .user }) {
            model.prepareRewind(doc, to: first.id)
            for _ in 0..<50 where doc.rewindFiles == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
            let listed = doc.rewindFiles?.map(\.lastPathComponent) ?? []
            if listed != ["hello.txt"] { say("rewind lists \(listed)"); ok = false }
            model.rewind(doc, files: true, chat: true)
            let hello = repo.appendingPathComponent("hello.txt").path
            for _ in 0..<50 where FileManager.default.fileExists(atPath: hello) { try? await Task.sleep(nanoseconds: 100_000_000) }
            if FileManager.default.fileExists(atPath: hello) { say("rewind left hello.txt"); ok = false }
            if doc.draft != first.text || !doc.transcript.turns.isEmpty {
                say("rewind: draft \"\(doc.draft)\", \(doc.transcript.turns.count) turns shown"); ok = false
            }
            say("rewind: \(ok ? "files and chat restored" : "failed")")
        } else { say("no user turn to rewind to"); ok = false }

        // Plan mode: refused without a question; the file stays as it is.
        try? FileManager.default.removeItem(at: repo.appendingPathComponent("hello.txt"))
        asked = []
        doc.mode = .plan
        let planned = await send("create hello.txt and show it")
        let planTools = doc.transcript.turns.filter { $0.role == .tool }.suffix(2)
        if !asked.isEmpty { say("plan mode asked: \(asked)"); ok = false }
        if FileManager.default.fileExists(atPath: repo.appendingPathComponent("hello.txt").path) { say("plan mode wrote"); ok = false }
        if planTools.map(\.toolStatus) != ["denied", "denied"] { say("plan tools: \(planTools.map(\.toolStatus))"); ok = false }
        say("plan mode reply: \(planned?.text.prefix(80) ?? "none")")
        answerer.cancel()

        // Saved with its folder, mode and answers.
        if let saved = model.chatStore.load(doc.id) {
            if saved.projectPath != repo.standardizedFileURL.path || saved.permissionMode != "plan" { say("not saved as a coding chat"); ok = false }
        } else { say("not saved"); ok = false }
        model.closeChat(doc.id)
        model.deleteChat(doc.id)
        try? FileManager.default.removeItem(at: repo)
        say(ok ? "OK" : "FAILED")
        return ok
    }
}

extension SmokeTest {
    /// `--smoke-graph-helper <url>` (a server with `AISTUDIO_FAKE_LLM=1`): the active tab's graph helper is asked to
    /// "add an upscale after Save". The fake model reads the graph, then puts an ImageScale between VAE Decode and
    /// Save Image: the edit is checked, then one undo must bring the original graph back. Then plan mode: the edit is
    /// refused and the graph doesn't change.
    static func runGraphHelper(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-graph-helper] \(s)"); fflush(stdout) }
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/helper" }?.id
        if mid == nil {
            do { mid = try await client.install(LLMInstallRequest(url: "fake/helper")).id } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            for _ in 0..<100 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        guard let mid else { say("no model"); return false }

        let wf = model.active
        let original = wf.graph
        guard original["decode"]?.type == "VAEDecode", original["save"]?.type == "SaveImage" else {
            say("the first tab isn't the text-to-image template: \(original.nodes.map(\.id))"); return false
        }
        model.showHelper = true
        let chat = model.ensureHelper(wf)
        model.setModel(LLMKey(serverID: conn.id, modelID: mid), for: chat)
        var ok = true
        func send(_ text: String) async -> ChatTurn? {
            chat.draft = text
            model.send(chat)
            let start = Date()
            while chat.isStreaming, Date().timeIntervalSince(start) < 60 { try? await Task.sleep(nanoseconds: 100_000_000) }
            return chat.transcript.turns.last
        }

        let reply = await send("Add an upscale after Save")
        say("reply: \(reply?.text ?? reply?.error ?? "none")")
        let g = wf.graph
        let up = g["upscale"]
        if up?.type != "ImageScale" { say("no ImageScale node: \(g.nodes.map(\.id))"); ok = false }
        if up?.inputs["image"] != .link(node: "decode", output: "image") { say("upscale.image isn't decode.image"); ok = false }
        if up?.inputs["factor"] != .value(.double(2)) { say("factor: \(String(describing: up?.inputs["factor"]))"); ok = false }
        if g["save"]?.inputs["image"] != .link(node: "upscale", output: "image") { say("save.image isn't upscale.image"); ok = false }
        if up?.position == nil { say("the new node wasn't placed"); ok = false }
        if model.validationError(of: g) != nil { say("not valid: \(model.validationError(of: g)!)"); ok = false }
        let tools = chat.transcript.turns.filter { $0.role == .tool }
        if tools.map(\.toolStatus) != ["done", "done"] { say("tool turns: \(tools.map { "\($0.toolTitle ?? "?"): \($0.toolStatus ?? "?")" })"); ok = false }
        if reply?.text.hasPrefix("Done: added upscale (ImageScale)") != true { say("the edit's result didn't come back"); ok = false }
        if wf.undoManager.undoActionName != GraphHelper.undoName { say("undo step: \(wf.undoManager.undoActionName)"); ok = false }
        if chat.lastHelperEdit?.changes.count != 2 { say("changes: \(chat.lastHelperEdit?.changes ?? [])"); ok = false }

        // One undo step brings it all back.
        model.undoHelperEdit(chat)
        if wf.graph != original { say("undo didn't restore the graph: \(wf.graph.nodes.map(\.id))"); ok = false }
        if !wf.undoManager.canRedo { say("nothing to redo"); ok = false }

        // Plan mode: it reads the graph, the edit is refused, nothing changes.
        chat.mode = .plan
        _ = await send("Add an upscale after Save")
        let planTools = chat.transcript.turns.filter { $0.role == .tool }.suffix(2)
        if planTools.map(\.toolStatus) != ["done", "denied"] { say("plan tools: \(planTools.map(\.toolStatus))"); ok = false }
        if wf.graph != original { say("plan mode changed the graph"); ok = false }

        // Saved as this tab's helper, with the session.
        if model.chatStore.load(chat.id)?.graphHelper != true { say("not saved as a graph helper"); ok = false }
        if model.sessionManifest().tabs.first(where: { $0.id == wf.id })?.helperChat != chat.id { say("the tab doesn't keep its helper"); ok = false }
        model.closeHelper(of: wf)
        model.deleteChat(chat.id)
        say(ok ? "OK" : "FAILED")
        return ok
    }

    /// Two seconds of a moving square (H.264, 1280×720, 10 fps).
    static func writeTestVideo(to url: URL) async -> Bool {
        try? FileManager.default.removeItem(at: url)
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return false }
        let (w, h) = (1280, 720)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h])
        writer.add(input)
        guard writer.startWriting() else { return false }
        writer.startSession(atSourceTime: .zero)
        for i in 0..<20 {
            while !input.isReadyForMoreMediaData { try? await Task.sleep(for: .milliseconds(5)) }
            guard let pool = adaptor.pixelBufferPool else { return false }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { return false }
            CVPixelBufferLockBaseAddress(buffer, [])
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let row = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<h {
                for x in 0..<w {
                    let inside = abs(x - 100 - i * 50) < 80 && abs(y - h / 2) < 80
                    let px = base + y * row + x * 4
                    px[0] = inside ? 40 : 200; px[1] = inside ? 40 : 120; px[2] = inside ? 230 : 30; px[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 10))
        }
        input.markAsFinished()
        await writer.finishWriting()
        return writer.status == .completed
    }

    /// The chat tools with the fake engine: a question card answered by clicking and by typing, a widget (its page
    /// in a real web view: height, state, fetch blocked), follow-up chips, and a quoted reply.
    static func runInteractive(_ model: AppModel, doc: ChatDocument, waitReply: () async -> ChatTurn?,
                               say: (String) -> Void) async -> Bool {
        var ok = true
        func waitQuestion() async -> PendingQuestion? {
            for _ in 0..<300 where doc.question == nil && doc.isStreaming { try? await Task.sleep(for: .milliseconds(50)) }
            return doc.question
        }
        // A question, answered on its card.
        model.sendText(doc, "ask me which database")
        if let q = await waitQuestion() {
            if q.questions.first?.label != "Database" { say("question: \(q.questions)"); ok = false }
            doc.answerQuestion([UserAnswer(picked: ["Postgres"])])
            let reply = await waitReply()
            if reply?.text != "Done: The user answered: Database: Postgres" { say("answered: \(reply?.text ?? "none")"); ok = false }
            let card = doc.transcript.turns.last { $0.role == .tool }
            if card?.toolDetail != "Database: Postgres" || card?.toolStatus != "done" {
                say("question card: \(card?.toolDetail ?? "none") \(card?.toolStatus ?? "")"); ok = false
            }
        } else { say("no question card"); ok = false }
        // Answered by typing in the composer: "Other".
        model.sendText(doc, "ask me again")
        if await waitQuestion() != nil {
            doc.draft = "a flat file"
            model.send(doc)
            let reply = await waitReply()
            if reply?.text.contains("Other: \"a flat file\"") != true { say("typed answer: \(reply?.text ?? "none")"); ok = false }
        } else { say("no second question card"); ok = false }
        // A widget: the call, and its page in a web view (the real one, offscreen).
        model.sendText(doc, "show a widget")
        let widget = await waitReply()
        let call = doc.transcript.turns.flatMap(\.toolCalls).last { $0.name == "show_widget" }
        if call == nil || widget?.text.hasPrefix("Done: Shown to the user") != true {
            say("widget: \(widget?.text ?? "none")"); ok = false
        }
        let page = await probeWidget("""
            <p id="x">probe</p><script>
            fetch("https://example.com/").then(() => aistudio.reportState({fetch: "allowed"}),
                                               () => aistudio.reportState({fetch: "blocked"}));
            </script>
            """)
        if page.height == nil || page.height! >= 300 { say("the widget's height: \(page.height.map { "\($0)" } ?? "none")"); ok = false }
        if page.state != "{\"fetch\":\"blocked\"}" { say("widget fetch: \(page.state ?? "no state")"); ok = false }
        // Its state goes with the next message.
        if let id = call?.id { doc.widgetStates[id] = "{\"strength\":0.5}" }
        // Follow-ups.
        model.sendText(doc, "suggest something")
        let suggested = await waitReply()
        if suggested?.followUps != ["Tell me more", "Show an example"] || suggested?.text.hasPrefix("You said: suggest something") != true || suggested?.text.contains("<followups") == true {
            say("follow-ups: \(suggested?.followUps ?? []) in \(suggested?.text ?? "none")"); ok = false
        }
        let sentTurn = doc.transcript.turns.last { $0.role == .user }
        if sentTurn?.context?.contains("[Widget \"Strength\" state: {\"strength\":0.5}]") != true {
            say("widget state not sent: \(sentTurn?.context ?? "none")"); ok = false
        }
        // A quoted reply.
        if let s = suggested { model.quote(doc, "You said", from: s) }
        doc.draft = "why that?"
        model.send(doc)
        let quoted = await waitReply()
        if quoted?.text.contains("> You said") != true || doc.transcript.turns.dropLast().last?.quote == nil {
            say("quote: \(quoted?.text ?? "none")"); ok = false
        }
        say("interactive: question cards, widget (height \(page.height.map { "\(Int($0))" } ?? "?"), \(page.state ?? "-")), follow-ups, quote")
        return ok
    }

    /// The live run is answering a question card itself (the run's answerer leaves it alone).
    @MainActor static var liveOwnsQuestions = false

    /// `--interactive` with a real model: what it does with the chat tools. Reports more than it checks (a model may
    /// word things its own way): a question card (answered with each question's first option), a widget (its page
    /// loaded in a real web view), follow-ups and a quoted reply.
    static func runLive(_ model: AppModel, doc: ChatDocument, waitReply: () async -> ChatTurn?,
                        say: (String) -> Void) async -> Bool {
        var ok = true
        liveOwnsQuestions = true
        defer { liveOwnsQuestions = false }
        func clip(_ s: String?, _ n: Int = 300) -> String { (s ?? "none").replacingOccurrences(of: "\n", with: " ").prefix(n).description }
        // A question card.
        model.sendText(doc, "I want to build a small note-taking app. Before suggesting anything, ask me which platform and which storage I prefer.")
        for _ in 0..<6000 where doc.question == nil && doc.isStreaming { try? await Task.sleep(for: .milliseconds(100)) }
        if let q = doc.question {
            for x in q.questions { say("asked: \(x.label): \(x.question) [\(x.options.map(\.label).joined(separator: " | "))]\(x.multiSelect ? " (multi)" : "")") }
            doc.answerQuestion(q.questions.map { UserAnswer(picked: [$0.options[0].label]) })
            let r = await waitReply()
            say("after the answer: \(clip(r?.text)) · follow-ups: \(r?.followUps ?? [])")
        } else {
            let r = doc.transcript.turns.last
            say("no question card; it said: \(clip(r?.text)) \(r?.error ?? "")"); ok = false
        }
        liveOwnsQuestions = false  // later questions: the answerer
        // A widget.
        model.sendText(doc, "Show me an interactive widget: a slider for a LoRA strength from 0 to 1 and a Chart.js bar chart (from a CDN) of how much of the style is kept at a few strengths, with the current one highlighted. Add a button that asks you to explain the current value.")
        let w = await waitReply()
        if let call = doc.transcript.turns.flatMap(\.toolCalls).last(where: { $0.name == "show_widget" }),
           let args = try? ToolArgs(json: call.arguments), let html = args.string("html") {
            say("widget \"\(args.string("title") ?? "?")\": \(html.count) chars, CDN: \(WidgetPage.cdnHosts.filter { html.contains($0) }), "
                + "sendPrompt: \(html.contains("sendPrompt")), reportState: \(html.contains("reportState")), theme vars: \(html.contains("--mg-"))")
            let probe = await probeWidget(html + """
                <script>addEventListener("load", function () { setTimeout(function () {
                  aistudio.reportState({ chart: typeof Chart, canvases: document.querySelectorAll("canvas").length,
                                         errors: (window.__errs || []).slice(0, 3) }); }, 1500); });
                </script>
                """, errorsFirst: true)
            say("widget page: height \(probe.height.map { "\(Int($0))" } ?? "none"), \(probe.state ?? "no state")")
            if probe.height == nil { ok = false }
            say("then: \(clip(w?.text, 200))")
        } else {
            say("no widget; it said: \(clip(w?.text)) \(w?.error ?? "")"); ok = false
        }
        // A quoted reply.
        if let last = doc.transcript.turns.last(where: { $0.role == .assistant && !$0.text.isEmpty }) {
            let passage = String(last.text.prefix(80))
            model.quote(doc, passage, from: last)
            doc.draft = "Explain this part in one sentence."
            model.send(doc)
            let r = await waitReply()
            say("quote \"\(clip(passage, 60))\" → \(clip(r?.text, 200))")
        }
        return ok
    }

    /// Loads a widget page in `WidgetWebView` in an offscreen window; returns its first height and state reports.
    /// `errorsFirst` collects the page's script errors in `window.__errs` (for the live run to report).
    static func probeWidget(_ html: String, errorsFirst: Bool = false) async -> (height: CGFloat?, state: String?) {
        final class Box { var height: CGFloat?; var state: String? }
        let box = Box()
        let page = errorsFirst ? """
            <script>window.__errs = []; addEventListener("error", function (e) {
              window.__errs.push(String(e.message || (e.target && (e.target.src || e.target.href)) || e)); }, true);</script>
            """ + html : html
        let view = WidgetWebView(html: page, title: "probe", onHeight: { box.height = $0 }, onState: { box.state = $0 })
        let window = NSWindow(contentRect: NSRect(x: -3000, y: -3000, width: 600, height: 300), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.contentView = NSHostingView(rootView: view.frame(width: 600, height: 300))
        window.orderFrontRegardless()
        for _ in 0..<(errorsFirst ? 400 : 200) where box.height == nil || box.state == nil || (errorsFirst && !(box.state ?? "").contains("canvases")) {
            try? await Task.sleep(for: .milliseconds(50))
        }
        window.orderOut(nil)
        return (box.height, box.state)
    }
}
