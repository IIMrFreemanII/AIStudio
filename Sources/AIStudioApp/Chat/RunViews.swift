import LLMClient
import SwiftUI

/// A finished run's folded work: "✓ Worked 2 min 14 s · 12 actions" over what it did. A click opens the timeline
/// under it (the steps on a rail); again closes it.
struct RunHeaderView: View {
    let doc: ChatDocument
    let info: RunHeaderInfo

    var body: some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) { doc.toggleOpen(info.key) }
        } label: {
            RunSummaryLabel(summary: info.summary, open: info.open)
                .padding(.horizontal, 10).padding(.vertical, 7)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .mgHelp(info.open ? "Hide the steps" : "Show every step of this run")
        .mgTestAnchor("run-summary:\(info.key)")
    }
}

/// The header's text: chevron, status, "Worked 2 min 14 s" (or a running clock), and what the run did.
struct RunSummaryLabel: View {
    let summary: RunSummary
    let open: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                .foregroundStyle(MG.tertiaryLabel).rotationEffect(.degrees(open ? 90 : 0))
            statusIcon.frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    title.font(.mgCallout).foregroundStyle(MG.label)
                    if summary.actions > 0 {
                        Text("· \(summary.actions) \(summary.actions == 1 ? "action" : "actions")")
                            .font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                    }
                }
                if !summary.kinds.isEmpty || summary.failed > 0 {
                    Text(detail).font(.mgFootnote).lineLimit(1).truncationMode(.tail)
                }
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder var title: some View {
        switch summary.status {
        case .running:
            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                Text("Working \(RunSummary.clock(ctx.date.timeIntervalSince(summary.started)))").monospacedDigit()
            }
        case .done: Text("Worked \(RunSummary.duration(summary.seconds))")
        case .stopped: Text("Stopped after \(RunSummary.duration(summary.seconds))")
        case .failed: Text("Failed after \(RunSummary.duration(summary.seconds))")
        }
    }

    @ViewBuilder var statusIcon: some View {
        switch summary.status {
        case .running: MGSpinner().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        case .stopped: Image(systemName: "stop.circle").foregroundStyle(MG.secondaryLabel)
        case .failed: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(MG.destructive)
        }
    }

    /// "read 5 · edited 2 +18 −4 · ran 3 · 1 failed", the counts coloured.
    var detail: AttributedString {
        var out = AttributedString()
        func add(_ s: String, _ color: Color) {
            var a = AttributedString(s)
            a.foregroundColor = color
            out += a
        }
        for (k, kind) in summary.kinds.enumerated() {
            if k > 0 { add(" · ", MG.tertiaryLabel) }
            add(kind, MG.secondaryLabel)
            if kind.hasPrefix("edited") {
                if summary.added > 0 { add(" +\(summary.added)", MG.success) }
                if summary.removed > 0 { add(" −\(summary.removed)", MG.destructive) }
            }
        }
        if summary.failed > 0 {
            if !summary.kinds.isEmpty { add(" · ", MG.tertiaryLabel) }
            add("\(summary.failed) failed", MG.destructive)
        }
        return out
    }
}

/// A live run: its header, then its steps in a small box that scrolls and keeps the newest in view. A click on the
/// header opens the whole run as a timeline; a click on a step opens its card in the box. When the run ends the box
/// gives way to the folded header.
struct LiveRunBox: View {
    let doc: ChatDocument
    let info: LiveRunInfo
    let results: [String: ChatTurn]

    /// One line of the box.
    enum Entry: Identifiable {
        case thinking(ChatTurn)
        case prose(ChatTurn)
        case call(ToolCall)
        case waiting

        var id: String {
            switch self {
            case .thinking(let t): "think:\(t.id)"
            case .prose(let t): "prose:\(t.id)"
            case .call(let c): "call:\(c.id)"
            case .waiting: "waiting"
            }
        }
    }

    var entries: [Entry] {
        var out: [Entry] = []
        let last = info.steps.last
        for s in info.steps {
            if !(s.reasoning ?? "").isEmpty { out.append(.thinking(s)) }
            if !s.text.isEmpty, !(s.id == last?.id && info.lastTextOutside) { out.append(.prose(s)) }
            for call in s.toolCalls where !RunFolding.pinnedTools.contains(call.name) && call.id != doc.permission?.call.id {
                out.append(.call(call))
            }
        }
        // Between steps, nothing streaming and nothing running: the next request is on its way.
        if let last, last.toolCalls.isEmpty, last.text.isEmpty, (last.reasoning ?? "").isEmpty, doc.liveTurnID == last.id {
            out.append(.waiting)
        }
        return out
    }

    var body: some View {
        let entries = entries
        // Taller while a step is opened in it, so its card fits.
        let opened = entries.contains { if case .call(let c) = $0 { doc.toolOpen[c.id] == true } else { false } }
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { doc.toggleOpen(info.key) }
            } label: {
                RunSummaryLabel(summary: info.summary, open: false)
                    .padding(.horizontal, 10).padding(.vertical, 7)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .mgHelp("Show every step of this run")
            if !entries.isEmpty {
                MGDivider()
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(entries) { entry(for: $0) }
                            Color.clear.frame(height: 1).id("end")
                        }
                        .padding(.horizontal, 8).padding(.vertical, 4)
                    }
                    .frame(maxHeight: opened ? 360 : 140)
                    .scrollBounceBehavior(.basedOnSize)
                    .mgScrollIndicators()
                    .onAppear { proxy.scrollTo("end", anchor: .bottom) }
                    .onChange(of: entries.count) { proxy.scrollTo("end", anchor: .bottom) }
                    .onChange(of: info.steps.last?.toolCalls.count) { proxy.scrollTo("end", anchor: .bottom) }
                }
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .mgTestAnchor("live-box:\(info.key)")
    }

    @ViewBuilder func entry(for e: Entry) -> some View {
        switch e {
        case .thinking(let t):
            ThinkingDisclosure(doc: doc, turnID: t.id, text: t.reasoning ?? "", seconds: t.thinkingSeconds,
                               live: doc.liveTurnID == t.id && t.text.isEmpty && t.toolCalls.isEmpty, compact: true)
                .padding(.horizontal, 2)
        case .prose(let t):
            Text(inlineMarkdown(String(t.text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(600)),
                                files: doc.isCoding ? doc.fileIndex : nil))
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2).truncationMode(.tail)
                .padding(.horizontal, 2).padding(.vertical, 3)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .call(let call):
            if call.name == "todo_write" { TodoUpdateRow(call: call) }
            else { ToolCard(doc: doc, call: call, result: results[call.id], compact: true) }
        case .waiting:
            WaitingLine(doc: doc).padding(.horizontal, 2).frame(height: 24)
        }
    }
}

/// A step's text in an opened run: secondary, three lines until opened.
struct RunProse: View {
    let doc: ChatDocument
    let turn: ChatTurn

    var body: some View {
        let key = "prose:\(turn.id.uuidString)"
        let open = doc.openRuns.contains(key)
        let text = turn.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let long = CollapsedText.lineCount(text) > 3 || text.utf16.count > 320
        VStack(alignment: .leading, spacing: 2) {
            Text(inlineMarkdown(open || !long ? text : String(text.prefix(1200)), files: doc.isCoding ? doc.fileIndex : nil))
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .lineLimit(open ? nil : 3)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if long {
                Button(open ? "Show less" : "Show more") {
                    withAnimation(.easeOut(duration: 0.15)) { doc.toggleOpen(key) }
                }
                .buttonStyle(.mg(.borderless)).font(.mgFootnote)
            }
        }
        .padding(.vertical, 2)
    }
}

extension View {
    /// A step of an opened run: indented on a thin line, so the steps read as one timeline under the run's header.
    func runRail() -> some View {
        padding(.top, 4)
            .padding(.leading, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .leading) {
                Rectangle().fill(MG.separator).frame(width: 2).padding(.leading, 6)
            }
    }
}
