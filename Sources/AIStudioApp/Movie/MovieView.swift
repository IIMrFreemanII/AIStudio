import MovieCore
import SwiftUI

/// A movie in the main area: the stages on the left (each unlocks when the one before is approved), the stage's
/// page on the right, and what's running on top of it.
struct MovieView: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        HStack(spacing: 0) {
            MovieStageRail(doc: doc)
                .frame(width: 196)
                .background(MG.groupedBackground)
            MGDivider(vertical: true)
            VStack(spacing: 0) {
                if let job = doc.job { MovieJobBar(job: job) { model.stopMovieJob(doc) } }
                if let error = doc.error { MovieErrorBar(message: error) { doc.error = nil } }
                Group {
                    switch doc.stage {
                    case .idea: MovieIdeaPage(doc: doc)
                    case .script: MovieScriptPage(doc: doc)
                    case .shots: MovieShotsPage(doc: doc)
                    default: MovieLaterPage(stage: doc.stage)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .mgSheet("movie-cost-gate", item: $doc.pendingGate) { gate in CostGateSheet(gate: gate) }
        .mgTestAnchor("movie-view")
    }
}

// MARK: stage rail

private struct MovieStageRail: View {
    @Bindable var doc: MovieDocument

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("STAGES").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                .padding(.horizontal, 12).padding(.top, 14).padding(.bottom, 6)
            ForEach(MovieStage.allCases) { stage in
                StageRow(stage: stage, selected: doc.stage == stage, approved: doc.movie.approved.contains(stage),
                         unlocked: doc.movie.isUnlocked(stage) && stage.isAvailable,
                         later: !stage.isAvailable) { doc.stage = stage }
            }
            Spacer()
            VStack(alignment: .leading, spacing: 4) {
                summaryLine("film", "\(doc.movie.shots.count) shots · \(MovieFormat.time(doc.movie.runtime))")
                summaryLine("target", "aim \(MovieFormat.time(Double(doc.movie.targetSeconds)))")
                summaryLine("dollarsign.circle", "spent \(CostEstimate.format(doc.movie.costSpent))")
                    .mgHelp("What this movie's GPU work has cost so far, from the servers' figures")
            }
            .padding(12)
        }
    }

    private func summaryLine(_ symbol: String, _ text: String) -> some View {
        Label(text, systemImage: symbol).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).labelStyle(.titleAndIcon)
    }
}

private struct StageRow: View {
    let stage: MovieStage
    let selected: Bool
    let approved: Bool
    let unlocked: Bool
    let later: Bool
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: stage.symbol).frame(width: 18)
                    .foregroundStyle(selected ? MG.accent : unlocked ? MG.label : MG.tertiaryLabel)
                Text(stage.title).foregroundStyle(unlocked ? MG.label : MG.tertiaryLabel)
                Spacer(minLength: 4)
                if approved {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
                } else if later {
                    Text("soon").font(.mg(size: 9.5, weight: .medium)).foregroundStyle(MG.tertiaryLabel)
                } else if !unlocked {
                    Image(systemName: "lock").font(.mg(size: 10)).foregroundStyle(MG.tertiaryLabel)
                }
            }
            .font(.mgBody)
            .padding(.horizontal, 10).frame(height: 30)
            .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(selected ? MG.accent.opacity(0.15) : hover && unlocked ? MG.hover : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!unlocked)
        .onHover { hover = $0 }
        .padding(.horizontal, 6)
        .mgHelp(later ? "\(stage.title) comes in a later update"
                : unlocked ? stage.title : "Approve the stage before this one first")
        .mgTestAnchor("movie-stage-\(stage.rawValue)")
    }
}

// MARK: bars

private struct MovieJobBar: View {
    let job: MovieJob
    let stop: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            MGSpinner()
            VStack(alignment: .leading, spacing: 2) {
                Text(job.title).font(.mgHeadline)
                if !job.detail.isEmpty { Text(job.detail).font(.mgFootnote).foregroundStyle(MG.secondaryLabel) }
            }
            if let p = job.progress { MGProgress(value: p).frame(width: 160) }
            Spacer()
            Button("Stop", action: stop).buttonStyle(.mg(.bordered)).mgHelp("Stop and keep what's done so far")
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(MG.accent.opacity(0.08))
        .overlay(alignment: .bottom) { MGDivider() }
        .mgTestAnchor("movie-job")
    }
}

private struct MovieErrorBar: View {
    let message: String
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(MG.warning)
            Text(message).font(.mgCallout).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button { dismiss() } label: { Image(systemName: "xmark") }.buttonStyle(.mg(.icon)).mgHelp("Dismiss")
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(MG.warning.opacity(0.10))
        .overlay(alignment: .bottom) { MGDivider() }
        .mgTestAnchor("movie-error")
    }
}

// MARK: pages shared bits

/// A stage page: a scrolling column with the stage's title and actions on top.
struct MoviePage<Actions: View, Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder var actions: Actions
    @ViewBuilder var content: Content

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.mgTitle2)
                    Text(subtitle).font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                HStack(spacing: 8) { actions }
            }
            .padding(.horizontal, 24).padding(.top, 18).padding(.bottom, 14)
            MGDivider()
            ScrollView {
                content
                    .padding(.horizontal, 24).padding(.vertical, 18)
                    .frame(maxWidth: 1100, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

enum MovieFormat {
    /// "1:05", "0:08".
    static func time(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    static func seconds(_ s: Double) -> String {
        s == s.rounded() ? "\(Int(s)) s" : String(format: "%.1f s", s)
    }
}

/// A labelled field on a stage page.
struct MovieField<Content: View>: View {
    let label: String
    var hint: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(label).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                if let hint { Text(hint).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel) }
            }
            content
        }
    }
}

private struct MovieLaterPage: View {
    let stage: MovieStage
    var body: some View {
        MGEmptyState(symbol: stage.symbol, title: "\(stage.title) is coming next",
                     message: "This stage is part of the next Movie Studio update. The script and shot list you approve now carry over.")
    }
}

// MARK: cost gate

/// What a stage will spend, before it runs: Cancel, or Run for the estimated amount.
struct CostGateSheet: View {
    let gate: CostGate
    @Environment(\.mgDismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(gate.estimate.title).font(.mgTitle3)
            Text("This runs on a cloud GPU and costs money. Nothing starts until you press Run.")
                .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
            VStack(spacing: 0) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 8) {
                    GridRow {
                        Text("Work"); Text("Jobs"); Text("GPU"); Text("GPU time"); Text("Est.").gridColumnAlignment(.trailing)
                    }
                    .font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                    ForEach(gate.estimate.lines) { line in
                        GridRow {
                            Text(line.label)
                            Text("\(line.count)").monospacedDigit()
                            Text(line.gpu)
                            Text(MovieFormat.time(line.seconds)).monospacedDigit()
                            Text(CostEstimate.format(line.usd)).monospacedDigit()
                        }
                    }
                    Divider().gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        Text("Total").font(.mgHeadline)
                        Text(""); Text("")
                        Text(MovieFormat.time(gate.estimate.seconds)).monospacedDigit()
                        Text("≈ " + CostEstimate.format(gate.estimate.usd)).font(.mgHeadline).monospacedDigit()
                    }
                }
                .font(.mgBody)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
            if !gate.estimate.notes.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(gate.estimate.notes, id: \.self) { Text("• " + $0) }
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Run (≈ \(CostEstimate.format(gate.estimate.usd)))") {
                    let run = gate.run
                    dismiss()
                    run()
                }
                .buttonStyle(.mg(.prominent))
                .keyboardShortcut(.defaultAction)
                .mgTestAnchor("movie-gate-run")
            }
        }
        .padding(20)
        .frame(width: 520)
        .mgTestAnchor("movie-cost-gate")
    }
}
