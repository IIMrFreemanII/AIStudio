import Agent
import CharacterCore
import Foundation
import MovieCore

/// The prompt tools in the Studio agent: drafts wait on Studio's run sheet (`studioRunGate`), with its pointer and
/// caption, like `run_workflow`. One record per reply.
@MainActor
final class StudioPromptHost: PromptToolHost {
    weak var app: AppModel?
    let studio: StudioController
    let record = PromptEngineRecord()
    let fixedModel: PromptModel? = nil
    let characterID: UUID? = nil

    init(app: AppModel, studio: StudioController) {
        self.app = app
        self.studio = studio
    }

    func approve(_ estimate: CostEstimate) async throws -> Bool {
        guard let app else { return false }
        try await studio.lock("sheet", caption: studio.actorPrefix + "Waiting for another cost sheet")
        defer { studio.unlock("sheet") }
        final class Pressed { var run = false }
        let pressed = Pressed()
        app.studioRunGate = CostGate(estimate: estimate, run: { pressed.run = true })
        studio.waitingOnUser = studio.actorPrefix
            + "Cost sheet ≈ \(CostEstimate.format(estimate.usd)): waiting for you to press Run"
        await studio.point(at: "movie-gate-run", fallback: "movie-cost-gate", caption: studio.waitingOnUser ?? "")
        defer { studio.waitingOnUser = nil }
        do {
            while app.studioRunGate != nil { try await Task.sleep(for: .milliseconds(200)) }
        } catch {
            app.studioRunGate = nil
            throw error
        }
        return pressed.run
    }
}
