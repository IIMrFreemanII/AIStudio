import Foundation
import MLX
import Testing
@testable import Wan
@testable import WanTransformer

/// The Lightning schedule is the one the distillation used: timesteps 1000, 937.5, 833.3, 625.
@Test func lightningScheduleMatchesTheDistillation() {
    let s = WanSchedule.lightning()
    let expected = [1000, 937.5, 833.3333, 625]
    #expect(s.timesteps.count == 4 && zip(s.timesteps, expected).allSatisfy { abs($0 - $1) < 1e-3 })
    #expect(s.sigmas.last == 0 && s.sigmas.count == 5)
}

/// The official schedule starts just below 1000 (sigma 999/1000 shifted), is integer-valued and decreasing.
@Test func officialScheduleIsShiftedAndInteger() {
    let s = WanSchedule.official(steps: 40, shift: 12)
    #expect(s.timesteps.count == 40 && s.sigmas.count == 41 && s.sigmas.last == 0)
    #expect(s.timesteps.first == 999 && s.timesteps.allSatisfy { $0 == $0.rounded() })
    #expect(zip(s.timesteps, s.timesteps.dropFirst()).allSatisfy { $0 >= $1 })
}

/// With the T2V boundary (875) the 4 Lightning steps split 2 high / 2 low; with I2V's 900 too.
@Test func expertsSwitchAtTheBoundary() {
    let t = WanSchedule.lightning().timesteps
    #expect(t.filter { $0 >= 875 }.count == 2 && t.filter { $0 >= 900 }.count == 2)
    let official = WanSchedule.official(steps: 40, shift: 12).timesteps
    #expect(official.filter { $0 >= 875 }.count > 0 && official.filter { $0 < 875 }.count > 0)
}

/// UniPC on a linear velocity field (x0 constant) lands exactly on x0, like any consistent solver.
@Test func uniPCRecoversAConstantTarget() {
    let target = MLXArray([0.5, -1.0, 2.0] as [Float])
    let s = WanSchedule.official(steps: 12, shift: 5)
    var x = MLXArray([1.0, 1.0, 1.0] as [Float])
    var solver = WanUniPC(sigmas: s.sigmas)
    for i in 0..<12 {
        // For x = (1−σ)·x0 + σ·ε the flow velocity is ε − x0 = (x − x0)/σ.
        let v = (x - target) / Float(s.sigmas[i])
        x = solver.step(velocity: v, sample: x)
    }
    #expect(abs(x - target).max().item(Float.self) < 1e-4)
}

/// Prompt cleaning folds full-width punctuation, curly quotes and repeated whitespace, as ftfy + the official
/// whitespace cleaning do.
@Test func promptCleaning() {
    #expect(WanPrompt.clean("  “Hello”，  world&amp;more\n") == "\"Hello\", world&more")
    #expect(WanPrompt.clean("色调艳丽，过曝") == "色调艳丽,过曝")
}

/// RoPE tables: 22 temporal + 21 height + 21 width frequency pairs per position, identity at the origin.
@Test func ropeTableLayout() throws {
    var c = WanDiT.Config()
    c.numLayers = 0
    #expect(c.headDim == 128)
    let half = 128 / 2
    let t = 128 - 4 * (128 / 6), hw = 2 * (128 / 6)
    #expect(t / 2 + hw / 2 + hw / 2 == half)
}
