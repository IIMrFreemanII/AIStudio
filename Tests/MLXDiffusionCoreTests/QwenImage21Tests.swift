import MLX
import Testing
@testable import QwenImage21

@Test func centredPositionsMatchReference() {
    // range(-(n - n // 2), n // 2)
    #expect(QwenImage21Transformer.centred(4) == [-2, -1, 0, 1])
    #expect(QwenImage21Transformer.centred(5) == [-3, -2, -1, 0, 1])
}

@Test func avgDownAveragesWithZeroFrame() {
    // One real frame, front-padded with one zero frame: every group mixes ones and zeros equally.
    let x = MLXArray.ones([1, 4, 4, 2])
    let y = avgDown(x, outChannels: 4, factorT: 2, factorS: 2)
    #expect(y.shape == [1, 2, 2, 4])
    #expect(abs(y.mean().item(Float.self) - 0.5) < 1e-6)
}

@Test func dupUpUnfoldsChannelsToSpace() {
    // Channel c of the input should land in the 2x2 block it is unfolded into.
    let x = MLXArray((0 ..< 8).map { Float($0) }, [1, 1, 1, 8])
    let y = dupUp(x, outChannels: 2, factorT: 1, factorS: 2)  // repeats = 2*4/8 = 1
    #expect(y.shape == [1, 2, 2, 2])
    // out channel 0 takes input channels 0..3 as (i, j) = (0,0),(0,1),(1,0),(1,1)
    #expect(y[0, 0, 0, 0].item(Float.self) == 0)
    #expect(y[0, 0, 1, 0].item(Float.self) == 1)
    #expect(y[0, 1, 0, 0].item(Float.self) == 2)
    #expect(y[0, 1, 1, 1].item(Float.self) == 7)
}

@Test func dupUpKeepsLastTemporalFrame() {
    // factorT 2: channels split (o, t, i, j); only t = 1 survives the first-chunk trim.
    let x = MLXArray((0 ..< 8).map { Float($0) }, [1, 1, 1, 8])
    let y = dupUp(x, outChannels: 1, factorT: 2, factorS: 2)
    #expect(y.shape == [1, 2, 2, 1])
    #expect(y[0, 0, 0, 0].item(Float.self) == 4)
    #expect(y[0, 1, 1, 0].item(Float.self) == 7)
}

@Test func upsampleNearestRepeatsPixels() {
    let x = MLXArray([Float(1), 2, 3, 4], [1, 2, 2, 1])
    let y = upsampleNearest2x(x)
    #expect(y.shape == [1, 4, 4, 1])
    #expect(y[0, 1, 1, 0].item(Float.self) == 1)
    #expect(y[0, 2, 3, 0].item(Float.self) == 4)
}
