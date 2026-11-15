import Foundation
@testable import MovieCore
import Testing

@Test func viewerTitlesEachTakeAndStartsAtTheOneAskedFor() {
    let takes = [Take(file: "shots/a.mp4"), Take(file: "shots/b.mp4"), Take(file: "shots/c.mp4")]
    let items = MovieViewerItems.takes(takes, name: "Shot 9")
    #expect(items.files == ["shots/a.mp4", "shots/b.mp4", "shots/c.mp4"])
    #expect(items.titles == ["Shot 9 · Take 1", "Shot 9 · Take 2", "Shot 9 · Take 3"])
    #expect(MovieViewerItems.index(of: takes[1].id, in: takes) == 1)
    #expect(MovieViewerItems.index(of: nil, in: takes) == 0)
    #expect(MovieViewerItems.index(of: UUID(), in: takes) == 0)
}

@Test func aClickOnTheBarPlaysFromThatPointWithinTheTake() {
    #expect(MoviePlayback.time(at: 0.5, of: 8) == 4)
    #expect(MoviePlayback.time(at: -0.2, of: 8) == 0)
    #expect(MoviePlayback.time(at: 1.4, of: 8) == 8)
    #expect(MoviePlayback.time(at: .nan, of: 8) == 0)
    #expect(MoviePlayback.time(at: 0.5, of: 0) == 0)
    // Play again: from where it paused, or from the top once it played to its end.
    #expect(MoviePlayback.resume(at: 3.2, of: 8) == 3.2)
    #expect(MoviePlayback.resume(at: 8, of: 8) == 0)
    #expect(MoviePlayback.resume(at: 7.995, of: 8) == 0)
    #expect(MoviePlayback.clock(61.25) == "1:01.2" || MoviePlayback.clock(61.25) == "1:01.3")
    #expect(MoviePlayback.clock(4.0) == "0:04.0")
    #expect(MoviePlayback.clock(-1) == "0:00.0")
}
