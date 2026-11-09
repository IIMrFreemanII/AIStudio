import CoreGraphics
import Foundation

/// The design system's scroll indicator: where its thumb sits along the track. Lengths are along the scroll axis.
public struct ScrollThumb: Equatable, Sendable {
    /// Distance of the thumb's leading end from the track's start.
    public var start: CGFloat
    public var length: CGFloat

    public static let minimumLength: CGFloat = 20

    /// The thumb for `offset` into `content` seen through `viewport`, on a track `track` long; nil when everything
    /// fits. Past either end (rubber-banding) the thumb shrinks by the overshoot, down to the minimum length.
    public static func make(content: CGFloat, viewport: CGFloat, offset: CGFloat, track: CGFloat) -> ScrollThumb? {
        guard content > viewport + 0.5, viewport > 0, track > minimumLength else { return nil }
        let maxOffset = content - viewport
        let overshoot = offset < 0 ? -offset : max(0, offset - maxOffset)
        let full = max(minimumLength, track * viewport / content)
        let length = max(minimumLength, full - overshoot)
        let progress = min(max(offset / maxOffset, 0), 1)
        return ScrollThumb(start: (track - length) * progress, length: length)
    }

    /// The content offset that puts the thumb's leading end at `start` (for dragging the thumb).
    public static func offset(forStart start: CGFloat, content: CGFloat, viewport: CGFloat, track: CGFloat) -> CGFloat {
        let maxOffset = max(0, content - viewport)
        guard maxOffset > 0 else { return 0 }
        let length = max(minimumLength, track * viewport / content)
        let room = track - length
        guard room > 0 else { return 0 }
        return min(max(start / room, 0), 1) * maxOffset
    }
}
