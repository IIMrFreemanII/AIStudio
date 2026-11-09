import CoreGraphics
import Foundation

/// Lays a graph out left to right in columns (a small Sugiyama-style layout): sources on the left, every link pointing
/// right, rows ordered to reduce crossings. Pure geometry: callers pass node sizes and links.
public enum GraphArranger {
    public struct Edge: Hashable, Sendable {
        public var source: String
        public var target: String
        /// Still used for columns, but not for row order or vertical alignment (e.g. hidden model wires).
        public var weak: Bool
        /// Closes a loop (a group block both feeding and fed by the same nodes): ignored for columns, so it is drawn
        /// as the one backward link.
        var loop = false
        public init(_ source: String, _ target: String, weak: Bool = false) {
            self.source = source
            self.target = target
            self.weak = weak
        }
    }

    public struct Spacing: Sendable {
        public var column: CGFloat = 80
        public var row: CGFloat = 40
        /// Inside a group: padding around the members, and room for the title bar.
        public var groupPadding: CGFloat = 24
        public var groupHeader: CGFloat = 36
        public init() {}
    }

    public struct Result: Sendable {
        public var positions: [String: CGPoint]
        public var groupFrames: [String: CGRect]
    }

    /// Arranges `sizes` (node id → size). `groups` (group id → member ids) are laid out as blocks: members are arranged
    /// inside, then each group is placed as one unit. `initial` positions break ties so the result stays close to the
    /// current order. The layout's top-left is `origin`.
    public static func arrange(sizes: [String: CGSize], edges: [Edge], groups: [String: Set<String>] = [:],
                               initial: [String: CGPoint] = [:], origin: CGPoint = .zero,
                               spacing: Spacing = Spacing()) -> Result {
        // Each node belongs to at most one group (the first by id).
        var owner: [String: String] = [:]
        for gid in groups.keys.sorted() {
            for m in groups[gid]!.sorted() where sizes[m] != nil && owner[m] == nil { owner[m] = gid }
        }
        let liveGroups = Set(owner.values)

        // 1. Arrange each group's members.
        var inner: [String: [String: CGPoint]] = [:]
        var blockSize: [String: CGSize] = [:]
        for gid in liveGroups.sorted() {
            let members = Set(owner.filter { $0.value == gid }.map(\.key))
            let s = sizes.filter { members.contains($0.key) }
            let e = edges.filter { members.contains($0.source) && members.contains($0.target) }
            let pos = flat(sizes: s, edges: e, initial: initial, spacing: spacing)
            let box = bounds(pos, s)
            inner[gid] = pos
            blockSize[gid] = CGSize(width: box.width + 2 * spacing.groupPadding,
                                    height: box.height + spacing.groupPadding + spacing.groupHeader)
        }

        // 2. Arrange units: ungrouped nodes and group blocks.
        func unit(_ id: String) -> String { owner[id].map { "group:" + $0 } ?? id }
        var unitSizes: [String: CGSize] = [:]
        var unitInitial: [String: CGPoint] = [:]
        for (id, size) in sizes where owner[id] == nil {
            unitSizes[id] = size
            unitInitial[id] = initial[id]
        }
        for gid in liveGroups {
            unitSizes["group:" + gid] = blockSize[gid]
            let ys = owner.filter { $0.value == gid }.compactMap { initial[$0.key] }
            if let minY = ys.map(\.y).min(), let minX = ys.map(\.x).min() { unitInitial["group:" + gid] = CGPoint(x: minX, y: minY) }
        }
        var unitEdges: [Edge] = []
        var seen = Set<String>()
        for e in edges where sizes[e.source] != nil && sizes[e.target] != nil {
            let (s, t) = (unit(e.source), unit(e.target))
            guard s != t else { continue }
            let key = "\(s)\u{1}\(t)\u{1}\(e.weak)"
            if seen.insert(key).inserted { unitEdges.append(Edge(s, t, weak: e.weak)) }
        }
        // A strong edge wins over a weak one between the same units.
        let allUnitEdges = unitEdges
        unitEdges = allUnitEdges.filter { e in
            !e.weak || !allUnitEdges.contains { !$0.weak && $0.source == e.source && $0.target == e.target }
        }
        // A group block in a loop (e.g. Outpainting feeds the sampler and takes the decoded image) is placed after the
        // nodes it takes from; its links back into the loop become the backward ones.
        let scc = components(Array(unitSizes.keys), unitEdges.filter { !$0.weak })
        for i in unitEdges.indices where unitEdges[i].source.hasPrefix("group:") {
            let (s, t) = (unitEdges[i].source, unitEdges[i].target)
            if let c = scc[s], c == scc[t] { unitEdges[i].loop = true }
        }
        let unitPos = flat(sizes: unitSizes, edges: unitEdges, initial: unitInitial, spacing: spacing)

        // 3. Final positions.
        var positions: [String: CGPoint] = [:]
        var groupFrames: [String: CGRect] = [:]
        for (u, p) in unitPos {
            let at = CGPoint(x: p.x + origin.x, y: p.y + origin.y)
            if u.hasPrefix("group:") {
                let gid = String(u.dropFirst(6))
                groupFrames[gid] = CGRect(origin: at, size: blockSize[gid]!)
                let pos = inner[gid] ?? [:]
                for (id, q) in pos {
                    positions[id] = CGPoint(x: at.x + spacing.groupPadding + q.x, y: at.y + spacing.groupHeader + q.y)
                }
            } else {
                positions[u] = at
            }
        }
        return Result(positions: positions, groupFrames: groupFrames)
    }

    /// The bounding box of nodes at `positions` with `sizes`.
    public static func bounds(_ positions: [String: CGPoint], _ sizes: [String: CGSize]) -> CGRect {
        var box: CGRect?
        for (id, p) in positions {
            let r = CGRect(origin: p, size: sizes[id] ?? .zero)
            box = box?.union(r) ?? r
        }
        return box ?? .zero
    }

    /// Strongly connected components (Tarjan): node → component index, only for components with more than one node.
    static func components(_ ids: [String], _ edges: [Edge]) -> [String: Int] {
        var succ: [String: [String]] = [:]
        for e in edges { succ[e.source, default: []].append(e.target) }
        var index = 0, comp = 0
        var idx: [String: Int] = [:], low: [String: Int] = [:], onStack = Set<String>(), stack: [String] = []
        var out: [String: Int] = [:]
        func visit(_ v: String) {
            idx[v] = index; low[v] = index; index += 1
            stack.append(v); onStack.insert(v)
            for w in succ[v] ?? [] {
                if idx[w] == nil { visit(w); low[v] = min(low[v]!, low[w]!) }
                else if onStack.contains(w) { low[v] = min(low[v]!, idx[w]!) }
            }
            if low[v] == idx[v] {
                var members: [String] = []
                while let w = stack.popLast() {
                    onStack.remove(w)
                    members.append(w)
                    if w == v { break }
                }
                if members.count > 1 { for m in members { out[m] = comp }; comp += 1 }
            }
        }
        for v in ids.sorted() where idx[v] == nil { visit(v) }
        return out
    }

    /// One level without groups; the result's top-left is (0, 0).
    static func flat(sizes: [String: CGSize], edges: [Edge], initial: [String: CGPoint], spacing: Spacing) -> [String: CGPoint] {
        let ids = sizes.keys.sorted()
        guard !ids.isEmpty else { return [:] }
        let all = edges.filter { sizes[$0.source] != nil && sizes[$0.target] != nil && $0.source != $0.target }
        let valid = all.filter { !$0.loop }
        var preds: [String: [String]] = [:], succs: [String: [String]] = [:]
        var strongPreds: [String: [String]] = [:], strongSuccs: [String: [String]] = [:]
        for e in valid {
            preds[e.target, default: []].append(e.source)
            succs[e.source, default: []].append(e.target)
            if !e.weak {
                strongPreds[e.target, default: []].append(e.source)
                strongSuccs[e.source, default: []].append(e.target)
            }
        }
        let connected = Set(all.flatMap { [$0.source, $0.target] })

        // Columns: longest path from the sources (cycle-safe).
        var layer: [String: Int] = [:]
        var visiting = Set<String>()
        func depth(_ id: String) -> Int {
            if let l = layer[id] { return l }
            if !visiting.insert(id).inserted { return 0 }
            let l = (preds[id] ?? []).map { depth($0) + 1 }.max() ?? 0
            visiting.remove(id)
            layer[id] = l
            return l
        }
        for id in ids where connected.contains(id) { _ = depth(id) }
        // Sources sit just left of their first consumer (e.g. Empty Latent next to the sampler).
        for id in ids where connected.contains(id) && (preds[id] ?? []).isEmpty {
            if let m = (succs[id] ?? []).compactMap({ layer[$0] }).min() { layer[id] = max(0, m - 1) }
        }
        let maxLayer = layer.values.max() ?? 0
        var columns: [[String]] = Array(repeating: [], count: maxLayer + 1)
        func initialKey(_ id: String) -> (CGFloat, String) { (initial[id]?.y ?? .greatestFiniteMagnitude, id) }
        for id in ids where connected.contains(id) { columns[layer[id]!].append(id) }
        for i in columns.indices { columns[i].sort { initialKey($0) < initialKey($1) } }

        // Row order: barycenter sweeps.
        func index() -> [String: Double] {
            var idx: [String: Double] = [:]
            for col in columns { for (i, id) in col.enumerated() { idx[id] = Double(i) } }
            return idx
        }
        func sweep(down: Bool) {
            let order = down ? Array(columns.indices.dropFirst()) : Array(columns.indices.dropLast().reversed())
            for c in order {
                let idx = index()
                let col = columns[c]
                func bary(_ id: String) -> Double {
                    let ns = (down ? strongPreds[id] : strongSuccs[id]) ?? []
                    let vs = ns.compactMap { idx[$0] }
                    return vs.isEmpty ? idx[id]! : vs.reduce(0, +) / Double(vs.count)
                }
                columns[c] = col.enumerated().sorted { l, r in
                    let (bl, br) = (bary(l.element), bary(r.element))
                    return bl != br ? bl < br : l.offset < r.offset
                }.map(\.element)
            }
        }
        for k in 0..<5 { sweep(down: k % 2 == 0) }

        // x: column offsets.
        var xs: [CGFloat] = []
        var x: CGFloat = 0
        for col in columns {
            xs.append(x)
            x += (col.map { sizes[$0]!.width }.max() ?? 0) + spacing.column
        }

        // y: align with strong neighbours, then push apart within each column.
        var y: [String: CGFloat] = [:]
        func centre(_ id: String) -> CGFloat? { y[id].map { $0 + sizes[id]!.height / 2 } }
        func median(_ v: [CGFloat]) -> CGFloat? {
            guard !v.isEmpty else { return nil }
            let s = v.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        func settle(_ c: Int, desired: [String: CGFloat]) {
            var bottom = -CGFloat.greatestFiniteMagnitude
            var placed: [String: CGFloat] = [:]
            for id in columns[c] {
                let want = desired[id] ?? (bottom == -CGFloat.greatestFiniteMagnitude ? 0 : bottom)
                let top = max(want, bottom)
                placed[id] = top
                bottom = top + sizes[id]!.height + spacing.row
            }
            // Pull the column back up as a block if everything got pushed below where it wanted to be.
            let shift = columns[c].compactMap { id in desired[id].map { placed[id]! - $0 } }.min() ?? 0
            for id in columns[c] { y[id] = placed[id]! - max(0, shift) }
        }
        for c in columns.indices {
            var desired: [String: CGFloat] = [:]
            for id in columns[c] {
                if let m = median((strongPreds[id] ?? []).compactMap(centre)) { desired[id] = m - sizes[id]!.height / 2 }
            }
            settle(c, desired: desired)
        }
        // Sources follow their consumers.
        for c in columns.indices.reversed() {
            var desired: [String: CGFloat] = [:]
            for id in columns[c] {
                let ps = (strongPreds[id] ?? []).compactMap(centre)
                let anchor = ps.isEmpty ? (strongSuccs[id] ?? []).compactMap(centre) : ps
                if let m = median(anchor) { desired[id] = m - sizes[id]!.height / 2 } else { desired[id] = y[id] }
            }
            settle(c, desired: desired)
        }

        var out: [String: CGPoint] = [:]
        for (c, col) in columns.enumerated() { for id in col { out[id] = CGPoint(x: xs[c], y: y[id]!) } }

        // Unconnected nodes (notes, lone loaders) go in a row underneath.
        let loose = ids.filter { !connected.contains($0) }.sorted { initialKey($0) < initialKey($1) }
        if !loose.isEmpty {
            var lx: CGFloat = 0
            let top = out.isEmpty ? 0 : bounds(out, sizes).maxY + spacing.row * 2
            for id in loose {
                out[id] = CGPoint(x: lx, y: top)
                lx += sizes[id]!.width + spacing.row
            }
        }

        // Normalise to (0, 0).
        let box = bounds(out, sizes)
        return out.mapValues { CGPoint(x: $0.x - box.minX, y: $0.y - box.minY) }
    }

    /// The free spot for a `size` rect nearest to `near` (its preferred top-left) that keeps `margin` from every rect in
    /// `avoiding`, searched in `step` increments.
    public static func freeSpot(size: CGSize, near: CGPoint, avoiding: [CGRect], margin: CGFloat = 40,
                                step: CGFloat = 24) -> CGPoint {
        func fits(_ p: CGPoint) -> Bool {
            let r = CGRect(origin: p, size: size).insetBy(dx: -margin, dy: -margin)
            return !avoiding.contains { $0.intersects(r) }
        }
        if fits(near) { return near }
        for ring in 1...200 {
            var candidates: [CGPoint] = []
            for i in -ring...ring {
                for j in [-ring, ring] {
                    candidates.append(CGPoint(x: near.x + CGFloat(i) * step, y: near.y + CGFloat(j) * step))
                    candidates.append(CGPoint(x: near.x + CGFloat(j) * step, y: near.y + CGFloat(i) * step))
                }
            }
            candidates.sort { l, r in
                let dl = hypot(l.x - near.x, l.y - near.y), dr = hypot(r.x - near.x, r.y - near.y)
                return dl != dr ? dl < dr : (l.y, l.x) < (r.y, r.x)
            }
            if let p = candidates.first(where: fits) { return p }
        }
        let below = avoiding.map(\.maxY).max() ?? near.y
        return CGPoint(x: near.x, y: below + margin)
    }
}
