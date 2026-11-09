import GraphRuntime
import SwiftUI

/// Sets up an XY plot: up to two inputs of the current graph and the values each takes.
struct PlotSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    @State private var xKey = ""
    @State private var xText = ""
    @State private var xReplace = false
    @State private var yKey = ""
    @State private var yText = ""
    @State private var yReplace = false

    var body: some View {
        let candidates = model.plotCandidates
        let x = axis(xKey, xText, xReplace, candidates)
        let y = yKey.isEmpty ? nil : axis(yKey, yText, yReplace, candidates)
        VStack(alignment: .leading, spacing: 0) {
            MGForm {
                Text("XY Plot").font(.mgTitle3)
                axisSection("X axis (columns)", key: $xKey, text: $xText, replace: $xReplace, result: x,
                            candidates: candidates, optional: false)
                axisSection("Y axis (rows)", key: $yKey, text: $yText, replace: $yReplace, result: y,
                            candidates: candidates, optional: true)
                Text("Each cell is queued as its own job; text encoding and other unchanged steps are reused from "
                     + "the cache. The finished grid is saved as one PNG with the graph embedded.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                    .padding(.horizontal, 10)
            }
            HStack {
                if case .success(let xa)? = x, y == nil || { if case .success = y! { return true }; return false }() {
                    let rows: Int = { if case .success(let ya)? = y { return ya.values.count }; return 1 }()
                    Text("\(xa.values.count) × \(rows) = \(xa.values.count * rows) images")
                        .foregroundStyle(MG.secondaryLabel)
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Queue Plot") {
                    guard case .success(let xa)? = x else { return }
                    var ya: PlotAxis?
                    if case .success(let a)? = y { ya = a }
                    model.enqueuePlot(PlotSpec(x: xa, y: ya))
                    dismiss()
                }
                .buttonStyle(.mg(.prominent))
                .keyboardShortcut(.defaultAction)
                .disabled(!ready(x, y))
            }
            .padding(16)
            .mgChrome(MG.barTint, material: .headerView)
            .overlay(alignment: .top) { MGDivider() }
        }
        .frame(width: 540, height: 560)
        .presentationBackground { MGGlassBackground(role: .sheet, shape: Rectangle()) }
        .onAppear { setDefaults(candidates) }
    }

    @ViewBuilder
    func axisSection(_ title: String, key: Binding<String>, text: Binding<String>, replace: Binding<Bool>,
                     result: Result<PlotAxis, Error>?, candidates: [(node: String, input: String, type: SocketType, label: String)],
                     optional: Bool) -> some View {
        MGSection(title) {
            MGFormRow("Input") {
                Picker("Input", selection: key) {
                    if optional { Text("None").tag("") }
                    ForEach(candidates, id: \.label) { c in Text(c.label).tag("\(c.node)\u{1F}\(c.input)") }
                }
                .labelsHidden()
                .fixedSize()
            }
            if !key.wrappedValue.isEmpty, let c = candidate(key.wrappedValue, candidates) {
                MGFormRow("Values") { TextField("Values", text: text, prompt: Text(placeholder(c.type))).frame(maxWidth: 280) }
                if c.type == .string {
                    Toggle("Search and replace in the current text", isOn: replace)
                }
                switch result {
                case .failure(let e)?: Text(String(describing: e)).font(.mgSubheadline).foregroundStyle(MG.destructive)
                case .success(let a)?: Text(a.labels.joined(separator: " · ")).font(.mgSubheadline)
                    .foregroundStyle(MG.secondaryLabel).lineLimit(2)
                case nil: EmptyView()
                }
            }
        }
    }

    func placeholder(_ t: SocketType) -> String {
        switch t {
        case .int: return "1-4, 7  or  4-12:4"
        case .float: return "0.3, 0.5  or  0.2-0.6:0.1"
        case .bool: return "true, false"
        default: return xReplace || yReplace ? "first word, replacement, …" : "option | option | …"
        }
    }

    func candidate(_ key: String, _ list: [(node: String, input: String, type: SocketType, label: String)])
        -> (node: String, input: String, type: SocketType, label: String)?
    {
        let parts = key.split(separator: "\u{1F}", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        return list.first { $0.node == parts[0] && $0.input == parts[1] }
    }

    func axis(_ key: String, _ text: String, _ replace: Bool,
              _ list: [(node: String, input: String, type: SocketType, label: String)]) -> Result<PlotAxis, Error>? {
        guard let c = candidate(key, list), !text.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
        return Result {
            try PlotAxis.parse(text, type: c.type, node: c.node, input: c.input,
                               current: model.literal(c.node, c.input), searchReplace: replace)
        }
    }

    func ready(_ x: Result<PlotAxis, Error>?, _ y: Result<PlotAxis, Error>?) -> Bool {
        guard case .success? = x else { return false }
        if yKey.isEmpty { return true }
        if case .success? = y { return true }
        return false
    }

    /// X: the first Sampler's seed (four seeds from the current one); Y: none.
    func setDefaults(_ list: [(node: String, input: String, type: SocketType, label: String)]) {
        guard xKey.isEmpty else { return }
        if let s = list.first(where: { model.graph[$0.node]?.type == "Sampler" && $0.input == "seed" }) {
            xKey = "\(s.node)\u{1F}\(s.input)"
            let cur: Int = { if case .int(let v)? = model.literal(s.node, s.input) { return v }; return 1 }()
            xText = "\(cur)-\(cur + 3)"
        } else if let first = list.first {
            xKey = "\(first.node)\u{1F}\(first.input)"
        }
    }
}
