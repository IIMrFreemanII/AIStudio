import AIStudioLibrary
import AppKit
import SwiftUI

/// A model's reply as Markdown: the blocks from `Markdown.blocks`, inline styles from `AttributedString(markdown:)`,
/// selectable text, code blocks with a Copy button.
struct MarkdownView: View {
    let text: String

    var body: some View {
        let blocks = Markdown.blocks(text)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                MarkdownBlockView(block: block)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Inline Markdown (emphasis, code spans, links); falls back to the plain text.
func inlineMarkdown(_ s: String) -> AttributedString {
    let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace,
                                                          failurePolicy: .returnPartiallyParsedIfPossible)
    var a = (try? AttributedString(markdown: s, options: options)) ?? AttributedString(s)
    for run in a.runs where run.inlinePresentationIntent?.contains(.code) == true {
        a[run.range].backgroundColor = MG.fill
    }
    return a
}

struct MarkdownBlockView: View {
    let block: MarkdownBlock

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(inlineMarkdown(text))
                .font(level <= 1 ? .mgTitle3 : level == 2 ? .mgHeadline : .mgBody.weight(.semibold))
                .padding(.top, level <= 2 ? 4 : 0)
        case .paragraph(let text):
            Text(inlineMarkdown(text)).font(.mgBody).fixedSize(horizontal: false, vertical: true)
        case .code(let language, let text, _):
            CodeBlockView(language: language, code: text)
        case .list(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Group {
                            if let checked = item.checked {
                                Image(systemName: checked ? "checkmark.square" : "square")
                            } else if let n = item.number {
                                Text("\(n).")
                            } else {
                                Text(item.depth == 0 ? "•" : "◦")
                            }
                        }
                        .font(.mgBody).foregroundStyle(MG.secondaryLabel)
                        .frame(minWidth: 14, alignment: .trailing)
                        Text(inlineMarkdown(item.text)).font(.mgBody).fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.depth) * 18)
                }
            }
        case .quote(let text):
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 1.5).fill(MG.separator).frame(width: 3)
                Text(inlineMarkdown(text)).font(.mgBody).foregroundStyle(MG.secondaryLabel)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .table(let header, let rows):
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        ForEach(Array(header.enumerated()), id: \.offset) { _, h in
                            Text(inlineMarkdown(h)).font(.mgCallout.weight(.semibold))
                        }
                    }
                    Divider()
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, c in
                                Text(inlineMarkdown(c)).font(.mgCallout)
                            }
                        }
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
            }
        case .rule:
            MGDivider()
        }
    }
}

struct CodeBlockView: View {
    let language: String
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.mgFootnote)
                }
                .buttonStyle(.mg(.borderless))
                if ChatQuoting.active {
                    Button { ChatQuoting.post(code) } label: {
                        Label("Reply", systemImage: "arrowshape.turn.up.left").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Reply to this code")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            MGDivider()
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code).font(.mgMono).fixedSize(horizontal: true, vertical: true).padding(10)
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
    }
}
