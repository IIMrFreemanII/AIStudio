import Foundation
import LLMClient

// Tools every chat gets (plain, coding, graph helper): questions for the user as cards, and HTML widgets drawn in the
// chat. Both have no side effects (effect `.read`, no paths), so every permission mode allows them.

/// A question of an `ask_user` call, as the card shows it.
public struct UserQuestion: Codable, Equatable, Sendable, Hashable {
    public struct Option: Codable, Equatable, Sendable, Hashable {
        public var label: String
        public var description: String?
        public init(label: String, description: String? = nil) {
            self.label = label
            self.description = description
        }
    }

    public var question: String
    /// A short label for the summary ("Database").
    public var header: String?
    public var multiSelect: Bool
    public var options: [Option]

    public init(question: String, header: String? = nil, multiSelect: Bool = false, options: [Option]) {
        self.question = question
        self.header = header
        self.multiSelect = multiSelect
        self.options = options
    }

    /// The label the summary uses: the header, else the question.
    public var label: String { (header?.isEmpty == false ? header : nil) ?? question }

    /// The questions of an `ask_user` call's arguments (1–4, each with 2–4 options).
    public static func parse(_ args: ToolArgs) throws -> [UserQuestion] {
        guard let list = args.raw["questions"] as? [[String: Any]], !list.isEmpty else {
            throw ToolError("expected \"questions\": a list of 1–4 questions")
        }
        if list.count > 4 { throw ToolError("at most 4 questions at once") }
        return try list.map { q in
            guard let text = (q["question"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
                throw ToolError("each question needs \"question\"")
            }
            let options = (q["options"] as? [Any] ?? []).compactMap { o -> Option? in
                if let s = o as? String { return Option(label: s) }
                guard let d = o as? [String: Any], let l = d["label"] as? String, !l.isEmpty else { return nil }
                return Option(label: l, description: (d["description"] as? String).flatMap { $0.isEmpty ? nil : $0 })
            }
            guard (2...4).contains(options.count) else { throw ToolError("\"\(text)\" needs 2–4 options") }
            let multi = (q["multiSelect"] as? Bool) ?? ((q["multiSelect"] as? String) == "true")
            return UserQuestion(question: text, header: q["header"] as? String, multiSelect: multi, options: options)
        }
    }
}

/// The user's answer to one question: the options picked and/or their own words ("Other").
public struct UserAnswer: Codable, Equatable, Sendable, Hashable {
    public var picked: [String]
    public var other: String?
    public init(picked: [String] = [], other: String? = nil) {
        self.picked = picked
        self.other = other?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? other : nil
    }

    public var isEmpty: Bool { picked.isEmpty && other == nil }

    /// "SQLite", "SQLite, Redis", `Other: "a flat file"`.
    public var summary: String {
        var parts = picked
        if let other { parts.append("Other: \"\(other)\"") }
        return parts.isEmpty ? "(no answer)" : parts.joined(separator: ", ")
    }

    /// One line per question: `Database: SQLite`.
    public static func summary(_ answers: [UserAnswer], for questions: [UserQuestion]) -> String {
        zip(questions, answers).map { "\($0.label): \($1.summary)" }.joined(separator: "\n")
    }
}

public struct AskUserTool: AgentTool {
    public init() {}
    public let name = "ask_user"
    public let description = """
        Ask the user 1–4 multiple-choice questions, shown as a card they click; the reply waits for their answer. Use \
        it when a choice is genuinely theirs and you can't settle it from what they said (an approach, a preference, \
        scope). Each question has 2–4 short options (a label, and a description of what it means); put your \
        recommendation first with " (Recommended)" at the end of its label. They can always type their own answer \
        instead. Don't ask what you can find out or decide sensibly yourself.
        """
    public var parameters: JSONValue {
        let option: JSONValue = .object(["type": .string("object"), "required": .array([.string("label")]),
                                         "properties": .object([
                                             "label": .object(["type": .string("string"), "description": .string("1–5 words")]),
                                             "description": .object(["type": .string("string"),
                                                                     "description": .string("What picking it means")]),
                                         ])])
        let question: JSONValue = .object([
            "type": .string("object"), "required": .array([.string("question"), .string("options")]),
            "properties": .object([
                "question": .object(["type": .string("string"), "description": .string("The whole question, ending with ?")]),
                "header": .object(["type": .string("string"), "description": .string("A 1–2 word label for it")]),
                "multiSelect": .object(["type": .string("boolean"), "description": .string("Several options may be picked")]),
                "options": .object(["type": .string("array"), "items": option, "minItems": .number(2), "maxItems": .number(4)]),
            ]),
        ])
        return .object(["type": .string("object"), "required": .array([.string("questions")]), "properties": .object([
            "questions": .object(["type": .string("array"), "items": question, "minItems": .number(1), "maxItems": .number(4)]),
        ])])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let qs = try UserQuestion.parse(args)
        return ToolPreview(kind: .read, title: qs.count == 1 ? qs[0].question : "\(qs.count) questions")
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let qs = try UserQuestion.parse(args)
        guard let ask = ctx.askUser else {
            return ToolOutput("The user can't be asked right now: decide sensibly yourself and say what you chose.",
                              isError: true)
        }
        let answers = await ask(ctx.callID, qs)
        try Task.checkCancellation()
        guard let answers, answers.count == qs.count, !answers.allSatisfy(\.isEmpty) else {
            return ToolOutput("The user skipped the questions: decide sensibly yourself and say what you chose.",
                              detail: "Skipped")
        }
        let summary = UserAnswer.summary(answers, for: qs)
        return ToolOutput("The user answered:\n" + summary, detail: summary)
    }
}

public struct ShowWidgetTool: AgentTool {
    public init() {}
    public let name = "show_widget"
    public let description = """
        Show the user an interactive HTML widget in the chat (a chart, a diagram, a calculator, a small simulation, a \
        form). Write a self-contained HTML fragment or page with inline <style> and <script>. Scripts and styles may \
        load from cdnjs.cloudflare.com, cdn.jsdelivr.net or unpkg.com (pin exact versions); nothing else can be \
        reached: no fetch, no other sites, no storage that outlives the widget. Use the theme's CSS variables \
        (var(--mg-bg), --mg-surface, --mg-text, --mg-text-2, --mg-accent, --mg-border, --mg-fill, --mg-radius, \
        --mg-font, --mg-mono, --mg-series-1 … --mg-series-6 for chart series) so it fits light and dark mode; keep \
        the background transparent. `aistudio.sendPrompt(text)` sends a message to the chat as if the user typed it \
        (for buttons like "Explain this"); `aistudio.reportState(object)` tells you the widget's current values \
        (sliders, selections), which you get with the user's next message. The widget is as wide as the chat and up \
        to 600 px tall.
        """
    public var parameters: JSONValue {
        schema(["title": ("string", "A short title shown above it"),
                "html": ("string", "The widget: HTML with inline <style> and <script>")],
               required: ["title", "html"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let title = try args.required("title")
        _ = try args.required("html")
        return ToolPreview(kind: .read, title: title)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let title = try args.required("title")
        let html = try args.required("html")
        if html.utf8.count > 512 * 1024 { throw ToolError("the widget is over 512 KB: make it smaller") }
        return ToolOutput("Shown to the user as the widget \"\(title)\". If it reports its state, that comes with their next message.")
    }
}

public enum ChatTools {
    /// The chat tools, for models that call tools.
    public static let all: [any AgentTool] = [AskUserTool(), ShowWidgetTool()]
    public static let names: Set<String> = ["ask_user", "show_widget"]
}

/// The system prompt's parts every chat shares.
public enum ChatPrompt {
    /// How to end a reply with follow-up suggestions (the app shows them as chips, see `FollowUps`).
    public static let followUps = """
        When natural next questions or requests come to mind, end your reply with up to three short follow-ups the \
        user might send next, written as they would type them, in this exact form (the app shows them as buttons and \
        hides the block):
        <followups>
        - First follow-up
        - Second follow-up
        </followups>
        Leave the block out when there's nothing worth suggesting, or when you've just asked the user a question.
        """

    public static let tools = """
        You can ask the user multiple-choice questions with ask_user (the reply waits for their answer) and show \
        interactive HTML widgets in the chat with show_widget (charts, diagrams, calculators, small simulations). \
        Prefer a widget when something is clearer to see or play with than to read; prefer plain text and Markdown \
        tables otherwise.
        """

    /// A plain chat's system prompt: the user's own, then the chat features the model can use.
    public static func system(user: String?, tools: Bool) -> String {
        var parts: [String] = []
        if let user, !user.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { parts.append(user) }
        parts.append("You are an assistant in AIStudio, a Mac app. Replies are rendered as Markdown. Today is \(today()).")
        if tools { parts.append(Self.tools) }
        parts.append(followUps)
        return parts.joined(separator: "\n\n")
    }

    /// The section added to the coding agent's and the graph helper's prompts.
    public static func section(tools: Bool) -> String {
        (tools ? Self.tools + "\n\n" : "") + followUps
    }

    static func today() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: Date())
    }
}

/// The page a widget runs in: the model's HTML after a head of ours, so our Content-Security-Policy is the first
/// thing parsed (a policy the model adds can only tighten it). Then the theme and the bridge to the app
/// (`aistudio.sendPrompt`, `aistudio.reportState`, the page's height for the frame).
public enum WidgetPage {
    public static let cdnHosts = ["cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com"]

    public static var contentSecurityPolicy: String {
        let cdns = cdnHosts.map { "https://\($0)" }.joined(separator: " ")
        return "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' \(cdns); style-src 'unsafe-inline' \(cdns); "
            + "img-src data: blob: \(cdns); font-src data: \(cdns); media-src data: blob:; connect-src 'none'; "
            + "frame-src 'none'; worker-src blob:; form-action 'none'; base-uri 'none'"
    }

    static let bridge = """
        (function () {
          var post = function (m) { try { window.webkit.messageHandlers.aistudio.postMessage(m); } catch (e) {} };
          window.aistudio = {
            sendPrompt: function (text) { post({ type: "sendPrompt", text: String(text) }); },
            reportState: function (state) { try { post({ type: "state", state: JSON.stringify(state) }); } catch (e) {} }
          };
          window.sendPrompt = window.aistudio.sendPrompt;
          var last = 0;
          // The content's height (the body's, not the document's: that never drops below the frame).
          var report = function () {
            var b = document.body;
            if (!b) return;
            var h = Math.ceil(Math.max(b.scrollHeight, b.getBoundingClientRect().height));
            if (h !== last) { last = h; post({ type: "height", height: h }); }
          };
          var observer = new ResizeObserver(report);
          addEventListener("DOMContentLoaded", function () { observer.observe(document.body); report(); });
          addEventListener("load", report);
          addEventListener("pointerdown", function () { post({ type: "interacted" }); }, true);
          addEventListener("keydown", function () { post({ type: "interacted" }); }, true);
        })();
        """

    /// The whole page. `theme` is the app's stylesheet (CSS variables for light and dark); `bridge` false leaves
    /// the app bridge out (a saved .html file: `aistudio.*` calls do nothing there).
    public static func html(body: String, title: String, theme: String, bridge: Bool = true) -> String {
        var model = body
        if let r = model.range(of: "<!doctype[^>]*>", options: [.regularExpression, .caseInsensitive]),
           model[..<r.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            model.removeSubrange(model.startIndex..<r.upperBound)
        }
        let script = bridge ? bridgeScript : """
            window.aistudio = { sendPrompt: function () {}, reportState: function () {} }; window.sendPrompt = window.aistudio.sendPrompt;
            """
        return """
            <!doctype html><html><head>\
            <meta http-equiv="Content-Security-Policy" content="\(contentSecurityPolicy)">\
            <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\
            <title>\(escape(title))</title><style>\(theme)</style><script>\(script)</script></head>
            \(model)
            """
    }

    static var bridgeScript: String { bridge }

    public static func escape(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
