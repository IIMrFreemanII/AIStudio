## graphify

This project has a knowledge graph at graphify-out/ (local only, gitignored) with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- The graph is a map, not the source: confirm anything it says against the files before editing or reporting it as fact. INFERRED edges are guesses.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost). `.graphifyignore` keeps the Kokoro lexicon JSON out of the graph.
