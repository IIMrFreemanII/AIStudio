---
name: related-tests
description: Run only the tests a change can affect (Swift test targets, server Python tests, and the smoke tests for that area), never the full suites. Use whenever you implement a feature or milestone, when asked to "run related/affected tests" or "test what I changed", or after editing sources and before committing. Works out the targets from the changed files and Package.swift's dependency graph, then runs xcodebuild test (Debug) with -only-testing.
---

# Related tests

The script maps each changed file to its SwiftPM target using `swift package dump-package`. It then selects every test target that is that target or depends on it, directly or through other targets.

Examples:
- a change in `Sources/Flux` selects `FluxTests` only;
- a change in `Sources/MLXDiffusionCore` selects every test target built on core;
- a change in `Package.swift` or `Package.resolved` selects all of them.

## Steps

1. **Show the selection** (fast, no build, fine inside the sandbox):
   ```bash
   bash .claude/skills/related-tests/scripts/related-tests.sh
   ```
   - The default is the working tree against `HEAD`, plus untracked files.
   - `--base <ref>` compares against a ref, for example `--base HEAD~1` for the last commit or `--base origin/main`.
   - `--files <paths…>` takes an explicit list.
2. **If the result says "none"**, there are no unit tests to run; go on to the smoke tests and Python tests below.
   - When it lists "changed targets with no unit tests", run the matching smoke test from the table below.
   - Smoke tests that load real image, video or audio models take minutes: run them when the change touches that path,
     and say so if you skip one.
3. **Run the selection:** add `--run`, with `dangerouslyDisableSandbox: true`, because xcodebuild needs it. Use `run_in_background` when more than one or two targets are selected.
   - It runs `xcodebuild test -scheme AIStudio-Package … -only-testing:<Target>` in **Debug**. Never pass `-configuration Release`: the tests fail in Release.
   - It prints the Swift Testing summary lines and `** TEST SUCCEEDED/FAILED **`, and exits with xcodebuild's status.
   - The full log goes to `$TMPDIR/related-tests.log`. Under an unsandboxed run, `$TMPDIR` is a different folder from the sandbox's, so the script prints the path.
4. **Report** the selected targets, the pass/fail counts and any failing test names.

## Rules

- Never start a second xcodebuild on `.build/xcode` while one is running: the build database is locked and both fail. Wait for any running build or test first.
- **Related tests only, milestones included.** When implementing a feature, run only:
  - the Swift test targets this script selects;
  - the server Python test files for the changed modules (see below);
  - the smoke tests for the areas you changed.

  Don't run every test target, every Python file or every smoke test. Don't list "full suite not run" as a caveat.
- Run the image goldens (txt2img vs M1, feather-0 inpaint vs M4, `--smoke-test` vs M1) only when the change touches
  the image pipeline or the code they cover.
- `-only-testing` limits which tests *run*; xcodebuild may still build the other test bundles in the scheme. The first build after a clean is slow either way.

## Server Python tests

For changes under `server/gpu/`, run only the matching files from `server/gpu`, for example
`python -m pytest tests/test_llm.py tests/test_llm_sources.py` for `aistudio_gpu/llm/`. Tests that bind local ports
need `dangerouslyDisableSandbox: true`.

## Smoke tests for the changed area

Build the app with `scripts/build-app.sh` (unsandboxed) and run `build/AIStudio.app/Contents/MacOS/AIStudio <flag>`.

The chat and agent smoke tests need the local GPU server with the fake model running in the background first
(unsandboxed, because it binds a port):
`cd server/gpu && AISTUDIO_FAKE_LLM=1 python -m aistudio_gpu serve --port 8189`.
Then pass `http://127.0.0.1:8189` as the flag's URL.

| Changed area | Smoke flag |
|---|---|
| Image panel, queue, Qwen paths | `--smoke-test`, `--smoke-image` |
| Mask / outpaint / hi-res | `--smoke-mask`, `--smoke-outpaint`, `--smoke-hires` |
| Prompt assistant | `--smoke-assist` |
| XY plot | `--smoke-xy` |
| Flux UI, LoRA section, img2img | `--smoke-flux` |
| Video panel (LTX) / Wan | `--smoke-video` / `--smoke-wan` |
| Sound / music / speech panels | `--smoke-sound` / `--smoke-music` / `--smoke-speech` |
| Video/audio I/O, model eviction (no model needed) | `--smoke-media` |
| Chats, LLM models, `LLMClient`, `server/gpu/aistudio_gpu/llm` | `--smoke-chat <url>` |
| Coding chats, `Sources/Agent`, tool cards | `--smoke-agent <url>` (and `--smoke-chat <url>` if chat code changed) |
| Graph helper, `GraphOps`, `enqueue(doc:)` | `--smoke-graph-helper <url>` |
| Chat or tool-card UI layout | `--snapshot <dir>`, then check the chat images in light and dark |
| `aistudio-cli`, `*-probe` tools | run the changed command by hand |
