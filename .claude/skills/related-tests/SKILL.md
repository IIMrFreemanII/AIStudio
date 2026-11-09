---
name: related-tests
description: Run only the unit-test targets a change can affect, instead of all nine. Use when asked to "run related/affected tests", "test what I changed", or after editing Swift sources and before committing. Works out the targets from the changed files and Package.swift's dependency graph, then runs xcodebuild test (Debug) with -only-testing.
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
2. **If the result says "none"**, stop and say so.
   - When it lists "changed targets with no unit tests", suggest the matching smoke test from the table below. Don't run it unasked: smoke tests load real models and take minutes.
3. **Run the selection:** add `--run`, with `dangerouslyDisableSandbox: true`, because xcodebuild needs it. Use `run_in_background` when more than one or two targets are selected.
   - It runs `xcodebuild test -scheme MLXComfyUI-Package … -only-testing:<Target>` in **Debug**. Never pass `-configuration Release`: the tests fail in Release.
   - It prints the Swift Testing summary lines and `** TEST SUCCEEDED/FAILED **`, and exits with xcodebuild's status.
   - The full log goes to `$TMPDIR/related-tests.log`. Under an unsandboxed run, `$TMPDIR` is a different folder from the sandbox's, so the script prints the path.
4. **Report** the selected targets, the pass/fail counts and any failing test names.

## Rules

- Never start a second xcodebuild on `.build/xcode` while one is running: the build database is locked and both fail. Wait for any running build or test first.
- This skill doesn't replace the milestone routine. A milestone still needs:
  - all tests;
  - the image goldens (txt2img vs M1, feather-0 inpaint vs M4, `--smoke-test` vs M1);
  - every smoke test.
- `-only-testing` limits which tests *run*; xcodebuild may still build the other test bundles in the scheme. The first build after a clean is slow either way.

## Smoke tests for code without unit tests

These are suggestions only. Build the app with `scripts/build-app.sh` and run `build/MLXComfyUI.app/Contents/MacOS/MLXComfyUI <flag>`.

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
| `mlxcomfy-cli`, `*-probe` tools | run the changed command by hand |
