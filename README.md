# MLXComfyUI

A native macOS image generator with a node graph in the style of ComfyUI. It's pure Swift on [mlx-swift](https://github.com/ml-explore/mlx-swift), with no Python at runtime. It supports text-to-image, editing with a reference image, inpainting, outpainting, a hi-res fix and LoRA. A prompt assistant uses the model's own language model to write prompts. The first supported model is Qwen-Image 2.1 Turbo.

This is an independent project and is not affiliated with ComfyUI.

## Requirements

- An Apple Silicon Mac. With every model kept loaded, it uses about 24 GB, peaking around 30 GB at 1024²; 48 GB of RAM is comfortable.
- macOS 15 or later, and Xcode 16 or later.
- Qwen-Image 2.1 Turbo weights in the diffusers folder layout (`model_index.json`, `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/`). The transformer can be bf16 or MLX-quantized. The weights aren't in this repo; put them in `qwen-turbo-mlx/` or choose the folder in the app.

## App

Build `build/MLXComfyUI.app` and open it:

```bash
scripts/build-app.sh
```

```bash
open build/MLXComfyUI.app
```

- **Image** (⌘1): the result, with a live preview after every sampler step. The inspector holds the Simple controls: prompt, size presets, seed with a randomize option, steps, batch count and model folder. The Simple controls edit the first TextEncode, EmptyLatent and Sampler nodes of whatever graph is loaded.
- **Prompt assistant:** under the prompt, **Enhance** rewrites a short idea as a detailed prompt, and **Describe Image** writes a prompt from the reference image. With a reference image set, Enhance turns a short edit request into a clear, specific instruction. The text streams into the prompt field; **Stop** keeps what's written so far, and ⌘Z restores the previous prompt. The instructions the assistant follows can be edited in Settings.
- **Graph** (⌘2): the node editor working on the same graph.
  - Drag a node by its header. ⇧-drag on the background selects with a rectangle.
  - Drag an output onto an input to connect them. Drag a connected input away to detach it.
  - Scroll to pan. ⌘-scroll or pinch to zoom.
  - Double-click the background or press ⌘K to add a node. ⌘D duplicates the selection and ⌫ deletes it.
  - Every edit can be undone.
  - The running node is outlined, and each node shows its time or a "cached" badge.
  - **Notes:** add a Note node (⌘K) for a sticky note you edit in place. Notes never run.
  - **Reroutes:** ⌥-click a wire to add a bend point, or add a Reroute node. A reroute takes the type of what flows into it, so type checks still apply end to end.
  - **Groups:** ⌘G puts a labelled frame around the selection. Drag its title bar to move it with the nodes inside, drag the corner to resize, and rename or recolor it in the inspector.
  - **Subgraphs:** ⌥⌘G collapses the selection into one node. Links that cross the boundary become its inputs and outputs. Double-click the node (or use the inspector) to edit inside it, with a breadcrumb bar to go back (Esc or ⌘↑). ⇧⌘G expands it again. Every instance shares the same definition. The Simple panel only edits nodes at the top level.
- **Sidebar:** the queue (cancel or reorder jobs) and history (the PNGs in the output folder). Right-click a history item to restore its graph.
  - Search filters history by prompt. ★ marks favorites (stored on the file, so they survive moving it) and the star button shows only favorites. The grid button switches between a list and a thumbnail grid.
- **Styles and wildcards:** pick a **Style** under the prompt to wrap it in a template (edit the styles in Settings). In any prompt, `{red|green|blue}` picks one option per image and `__name__` picks a line from `~/Library/Application Support/MLXComfyUI/wildcards/name.txt`. Choices are seeded by the image's seed, and the saved PNG records both the final prompt and the template.
- **XY plot** (toolbar): vary one or two inputs of the current graph, such as seeds `1-4`, steps `4-12:4`, denoise `0.3, 0.5`, or a word in the prompt (search and replace). Each cell runs as a queued job, shown as one collapsed row, and the result is saved as a labelled grid PNG with the graph embedded.
- **Compare:** ⌘-click a history item, or choose **Compare with Shown Image**, to see it against the current image: a draggable split, side by side, or the difference (brightened ×4 so small changes show).
- **Restoring a graph:** drop a PNG made by MLXComfyUI anywhere on the window. ⌘O opens a graph JSON file or such a PNG. ⌘S saves the graph.
- **Reference image:** drop an image on the Simple panel to edit it. The prompt describes the change, and the output keeps the image's aspect ratio.
- **Inpainting:** use **Paint Mask…** to mark an area. Only that area is regenerated. **Feather** softens the edge so the change blends in; pixels outside the feathered edge stay identical to the original.
- **Outpainting:** **Extend…** grows the reference image's canvas. Set the pixels to add on each side; the new border is generated to match.
- **Hi-res fix** (Sampling): upscales the result and refines it with a second pass at low denoise, for example 1024² → 1536².
- **Model:** loaded at launch and kept in memory. The toolbar gauge shows MLX memory use.

Images are saved to `~/Pictures/MLXComfyUI` by default; change this in Settings (⌘,).

To check the app without clicking through it, use these launch flags:

```bash
build/MLXComfyUI.app/Contents/MacOS/MLXComfyUI --smoke-test "a lighthouse at dusk"
```

```bash
build/MLXComfyUI.app/Contents/MacOS/MLXComfyUI --snapshot /tmp/snaps
```

`--smoke-test` queues one job through the same path as the Generate button and exits with status 0 on success. `--snapshot` renders the window offscreen to PNG files.

## Build

SwiftPM on the command line cannot compile MLX's Metal shaders, so build with `xcodebuild`:

```bash
xcodebuild build -scheme mlxcomfy-cli -destination 'platform=macOS' -derivedDataPath .build/xcode -configuration Release -skipPackagePluginValidation
```

Run the tests:

```bash
xcodebuild test -scheme MLXComfyUI-Package -destination 'platform=macOS' -derivedDataPath .build/xcode -skipPackagePluginValidation
```

## CLI

Run a graph:

```bash
.build/xcode/Build/Products/Release/mlxcomfy-cli run examples/txt2img.json --set prompt.prompt="A lighthouse at dusk" --set sampler.seed=3
```

Generate without a graph:

```bash
.build/xcode/Build/Products/Release/mlxcomfy-cli generate --prompt "A red fox in fresh snow" --seed 1 --out outputs/fox.png
```

| Command | What it does |
|---|---|
| `run` | Runs a graph JSON file, or the graph embedded in a PNG that `SaveImage` wrote. `--set node.input=value` overrides a literal input. `--then-set` runs a second time in the same process to show the cache at work |
| `nodes` | Lists node types with their inputs and outputs |
| `graph-from-png` | Prints the graph embedded in a saved PNG |
| `compare` | Pixel diff of two images; exits with status 1 if they differ |
| `generate` | Text to image. Options: `--image` (condition image, repeatable), `--lora path[:strength]` (repeatable), `--size WxH`, `--seed`, `--save-steps`, `--dequantize` |
| `make-test-lora` | Writes a synthetic LoRA. `--up-scale 0` makes a no-op one |
| `vae-roundtrip` | Encodes and decodes an image and prints PSNR. Use it to check the VAE. `--tiled on` uses tiles |
| `assist` | Prompt assistant: `--prompt "a cat"` enhances a prompt; `--mode describe --image x.png` describes an image; `--mode custom --instruction "…"` follows any instruction. `--temperature 0` is deterministic |
| `tokenize` | Prints the Qwen token ids for a string |

Measured on an M4 Max with 48 GB, at 1024² and 8 steps: about 5.7 s per step and about 53 s end to end, with a peak of about 29 GB.

## Editing, inpainting and LoRA

- **Edit or reference:** connect a `LoadImage` to `TextEncode.image` (and optionally `image2`).
  - Qwen3-VL sees the image through its vision tower, and its VAE latents become an image block in the transformer prefix.
  - Set `EmptyLatent` to 0×0 to match the image's aspect ratio at about 1 MP.
  - From the CLI: `generate --image refs/girl.png --prompt "Put her on a snowy mountain top"`.
- **Inpainting:** see [examples/inpaint.json](examples/inpaint.json).
  - The chain is `VAEEncode` → `SetLatentMask` (from `LoadMask` → `FeatherMask`, white = regenerate) → `Sampler`, then `ImageCompositeMasked` blends the result over the original with the same feathered mask. Pixels where the feathered mask is 0 are bit-identical.
  - `FeatherMask` grows (or shrinks, with a negative value) the mask and blurs its edge. With feather 0 the result is the same as without it.
  - `SetLatentMask.blend`: `hard` (default) fully regenerates every latent the mask touches; `soft` follows the mask's coverage.
- **Outpainting:** see [examples/outpaint.json](examples/outpaint.json). `PadImage` extends the canvas and outputs a mask over the new border, which drives the same chain as inpainting.
- **Hi-res fix:** see [examples/hires.json](examples/hires.json): `VAEDecode` → `ImageScale` → `VAEEncode` → a second `Sampler` with `steps` 8 and `denoise` 0.4 → `VAEDecode`.
  - `steps` 8 gives an evenly spaced schedule, so denoise 0.4 runs 3 refinement steps (about 45 s at 1536²). With `steps` 0 the checkpoint's distilled schedule is used; it ends at σ 0.41, so any denoise below 0.7 gets a single step and less added detail.
- **Large images:** `VAEEncode` and `VAEDecode` process the image in overlapping tiles above about 1.25 MP (`tiled`: `auto`, `on` or `off`). A whole-image VAE pass peaks at about 16 GB at 1536² and 28 GB at 2048²; tiled, it stays near 8 GB at the same accuracy.
  - `Sampler.denoise` < 1 on an encoded latent gives img2img instead.
- **Prompt assistant:** the `PromptAssistant` node (model, prompt, optional image, mode `enhance`, `describe` or `custom`) outputs a STRING; link it into `TextEncode.prompt`. See [examples/assistant.json](examples/assistant.json). It is cached like any node, so it reruns only when its own inputs change.
  - Qwen-Image's text encoder is the full Qwen3-VL 8B chat model. Its LM head isn't needed for images, so it is loaded only on first use (8-bit, about 0.6 GB). Generation runs at about 28 tokens per second; a prompt takes about 5 s.
- **LoRA:** the `LoRA` node (model, path, strength) returns a model that shares the base weights. Each target layer runs the LoRA as a low-rank side branch, because the base weights are 8-bit and can't be merged into.
  - Supported key formats: diffusers/PEFT (`lora_A`/`lora_B`), kohya/Comfy (`lora_down`/`lora_up` + `alpha`) and `lora_unet_…` underscore names.
  - Connect the LoRA model to both TextEncode and Sampler.
  - Chain LoRA nodes to stack them.

## Graphs

A graph is JSON with a `nodes` list. Each input is either a literal value or a link written as `{"node": "<id>", "output": "<name>"}`. See [examples/txt2img.json](examples/txt2img.json).

The executor:

- checks node types, links, socket types and required inputs;
- orders nodes topologically and runs only the nodes that feed output nodes;
- caches each node's outputs under a hash of its type, its literal inputs and its upstream hashes. Changing a seed reruns only the Sampler, VAE Decode and Save nodes;
- keeps loaded models in a resource cache, so they stay loaded across runs and graph edits.

`SaveImage` embeds the full graph in the PNG as an `iTXt` chunk under the key `mlxcomfy.graph`.

Graphs can also hold `groups` (visual frames), `subgraphs` (definitions shared by `Subgraph` instance nodes) and `metadata`; all are optional, so older graphs still load. Before running, the executor flattens the graph: subgraph instances are inlined (inner nodes run as `instance/node`), reroutes are spliced out and notes are dropped. Caching works the same as in a flat graph, because signatures don't include node ids. `--set instance/node.input=value` reaches a node inside a subgraph. See [examples/structured.json](examples/structured.json), which renders the same image as `txt2img.json`.

## Layout

- `Sources/MLXDiffusionCore`: safetensors weight store, dense and quantized `Lin`, `Conv2d`, the flow-match Euler scheduler, PNG I/O, and mask and tiling helpers (`MaskOps`, `Tiling`).
- `Sources/QwenImage21`:
  - `Qwen3TextEncoder`: features from the last layer, taken before the final norm. For image prompts it adds the vision tower (`Qwen3VisionTower`), interleaved mRoPE and deepstack.
  - `QwenImage21Transformer`: block-causal DiT. The prompt's K/V are cached once and each step runs only the image tokens.
  - `QwenImage21VAE`: the single-frame form of the causal 3D VAE.
  - `QwenImage21Pipeline`.
  - `TextGenerator.swift`: KV-cached text generation with the same decoder stack (the prompt assistant).
- `Sources/GraphRuntime`: graph model (Codable), node schemas and registry, the caching executor, events and cancellation. Also subgraph flattening, collapse and expand (`Subgraphs.swift`), and the Note and Reroute nodes. Also `PromptTemplate` (choices, wildcards, styles) and `PlotAxis` / `PlotSpec` (XY plots). It doesn't depend on MLX.
- `Sources/NodesStandard`: LoadModel, LoRA, TextEncode, PromptAssistant, EmptyLatent, Sampler, VAEEncode, VAEDecode, LoadImage, LoadMask, SetLatentMask, ImageCompositeMasked, FeatherMask, PadImage, ImageScale, SaveImage and PreviewImage.
- `Sources/MLXComfyApp`: the SwiftUI app.
  - `AppModel`: the document graph with undo, the queue, run state and history.
  - `Engine`: an actor that owns the executor and the loaded models.
  - `GraphEditor/`: the node editor.
- `Sources/mlxcomfy-cli`: the command line front end.

The architecture is ported from diffusers' `pipeline_qwenimage21.py`, `transformer_qwenimage21.py` and `autoencoder_kl_qwenimage21.py`.

## License

Apache-2.0; see [LICENSE](LICENSE). The model code is ported from [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers), both Apache-2.0. Model weights are covered by their own licenses.
