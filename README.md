# MLXComfyUI

A native macOS image generator with a node graph in the style of ComfyUI. It's pure Swift on [mlx-swift](https://github.com/ml-explore/mlx-swift), with no Python at runtime. It supports text-to-image, editing with a reference image, inpainting, outpainting, a hi-res fix and LoRA, with Qwen-Image 2.1 Turbo, and text-to-image, image-to-image, inpainting and **LoRAs** (kohya, diffusers/PEFT and ai-toolkit files) with **FLUX.1 [dev]**. It also makes **video with a synchronized soundtrack**, from text or from a still image, with LTX-2.3, and silent video with **Wan 2.2 A14B** (text-to-video and image-to-video). A prompt assistant uses the image model's own language model to write prompts. It also generates **sound effects, ambience and loops** from text with Stable Audio Open Small, **songs with or without lyrics** with ACE-Step 1.5, and **narration** in 28 English voices with Kokoro-82M, whose pronunciation step is also pure Swift.

This is an independent project and is not affiliated with ComfyUI.

## Requirements

- An Apple Silicon Mac. Images: about 24 GB resident, peaking around 30 GB at 1024² (FLUX.1 [dev]: about 14 GB resident, peaking around 20 GB). Video: about 27 GB resident, peaking around 35 GB for 5 s at 768×512. Models are swapped as needed, so 48 GB of RAM is comfortable for both.
- macOS 15 or later, and Xcode 16 or later.
- Qwen-Image 2.1 Turbo weights in the diffusers folder layout (`model_index.json`, `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/`). The transformer can be bf16 or MLX-quantized. The weights aren't in this repo; download them from the Models window, or put them in `~/Library/Application Support/MLXComfyUI/models/qwen-turbo-mlx/` yourself, or choose another folder in the app.
- For FLUX.1 [dev]: the official [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) diffusers folders (bf16, about 34 GB), downloaded from the Models window. The repo is gated under the FLUX.1 [dev] Non-Commercial License: accept it on the model page and paste a read token in the Models window. On first use the 24 GB transformer is quantized to 8-bit once and cached in the model folder (`.mlxcomfy-cache/`, 12.6 GB; about 10 s).
- For Wan 2.2 video: downloaded from the Models window. **Wan 2.2 T2V A14B** is about 31 GB: the 4-bit MLX conversion [PocketAiHub/Wan2.2-T2V-A14B-MLX](https://huggingface.co/PocketAiHub/Wan2.2-T2V-A14B-MLX). **Wan 2.2 I2V A14B** is about 45 GB: the 8-bit [Anes1032/Wan2.2-I2V-A14B-mlx-q8](https://huggingface.co/Anes1032/Wan2.2-I2V-A14B-mlx-q8). Each includes the 4-step [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) LoRAs. All Apache-2.0, not gated.
- For video: LTX-2.3, downloaded from the Models window (about 28.6 GB: the 4-bit MLX conversion [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) plus the Gemma 3 12B text encoder [mlx-community/gemma-3-12b-it-4bit](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit)).

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
- **LoRA:** **Add LoRA…** applies a `.safetensors` LoRA to the image model, with a strength slider for each. Several stack in order. It works with Qwen-Image and FLUX.1 [dev], and is the same `LoRA` node as in the graph.
- **FLUX.1 [dev]:** with both image models installed, the Image panel has a **Model** picker. Flux runs 28 steps by default with distilled **Guidance** 3.5 (Sampling). It has no reference editing, so a dropped image is redrawn **image to image** at the chosen **Strength**, scaled to about 1 MP. **Paint Mask…** and **Extend…** inpaint and outpaint as with Qwen. Enhance and Describe keep using Qwen-Image's language model. On an M4 Max a 1024² image takes about 3 min 20 s (7 s per step), peaking at about 20 GB; 512² with 8 steps takes about 18 s.
- **Models** (⇧⌘M): lists the models the app supports, with size and license, and downloads them from Hugging Face when you click Download and accept the license. Downloads resume if interrupted and are checked against their SHA-256. Gated models need a Hugging Face token, which is kept in the Keychain. You can also add a folder you downloaded yourself. Models are stored in `~/Library/Application Support/MLXComfyUI/models` (changeable in the window). A model path in a graph or a `--model` option that doesn't exist as given, such as the examples' `qwen-turbo-mlx`, is looked up by name in that folder.
- **Memory:** models load on first use and stay in memory while they fit. When a new model doesn't fit the budget (80% of RAM by default; change it in Settings), the least recently used one is unloaded first, never one the running job uses. Click the toolbar memory gauge to see loaded models and unload one.
- **Modes:** the Simple panel switches between Image, Video, Music, Speech and Sound. Each mode has its own graph; switching swaps it (and switching back restores it), and undo covers the switch. A mode whose model isn't installed shows which model it needs and opens the Models window.
- **Video** (LTX-2.3): a prompt describing the action, camera and sound; an optional **first frame** image to animate; size presets (multiples of 64, e.g. 768×512, 1024×576, portrait and square); length up to 10 s; 24, 25 or 30 fps; seed. Generation runs 8 steps at half resolution, upscales the latents 2×, and refines with 3 steps, showing a preview of the middle frame after each. The result is an MP4 with AAC stereo sound at 48 kHz. A 5 s 768×512 clip takes about 2 minutes on an M4 Max (2.6 s prompt, 114 s sampling, 13 s decoding), peaking at about 35 GB.
- **Video with Wan 2.2 A14B:** with both video models installed, the Video panel has a **Model** picker. Wan makes silent clips at 16 fps, up to 5 s, at 832×480, 480×832 or 624×624 (or 512×288 for quick tries). Choosing a first frame switches to the I2V model automatically.
  - **Quality** is **Fast** by default: 4 steps with the Lightning LoRAs, guidance 1. **Full** is the official 40-step sampler with guidance and the default negative prompt, about 20× slower.
  - Two 14B experts share the work: the high-noise expert does the first steps and the low-noise expert the rest. The step preview shows which one is running.
  - On an M4 Max a 5 s clip at 832×480 takes about 12 minutes in Fast mode: about 172 s per step (attention over 32,760 tokens is more than half of it), then about 40 s of decoding. Small clips are much quicker, since attention cost grows with the square of the clip size: 2 s at 512×288 takes about a minute. T2V (4-bit) peaks at about 28 GB; I2V (8-bit) at about 41 GB, so it needs the other models out of memory, which the app does on its own.
- **Music** (ACE-Step 1.5 Turbo): a style description (genre, instruments, mood, vocals), lyrics with `[Verse]` / `[Chorus]` markers or **Instrumental**, a length from 10 s to 4 min, and optionally tempo, key, time signature and vocal language (each "Auto" by default). With **Plan the song** on, a 1.7B language model first writes the song's blueprint: it fills in tempo, key and time signature, expands the description, and writes 5 audio codes per second that lay out the arrangement. The music model then renders the song in 8 steps. The result is a stereo 48 kHz WAV. On an M4 Max a 30 s song takes about 9 s with the planner (5 s without) and a 2-minute song about 21 s, peaking at about 16 GB. The model is MIT-licensed and not gated.
- **Speech** (Kokoro-82M): text of any length (read in chunks of up to 510 phonemes, cut at sentence ends), one of 28 voices (American and British, female and male, with the authors' quality grades), and speed. **Preview Voice** speaks a short sample in the chosen voice without saving it. Pronunciation is a pure-Swift port of misaki: its US/UK dictionaries, spaCy's tokenizer rules, Apple's NaturalLanguage for word classes, number/date/money reading, and a rule-based fallback for unknown words (no espeak). `[word](/phonemes/)` forces a pronunciation. Mono 24 kHz WAV; about 40× faster than real time on an M4 Max, using under 3 GB.
- **Sound** (Stable Audio Open Small): a prompt describing the sound (sources, setting, texture; a tempo for loops, e.g. "128 BPM tech house drum loop"), a length up to 11 s, seed and batch. 8 ping-pong steps, stereo 44.1 kHz WAV, peak-normalized. About 0.7 s per clip, peaking around 5 GB. The model is gated on Hugging Face: accept its license on the model page, paste a read token in the Models window, then Download.
- **Video and audio results** play in the canvas (a player for video, a waveform for audio) and appear in history, with the graph embedded in the file so dropping it on the window restores it.

Images are saved to `~/Pictures/MLXComfyUI` by default; change this in Settings (⌘,).

To check the app without clicking through it, use these launch flags:

```bash
build/MLXComfyUI.app/Contents/MacOS/MLXComfyUI --smoke-test "a lighthouse at dusk"
```

```bash
build/MLXComfyUI.app/Contents/MacOS/MLXComfyUI --snapshot /tmp/snaps
```

`--smoke-test` queues one job through the same path as the Generate button and exits with status 0 on success. `--smoke-flux` (FLUX.1 [dev] with a LoRA, then image to image and back to Qwen), `--smoke-wan`, `--smoke-video`, `--smoke-sound`, `--smoke-music` and `--smoke-speech` do the same for the other modes with their installed models, and `--smoke-media` checks video/audio I/O and model eviction without any model. `--snapshot` renders the window offscreen to PNG files. `--open <file>` opens a graph (JSON, or a PNG/MP4/audio file saved by the app) at launch, and `--generate` also queues it, e.g. `open build/MLXComfyUI.app --args --open ~/graph.json --generate`.

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
| `generate-video` | Text or image to video with sound. `--model <ltx folder> --prompt … --size 768x512 --seconds 5 --fps 24 --seed`, `--image first.png` for image-to-video, `--contact-sheet x.png` writes 6 frames, `--preview-prefix p` saves each step's preview. With `--model wan-2.2-t2v` or `wan-2.2-i2v` (sizes in multiples of 16, e.g. `--size 832x480`), `--quality full` uses the official 40-step sampler |
| `generate-audio` | Text to sound: `--model <folder> --prompt … --seconds 10 --seed`, `--out x.wav` or `.m4a` |
| `run examples/txt2sound.json` | The Sound mode's graph |
| `generate-music` | Text and lyrics to music: `--model <ace-step folder> --tags … --lyrics "…"` (or a path to a text file) `--seconds 30 --seed`, optional `--bpm --key "A minor" --time-signature 4 --language en`, `--no-planner`, `--out x.wav` or `.m4a` |
| `run examples/txt2music.json` | The Music mode's graph |
| `tts` | Text to speech: `--model <kokoro folder> --text …` (or a path to a text file) `--voice af_heart --speed 1`, `--phonemes` prints the phonemes instead, `--out x.wav` or `.m4a` |
| `run examples/tts.json` | The Speech mode's graph |
| `run examples/txt2video.json` | The Video mode's graph. Try `--set latent.width=512 --set latent.height=320 --set latent.seconds=2` for a quick run |
| `generate` | Text to image. Options: `--image` (condition image, repeatable), `--lora path[:strength]` (repeatable), `--size WxH`, `--seed`, `--save-steps`, `--dequantize` |
| `run examples/flux-txt2img.json` | Text to image with FLUX.1 [dev] (28 steps, guidance 3.5) |
| `fit-preview` | Fits the linear latent → RGB step preview from VAE-encoded images (`--model` Qwen-Image or FLUX.1 [dev]) |
| `make-test-lora` | Writes a synthetic LoRA. `--up-scale 0` makes a no-op one |
| `vae-roundtrip` | Encodes and decodes an image and prints PSNR. Use it to check the VAE. `--tiled on` uses tiles |
| `run examples/media-test.json` | Test pattern + tones → mixed → video with soundtrack (MP4) and audio (M4A). Needs no model |
| `assist` | Prompt assistant: `--prompt "a cat"` enhances a prompt; `--mode describe --image x.png` describes an image; `--mode custom --instruction "…"` follows any instruction. `--temperature 0` is deterministic |
| `tokenize` | Prints the Qwen token ids for a string |

Measured on an M4 Max with 48 GB, at 1024² and 8 steps: about 5.7 s per step and about 53 s end to end, with a peak of about 29 GB.

## Editing, inpainting and LoRA

- **Edit or reference:** connect a `LoadImage` to `TextEncode.image` (and optionally `image2`).
  - Qwen3-VL sees the image through its vision tower, and its VAE latents become an image block in the transformer prefix.
  - Set `EmptyLatent` to 0×0 to match the image's aspect ratio at about 1 MP.
  - From the CLI: `generate --image examples/girl.png --prompt "Put her on a snowy mountain top"`.
- **Inpainting:** see [examples/inpaint.json](examples/inpaint.json).
  - The chain is `VAEEncode` → `SetLatentMask` (from `LoadMask` → `FeatherMask`, white = regenerate) → `Sampler`, then `ImageCompositeMasked` blends the result over the original with the same feathered mask. Pixels where the feathered mask is 0 are bit-identical.
  - `FeatherMask` grows (or shrinks, with a negative value) the mask and blurs its edge. With feather 0 the result is the same as without it.
  - `SetLatentMask.blend`: `hard` (default) fully regenerates every latent the mask touches; `soft` follows the mask's coverage.
- **Outpainting:** see [examples/outpaint.json](examples/outpaint.json). `PadImage` extends the canvas and outputs a mask over the new border, which drives the same chain as inpainting.
- **Hi-res fix:** see [examples/hires.json](examples/hires.json): `VAEDecode` → `ImageScale` → `VAEEncode` → a second `Sampler` with `steps` 8 and `denoise` 0.4 → `VAEDecode`.
  - `steps` 8 gives an evenly spaced schedule, so denoise 0.4 runs 3 refinement steps (about 45 s at 1536²). With `steps` 0 the checkpoint's distilled schedule is used; it ends at σ 0.41, so any denoise below 0.7 gets a single step and less added detail.
- **Large images:** `VAEEncode` and `VAEDecode` process the image in overlapping tiles above about 1.25 MP (`tiled`: `auto`, `on` or `off`). A whole-image VAE pass peaks at about 16 GB at 1536² and 28 GB at 2048²; tiled, it stays near 8 GB at the same accuracy.
  - `Sampler.denoise` < 1 on an encoded latent gives img2img instead.
- **Video and audio:** `TestPattern`, `TestTone`, `LoadVideo`, `LoadAudio`, `MixAudio` (with offset and gain, resampling as needed), `AddAudioToVideo` (replace or mix the soundtrack, fitted to the video), `SaveVideo` (MP4 or MOV, H.264 or HEVC, AAC audio), `SaveAudio` (WAV or M4A), `PreviewVideo` and `PreviewAudio`. Files carry the graph: MP4/M4A in the iTunes comment field, WAV in a custom `mlxc` chunk. See [examples/media-test.json](examples/media-test.json).
- **Flux:** `Sampler.guidance` sets FLUX.1 [dev]'s distilled guidance (0 = its default 3.5; other models ignore it). `TextEncode` ignores connected images for Flux, since it has no reference conditioning, so a `LoadImage` feeds only the latent path (`VAEEncode` for image to image, inpainting and outpainting). See [examples/flux-txt2img.json](examples/flux-txt2img.json).
- **Video models:** `TextEncode` accepts a video model too. `EmptyVideoLatent` (width, height, seconds, fps) → `VideoSampler` (model, conditioning, latent, seed, steps, optional `image` for the first frame, `quality` `fast` or `full`, used by Wan) → `VideoDecode` → `video` and `audio`, then `SaveVideo`. See [examples/txt2video.json](examples/txt2video.json).
- **Audio models:** `TextToAudio` (model, prompt, seconds, seed, steps) → `audio`, then `SaveAudio`. See [examples/txt2sound.json](examples/txt2sound.json).
- **Speech:** `TextToSpeech` (model, text, voice, speed, seed) → `audio`. See [examples/tts.json](examples/tts.json).
- **Music:** `TextToMusic` (model, tags, lyrics, seconds, seed, steps, planner, bpm, key, time_signature, language; 0 / "auto" leave a field to the model) → `audio`. See [examples/txt2music.json](examples/txt2music.json).
- **Prompt assistant:** the `PromptAssistant` node (model, prompt, optional image, mode `enhance`, `describe` or `custom`) outputs a STRING; link it into `TextEncode.prompt`. See [examples/assistant.json](examples/assistant.json). It is cached like any node, so it reruns only when its own inputs change.
  - Qwen-Image's text encoder is the full Qwen3-VL 8B chat model. Its LM head isn't needed for images, so it is loaded only on first use (8-bit, about 0.6 GB). Generation runs at about 28 tokens per second; a prompt takes about 5 s.
- **LoRA:** the `LoRA` node (model, path, strength) returns a model that shares the base weights. Each target layer runs the LoRA as a low-rank side branch, because the base weights are 8-bit and can't be merged into.
  - Supported key formats: diffusers/PEFT (`lora_A`/`lora_B`), kohya/Comfy (`lora_down`/`lora_up` + `alpha`) and `lora_unet_…` underscore names.
  - Flux also takes kohya/BFL files with the original names (`lora_unet_double_blocks_0_img_attn_qkv`, `single_blocks_N_linear1`, or `diffusion_model.double_blocks…`), whose fused q/k/v (and q/k/v/MLP) projections are split by rows onto diffusers' separate layers. That gives exactly the fused update. ai-toolkit files use diffusers names. Text-encoder keys (`lora_te…`) are ignored and counted in the report.
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

- `Sources/MLXDiffusionCore`: safetensors weight store, dense and quantized `Lin`, `Conv2d`, the flow-match Euler scheduler, media clips and file I/O (`Media.swift`, `MediaIO.swift`), the model catalog and resumable downloader (`ModelDownloads.swift`), PNG I/O, and mask and tiling helpers (`MaskOps`, `Tiling`).
- `Sources/QwenImage21`:
  - `Qwen3TextEncoder`: features from the last layer, taken before the final norm. For image prompts it adds the vision tower (`Qwen3VisionTower`), interleaved mRoPE and deepstack.
  - `QwenImage21Transformer`: block-causal DiT. The prompt's K/V are cached once and each step runs only the image tokens.
  - `QwenImage21VAE`: the single-frame form of the causal 3D VAE.
  - `QwenImage21Pipeline`.
  - `TextGenerator.swift`: KV-cached text generation with the same decoder stack (the prompt assistant).
- `Sources/LTX2*`: LTX-2.3, ported from the MIT-licensed MLX port [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) of [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2):
  - `LTX2Text`: Gemma 3 12B (all 49 hidden states), the per-token RMS feature stack and the video/audio embedding connectors.
  - `LTX2Transformer`: the 48-block dual-stream DiT (video 4096-wide, audio 2048-wide, bidirectional audio↔video cross-attention, per-token timesteps), positions and RoPE.
  - `LTX2VAE`: the causal video VAE encoder and decoder (tiled, streaming) and the 2× latent upsampler.
  - `LTX2Audio`: the audio VAE decoder (latents → mel), the BigVGAN vocoder (16 kHz) and bandwidth extension to 48 kHz stereo.
  - `LTX2`: the distilled two-stage pipeline, sampling loop, image conditioning (with an H.264 round trip, as the reference does with libx264 CRF 33), and the sampler preview.
  - Each component was checked against the Python reference on the real weights: the VAE, audio VAE and transformer blocks match bit for bit, the vocoder to float32 rounding, and the text features to about 0.5% (an rms-norm kernel rounds differently in one place). `ltx-probe-*` are the comparison tools used for that.
- `Sources/StableAudio`: Stable Audio Open Small, ported from Stability AI's [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (MIT), checked module by module against the MLX port [sandst1/stable-audio-mlx](https://github.com/sandst1/stable-audio-mlx) (MIT): the T5-base encoder, the seconds conditioner, the 16-block DiT (T5 cross-attention, prepended global token, partial rotary), ping-pong sampling, and the Oobleck VAE decoder (also used by ACE-Step, with diffusers' key layout and tiled decoding). The DiT and T5 match the reference to float32 rounding.
- `Sources/ACEStep*`: ACE-Step 1.5 Turbo, ported from the official [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) (MIT):
  - `ACEStepText`: a plain Qwen3 decoder (Qwen3-Embedding-0.6B for the caption; also the planner's backbone) and the condition encoder (caption projector, 8-layer lyric encoder, 4-layer timbre encoder, sequence packing).
  - `ACEStepDiT`: the 24-layer DiT (alternating sliding-window and full attention, cross-attention, AdaLN), the 8-step Euler sampler and DCW (wavelet-domain correction, on by default for turbo like the official pipeline).
  - `ACEStepLM`: the 5 Hz planner (Qwen3-1.7B): chat prompts, the metadata-constrained decoder for its reasoning block, CFG sampling of audio codes, and the FSQ code decoder that turns codes into 25 Hz hints.
  - `ACEStep`: the pipeline, prompt formats and the silence latent (read straight from the `.pt` zip, see `TorchArchive`).
  - Each module was checked against the official code running on the same weights: the text and condition encoders, the DiT (single steps and the full sampler with DCW), the VAE and the code decoder match to float32 rounding. The planner reproduces the reference's greedy output token for token in float32; the app runs it in bf16 (about 120 tokens/s). Given the same noise, the whole pipeline's audio matches the reference to 4e-4 (relative L2). `ace-probe-*` and `mlxcomfy-cli probe-ace-step` are the comparison tools.
- `Sources/Wan*` and `Sources/T5Text`: Wan 2.2 A14B, ported from the official [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) (Apache-2.0), with [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) (MIT, which made the MLX conversions) as the numerical reference:
  - `T5Text`: the T5 v1.1 / umT5-XXL encoder (per-layer or shared relative position bias, gated GELU), with float32 activations.
  - `WanTransformer`: one 14B expert (40 blocks, AdaLN modulation in float32, 3D RoPE, text cross-attention) with Lightning LoRA side branches.
  - `WanVAE`: the Wan 2.1 VAE, chunked with the official causal feature cache (81 frames from 21 latent frames). The full-resolution stages run frame by frame and the large convolutions in bands, so decoding a 5 s clip peaks at about 9 GB.
  - `Wan`: the pipeline (expert switching at the boundary timestep, the Lightning Euler and official UniPC samplers, image-to-video conditioning, previews).
  - Checked against the references on the same weights: one expert's output matches mlx-video to bf16 noise (0.4%, the same change a 0.01% input nudge causes), umT5 to 3e-6, the VAE matches the official PyTorch code to 1e-6, token ids and the UniPC steps exactly. `wan-probe` is the comparison tool.
- `Sources/Flux*` and `Sources/CLIPText`: FLUX.1 [dev], ported from diffusers' `FluxPipeline`, `FluxTransformer2DModel` and `AutoencoderKL` (Apache-2.0), which were also the numerical reference:
  - `CLIPText`: the CLIP-L text encoder (pooled output) and a CLIP BPE tokenizer that matches transformers' slow `CLIPTokenizer` (including CJK splitting). T5-XXL comes from `T5Text` in its shared-bias Hugging Face layout, padded to 512 tokens as the reference does, and is loaded only while a prompt is encoded.
  - `FluxTransformer`: the 12B MMDiT (19 double-stream and 38 single-stream blocks, 3-axis RoPE, guidance embedding). Residual streams, norms and modulation are float32, matmuls bf16. The timestep and guidance are rounded to bf16 as the reference and BFL's code do. It has LoRA side branches and the Flux LoRA key mapping.
  - `FluxVAE`: the 16-channel 2D autoencoder, with banded convolutions at full resolution.
  - `Flux`: the pipeline (resolution-dependent schedule shift, 2×2 latent packing, image to image and inpainting through the generic sampler, previews). The transformer is quantized to 8-bit once by `WeightStore.quantizedCache`.
  - Checked against diffusers on the same weights. Token ids are identical (CLIP and T5, including punctuation, accents, CJK and emoji), CLIP's pooled output agrees to 3e-6, T5 to 1e-5, the VAE to 5e-6 (decode) and 5e-5 (encode), and the schedule exactly. With two blocks of each kind in float32 the transformer agrees to 3e-6. The full bf16 model agrees to 2.3% (2.2% at 8-bit), against the reference's own 1.3% change from a 0.01% input nudge. `flux-probe` is the comparison tool.
- `Sources/Kokoro*`: Kokoro-82M, ported from the official [hexgrad/kokoro](https://github.com/hexgrad/kokoro) and [hexgrad/misaki](https://github.com/hexgrad/misaki) (both Apache-2.0):
  - `KokoroG2P`: English G2P with no dependencies beyond Foundation and NaturalLanguage. misaki's lexicons are bundled (`Lexicons/`), spaCy's tokenizer exceptions are ported (MIT), `num2words` is ported exactly. It agrees with misaki (with spaCy) on 99.8% of words in held-out text, US and UK. Unknown words go through morphology (known stem + affixes, compounds) and then public-domain NRL letter-to-sound rules with stress assignment.
  - `Kokoro`: the ALBERT encoder, prosody predictor (bidirectional LSTMs as a single Metal kernel), text encoder and the iSTFTNet decoder with its harmonic source. Every module matches the official code to about 4e-6 and durations match exactly; the waveform matches to 1e-5 given the same noise.
  - `KokoroSpeech`: the pipeline (paragraph splitting, chunking at punctuation, voice rows by phoneme count).
  - `TorchPickle` (in MLXDiffusionCore) reads the `.pth`/`.pt` weights with a pickle interpreter that only understands tensors and dicts and never executes code.
- `Sources/GraphRuntime`: graph model (Codable), node schemas and registry, the caching executor, events and cancellation. Also subgraph flattening, collapse and expand (`Subgraphs.swift`), and the Note and Reroute nodes. Also `PromptTemplate` (choices, wildcards, styles) and `PlotAxis` / `PlotSpec` (XY plots). It doesn't depend on MLX.
- `Sources/NodesStandard`: LoadModel, LoRA, TextEncode, EmptyVideoLatent, VideoSampler, VideoDecode, TextToAudio, TextToMusic, TextToSpeech, the media nodes, PromptAssistant, EmptyLatent, Sampler, VAEEncode, VAEDecode, LoadImage, LoadMask, SetLatentMask, ImageCompositeMasked, FeatherMask, PadImage, ImageScale, SaveImage and PreviewImage.
- `Sources/MLXComfyApp`: the SwiftUI app.
  - `AppModel`: the document graph with undo, the queue, run state and history.
  - `Engine`: an actor that owns the executor and the loaded models.
  - `GraphEditor/`: the node editor.
- `Sources/mlxcomfy-cli`: the command line front end.

The architecture is ported from diffusers' `pipeline_qwenimage21.py`, `transformer_qwenimage21.py` and `autoencoder_kl_qwenimage21.py`.

## License

Apache-2.0; see [LICENSE](LICENSE). The model code is ported from [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers), both Apache-2.0. The LTX-2.3 code is ported from [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) (MIT). Wan code follows Wan-Video/Wan2.2 (Apache-2.0) and mlx-video (MIT). Model weights are covered by their own licenses (FLUX.1 [dev]: FLUX.1 [dev] Non-Commercial License; LTX-2.3: LTX-2 Community License; Gemma: Gemma Terms of Use; Stable Audio Open: Stability AI Community License; T5: Apache-2.0; ACE-Step 1.5: MIT; Kokoro-82M: Apache-2.0; Wan 2.2, its MLX conversions and the Lightning LoRAs: Apache-2.0). Stable Audio code follows stable-audio-tools (MIT); ACE-Step code follows ace-step/ACE-Step-1.5 (MIT); Kokoro code follows hexgrad/kokoro and hexgrad/misaki (Apache-2.0). The bundled pronunciation lexicons are misaki's (Apache-2.0) and the tokenizer exceptions come from spaCy (MIT).
