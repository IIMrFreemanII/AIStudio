# AIStudio

A native macOS image generator with a node graph in the style of ComfyUI. It's pure Swift on [mlx-swift](https://github.com/ml-explore/mlx-swift), with no Python at runtime. It supports text-to-image, editing with a reference image, inpainting, outpainting, a hi-res fix and LoRA, with Qwen-Image 2.1 Turbo, and text-to-image, image-to-image, inpainting and **LoRAs** (kohya, diffusers/PEFT and ai-toolkit files) with **FLUX.1 [dev]**. It also makes **video with a synchronized soundtrack**, from text or from a still image, with LTX-2.3, and silent video with **Wan 2.2 A14B** (text-to-video and image-to-video). A prompt assistant uses the image model's own language model to write prompts. It also generates **sound effects, ambience and loops** from text with Stable Audio Open Small, **songs with or without lyrics** with ACE-Step 1.5, and **narration** in 28 English voices with Kokoro-82M, whose pronunciation step is also pure Swift.

This is an independent project and is not affiliated with ComfyUI.

## Requirements

- An Apple Silicon Mac. Images: about 24 GB resident, peaking around 30 GB at 1024² (FLUX.1 [dev]: about 14 GB resident, peaking around 20 GB). Video: about 27 GB resident, peaking around 35 GB for 5 s at 768×512. Models are swapped as needed, so 48 GB of RAM is comfortable for both.
- macOS 15 or later, and Xcode 16 or later.
- Qwen-Image 2.1 Turbo weights in the diffusers folder layout (`model_index.json`, `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/`). The transformer can be bf16 or MLX-quantized. The weights aren't in this repo; download them from the Models window, or put them in `~/Library/Application Support/AIStudio/models/qwen-turbo-mlx/` yourself, or choose another folder in the app.
- For FLUX.1 [dev]: the official [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) diffusers folders (bf16, about 34 GB), downloaded from the Models window. The repo is gated under the FLUX.1 [dev] Non-Commercial License: accept it on the model page and paste a read token in the Models window. On first use the 24 GB transformer is quantized to 8-bit once and cached in the model folder (`.aistudio-cache/`, 12.6 GB; about 10 s).
- For Wan 2.2 video: downloaded from the Models window. **Wan 2.2 T2V A14B** is about 31 GB: the 4-bit MLX conversion [PocketAiHub/Wan2.2-T2V-A14B-MLX](https://huggingface.co/PocketAiHub/Wan2.2-T2V-A14B-MLX). **Wan 2.2 I2V A14B** is about 45 GB: the 8-bit [Anes1032/Wan2.2-I2V-A14B-mlx-q8](https://huggingface.co/Anes1032/Wan2.2-I2V-A14B-mlx-q8). Each includes the 4-step [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) LoRAs. All Apache-2.0, not gated.
- For video: LTX-2.3, downloaded from the Models window (about 28.6 GB: the 4-bit MLX conversion [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) plus the Gemma 3 12B text encoder [mlx-community/gemma-3-12b-it-4bit](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit)).

## App

Build `build/AIStudio.app` and open it (the script signs it ad hoc; to stop macOS asking again for Keychain access after every rebuild, create a self-signed code-signing certificate named "AIStudio Local Signing" in Keychain Access, and the script uses it; one named "MLXComfyUI Local Signing" from before the rename works too):

```bash
scripts/build-app.sh
```

```bash
open build/AIStudio.app
```

The app is a node graph, as in ComfyUI: every job is a workflow, and the window is organised around it.

**Look:** the app uses the **MetalGraphics Frosted Glass** design system:
- **Type:** JetBrains Mono (with its ligatures) everywhere, at macOS sizes (body 13).
- **Colour:** role colours with a light and a dark value; the app follows macOS, or set Light or Dark in Settings › Appearance or View › Appearance.
- **Surfaces:** text sits on opaque surfaces (cards, the grouped background, the canvas). Glass, the blurred backdrop under a tint, is only for chrome and overlays: the sidebar and bars, the zoom bar, minimap, viewer and sheets.
- **Lines and controls:** 0.5 pt hairlines; rows 24 tall; buttons and fields 22 tall.
- **Graph:** nodes are cards with a coloured category tile. Each data type has its own palette hue: model purple, conditioning orange, latent pink, image blue, mask green, text mint, video red, audio teal.

- **Chrome of its own:** everything inside the windows is drawn by the design system, including what macOS would normally draw:
  - the window bar (the title and tools in a 52 pt bar beside the traffic lights);
  - context menus and pop-up menus, with submenus, shortcuts and type-to-select (long lists get a search field);
  - tooltips, alerts, confirmation dialogs, rename prompts and sheets;
  - the loaded-models popover;
  - sliders, steppers, progress bars and spinners, and scroll bars;
  - the edit menu of text fields (Undo … Select All).
- **Files:** Open, Save As and choosing an image, video or sound use an in-app picker over the app's own folders (the workflow library, Assets › Input and Output). **Browse in Finder…** hands over to the system's panel for anywhere else; choices that are always elsewhere on disk (Import, folder settings, Add Model Folder) go straight to Finder.

The menu bar, the traffic lights, Finder's panels, the Dock menu and system prompts are drawn by macOS and stay native.

- **Workflow tabs:** several workflows are open at once as tabs above the graph. ⌘T opens a blank text-to-image tab, ⌘N the preset gallery, ⌘W closes a tab (asking to save unsaved changes), ⇧⌘] and ⇧⌘[ switch tabs, and tabs can be dragged to reorder. Each tab has its own undo history and its own results. A dot marks unsaved changes. Every open tab, saved or not, comes back when you reopen the app.
- **Workflows** (left rail, folder icon): your saved workflows, the `.json` files in `~/Documents/AIStudio/workflows`, in folders. ⌘S saves the current tab there (the first time, it asks for a name), ⇧⌘S saves a copy. Click to open; right-click to rename, duplicate, move to a folder, reveal in Finder or move to the Trash. Drag a workflow onto a folder to move it. Open tabs follow a renamed or moved file.
- **Presets** (left rail, grid icon, and ⌘N): ready-made workflows to start from. They cover Qwen-Image text to image, image editing, inpainting, outpainting and hi-res fix; FLUX.1 [dev] text to image, image to image and inpainting; LTX-2.3 text and image to video; Wan 2.2 text and image to video; Stable Audio sound effects; ACE-Step songs; and Kokoro speech. Each opens in a new tab with the model's folder filled in. A preset whose model isn't installed is dimmed, with a link to the Models window. **Save as Preset…** (File menu or the inspector) turns the current workflow into your own preset, in `~/Documents/AIStudio/presets`, with its latest result as the card's picture. Model folders and the output folder are stored by name, so a preset also works on another Mac.
- **Graph editor:**
  - Drag a node by its header. It snaps to the grid; hold ⌘ while dragging to place it freely. ⇧-drag on the background selects with a rectangle.
  - Drag an output onto an input to connect them. Drag a connected input away to detach it. Drag from an empty input to an output to connect backwards.
  - **Release a wire on empty canvas** to open node search, filtered to the nodes that can take it (or, from an input, feed it). The node you pick is added there and connected.
  - **Wires** are drawn right-angled with rounded corners by default; View › Wire Style (or Settings) switches to curved or straight. A link that runs backwards goes around the nodes rather than through them. Hover a wire to see its type and ends. Click it to select it, and ⌫ to delete it. Selecting a node highlights the paths through it and dims the rest.
  - **Model wires** are hidden by default, because one model feeds almost every node. Each model input names its source instead ("⟵ Load Model"; double-click it to jump there). The wires show while you hover or select either end, or turn them all on with View › Show Model Wires (⌥⌘M).
  - **Arrange** (⇧⌘L, the canvas menu or the zoom bar) lays the graph out in left-to-right columns with fewer crossings, keeping each group together as a block. With several nodes selected it arranges just those. **Align** (Graph menu or the node menu) lines up edges or centres, or spaces the selection evenly.
  - **Collapse** a node to its title bar with the chevron in its header or ⌥C; its wires meet at its ends.
  - **Copy and paste** nodes with ⌘C, ⌘X and ⌘V (at the pointer), also between tabs and into subgraphs. Pasting drops links to nodes that weren't copied; Graph › Paste Nodes with Connections (⇧⌘V) keeps them.
  - Scroll to pan. ⌘-scroll or pinch to zoom. Each tab keeps its own zoom and position. The **minimap** (bottom left, when the graph doesn't fit) shows where you are; click or drag in it to move. Zoomed out, nodes show values as plain text and then just big titles, so the graph stays readable.
  - Double-click the background or press ⌘K to add a node, or click or drag one from the **Node Library** (left rail). ⌘D duplicates the selection and ⌫ deletes it.
  - Every edit can be undone.
  - The running node is outlined, and each node shows its time or a "cached" badge. A sampler shows its live step preview just under it while it runs.
  - **Results on nodes:** Save and Preview nodes show their latest image, video or sound inside the node. Double-click it to open the viewer.
  - **Notes:** add a Note node (⌘K) for a sticky note you edit in place. Notes never run.
  - **Reroutes:** ⌥-click a wire to add a bend point, or add a Reroute node. A reroute takes the type of what flows into it, so type checks still apply end to end.
  - **Groups:** ⌘G puts a labelled frame around the selection. Drag its title bar to move it with the nodes inside, drag the corner to resize, and rename or recolor it in the inspector.
  - **Subgraphs:** ⌥⌘G collapses the selection into one node. Links that cross the boundary become its inputs and outputs. Double-click the node (or use the inspector) to edit inside it, with a breadcrumb bar to go back (Esc or ⌘↑). ⇧⌘G expands it again. Every instance shares the same definition.
- **Results pane** (under the graph, ⌥⌘R): the tab's live preview while it runs, then its result, with a player for video and a waveform for audio, and a strip of the tab's earlier results. Double-click (or ⌘Y) opens the **viewer**, which fills the window. Pinch or double-click to zoom, drag to pan, ← and → step through the results, and Esc closes it.
- **Inspector** (right): with a node selected, all of its inputs, plus shortcuts for some nodes:
  - **Load Model:** a menu of installed models.
  - **Empty Latent** and **Empty Video Latent:** size presets.
  - **Text Encode:** the prompt assistant and a style.
  - **Load Image:** a thumbnail, **Choose…** and **Paint Mask…**.
  - **Text to Speech:** **Preview Voice**.

  With nothing selected, it shows the workflow itself: name and file, batch count and seed randomizing, a Generate button, and **Add to Workflow**.
- **Add to Workflow** (inspector, Graph menu, or right-click the canvas): splices a ready-made chain into the workflow, wired to its sampler. It can add a **Reference Image** (to edit, or for Flux to redraw image to image), an **Inpaint Mask** (painted in the mask editor), **Outpaint**, a **Hi-res Fix**, a **LoRA** or a video's **First Frame**. Multi-node chains arrive as a labelled group, laid out left to right and placed in free space next to the nodes that feed them (a Save node fed only by the chain moves to its right), and **Remove** takes them out again.
- **Prompt assistant:** on a selected Text Encode node, **Enhance** rewrites a short idea as a detailed prompt, and **Describe Image** writes a prompt from the image wired to it. With a reference image wired in, Enhance turns a short edit request into a clear, specific instruction. The text streams into the prompt; **Stop** keeps what's written so far, and ⌘Z restores the previous prompt. The assistant always uses Qwen-Image's language model (Flux can't write text). Its instructions can be edited in Settings.
- **Assets** (left rail, photo-stack icon), with two tabs, like ComfyUI's input and output folders:
  - **Input:** files to use in workflows, kept in `~/Documents/AIStudio/input` (masks in `input/masks`). **Import** (or drop files on the panel) copies them in; importing the same file again reuses the copy. Load nodes refer to these files by name, so a workflow keeps working when the originals move. Filter by images, masks, videos or sounds, search by name, click to show one in the results pane, double-click for the viewer. Right-click to add it to the workflow as a Load node, use it as the reference image or first frame, rename it (Load nodes in open tabs follow), reveal it in Finder or move it to the Trash.
  - **Output:** everything generated, the files in the output folder and its workflow subfolders. Search by prompt, filter by workflow, show only favorites (★, stored on the file, so they survive moving it), and switch between a list and a grid. Click to show, double-click for the viewer, ⌘-click to compare, or right-click to restore the graph in a new tab or copy the file into Input.
- **Dropping files on the graph:** an image, video or sound becomes a Load Image / Load Video / Load Audio node where you drop it (copied into Input first). Dropped on a Load node, it replaces that node's file; an image dropped on Load Mask goes to `input/masks`. A workflow JSON, or a PNG, video or sound made by the app, opens its graph in a tab instead; hold ⌥ while dropping to use such an output as an input. Files dropped elsewhere on the window open as workflows. The inspector's **Choose…** and Add to Workflow's reference image and first frame also copy the file into Input.
- **Queue** (left rail, list icon): pending, running and finished jobs, each labelled with its workflow. Cancel, remove or reorder them; an XY plot is one collapsed row.
- **Styles and wildcards:** a **Style** on a Text Encode node wraps its prompt in a template (edit the styles in Settings). In any prompt, `{red|green|blue}` picks one option per run and `__name__` picks a line from `~/Library/Application Support/AIStudio/wildcards/name.txt`. Choices are seeded by the run's seed, and the saved file records both the final prompt and the template.
- **XY plot** (toolbar): vary one or two inputs of the current workflow, such as seeds `1-4`, steps `4-12:4`, denoise `0.3, 0.5`, or a word in the prompt (search and replace). Each cell runs as a queued job, shown as one collapsed row, and the result is saved as a labelled grid PNG with the graph embedded.
- **Compare:** ⌘-click a history item, or choose **Compare with Shown Image**, to see it against the current result in the results pane: a draggable split, side by side, or the difference (brightened ×4 so small changes show).
- **Opening files:** ⌘O, or dropping a file on the window, opens a workflow JSON in a tab, or restores the graph embedded in a PNG, video or sound made by the app.
- **Inpainting:** only the masked area is regenerated. Masks painted in the mask editor are saved in `input/masks`. **Feather** (on the Feather Mask node) softens the edge so the change blends in; pixels outside the feathered edge stay identical to the original.
- **Outpainting:** grows the reference image's canvas (set the pixels per side on the Pad Image node); the new border is generated to match.
- **Hi-res fix:** upscales the result and refines it with a second pass at low denoise, for example 1024² → 1536².
- **Fine-tunes from links:** **Models window › Models from Links › Add from Link…** takes a FLUX.1 dev, Qwen-Image 2.1, Qwen-Image or Wan 2.2 fine-tune: a Civitai checkpoint page or a Hugging Face file or diffusers repo. On this Mac (FLUX.1 dev, Qwen-Image 2.1 and Wan 2.2) it's downloaded and converted by `ModelImport` (BFL / ComfyUI / diffusers names; fp8, bf16 or fp16; GGUF and NF4 refused) to an 8-bit transformer in `<models>/<id>` (about 12 GB for FLUX; Wan's experts quantized like the installed base's), which links the base model's text encoders and VAE (so the base model must be installed). A Wan 2.2 fine-tune comes as a high-noise and a low-noise file (one Civitai version with both, two "High" / "Low" versions, or a diffusers repo); one expert alone keeps the base's other. It then shows in the Load Model menu as *<name> (fine-tune)*, or use the row's **Use in This Workflow**. The same entry can install on your GPU server for the cloud FLUX node, and fine-tunes of the original 20B Qwen-Image install there for the cloud Qwen-Image node (see docs/remote-server.md).
- **LoRA:** a `LoRA` node applies a LoRA to the model, with a strength. Several stack in order. It works with Qwen-Image, FLUX.1 [dev] and Wan 2.2 (kohya, diffusers/PEFT and ai-toolkit files). Wan 2.2 LoRAs usually come as a high-noise and a low-noise file: a Civitai version with both, "High" / "Low" versions of one model, or a Hugging Face repo with the two files becomes one library entry, and each file goes to its expert; a single file goes to both experts, or to the one its name says. They stack with the Lightning LoRAs in fast mode. LoRAs come from the library: **Models window › LoRAs › Add from Link…** takes a Civitai model page or a Hugging Face file or repo link, previews it (image, base model, size, trigger words, whether it can be downloaded), and installs it on this Mac (`<models>/loras`, checked against the site's SHA-256), on your GPU servers, or both; **Add File…** adds one from disk. The node's menu lists the LoRAs that fit the graph's model (and can add more); its inspector card shows the trigger words (Copy, Add to Prompt). A LoRA not on this Mac yet downloads on first run, with progress on the node. The library is `<models>/library.json`, so the CLI and a Mac server find the same LoRAs; `path` still works for older graphs. Most Civitai files need your API key: Models window › Civitai API Key (kept in the Keychain; `CIVITAI_TOKEN` for the CLI and a Mac server). Gated Hugging Face repos use the Hugging Face token.
- **FLUX.1 [dev]:** runs 28 steps by default with distilled guidance 3.5 (the Sampler's `guidance`; 0 = the model's default). It has no reference editing, so a reference image is redrawn **image to image** at the Sampler's `denoise`, scaled to about 1 MP. On an M4 Max a 1024² image takes about 3 min 20 s (7 s per step), peaking at about 20 GB; 512² with 8 steps takes about 18 s.
- **Models** (⇧⌘M): lists the models the app supports, with size and license, and downloads them from Hugging Face when you click Download and accept the license. Downloads resume if interrupted and are checked against their SHA-256. Gated models need a Hugging Face token, which is kept in the Keychain and read only when a gated download starts (at most once per launch). You can also add a folder you downloaded yourself. Models are stored in `~/Library/Application Support/AIStudio/models` (changeable in the window). A model path in a graph or a `--model` option that doesn't exist as given, such as the examples' `qwen-turbo-mlx`, is looked up by name in that folder.
- **Memory:** models load on first use and stay in memory while they fit. When a new model doesn't fit the budget (80% of RAM by default; change it in Settings), the least recently used one is unloaded first, never one the running job uses. Click the toolbar memory gauge to see loaded models and unload one.
- **Video** (LTX-2.3): a prompt describing the action, camera and sound; an optional **first frame** image to animate; any size in multiples of 64 (presets on Empty Video Latent: 768×512, 1024×576, portrait and square); length up to 10 s; 24, 25 or 30 fps; seed. Generation runs 8 steps at half resolution, upscales the latents 2×, and refines with 3 steps, showing a preview of the middle frame after each. The result is an MP4 with AAC stereo sound at 48 kHz. A 5 s 768×512 clip takes about 2 minutes on an M4 Max (2.6 s prompt, 114 s sampling, 13 s decoding), peaking at about 35 GB.
- **Video with Wan 2.2 A14B:** silent clips at 16 fps, up to 5 s, at 832×480, 480×832 or 624×624 (or 512×288 for quick tries). Text to video and image to video are separate presets, since they are separate models.
  - **Quality** (on the Video Sampler) is **fast** by default: 4 steps with the Lightning LoRAs, guidance 1. **full** is the official 40-step sampler with guidance and the default negative prompt, about 20× slower.
  - Two 14B experts share the work: the high-noise expert does the first steps and the low-noise expert the rest. The step preview shows which one is running.
  - On an M4 Max a 5 s clip at 832×480 takes about 12 minutes in Fast mode: about 172 s per step (attention over 32,760 tokens is more than half of it), then about 40 s of decoding. Small clips are much quicker, since attention cost grows with the square of the clip size: 2 s at 512×288 takes about a minute. T2V (4-bit) peaks at about 28 GB; I2V (8-bit) at about 41 GB, so it needs the other models out of memory, which the app does on its own.
- **Music** (ACE-Step 1.5 Turbo): a style description (genre, instruments, mood, vocals), lyrics with `[Verse]` / `[Chorus]` markers or `[Instrumental]`, a length from 10 s to 4 min, and optionally tempo, key, time signature and vocal language (each "auto" by default). With `planner` on, a 1.7B language model first writes the song's blueprint: it fills in tempo, key and time signature, expands the description, and writes 5 audio codes per second that lay out the arrangement. The music model then renders the song in 8 steps. The result is a stereo 48 kHz WAV. On an M4 Max a 30 s song takes about 9 s with the planner (5 s without) and a 2-minute song about 21 s, peaking at about 16 GB. The model is MIT-licensed and not gated.
- **Speech** (Kokoro-82M): text of any length (read in chunks of up to 510 phonemes, cut at sentence ends), one of 28 voices (American and British, female and male, with the authors' quality grades), and speed. **Preview Voice** (inspector) speaks a short sample in the chosen voice without saving it. Pronunciation is a pure-Swift port of misaki: its US/UK dictionaries, spaCy's tokenizer rules, Apple's NaturalLanguage for word classes, number/date/money reading, and a rule-based fallback for unknown words (no espeak). `[word](/phonemes/)` forces a pronunciation. Mono 24 kHz WAV; about 40× faster than real time on an M4 Max, using under 3 GB.
- **Sound** (Stable Audio Open Small): a prompt describing the sound (sources, setting, texture; a tempo for loops, e.g. "128 BPM tech house drum loop"), a length up to 11 s, seed and batch. 8 ping-pong steps, stereo 44.1 kHz WAV, peak-normalized. About 0.7 s per clip, peaking around 5 GB. The model is gated on Hugging Face: accept its license on the model page, paste a read token in the Models window, then Download.
- **Video and audio results** play on their Save node, in the results pane and in the viewer, and appear in history, with the graph embedded in the file so dropping it on the window restores it.

Relative Load paths are looked up in the input folder first, then in the working directory. Results are saved to `~/Pictures/AIStudio/<workflow name>/` by default (Save nodes pointed at another folder keep it); change the output folder in Settings (⌘,). Workflows, your presets and input assets live in `~/Documents/AIStudio` (also changeable in Settings); the open tabs and their unsaved changes are kept in `~/Library/Application Support/AIStudio/session`.

To check the app without clicking through it, use these launch flags:

```bash
build/AIStudio.app/Contents/MacOS/AIStudio --smoke-test "a lighthouse at dusk"
```

```bash
build/AIStudio.app/Contents/MacOS/AIStudio --snapshot /tmp/snaps
```

`--smoke-test` queues one job through the same path as the Generate button and exits with status 0 on success. `--smoke-flux` (FLUX.1 [dev] with a LoRA, then image to image and back to Qwen), `--smoke-wan`, `--smoke-video`, `--smoke-sound`, `--smoke-music` and `--smoke-speech` open the matching preset in a tab and do the same with their installed models. `--smoke-media` checks video/audio I/O and model eviction without any model, and `--smoke-workflows` (no model needed) checks tabs and per-tab undo, saving and the library, the restored session, every preset and Add to Workflow chain (placed in free space), Arrange, Align, copy and paste, deleting wires, adding a node from a dropped wire, collapsing, and results landing on their nodes and in the workflow's folder. `--smoke-chrome` (no model needed) drives the design-system chrome with real posted events: context menus on rows, nodes, wires and the canvas with their keyboard, a pop-up picker, the title bar, alerts, the save picker, tooltips and the text edit menu, and checks that no system menu opens. Smoke tests and snapshots use a throwaway library (or `--library-root <dir>`), never your workflows or open tabs. `--snapshot` renders the window offscreen to PNG files. `--export-presets Sources/AIStudioLibrary/Presets` regenerates the built-in presets from the graph templates and Add to Workflow chains. `--open <file>` opens a graph (JSON, or a PNG/MP4/audio file saved by the app) at launch, and `--generate` also queues it, e.g. `open build/AIStudio.app --args --open ~/graph.json --generate`.

### Upgrading from MLXComfyUI

The project used to be called MLXComfyUI. On its first launch AIStudio moves `~/Library/Application Support/MLXComfyUI`,
`~/Documents/MLXComfyUI` and `~/Pictures/MLXComfyUI` to `AIStudio` folders and leaves a link at each old path, so paths
saved in workflows and history keep working (delete the links when you no longer need them). It also copies the old
app's settings and moves the Hugging Face, Civitai and server tokens to their new Keychain names the first time each is
used (macOS may ask once for Keychain access). It still reads graphs embedded by the old app (`mlxcomfy.graph`) and
renames `mlxcomfy.json`, `.mlxcomfy-cache` and `mlxcomfy-link.json` in model folders when it loads them. `MLXCOMFY_*`
environment variables still work as their `AISTUDIO_*` names. For servers, re-run the bootstrap scripts; for a Modal
deployment, see [docs/remote-server.md](docs/remote-server.md#upgrading-a-deployment-from-before-the-rename).

## Build

SwiftPM on the command line cannot compile MLX's Metal shaders, so build with `xcodebuild`:

```bash
xcodebuild build -scheme aistudio-cli -destination 'platform=macOS' -derivedDataPath .build/xcode -configuration Release -skipPackagePluginValidation
```

Run the tests:

```bash
xcodebuild test -scheme AIStudio-Package -destination 'platform=macOS' -derivedDataPath .build/xcode -skipPackagePluginValidation
```

## CLI

Run a graph:

```bash
.build/xcode/Build/Products/Release/aistudio-cli run examples/txt2img.json --set prompt.prompt="A lighthouse at dusk" --set sampler.seed=3
```

Generate without a graph:

```bash
.build/xcode/Build/Products/Release/aistudio-cli generate --prompt "A red fox in fresh snow" --seed 1 --out outputs/fox.png
```

| Command | What it does |
|---|---|
| `run` | Runs a graph JSON file, or the graph embedded in a PNG that `SaveImage` wrote. `--set node.input=value` overrides a literal input. `--then-set` runs a second time in the same process to show the cache at work. Relative Load paths are looked up in `--input-dir` first (default `~/Documents/AIStudio/input`, the app's Assets › Input), then in the working directory |
| `nodes` | Lists node types with their inputs and outputs |
| `graph-from-png` | Prints the graph embedded in a saved PNG |
| `compare` | Pixel diff of two images; exits with status 1 if they differ |
| `generate-video` | Text or image to video with sound. `--model <ltx folder> --prompt … --size 768x512 --seconds 5 --fps 24 --seed`, `--image first.png` for image-to-video, `--contact-sheet x.png` writes 6 frames, `--preview-prefix p` saves each step's preview. With `--model wan-2.2-t2v` or `wan-2.2-i2v` (sizes in multiples of 16, e.g. `--size 832x480`), `--quality full` uses the official 40-step sampler |
| `generate-audio` | Text to sound: `--model <folder> --prompt … --seconds 10 --seed`, `--out x.wav` or `.m4a` |
| `run examples/txt2sound.json` | The Sound preset's graph |
| `generate-music` | Text and lyrics to music: `--model <ace-step folder> --tags … --lyrics "…"` (or a path to a text file) `--seconds 30 --seed`, optional `--bpm --key "A minor" --time-signature 4 --language en`, `--no-planner`, `--out x.wav` or `.m4a` |
| `run examples/txt2music.json` | The Song preset's graph |
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

## Running on another machine

A tab can run on a server instead of this Mac: another Mac running `aistudio-cli serve` executes the whole graph
(same models, same results), with live previews, cancel, and results downloaded into the usual output folders. The
server has a web dashboard (memory, GPU, queue, live run, models, charts, logs). Pick the target in the toolbar next to
Generate, and add servers in the **Servers** panel.

```bash
aistudio-cli serve --show-token                 # create and print the token
aistudio-cli serve                              # listen on 127.0.0.1:8188
tailscale serve --bg http://127.0.0.1:8188      # publish it on your tailnet as https://<mac>.<tailnet>.ts.net
aistudio-cli bench-remote https://studio.tail1234.ts.net --token-file token.txt
```

The app needs https for Tailscale addresses (macOS blocks plain http from apps to `100.x` and `*.ts.net`);
`scripts/bootstrap-mac-server.sh --tailscale` sets all of this up as a login service.

### Cloud GPU nodes (pay per second)

For models your Mac is slow at, **Cloud GPU** nodes (*Wan 2.2 Text to Video*, *Wan 2.2 Image to Video*, *FLUX.1 dev*,
*Qwen-Image* (the original 20B), *ACE-Step 1.5 XL Music*, *Stable Audio 3 Sound*, *Qwen3-TTS Speech* with voice
cloning; cloud FLUX and Qwen-Image take up to four of your own LoRAs through *LoRA (Cloud GPU)* nodes, and fine-tunes
from the library) send single jobs to the app's
own GPU server (`server/gpu`: Python, on NVIDIA hardware), and return ordinary images, videos and audio for the local
nodes after them. On [Modal](https://modal.com) the GPUs start per job
and stop when idle, so you pay for GPU seconds only; each node shows its status (*Starting GPU 0:41*, steps, live
previews) and, when done, the GPU time and estimated cost. Videos are saved exactly as the server encoded them.

```bash
modal run server/gpu/modal_download.py --models wan22,flux,audio   # once: weights into a Modal Volume
modal deploy server/gpu/modal_app.py                               # then add the URL in Servers › + › GPU server
```

The same server runs on a GPU box you rent by the hour (`scripts/bootstrap-gpu-server.sh aistudio-gpu`), or locally
for tests (`python -m aistudio_gpu serve`; on a Mac it uses Metal).

### Chats with language models

**Chats** (rail, or ⌥⌘N) talk to language models on the same GPU server: add any Hugging Face model by its link
(Servers › Language Models › +). The server reads the model's config, picks the cheapest GPUs it fits on (up to 8×
H100/H200/B200), downloads it to its volume and serves it with vLLM. Chats open as tabs next to the workflows, stream
the reply with the model's thinking in a collapsible block, render Markdown and code, and show tokens per second and
what each reply cost. The GPU stays warm while a chat is in front and stops a few minutes after the last message. See
docs/remote-server.md › Language models.

[docs/remote-server.md](docs/remote-server.md) covers choosing a host (your own Mac, a cloud Mac, serverless GPUs that
bill only for GPU seconds), networking with Tailscale or an SSH tunnel, the bootstrap scripts, getting models onto the
server, the dashboard, Modal setup and costs, and troubleshooting.

## Graphs

A graph is JSON with a `nodes` list. Each input is either a literal value or a link written as `{"node": "<id>", "output": "<name>"}`. See [examples/txt2img.json](examples/txt2img.json).

The executor:

- checks node types, links, socket types and required inputs;
- orders nodes topologically and runs only the nodes that feed output nodes;
- caches each node's outputs under a hash of its type, its literal inputs and its upstream hashes. Changing a seed reruns only the Sampler, VAE Decode and Save nodes;
- keeps loaded models in a resource cache, so they stay loaded across runs and graph edits.

`SaveImage` embeds the full graph in the PNG as an `iTXt` chunk under the key `aistudio.graph`.

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
  - Each module was checked against the official code running on the same weights: the text and condition encoders, the DiT (single steps and the full sampler with DCW), the VAE and the code decoder match to float32 rounding. The planner reproduces the reference's greedy output token for token in float32; the app runs it in bf16 (about 120 tokens/s). Given the same noise, the whole pipeline's audio matches the reference to 4e-4 (relative L2). `ace-probe-*` and `aistudio-cli probe-ace-step` are the comparison tools.
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
- `Sources/AIStudioLibrary`: the workflow library (`WorkflowStore`: folders, unique names, rename, move, duplicate), the restored session (`SessionStore`: open tabs and their drafts), presets (`Preset`, `PresetCatalog`, with the built-in presets in `Presets/`), input assets (`AssetStore`: import, kinds, relative paths), the editor's geometry (`GraphGeometry/`: wire routing, the Arrange layout and free-space placement, align and distribute, the node clipboard), and the chrome's (`UIGeometry/`: where menus, submenus and tooltips go on screen, scroll thumbs, menu keyboard navigation). Plain Foundation and CoreGraphics, unit-tested.
- `Sources/AIStudioApp`: the SwiftUI app.
  - `Design/`: the design system in SwiftUI:
    - `MGTokens.swift`: generated colours, text styles, spacing, radii, shadows and glass;
    - `MGTheme.swift`: JetBrains Mono, appearance, and the `.mgRoot()` every window goes through;
    - the controls (`MGControls`: buttons, fields, switch, segmented picker, rows, slider, stepper, progress, spinner; `MGForm`: sections; `MGGlass`);
    - floating surfaces in borderless child panels (`MGFloating`), with `MGMenu` (one item model for context menus, pop-up pickers and the menu bar), `MGTooltip` (`.mgHelp`) and `.mgPopover`;
    - `MGPresent.swift`: alerts, confirmation dialogs, prompts and sheets, drawn over the window by a presenter (`.mgPresentationHost`);
    - `MGWindowChrome.swift` (the title bars), `MGFilePicker.swift`, `MGScroll.swift` (scroll indicators) and `MGTextMenu.swift` (the text edit menu);
    - `MGMappings.swift`: which hue each data type, category and group colour uses.

    The tokens come from `design/metalgraphics/tokens.json`, the design system's own export. To re-sync, copy a new one over it and run `swift scripts/sync-design-tokens.swift`. The fonts are in `Resources/Fonts` with their OFL licence.
  - `AppModel`: the open tabs (`WorkflowDocument`: a graph with its own undo history, editor state and results), the queue, run state and history. `Workflows.swift` holds tabs, opening and saving, the library, presets and the session.
  - `Engine`: an actor that owns the executor and the loaded models; `Backends/` puts it behind `ExecutionBackend` next to `RemoteMLXBackend` (a server), with the Servers panel, the run-target menu and the event coalescer.
  - `GraphEditor/`: the node editor; `TabStrip`, `SidebarRail` (Workflows, Presets, Assets, Queue, Node Library), `QueuePanel` (with the Output list), `Assets.swift` (imports and drops), `ResultsPane` (with the viewer), `PresetGallery`, the inspector (`InspectorView`, `NodeExtras`) and `AddToWorkflow` (the chains in `SimpleChains.swift`).
- `Sources/AIStudioProtocol`: the remote protocol shared by the server, the app and the tests: 20-byte binary event records (`WireEvent`), JSON control messages, metrics ring buffers, `GraphTransport` (rewrites a graph's files for a server, content-addressed uploads) and `RemoteClient` (URLSession and WebSocket client with resumable transfers and reconnect).
- `Sources/AIStudioServer`: `aistudio-cli serve` on Hummingbird: the run queue (`Scheduler`), event fan-out with per-client coalescing and latest-wins previews (`Hub`), uploads (`FileStore`), auth and the path sandbox, and the web dashboard (built from `web/dashboard`, Svelte 5 + uPlot, by `scripts/build-dashboard.sh`).
- `Sources/LLMClient`: chats with LLMs on a GPU server: OpenAI-compatible streaming (SSE, `<think>` splitting, tool-call assembly), the server's `/v1/llm` routes, saved transcripts. The chat UI is `Sources/AIStudioApp/Chat`.
- `Sources/NodesRemote`: the Cloud GPU nodes and their client of the GPU server's job API (uploads, a WebSocket of the same binary event records as the Mac server, cancel, result download).
- `server/gpu`: the GPU server (Python): a FastAPI front desk and workers for Wan 2.2, FLUX.1 dev and Qwen-Image (diffusers), ACE-Step 1.5 XL, Stable Audio 3 and Qwen3-TTS, and language models from links (`aistudio_gpu/llm`: sizing, installs, vLLM, an SSE proxy); `modal_app.py` deploys it to Modal, `Dockerfile` to a GPU box; tests with `pytest` (CPU test tasks and tiny random-weight models).
- `Sources/aistudio-cli`: the command line front end (including `serve`, `download` and `bench-remote`).

The architecture is ported from diffusers' `pipeline_qwenimage21.py`, `transformer_qwenimage21.py` and `autoencoder_kl_qwenimage21.py`.

## License

Apache-2.0; see [LICENSE](LICENSE). The model code is ported from [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers), both Apache-2.0. The LTX-2.3 code is ported from [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) (MIT). Wan code follows Wan-Video/Wan2.2 (Apache-2.0) and mlx-video (MIT). Model weights are covered by their own licenses (FLUX.1 [dev]: FLUX.1 [dev] Non-Commercial License; LTX-2.3: LTX-2 Community License; Gemma: Gemma Terms of Use; Stable Audio Open: Stability AI Community License; T5: Apache-2.0; ACE-Step 1.5: MIT; Kokoro-82M: Apache-2.0; Wan 2.2, its MLX conversions and the Lightning LoRAs: Apache-2.0). Stable Audio code follows stable-audio-tools (MIT); ACE-Step code follows ace-step/ACE-Step-1.5 (MIT); Kokoro code follows hexgrad/kokoro and hexgrad/misaki (Apache-2.0). The bundled pronunciation lexicons are misaki's (Apache-2.0) and the tokenizer exceptions come from spaCy (MIT). The bundled JetBrains Mono font is under the SIL Open Font License 1.1 (`Sources/AIStudioApp/Resources/Fonts/OFL.txt`).
