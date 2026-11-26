# Writing prompts for each model

Each model the app sends a prompt to reads it differently. Some read tags and some read sentences. Some stop after 77
tokens and some read 4096. Some use the negative prompt and some ignore it. This page gives the rules for each one.
App code that writes prompts (`PicturePrompt.swift`, `CharacterArt`, Movie Studio) follows them, and so should you
when you type a prompt into a node.

**How to read the marks.** A rule with a link comes from that model's card or guide, paraphrased. **(code)** means
what our worker or the diffusers pipeline does, read from the source. **(ours)** is our own finding from real runs
or our own rule of thumb, not documentation. Model cards change, so check the link when a rule matters. Defaults are those in
`server/gpu/aistudio_gpu/tasks/__init__.py` and the task modules beside it, as of 2026-11-26.

**Content.** Explicit adult content is allowed with a minimal guard: adults only, nothing minor-coded, and no
real-person likeness. Models that take a negative prompt get `AgeGuard.negative`. Models that don't (MiniMax, the
distilled modes) only see what the prompt says, so the prompt states the adult age itself ("a 30-year-old adult
woman"). The examples here are SFW.

- [The basics](#the-basics)
- [Quick reference](#quick-reference)
- Image: [SD 1.5](#stable-diffusion-15) · [SDXL and its fine-tunes](#sdxl-plain-pony-illustrious-noobai) ·
  [SD 3.5](#stable-diffusion-35-large-and-turbo) · [FLUX.1 dev](#flux1-dev) · [Chroma](#chroma1-hd-and-chroma1-flash) ·
  [Anima](#anima) · [Qwen-Image](#qwen-image) · [Qwen-Image-Edit](#qwen-image-edit-2511) ·
  [MiniMax H3 Image](#minimax-h3-image)
- Video: [Wan 2.2](#wan-22-t2v-and-i2v) · [Wan S2V](#wan-22-s2v) · [LTX-2.3](#ltx-23) ·
  [MiniMax H3 video](#minimax-h3-video-and-references) · [LongCat Avatar](#longcat-video-avatar-16) ·
  [InfiniteTalk](#infinitetalk)
- Audio: [ACE-Step](#ace-step-15-xl-music) · [Stable Audio 3](#stable-audio-3-sound) · [Qwen3-TTS](#qwen3-tts) ·
  [Breeze TTS 2](#breeze-tts-2)
- [Rules for the app's prompt writers](#rules-for-the-apps-prompt-writers)

## The basics

**Token limits.** A text encoder reads a fixed number of tokens. A token is roughly a short word or a piece of a long
one, and every comma, hyphen and digit counts. Past the limit the pipeline cuts the rest silently. (diffusers logs a
warning, but nobody sees it in our containers.) So put what matters first, and with a short encoder put everything
that matters inside the limit.

**Tags or sentences.** CLIP models (SD 1.5, SDXL and its fine-tunes) learned from short captions and alt-text. They
read comma-separated phrases well and long grammar badly. Models with a language-model encoder (T5, Qwen, Gemma,
UMT5) read sentences, and they read relationships too: who is where, who does what, what is not there.

**Negative prompts need guidance.** Classifier-free guidance (CFG) runs the model twice per step, once on the prompt and
once on the negative, and pushes the result away from the negative. Distilled fast modes run once per step at guidance
1, so a negative prompt there does nothing (our workers report `"negative": "ignored"` in the result). The exceptions
are Wan 2.2 in fast mode, where we apply it through NAG inside the cross-attention (`tasks/nag.py`), and the models
whose fast mode bakes it in. Where there is no negative, say what you want instead: "a soft belly, no visible abs", not
a negative of "abs". **(ours)**

**Naming a thing can draw it.** A word in the prompt pulls towards what it names, even inside "no …", and models that
render text will print it. **(ours, PR #90):** "Glamour photo" in a MiniMax prompt came out as print on a man's hoodie
and pulled his body towards a fitness model's. Describe the picture by what is visible in it. Keep labels about the
picture ("glamour photo", "reference sheet", "masterpiece") to models whose card asks for them.

**Weights.** The `(word:1.3)` syntax is AUTOMATIC1111's. It raises attention on `word` 1.3 times, and each plain `( )`
multiplies by 1.1 ([A1111 wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)).
It is a feature of the UI, not of the models. diffusers does not parse it, so today our workers pass the brackets and
numbers to the encoder as text. **(code)** SDXL weights are being added on the server. No other model of ours reads
weights.

**LoRA trigger words.** A LoRA trained on a trigger word (`ohwx woman`, an artist tag, a style name) only works with
that word in the prompt, usually near the start. The LoRA's Civitai or Hugging Face page names the word. Put it where
the encoder will still read it, which on SDXL means inside the first 77 tokens.

## Quick reference

| Model | Reads with | Limit (what we send) | Style | Negative works | Our default |
|---|---|---|---|---|---|
| SD 1.5 | CLIP ViT-L | 77 tokens, rest cut | tags | yes (CFG 7) | 25 steps, CFG 7, DPM++ 2M Karras, 512×768 |
| SDXL + fine-tunes | CLIP ViT-L + OpenCLIP bigG | 77 tokens, rest cut | tags | yes (CFG 6) | 30 steps, CFG 6, DPM++ 2M Karras |
| SD 3.5 Large | 2 CLIP + T5-XXL | CLIP 77, T5 256 | sentences | Large only (3.5) | Turbo: 4 steps, no CFG |
| FLUX.1 dev | CLIP ViT-L + T5-XXL | CLIP 77, T5 512 | sentences | no (distilled) | 28 steps, guidance 3.5 |
| Chroma1-HD | T5-XXL | 512 | sentences | HD only (CFG 3) | Flash: 16 steps, no CFG |
| Anima | Qwen3-0.6B | 512 | tags and/or sentences | yes (CFG 4) | 30 steps, CFG 4 |
| Qwen-Image | Qwen2.5-VL 7B | 512 | sentences | full only (CFG 4) | Lightning: 8 steps, no CFG |
| Qwen-Image-Edit 2511 | Qwen2.5-VL 7B + the pictures | 512 | instructions | full only (CFG 4) | Lightning: 8 steps, no CFG |
| MiniMax H3 Image | Qwen3-VL (H3's) | 4096 | detailed paragraph | never | 6 steps, 4 MP canvas |
| Wan 2.2 T2V / I2V | UMT5-XXL | 512 | sentences | CFG (full) or NAG (fast) | Lightning: 4 steps, NAG |
| Wan 2.2 S2V | UMT5-XXL | 512 | scene and style | yes (CFG 4.5) | 40 steps |
| LTX-2.3 | Gemma 3 12B | 1024 (≤ 200 words advised) | one chronological paragraph | full only | distilled: 8 steps |
| MiniMax H3 video | Qwen3-VL-32B | not published | structured fields | never | Turbo: 8 evaluations |
| LongCat Avatar | UMT5-XXL | 512 | short action line | no (distilled) | 8 steps |
| InfiniteTalk | UMT5-XXL | 512 | short action line | no (LoRA, text CFG 1) | 4 steps |

---

## Image models

### Stable Diffusion 1.5

**Reads with:** one CLIP ViT-L/14 text encoder ([card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)),
77 tokens: 75 of yours plus start and end. `tasks/sd15.py` passes the plain string, so diffusers keeps the first 77
and cuts the rest. **(code)** A fine-tune from Civitai brings its own CLIP weights but the same limit.

**Style:** comma-separated phrases, most important first: subject, then action, place, light, medium. Aim for 40–70
tokens. The card names its weak spots: legible text, compositions like "a red cube on a blue sphere", and faces and
hands, which fine-tunes (Realistic Vision, DreamShaper) mostly fix.

**Control:**
- Negative prompt: yes, at CFG 7. Short and concrete works best: `lowres, blurry, deformed hands, watermark, text`.
- Weights: not parsed yet (see [the basics](#the-basics)).
- LoRAs: up to four; put the trigger word in the first 75 tokens.
- Fine-tune quality tags: only what the fine-tune's page asks for. Base SD 1.5 has none.

**Avoid:** long paragraphs, because everything after token 77 is gone. Avoid piles of "8k, uhd, trending on
artstation, masterpiece": they take tokens from the subject and do little. Avoid sizes far from 512 on the short
side, which give doubled heads and bodies. **(ours)**

**Example**
- Good: `photo of a 30 year old woman reading a book in a cafe, rainy window behind her, warm lamp light, film grain, sharp focus`
  with negative `lowres, blurry, deformed hands, watermark, text`
- Bad: `A beautiful, stunning, ultra-detailed 8k masterpiece photograph, trending on artstation, of a woman who is about thirty years old and is sitting in a cozy little cafe on a rainy afternoon while she reads a novel…` (it runs past 77 tokens before the café and the light, and the quality words use up the budget).

**In AIStudio:** node `RemoteSD15` (task `sd15`, L40S). Defaults: 25 steps, CFG 7, DPM++ 2M Karras, 512×768, fine-tunes
from links, four LoRAs. The app builds no SD 1.5 prompts itself; the preset `cloud-sd15` in
`Sources/AIStudioApp/PresetExport.swift` starts with a literal prompt.

### SDXL (plain, Pony, Illustrious, NoobAI)

**Reads with:** two CLIP encoders, CLIP ViT-L and OpenCLIP ViT-bigG
([card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)). Each reads 77 tokens. `tasks/sdxl.py:108`
passes the plain prompt string, so **only the first 77 tokens count and the rest is dropped silently**. **(code)** Our
character prompts run 90–130 tokens, so the photo's request and its place, which come last, are lost. **(ours)**
Long-prompt support (A1111-style 75-token chunks, concatenated) and `(word:1.3)` weights are being added on the server.
Until they land, keep SDXL prompts under about 70 tokens with the request near the front. A1111's `BREAK` keyword ends
a chunk early and starts the next one
([wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#break)); it will only mean something
once chunking is in.

**Style:** depends on the checkpoint.
- **Plain SDXL and photo fine-tunes (Juggernaut, RealVis):** short phrases or one plain sentence, then light and camera.
  The card's examples are plain sentences. It names the same weak spots as SD 1.5: text, compositions, and faces at
  small sizes.
- **Pony Diffusion V6 XL:** tags, opening with its score tags `score_9, score_8_up, score_7_up, score_6_up, score_5_up,
  score_4_up`, then `source_…` and `rating_…` tags, then Danbooru-style content tags
  ([Civitai page](https://civitai.com/models/257749/pony-diffusion-v6-xl)). The Civitai page refused our automated read
  on 2026-11-26, so this is the page's well-known recipe and has not been re-checked. Read the page before relying on
  it. The score preamble is about 40 tokens, so on Pony the 77-token cut bites hardest. **(ours)**
- **Illustrious XL:** Danbooru tags. Quality tags run from `worst quality` through `bad quality`, `average quality`,
  `good quality` and `best quality` to `masterpiece`. Framing tags: `portrait`, `upper body`, `cowboy shot`,
  `full body`; don't stack conflicting ones such as `close-up` with `cowboy shot`. The card suggests Euler a, 20–28
  steps, CFG 5–7.5
  ([card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)).
- **NoobAI-XL:** tags in the card's order: `1girl/1boy/…`, character, series, artists, special tags, general tags.
  Prefix `masterpiece, best quality, newest, absurdres, highres` (plus a rating tag). Quality tags follow popularity
  percentiles, and date tags run `old` (2005–10) to `newest` (2021–24). The card suggests CFG 5–6, 25–30 steps, Euler a,
  and buckets such as 832×1216 ([card](https://huggingface.co/Laxhar/noobai-XL-1.1)). Its v-prediction versions need a
  v-prediction scheduler, which our worker doesn't set. Install the eps versions. **(code)**

**Control:**
- Negative prompt: yes, with CFG (our default is 6). Illustrious and NoobAI cards open theirs with quality terms:
  `worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, signature, watermark`.
- Weights: coming with the long-prompt work. Until then the brackets are read as text. **(code)**
- Face reference: the optional `face` picture goes through IP-Adapter Plus Face (`face_strength` 0.6). It copies what
  it sees, so `CharacterArt.faceCrop` crops the portrait to the head. Otherwise the portrait's clothes and background
  win over the prompt's. **(ours)**
- Clip skip: SDXL was trained on the penultimate CLIP layer, so the setting changes nothing
  ([wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features)).
- LoRAs: up to four, trigger words inside the first 77 tokens.

**Avoid:** prose paragraphs, because the 77-token cut takes the end. Mixing tag dialects in one prompt (Pony score
tags on an Illustrious model, or Danbooru tags on Juggernaut) does little. **(ours)**

**Example** (a photo fine-tune)
- Good: `photo of a 30-year-old adult woman reading in a cafe, rainy window, warm lamp light, shallow depth of field, candid, natural skin`
  with negative `cartoon, anime, illustration, deformed hands, blurry, watermark, text`
- Bad: `Photo of a 30-year-old adult woman, long wavy auburn hair, green eyes, freckles, slim, wearing a cream knit sweater and jeans, silver earrings, soft smile. Reading a book. At a cafe by a rainy window. Candid smartphone photo, natural light, sharp focus, detailed skin.`
  The description of her look uses the 77 tokens, so "reading a book" and the café never reach the model. **(ours)**

**In AIStudio:** node `RemoteSDXL` (task `sdxl`, L40S). Defaults: 30 steps, CFG 6, DPM++ 2M Karras, fine-tunes from
links, four LoRAs, optional face picture. The app builds SDXL prompts in
`Sources/CharacterCore/CharacterArt.swift`: `prompt(_:request:scene:)` writes a sentence for photoreal characters and
tags for anime ones, and `CharacterArt.negative` adds the style's negative and `AgeGuard.negative`. The creator's option
pictures come from `CharacterOptionVisuals.examplePrompt` and `exampleNegative`.

### Stable Diffusion 3.5 Large and Turbo

**Reads with:** three encoders, CLIP ViT-L and OpenCLIP bigG (77 tokens each) and T5-XXL
([Large card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)). Our worker sets
`max_sequence_length=256` for T5. **(code)** Past token 77 only T5 reads on; past 256 the text is cut.

**Style:** natural-language sentences; the cards show long descriptive prompts. Subject and action first, then setting,
light and style.

**Control:**
- Negative prompt: **Large only** (28 steps, guidance 3.5). Turbo is distilled to 4 steps at guidance 0
  ([Turbo card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)), so it ignores the negative.
  Turbo is our default.
- LoRAs: up to four (SD 3.5 Large family). No fine-tunes.

**Avoid:** relying on a negative in fast mode. Avoid tag soup, which wastes T5.

**Example**
- Good: `A 30-year-old woman reads a paperback at a small café table beside a rain-streaked window. Warm lamp light falls on her face, the street outside is blue and blurred. Candid photograph, 35mm, shallow depth of field.`
- Bad: `woman, cafe, rain, book, masterpiece, best quality, 8k` with `fast` on and a long negative prompt (the tags give T5 nothing to work with, and Turbo ignores the negative).

**In AIStudio:** node `RemoteSD35` (task `sd35-large`, H100). `fast` on by default (Turbo, 4 steps); off is Large at 28
steps and guidance 3.5. Only presets use it; no app code writes SD 3.5 prompts.

### FLUX.1 dev

**Reads with:** CLIP ViT-L (77 tokens; it gives the pooled vector) and T5-XXL. diffusers' `FluxPipeline` reads 512 T5
tokens and cuts the rest **(code)**. The card's example uses `max_sequence_length=512`
([card](https://huggingface.co/black-forest-labs/FLUX.1-dev)).

**Style:** natural sentences. The card says prompt following depends heavily on how the prompt is written. Describe
the scene plainly, subject first, and quote any text it should write.

**Control:**
- Negative prompt: **none.** FLUX.1 dev is guidance-distilled; `guidance` (3.5) is an input to the model, not CFG. The
  node has no negative field.
- LoRAs: up to four, and fine-tunes from links. Trigger words matter.

**Avoid:** "no X" phrasing, because it draws X. Avoid SD-style quality-tag strings.

**Example**
- Good: `A 30-year-old woman reads a paperback at a café table by a rain-streaked window, warm lamp light on her face, the street outside blue and out of focus. A chalkboard behind her reads "SOUP OF THE DAY". Candid 35mm photograph.`
- Bad: `woman reading, cafe, no people in background, no blur` (FLUX has no negative, so "no people" and "no blur" pull people and blur in).

**In AIStudio:** node `RemoteFlux` (task `flux1-dev`, L40S): 28 steps, guidance 3.5, 1024². The local `Flux` pipeline
(MLX) reads prompts the same way. Presets `cloud-flux` and `cloud-flux-lora` in `PresetExport.swift`.

### Chroma1-HD and Chroma1-Flash

**Reads with:** T5-XXL only, no CLIP ([HD card](https://huggingface.co/lodestones/Chroma1-HD)). diffusers'
`ChromaPipeline` reads 512 tokens. **(code)**

**Style:** long natural-language descriptions. The card's own example is a full paragraph about a fashion portrait.
Several sentences are fine: subject and look, pose and action, place, light, camera and medium.

**Control:**
- Negative prompt: **Chroma1-HD with guidance only.** The card runs 40 steps at CFG 3 with a negative along the lines
  of "low quality, ugly, unfinished, out of focus, deformed, disfigured, blurry…". **Chroma1-Flash** has CFG baked in
  (16 steps, guidance 1 in our worker), so the worker doesn't send the negative at all. **(code)** The Flash card is
  empty ([Flash card](https://huggingface.co/lodestones/Chroma1-Flash)).
- Face reference: optional, through XLabs' FLUX IP-Adapter v2 (`face_strength` 0.6).
- LoRAs: Chroma family, up to four; a fine-tune always runs as full HD. **(code)**

**Avoid:** depending on the negative in Flash. Avoid short tag prompts: with T5 alone, sentences carry more than tags.
**(ours)**

**Example**
- Good: `A candid photograph of a 30-year-old woman reading a paperback at a small wooden café table. Rain streaks the window beside her and the street beyond is a blur of blue and amber. A brass lamp lights her face warmly; steam rises from a cup. Shot on 35mm film, shallow depth of field, natural skin texture.`
  with full mode and negative `low quality, ugly, unfinished, out of focus, deformed, blurry, child, minor, teen`
- Bad: the same prompt in Flash mode, relying on the negative for the age guard (Flash never sees it).

**In AIStudio:** node `RemoteChroma` (task `chroma`, H100). `fast` (Flash) is on by default, but character photos turn
it off. `CharacterArt.template(.chroma)` sets `fast: false` so the negative, which carries AgeGuard, applies.

### Anima

**Reads with:** Qwen3-0.6B through a small text conditioner ([card](https://huggingface.co/circlestone-labs/Anima)).
The modular pipeline's default is 512 tokens. **(code)**

**Style:** Danbooru tags, natural language, or both. Tags are lowercase with spaces, not underscores; the `score_` tags
are the exception. Order: quality/meta/year/safety tags, then character count, character, series, artist, general
tags. With natural language write at least two sentences; very short prompts behave unpredictably. With several
characters, name each and then describe them.

**Control:**
- Quality prefix (the card's): `masterpiece, best quality, score_7, safe,`. The Aesthetic version uses no `score_` tags.
- Negative prompt: yes, with CFG 4–5. Our default is the card's:
  `worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration`.
- Artist tags need a leading `@` (`@artist name`). Without it the effect is very weak.
- Weights: work in tools that parse them, but need bigger values than SDXL (the card's example is `(chibi:2)`). Our
  worker doesn't parse them. **(code)**
- LoRAs: up to four kohya-style Anima LoRAs.

**Avoid:** realism (it is an anime and illustration model), one-line prompts, and long text in the picture.

**Example**
- Good: `masterpiece, best quality, score_7, safe, 1girl, adult woman, long auburn hair, green eyes, cream sweater, reading a book, cafe, rainy window, warm lighting. She sits by the window and turns a page, smiling softly.`
- Bad: `photorealistic woman in cafe, 8k` (realism is out of scope, and the prompt is too short).

**In AIStudio:** node `RemoteAnima` (task `anima`, L40S): 30 steps, CFG 4, the card's negative as the default. Preset
`cloud-anima`. Non-commercial licence.

### Qwen-Image

**Reads with:** Qwen2.5-VL 7B as the text encoder. diffusers keeps the first 512 tokens of its output **(code)**.

**Style:** natural sentences, as long as needed. It is strong at writing text in pictures, English and Chinese: put the
exact words in double quotes, as the card's example does ([card](https://huggingface.co/Qwen/Qwen-Image)). The card
ends prompts with a "positive magic" suffix, `, Ultra HD, 4K, cinematic composition.`

**Control:**
- Negative prompt: only with `cfg` above 1 (the card's full setting is 50 steps at true CFG 4, with `" "` as the empty
  negative). Our default is the Lightning 8-step LoRA at CFG 1, which ignores it
  ([Lightning card](https://huggingface.co/lightx2v/Qwen-Image-Lightning)). Raising `cfg` in fast mode applies it but
  doubles the cost per step. NAG doesn't apply, because Qwen-Image's attention is joint. **(code)**
- Sizes: its training sizes, 1328², 1664×928, 928×1664, 1472×1140, 1140×1472, 1584×1056, 1056×1584.
- LoRAs and fine-tunes from links.

**Avoid:** unquoted text you want written, which gets paraphrased. Avoid depending on a negative in fast mode.

**Example**
- Good: `A wide shot of a small café on a rainy evening, seen through its front window. Inside, a 30-year-old woman reads at a table under a brass lamp. The window's gold lettering reads "Café Lumen". Wet cobblestones reflect the warm light, Ultra HD, 4K, cinematic composition.`
- Bad: `cafe with sign Cafe Lumen, woman, rain` (unquoted, so the words drift, and it is too thin for a 7B encoder).

**In AIStudio:** node `RemoteQwenImage` (task `qwen-image`, H100). Movie Studio draws character sheets and location
plates with it: `MovieArt.sheetPrompt` and `platePrompt`, with `MovieStyle.keyframeSuffix` at the end. The
template (`MovieArt.textToImageTemplate`) keeps `fast` on, so the `MovieStyle.negative` it sends is ignored. **(code)**

### Qwen-Image-Edit 2511

**Reads with:** the same Qwen2.5-VL encoder, which also sees the input pictures. The pipeline labels them itself as
"Picture 1:", "Picture 2:", "Picture 3:" before your instruction **(code)**, so refer to them by those names. 512 text
tokens. ([card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511))

**Style:** an instruction plus the final scene: say which picture is who or what, what to keep, and what the new
picture shows. The card's examples describe people by role and position ("the bear on the left"), which also works.

**Control:**
- One to three pictures (`image1`–`image3`); with width and height 0 it uses picture 1's shape at about 1 MP.
- Negative prompt: with `cfg` above 1 only (the card: 40 steps at true CFG 4). Our default is Lightning, 8 steps at
  CFG 1, which ignores it.
- LoRAs: edit LoRAs. Qwen-Image's own seldom fit. **(code)**

**Avoid:** vague keeps such as "same style". Name what is kept: face, hair, build, clothes, architecture, light.
Movie Studio adds "a single frame of a film, not a reference sheet, no plain background", because the model copied
the sheet's grey background and layout into keyframes. **(ours)**

**Example**
- Good: `Picture 1 is Mara. Picture 2 is the place, the lamp room: keep its architecture, materials, light and colours. Mara sits at the desk in the lamp room, reading a letter by lamplight. Medium shot. Keep Mara's face, hair, build and clothes exactly as in her picture; a single frame of a film, not a reference sheet, no plain background.`
- Bad: `make her sit in the room` (which picture is "her", which is the room, and what is kept?).

**In AIStudio:** node `RemoteQwenImageEdit` (task `qwen-image-edit`, H100). Movie Studio keyframes:
`MovieArt.keyframePrompt(for:in:references:)` writes the "Picture n is …" lines, the shot, its framing and the
style suffix, from at most three references (sheets, then the plate, then the previous keyframe).

### MiniMax H3 Image

**Reads with:** MiniMax H3's own Qwen3-VL text encoder. Up to 4096 tokens are used and the rest is truncated
([card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)).

**Style** (from MiniMax's
[h3-image-prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/skills/h3-image-prompt-writing/SKILL.md)):
one detailed English paragraph, in this order:
1. the picture's purpose (a portrait, a product shot, a film still);
2. layout and framing;
3. the subjects and what they do;
4. style, conveyed through visible light, colour and material.

The model was trained on rewritten, detailed prompts; MiniMax's API expands short prompts first. So write the long
version yourself.

**Control:**
- **No negative prompt**; the pipeline rejects one. Exclusions are plain sentences in the prompt: "The image contains
  no text, letters or numbers." Body differences go in positively: "a soft belly, no visible abs". **(ours, PR #90)**
- **No steps or CFG:** six distilled steps with guidance baked in.
- **Reference pictures** are `<Picture 1>` … `<Picture 4>`, in order. Give each a role: fully preserved, partially
  preserved (say what changes), attribute transfer (which attribute), or a weak reference. Then narrate the final
  picture, not the edit.
- **Text in the picture:** quote the exact words.
- **Size:** best at about 16 MP (4–20 MP works). We generate on a 4 MP canvas (`canvas_mp`) for cost and resize.
- No LoRAs or fine-tunes in our worker.

**Avoid:**
- Vague quality words ("premium", "high-end", "stunning"). Describe what makes it so: soft window light, matte linen,
  fine skin texture.
- Labels about the picture. **(ours, PR #90):** "Glamour photo" was printed on a hoodie and pulled a man towards a
  fitness-model body.
- Relying on `AgeGuard.negative`, which can't reach it. The prompt says "adult" and the age.

**Example**
- Good: `A candid lifestyle portrait for a café's website. A medium shot from across the table at eye level, the woman on the left third, the rain-streaked window filling the right. A 30-year-old adult woman with shoulder-length auburn hair sits reading a paperback, a cup of coffee steaming beside her, about to turn the page. Warm tungsten light from a brass lamp on her face, cool blue daylight from the window, matte oak table, soft cream knit sweater, shallow depth of field. The image contains no text, letters or numbers.`
- Bad: `Premium high-end glamour photo of a woman in a cafe, no text` ("glamour photo" may be printed, "premium" adds nothing, and "no text" without a sentence may itself draw text).

**In AIStudio:** node `RemoteH3Image` (task `h3-image`, B200). Character photos with the MiniMax art model: the face
reference is `<Picture 1>` and `CharacterArt.render` puts `minimaxFaceLead` in front ("The same person as in
<Picture 1>, with exactly that face and hair: …"). The creator's option pictures were redrawn with it in PR #90.
Licence: not for the US, EU, UK or Korea.

---

## Video models

### Wan 2.2 T2V and I2V

**Reads with:** UMT5-XXL, 512 tokens; longer prompts are cut and shorter ones padded
([config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/shared_config.py)). UMT5 is multilingual; Wan's
own default negative is Chinese.

**Style:** one paragraph of plain sentences: the subject, what they do over the clip, the setting, then camera and
light. The official examples name subject, action, background and framing
([repo](https://github.com/Wan-Video/Wan2.2)). Training data was labelled for lighting, composition, contrast and colour
tone, so those words work. Wan recommends prompt extension, an LLM that rewrites short prompts into detailed ones. For
I2V the picture already fixes the look, so spend the words on motion and camera.

**Control:**
- **Negative prompt, two ways (code):**
  - full mode (40 steps, guidance 4/3 for T2V, 3.5/3.5 for I2V): CFG applies it, at twice the cost per step;
  - `fast` (our default, the [Lightning LoRAs](https://huggingface.co/lightx2v/Wan2.2-Lightning), 4 steps, no CFG):
    [NAG](https://arxiv.org/abs/2505.21179) applies it inside the cross-attention for a few percent more (scale 11,
    tau 2.5, alpha 0.25, the paper's values for Wan).
  Short concrete negatives work best with NAG: `static, blurry, low resolution, subtitles, deformed hands`
  ([NAG repo](https://github.com/ChenDarYen/Normalized-Attention-Guidance)).
- I2V can end on a second picture (`last_image`).
- LoRAs: per expert (high, low or both) or as a high/low pair; fine-tunes replace one expert or both.

**Avoid:** big, fast motion in fast mode. The Lightning card warns of artifacts with very large motion and of vehicles
reversing direction. Avoid several actions in a 5-second clip. Avoid tags.

**Example** (I2V from a still of the woman in the café)
- Good: `She turns a page, glances up at the rain on the window and smiles to herself, then goes back to reading. Steam drifts from her cup. The camera holds steady with a slow, slight push in. Warm lamp light, soft rain outside.`
  with negative `static, blurry, subtitles, deformed hands, extra fingers`
- Bad: `1girl, cafe, rain, masterpiece, she stands up, runs outside, spins, camera orbits fast` (tags, too many actions,
  and big motion that the 4-step LoRA smears).

**In AIStudio:** nodes `RemoteWanTextToVideo` and `RemoteWanImageToVideo` (tasks `wan22-t2v`, `wan22-i2v`, H100):
832×480, 81 frames at 16 fps, `fast` on. The app writes Wan prompts in two places:
- character videos: `CharacterMedia.motionPrompt` (`Sources/CharacterCore/CharacterMedia.swift`), the age and look,
  the request, then "Natural, smooth motion; the camera holds steady." The negative is `CharacterArt.negative`.
- Movie Studio shots: `MovieShots.motionPrompt` (`Sources/MovieCore/MovieShots.swift`), the shot's motion, then
  "Camera: …", then `MovieStyle.motionSuffix`. The negative is `MovieStyle.negative`.

The local MLX Wan pipeline reads prompts the same way.

### Wan 2.2 S2V

**Reads with:** UMT5-XXL, 512 tokens, plus a wav2vec2 speech encoder
([config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_s2v_14B.py)).

**Style:** the speech drives timing and lips; the prompt sets the scene, style and gestures. The repo's examples are
one scene-and-style line ([card](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)). Say that the person speaks.

**Control:** negative prompt with CFG (40 steps at guidance 4.5 by default). We pass ours as given. Wan's default
negative, which also covers agitated emotion and rapid hand shaking, is Chinese. Clip length follows the audio (up to
`max_seconds`, 15 s).

**Avoid:** big gestures and camera moves; the clip continues pass by pass from the last frames, so they drift.
**(ours)**

**Example**
- Good: `A 30-year-old woman sits at a café table and talks to the camera, calm and warm, small hand gestures. Static medium shot, warm lamp light.`
- Bad: `she talks` (no scene or style).

**In AIStudio:** node `RemoteWanSpeechToVideo` (task `wan22-s2v`). Movie Studio's talking shots with the S2V engine:
`MovieShots.talkingPrompt`, which is `motionPrompt` plus "The person speaks, their lips moving with the words."

### LTX-2.3

**Reads with:** Gemma 3 12B. diffusers' `LTX2Pipeline` reads up to 1024 tokens **(code)**, but Lightricks advise
keeping the prompt within 200 words ([repo](https://github.com/Lightricks/LTX-2)). The repo now lists 2.3 as legacy;
we run 2.3.

**Style:** one flowing paragraph in chronological order, in literal shot-list language. Open with the main action in
one sentence, then movements and gestures, appearance, background, camera angle and moves, light and colour, and any
change. The clip has sound, so write it as prose in the same paragraph ("rain patters on the glass"). Speech is a
voice description plus the quoted line.

**Control:**
- Negative prompt: full mode only (30 steps, CFG 3 with spatio-temporal and modality guidance). A blank negative uses
  Lightricks' default from diffusers. Our default `fast` mode (the distilled transformer, 8 steps) runs without
  guidance and ignores it. **(code)**
- Start picture for image-to-video; LoRAs and fine-tunes (LTX-2.3 family only, as the repo notes).

**Avoid:** going over 200 words and prompts that are lists or tags. Describe the sound too, or the model picks it.
**(ours)**

**Example**
- Good: `A woman in her thirties turns a page at a café table by a rainy window and looks up, smiling. She lifts her cup and takes a sip. The camera holds a medium shot at eye level, slowly pushing in. Warm lamp light on her face, cool blue light from the window. Rain patters on the glass and a coffee machine hisses in the background. She says softly, in a warm low voice, "It's still raining."`
- Bad: `woman, cafe, rain, cinematic, 4k, sound` (no action, no order, no sound described).

**In AIStudio:** node `RemoteLTX23` (task `ltx23-video`, B200), 768×512, 5 s, `fast` on; presets `cloud-ltx23-t2v` and
`cloud-ltx23-i2v`. The local `LTX2` pipeline reads prompts the same way. No app code writes LTX prompts yet.

### MiniMax H3 video and references

**Reads with:** H3-Encoder, the hidden states of Qwen3-VL-32B, with special tokens such as `<d>` added
([repo](https://github.com/MiniMax-AI/MiniMax-H3)). No token limit is published.

**Style:** the structured form in MiniMax's prompt guide
([base-en](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)):
- `integrated_multimodal_description:` the timeline. `[Shot 1]` opens with the overall style ("Cinematic,
  live-action", "2D-animated"), then framing, action, camera and dialogue. Mark later shots `[Shot 2]` … with their cut
  times, and give Shot 1 no time. Camera moves are a sentence: the move, then amplitude and speed only when they matter.
- `overall_soundscape:` 1–4 sentences of ambience and physical sounds, or `N/A` only for real silence.
- `non_diegetic_music:` 1–3 sentences on instruments, tempo, rhythm and dynamics, without abstract mood words, or
  `none`.
- Dialogue: a speaker phrase with a stable ID, then the words verbatim in `<d>[English] …</d>`: `Mara (S1) says
  (quietly): <d>[English] It's still raining.</d>`. Don't translate the words. A voice-over "says in an off-screen
  voiceover" with closed lips; `<scenetrans>` carries speech across a cut, and `<cutoff>` marks speech the clip's end
  cuts off.
- Text on screen goes in double quotes, verbatim.
- From pictures: a first line saying which picture anchors which time ("at 0.00 seconds … `<Picture 1>` is fully
  referenced").

**References** (`h3-references`; the
[ref guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)): tags
`<Subject n>`, `<Picture n>` and `<Audio n>`. Each picture gets a role such as `fully_preserved` or `weak_reference`,
and each audio clip one such as `reference` for timbre only. Audio clips run 2–15 s.

**Control:** no negative prompt (the checkpoints are CFG-distilled, [card](https://huggingface.co/MiniMaxAI/MiniMax-H3)).
Our `fast` is LightX2V's Turbo LoRA at 8 evaluations, and off is 49. LoRAs per partition (FL2VA for `h3-video`,
REF2VA for `h3-references`).

**Avoid:** cuts that add nothing new (use a camera move instead), translated dialogue, mood words in the music line, and
labels that may be drawn (see [the basics](#the-basics)). With a whole three-view character sheet as a reference, the
sheet itself sometimes appeared in the clip, so we cut single views from it and say each picture is "never shown as a
picture of its own". **(ours)**

**Example**
- Good:
  ```
  For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
  integrated_multimodal_description: [Shot 1] Cinematic, live-action, a medium shot. It begins exactly on <Picture 1>, keeping its subjects, clothing, setting and light. Mara turns a page, looks up at the rainy window and smiles; the camera slowly pushes in. Mara (S1) says (quietly, to herself): <d>[English] It's still raining.</d>
  overall_soundscape: Steady rain on the glass, the low murmur of a café, a cup set down on a saucer.
  non_diegetic_music: none
  ```
- Bad: `Mara in a cafe says it's still raining, emotional music` (no fields, so the line may not be spoken verbatim,
  and the music is a mood word).

**In AIStudio:** nodes `RemoteH3Video` and `RemoteH3References` (tasks `h3-video` and `h3-references`, B200),
5–15 s, 768p. Movie Studio writes both prompts in `Sources/MovieCore/MovieH3.swift`: `MovieH3.prompt(for:in:start:end:)`
for shots without lines, and `referencePrompt(for:in:pictures:voices:)` for talking shots, with sheet views as
`<Picture 2…>` and voice samples as `<Audio n>`. Licence: not for the US, EU, UK or Korea.

### LongCat-Video-Avatar 1.6

**Reads with:** LongCat-Video's UMT5-XXL (Wan's), plus Whisper-large-v3 for the speech. **(code)**

**Style:** a short line about the person and that they are talking. The card advises a clear speaking cue ("talking",
"speaking") for natural lips ([card](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar)). The speech sets the
timing.

**Control:** distilled, 8 steps at text and audio guidance 1, so the prompt conditions the clip but no negative applies.
The worker passes a fixed English negative in the style of Wan's, which has no effect at guidance 1. **(code)** Two
speakers: `audio` is the left-hand person, `audio2` the right-hand one, on one timeline.

**Avoid:** describing the speaker's look again when the picture already shows it; spend the words on expression and
gesture. Avoid camera moves; it continues in segments, so they drift. **(ours)**

**Example**
- Good: `A 30-year-old woman talks to the camera, lips moving with the words, a warm natural expression, small nods.`
- Bad: `a woman` (no speaking cue).

**In AIStudio:** node `RemoteLongCatAvatar` (task `longcat-avatar`, H100). Character talking videos use
`CharacterMedia.talkPrompt`, and Movie Studio's LongCat shots use `MovieShots.talkingPrompt`.

### InfiniteTalk

**Reads with:** Wan 2.1 I2V 14B's UMT5-XXL, with a Chinese wav2vec2 speech encoder **(code)**.

**Style:** a short line about who speaks and how; the source clip sets the look and camera move, and the speech sets the
lips, head and body. The repo documents the guidance values, not prompt wording
([repo](https://github.com/MeiGen-AI/InfiniteTalk)).

**Control:** with the lightx2v LoRA: 4 steps, text guidance 1, audio guidance 2, shift 2 (the repo's advice for the
LoRA). Text guidance 1 means no negative prompt. Raise `audio_guidance` (3–5) for tighter lips.

**Avoid:** asking for new actions or camera moves the source clip doesn't have.

**Example**
- Good: `A woman speaks to someone off camera, natural lip movement, calm expression.`
- Bad: a prompt that restates the whole shot and adds a new move: `she stands up and walks to the window while talking`.

**In AIStudio:** node `RemoteInfiniteTalk` (task `infinitetalk`, H100). Movie Studio's dub engine:
`MovieShots.dubRender` sends `talkingPrompt` as `talk_prompt`.

---

## Audio models

### ACE-Step 1.5 XL (music)

**Fields:** `tags`, the caption, and `lyrics`
([tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md)).
- `tags`: the most important input. Give genre, mood, instruments, timbre, vocal type, era and production, comma
  separated. Several dimensions anchor it better than one, and specific beats vague. Keep BPM, key and time signature
  in their own fields (`bpm`, `key`, `time_signature`), not in the tags.
- `lyrics`: section tags `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, with one modifier at
  most (`[Chorus - anthemic]`), lines of about 6–10 syllables, a blank line between sections, and backing vocals in
  parentheses. `[Instrumental]` alone gives no vocals; it is our default.

**Avoid:** clashing styles in one caption, captions that contradict the lyrics (a "solo piano" caption with a
`[Guitar Solo]`), and stacking many tags in one bracket, which can make it sing the tag.

**In AIStudio:** node `RemoteTextToMusic` (task `ace-step-xl`). Movie Studio scores from `MovieStyle.musicTags`
(`Sources/MovieCore/MovieAudio.swift`).

### Stable Audio 3 (sound)

**Field:** `prompt`, a description of the sound. It is distilled: 8 steps, guidance baked in, **no negative prompt**
(`tasks/sound.py`). Its card is gated and we couldn't read it on 2026-11-26
([card](https://huggingface.co/stabilityai/stable-audio-3)), so these rules are ours:
- Name the sources and the space: `heavy rain on a café window, distant traffic, cups clinking, small indoor room`.
- For music, give genre, instruments, tempo (BPM) and mood.
- Movie Studio adds "sound effects only, no music, no speech" to effect prompts (`MovieAudio.effectsPrompt`). Without a
  negative, naming "music" and "speech" may pull them in (see [the basics](#the-basics)). Watch for it, and prefer
  describing only what is heard. **(ours, unverified)**

**In AIStudio:** node `RemoteTextToSound` (task `stable-audio-3`). Movie Studio's shot effects:
`MovieAudio.effectsPrompt`.

### Qwen3-TTS

**Fields** ([card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), [repo](https://github.com/QwenLM/Qwen3-TTS)):
- `text`: the words to say, exactly. Set `language` when it is known rather than Auto.
- `voice` + `description`: a built-in voice, which speaks best in its native language, with an optional style ("excited,
  a little breathless"). With `voice: designed`, the description makes a new voice from a full sentence about persona,
  age, timbre and tone.
- `reference` + `reference_text`: cloning. The exact transcript gives a better clone.

**In AIStudio:** node `RemoteTextToSpeech` (task `qwen3-tts`). In Movie Studio (`MovieAudio.swift`) a character's
voice sample is designed from their voice description, then each line is cloned from that sample. Qwen3-TTS can't
apply a line's delivery while cloning, so the delivery is lost on this engine. **(code)**

### Breeze TTS 2

**Fields** ([repo](https://github.com/breezeblue-ai/breeze-tts)):
- `text`: the words, with vocal events inline: `(laugh)`, `(sigh)`, `(clears throat)`, `(cough)`; in Chinese
  `[笑]`, `[叹气]`.
- `instruction`: with a reference clip, direction ("slowly, restrained"); without one, a voice description. Write it in
  the language of the text. CFG 4 is the repo's advice for both, and our default with an instruction.
- `reference` + `reference_text`: clone. The transcript must be exact and cover everything said, and the clip should be
  clean speech. Adding an instruction turns a clone into direction.

**In AIStudio:** node `RemoteBreezeSpeech` (task `breeze-tts`). Movie Studio sends each line's delivery as the
instruction. Research and non-commercial use only.

---

## Rules for the app's prompt writers

These are the rules for code that writes prompts (`PicturePrompt.swift`, `CharacterArt`, `CharacterMedia`, Movie
Studio). Each writer takes the same facts (age, look, request, place, outfit, time, style) and lays them out for its
model:

1. **SDXL and SD 1.5 (CLIP):** tags or short phrases, request and place **first**, look after, quality words last.
   Until long prompts land on the server, keep it under 75 tokens; afterwards keep the important part in the first
   chunk anyway. Fine-tune tags go first (Pony score tags, NoobAI/Illustrious quality tags), and only on that
   fine-tune. Use weights only once the server parses them. The negative carries the style terms and `AgeGuard.negative`.
2. **Chroma, FLUX, SD 3.5, Qwen-Image (T5 or Qwen):** sentences, subject and action first, then place, light, camera.
   Send a negative only where guidance runs (Chroma HD, SD 3.5 Large, Qwen with `cfg` above 1). FLUX never gets one.
3. **MiniMax H3 Image:** one English paragraph: purpose, framing, subjects and action, then light, colour and material.
   No negative, so state the adult age, and write exclusions and body differences as plain positive sentences. No
   labels or quality words. Give each `<Picture n>` its role.
4. **Wan, LTX, H3 video:** say what moves, in time order, and the camera. On H3, add the sound fields. Quote dialogue.
   Keep within each model's length (LTX ≤ 200 words).
5. **Every writer:** state the subject as an adult with an age of 18 or more, keep `AgeGuard` checks on the request,
   and never name a real person.
