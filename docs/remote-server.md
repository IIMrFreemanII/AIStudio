# Running models on another machine

AIStudio can send work to a server instead of running it on this Mac. There are two kinds of server, and they cost
very different amounts:

| Where the models run | What runs there | You pay | Status |
|---|---|---|---|
| **Another Mac you own** (studio Mac at home or in the office) | this app's own engine (`aistudio-cli serve`): every model and node, same results as local | nothing extra | ready |
| **A rented cloud Mac** (AWS EC2 Mac, Scaleway, MacStadium) | the same server | by the hour, **minimum 24 hours per lease** (Apple's macOS license) | ready |
| **Serverless NVIDIA GPU on Modal** | the app's GPU server (`server/gpu`: Wan 2.2, FLUX.1 dev on diffusers), called by the *Cloud GPU* nodes | **per second of GPU time actually used**; idle costs nothing | ready (section 7) |
| **An always-on GPU box** (RunPod/Vast/Lambda pod, your own Linux PC) | the same GPU server behind a token gate (`scripts/bootstrap-gpu-server.sh aistudio-gpu`) | by the hour while it's on | ready (section 7) |

MLX only runs on Apple Silicon, so a Mac server runs *this app's* models (Qwen-Image, FLUX, LTX, Wan, ACE-Step…) and
a whole tab's graph goes there. On NVIDIA, the app's own Python GPU server runs Wan 2.2 and FLUX.1 dev with Hugging
Face diffusers, and single **Cloud GPU nodes** in your graph send it jobs; their results come back as ordinary images
and videos for the local nodes after them.

Sections 1–6 set up a **Mac server**: build it, keep it running, reach it safely, connect the app, and watch it from
the web dashboard. Section 7 sets up the **GPU server**.

---

## 1. Pick the machine

The server keeps models in memory between runs and unloads the least recently used one when a new one doesn't fit the
memory budget (80 % of RAM by default). What a model needs while generating, measured on an M4 Max:

| Model | Peak memory | Suggested RAM |
|---|---|---|
| Kokoro (speech), Stable Audio Open Small | < 4 GB | any |
| ACE-Step 1.5 (music) | ~16 GB | 24 GB+ |
| FLUX.1 dev (8-bit) | ~14 GB resident, ~20 GB peak | 32 GB+ |
| Qwen-Image 2.1 Turbo | ~24 GB resident, ~30 GB peak at 1024² | 36 GB+ |
| LTX-2.3 (video + audio) | ~27 GB resident, ~35 GB peak (5 s, 768×512) | 48 GB+ |
| Wan 2.2 A14B (video) | more than LTX; the I2V 8-bit weights alone are 45 GB | 64–96 GB+ |

Speed follows the GPU: an M4 Max (40 GPU cores) does a 1024² Qwen-Image image in about 45 s; expect chips with fewer
GPU cores to be correspondingly slower. Disk: the weights above add up to ~180 GB for all models; get at least 256 GB
free.

**Cloud Mac notes**
- **AWS EC2 Mac** (`mac2-m2pro.metal`, `mac2-m2.metal`, …) runs on a *Dedicated Host* you allocate for at least
  24 hours. Use a macOS AMI with Xcode, or install Xcode yourself (step 3). You manage it over SSH and Screen Sharing.
- **Scaleway Apple silicon** and **MacStadium** rent whole Mac minis / Mac Studios, billed for at least 24 hours
  (often longer terms); check their current offers.
- Whichever you choose, the Mac must stay **logged in** (the server runs as a login service and needs the GPU of a
  user session) and must **not sleep**. The bootstrap script shows how.

## 2. Plan how you'll reach it (networking is yours)

The server listens on **127.0.0.1** by default and every request needs the **token**. Pick one way in:

1. **Tailscale (recommended).** Install Tailscale on the server and on your laptop and sign both into your tailnet.
   In the [admin console](https://login.tailscale.com/admin/dns) turn on **MagicDNS** and **HTTPS Certificates**
   (DNS page). Then bootstrap with `--tailscale`: the server stays on 127.0.0.1 and `tailscale serve` publishes it
   at **`https://<mac>.<tailnet>.ts.net`** with a real certificate. Nothing is exposed to the internet; only your
   tailnet can reach it. In the app use that https URL.

   > The app needs **https** for tailnet addresses: macOS's App Transport Security blocks plain `http://` from the
   > app to `100.x.y.z` and `*.ts.net` (it allows LAN addresses, `.local` names and 127.0.0.1). `aistudio-cli` is not
   > affected, so `bench-remote http://100.x.y.z:8188` still works if you bind the server there with `--host`.
2. **SSH tunnel.** Keep the default 127.0.0.1 and forward a port from your laptop:

   ```bash
   ssh -N -L 8188:127.0.0.1:8188 you@cloud-mac
   ```

   In the app use `http://127.0.0.1:8188` while the tunnel is open.
3. **HTTPS reverse proxy** (only if you must reach it from the open internet). Run Caddy on the server
   (`caddy reverse-proxy --from your.domain --to 127.0.0.1:8188`), open port 443 only, and use
   `https://your.domain`. Never expose the plain-HTTP port publicly: the token would travel unencrypted. The app
   warns when a server URL is plain http to a public address.

Provider firewalls: with Tailscale or SSH you need **no inbound port** besides SSH (22). On AWS keep port 8188 closed
in the security group.

## 3. Bootstrap the server

On the server Mac (over SSH or Screen Sharing), with the repository checked out or not (the download below works once
this script is on the `main` branch):

```bash
curl -fsSLO https://raw.githubusercontent.com/IIMrFreemanII/AIStudio/main/scripts/bootstrap-mac-server.sh
bash bootstrap-mac-server.sh --tailscale
```

or, from a checkout: `scripts/bootstrap-mac-server.sh --source . --tailscale`.

What it does (and asks before the steps that need your password):

1. Checks for Apple Silicon and **full Xcode** (MLX compiles Metal shaders; the Command Line Tools are not enough) and
   installs the Metal toolchain component if it's missing.
2. Clones or updates the repository (default `~/AIStudio`) and builds `aistudio-cli` in Release.
3. Installs the binary with its resource bundles into `~/Library/Application Support/AIStudio/server/bin`.
4. Creates the token in `~/.config/aistudio/token` (256 random bits, readable only by you) unless one exists.
5. Installs a **LaunchAgent** (`com.aistudio.server`): starts at login, restarts if it stops, logs to
   `~/Library/Logs/aistudio-server.log`.
6. Waits until the server answers `/v1/info` with the token.
7. Offers `pmset` settings so the Mac never sleeps and restarts after a power failure, and reminds you to turn on
   **automatic login** (System Settings › Users & Groups): after a reboot the service only starts once a user is
   logged in.

With `--tailscale` it also runs `tailscale serve --bg --https=443 http://127.0.0.1:8188` (the first time, Tailscale
may print a link to enable HTTPS for the tailnet), checks the https URL and prints it. Remove it again with
`tailscale serve --https=443 off`.

Options: `--host ADDR` (listen elsewhere, e.g. a LAN IP), `--port N`, `--models DIR`, `--branch NAME`, `--dry-run` (print only),
`--test` (installs under a temporary name and port, checks it, removes it again).

Show the token whenever you need it:

```bash
~/Library/Application\ Support/AIStudio/server/bin/aistudio-cli serve --show-token
```

## 4. Put models on it

Models live in the server's models folder (`~/Library/Application Support/AIStudio/models`, the same place the app
uses). Graphs refer to models by **folder name**, so a server can run a graph made on your laptop as long as it has a
folder with that name.

- **Download on the server** (fastest from a data center):

  ```bash
  aistudio-cli download                       # list the catalog
  aistudio-cli download qwen-image-2.1-turbo  # by catalog id or folder name
  HF_TOKEN=hf_... aistudio-cli download flux-1-dev   # gated repos need your Hugging Face token
  ```

  Downloads resume where they stopped and are checked against their SHA-256.
- **Copy from your Mac**:

  ```bash
  rsync -aP ~/Library/Application\ Support/AIStudio/models/qwen-turbo-mlx you@cloud-mac:Library/Application\ Support/AIStudio/models/
  ```

A run that needs a model the server doesn't have is refused right away with the missing folder names.

## 5. Connect the app

1. In AIStudio open the **Servers** panel (the rack icon in the left rail) and click **+**.
2. Enter a name, the URL (`https://<mac>.<tailnet>.ts.net` with Tailscale, `http://127.0.0.1:8188` with a tunnel,
   `http://192.168.x.y:8188` on your LAN, or `https://your.domain`) and the
   token. **Test** shows the server's chip, memory and model count. **Save** stores the token in your Keychain.
3. Choose where a tab runs with the **run target** menu in the toolbar (next to Generate), or click a server in the
   Servers panel. Each tab remembers its target; the queue shows `on <server>` for remote items.

What happens on Generate:
- Input files (images, masks, audio, video, LoRAs) are uploaded **once**, by content hash; the next run with the same
  file skips the upload. Uploads resume after a dropped connection.
- The server loads the run's models **while the files upload**.
- Progress and live previews stream back; Cancel stops the run on the server.
- Results are downloaded as soon as each Save node finishes, into the same folders a local run would use, with the
  workflow embedded exactly as on this Mac. They appear in Output history like local results.
- If the connection drops mid-run, the run continues on the server; the app reconnects and replays what it missed.
- A local run and a remote run can go at the same time (one queue worker per target).
- The prompt assistant (Enhance / Describe) runs where the tab runs.

## 6. The web dashboard

Open the server's URL in a browser (through the same tunnel or tailnet) and sign in with the token. The session lasts
12 hours; the token itself is never stored in the browser.

It shows, live:
- **Overview:** MLX memory against the budget, GPU and server CPU, queue state, thermal state, uptime.
- **Live run:** the running workflow, every node's state and time, the current step, and the sampler preview, with a
  Cancel button.
- **Queue & history:** waiting runs (cancel or clear them) and the last 100 runs with status, time, peak memory,
  errors and output thumbnails.
- **Models:** what's loaded (unload), what's installed and whether it fits the budget (preload).
- **Performance:** 10-minute charts of memory, utilization, network, event latency, events and preview frames per
  second.
- **Clients & transfers:** connected apps and dashboards, uploads and downloads in progress, cache and disk space
  (purge the upload cache).
- **Settings:** memory budget and preview defaults. The token is never shown; rotate it on the server.
- **Logs:** the server log, filterable.

The dashboard can't run graphs or change the token: those need the token itself (the app or the CLI).

## 7. Serverless GPUs: pay only while generating

For big models your Mac is slow at or can't fit (Wan 2.2 A14B video above all, and the larger music, sound and speech
models), rent NVIDIA GPUs **by the second** on
[Modal](https://modal.com): containers start when a job arrives and stop shortly after, so you pay for the seconds a GPU
works, plus its start-up, and nothing while you're idle. New Modal accounts get $30 of free compute a month.

How it fits together (all of it is `server/gpu/modal_app.py`):

- A **front desk** (FastAPI on a quarter of a CPU core, fractions of a cent per hour) that the app talks to. Checking
  the server, uploading images and the CPU test tasks **never start a GPU**. Modal's proxy auth rejects requests
  without your token before they reach it.
- **GPU workers**, one per model, each starting on its own GPU type when a job needs it and stopping
  `AISTUDIO_IDLE` seconds (default 120) after its last job:

  | Task | GPU (default) | Weights read per cold start |
  |---|---|---|
  | Wan 2.2 Text to Video (A14B) | H100 80 GB, $3.95/h | ≈ 63 GB (bf16, kept as fp8 on the GPU) |
  | Wan 2.2 Image to Video (A14B) | H100 80 GB, $3.95/h | ≈ 63 GB |
  | FLUX.1 dev | L40S 48 GB, $1.95/h | ≈ 34 GB |
  | Qwen-Image (20B) | H100 80 GB, $3.95/h | ≈ 58 GB (bf16; fp8 on a 48 GB card) |
  | ACE-Step 1.5 XL music (4B planner + 4B DiT) | L40S 48 GB, $1.95/h | ≈ 38 GB |
  | Stable Audio 3 medium sound | L40S 48 GB, $1.95/h | ≈ 10 GB |
  | Qwen3-TTS speech (three 1.7B checkpoints) | L40S 48 GB, $1.95/h | ≈ 14 GB |

- A **Volume** (`aistudio-gpu-models`) with the weights, written once by the download step.

### Set it up (once)

On this Mac, in the repository:

```bash
pip install modal                      # or: uv tool install modal
modal setup                            # opens the browser to sign in
modal secret create huggingface HF_TOKEN=hf_...   # for FLUX.1 dev and Stable Audio 3 (gated: accept their licences on Hugging Face first)
modal run server/gpu/modal_download.py --models wan22,flux,qwen,audio   # into the Volume, on Modal CPUs; ~20–25 min
modal deploy server/gpu/modal_app.py   # prints the front desk's URL: https://<you>--aistudio-gpu-api.<region>.modal.run
```

- `--models` takes `wan22` (both Wan tasks and the Lightning LoRAs, ≈ 140 GB), `flux` (≈ 34 GB), `qwen` (Qwen-Image
  and its Lightning LoRA, ≈ 58 GB, not gated), `audio` (music, sound and speech, ≈ 62 GB), `music`, `sound`, `speech`,
  or single ones (`wan22-t2v`, `wan22-i2v`, `wan22-lightning`, `flux1-dev`, `qwen-image`, `qwen-image-lightning`,
  `ace-step-xl`, `stable-audio-3`, `qwen3-tts`). Finished models are skipped, so it is safe to re-run.
  Without the Hugging Face secret, use `AISTUDIO_HF_SECRET= modal run … --models wan22,music,speech`.
- Gated models: [FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and
  [Stable Audio 3 medium](https://huggingface.co/stabilityai/stable-audio-3-medium). Accept each licence with the
  Hugging Face account whose token is in the secret. Stable Audio 3 is converted to the diffusers layout during the
  download, by diffusers' own script (fetched at the pinned release, checked by its hash).
- The Wan transformers come from bf16 copies of the official weights (`ai-toolkit/…`, `cbensimon/…`: safetensors,
  configs identical to Wan-AI's, pinned to a commit); everything else from `Wan-AI/…`. That halves the download.
- GPUs need a **payment method** on the Modal account (Settings › Usage & Billing), even while the free credit lasts.
  Usage is billed after the fact (the monthly credit first, then the card). Set a **spend limit** there too: past it,
  Modal stops work that would cost more.
- Then create a **proxy token**: Modal dashboard › Settings › Proxy Auth Tokens › New. It has an id (`wk-…`) and a
  secret (`ws-…`).

### Connect the app

Servers panel › **+** › kind **GPU server**, the URL from `modal deploy`, and the token as `wk-….ws-…` (id, a dot,
secret). **Test** lists the tasks, their GPUs and prices, and which models are installed — without starting a GPU.

Then open a Cloud GPU preset (Presets panel or gallery, marked with a cloud): *Wan 2.2 Text to Video · Cloud GPU*,
*Wan 2.2 Image to Video · Cloud GPU*, *FLUX.1 dev · Cloud GPU*, *Qwen-Image · Cloud GPU*, *Song · Cloud GPU*, *Sound Effect · Cloud GPU*,
*Voice Clone & TTS · Cloud GPU*. Or add the nodes yourself (Node Library › remote) and wire them to any local node: a
cloud Wan clip into Save Video, a cloud FLUX image into a local inpaint, a cloud song into Mix Audio. A node's server
input picks the GPU server (empty: the first one).

**LoRAs on cloud FLUX and Qwen-Image:** LoRAs live in the app's library: **Models window › LoRAs › Add from Link…** (or **Add
File…**), where you choose where each one is installed: this Mac, each GPU server, or both, with a status chip per
place. Open *FLUX.1 dev + LoRA · Cloud GPU* (or **Add to Workflow › LoRA** on any graph with the cloud FLUX or
Qwen-Image node) and pick the LoRA in the node's menu, which lists the LoRAs made for that model; the menu can also add
one from a link or a file. Each LoRA (Cloud GPU) node adds one LoRA with a strength; chain up to four. Files in the
kohya/ComfyUI, diffusers/PEFT, ai-toolkit and XLabs formats all work. A LoRA Civitai lists for another model is refused
before anything is sent (*This LoRA is for Flux.1 D, not Qwen-Image*); one that the server can't apply fails with
*LoRA 1 doesn't fit FLUX.1 dev*.

- **Links:** a Civitai model page (`civitai.com/models/…`, optionally with `?modelVersionId=`) or a Hugging Face
  file or repo link. The server downloads the file itself at datacenter speed (22 MB from Hugging Face took 2.5 s),
  so nothing goes through your upload; the node shows *Server downloading 230 / 612 MB · … left*. A Civitai page
  with versions for several models takes the newest one for a model the app runs, and a LoRA for another model
  (SDXL, Pony…) is refused before anything downloads. The node shows Civitai's trigger words when it's done; put them in the prompt.
  The Add sheet previews the link (image, title, base model, size, trigger words) and says whether this Mac and
  each GPU server can download it, checked without downloading. The picked LoRA's card in the inspector has the
  trigger words (Copy, Add to Prompt) and where it's installed. A LoRA removed from a server in the Models window is
  deleted there (`DELETE /v1/files/<sha256>.lora`).
  - **Civitai needs an API key for most files.** Create one on civitai.com (Account settings › API Keys), then
    `modal secret create civitai CIVITAI_TOKEN=…` and `modal deploy server/gpu/modal_app.py` again. The deploy picks
    the secret up when it exists (and the `huggingface` one, for gated Hugging Face repos); without it, Civitai links
    fail with that hint. A GPU box reads `CIVITAI_TOKEN` and `HF_TOKEN` from its environment.
  - The server only downloads from civitai.com, huggingface.co and hf.co, at most 4 GB, and only safetensors files
    that match the site's published SHA-256.
- **Upload:** a file from your Mac uploads at your connection's speed (45 MB took about a minute on ours, 612 MB
  about 4½); the node counts it up: *Uploading 230 / 612 MB · 1.9 MB/s · 3 min left*. A broken upload resumes where
  it stopped. Files are named by their SHA-256 and kept on the
  `aistudio-gpu-uploads` Volume in your Modal account, so later runs, in this session or another, send nothing.
  Start images and reference clips are kept the same way. The Volume holds up to 20 GB (`AISTUDIO_UPLOADS_GB=50
  modal deploy …` to change it); past that, the least recently used files are deleted. Delete them all with
  `modal volume rm -r aistudio-gpu-uploads /`. A GPU box keeps uploads in its `data/uploads` folder, with the same
  limit.
- **GPU:** loading a LoRA adds well under a second. A warm container keeps the last 8 loaded, so a re-run only
  switches them on. A 1024×1024, 28-step image took 16 s on the L40S with or without LoRAs (≈ $0.009).

**Fine-tunes from links (FLUX.1 dev, Qwen-Image, Wan 2.2):** **Models window › Models from Links › Add from Link…**
takes a Civitai checkpoint page or a Hugging Face link (a single `.safetensors`, or a diffusers repo with a
`transformer/` folder). Tick your GPU server and **Add**; the cloud node's **model** menu then lists it next to *FLUX.1
dev (base)*, *Qwen-Image (base)* or *Wan 2.2 T2V / I2V (base)*. (Qwen-Image 2.1 fine-tunes install on this Mac only;
the server runs the original Qwen-Image.)

- **What the server does:** a CPU function (`import_model`, no GPU) downloads the file onto the models Volume and
  converts it to a diffusers transformer next to FLUX.1 dev, whose text encoders and VAE it reuses
  (`variants/flux.1/<id>/transformer`, about 24 GB in bf16). It takes BFL / ComfyUI names and fp8, bf16 or fp16
  checkpoints; Civitai's file is picked in that order (bf16/fp16 first), and GGUF and NF4 files are refused.
  A Qwen-Image fine-tune goes to `variants/qwen-image/<id>/transformer` (about 41 GB in bf16): ComfyUI and diffusers
  files share Qwen-Image's names, so it's copied tensor by tensor (prefix stripped, a full checkpoint's VAE and text
  encoder left out, ComfyUI's scaled fp8 multiplied out); a file that lacks some of the transformer's weights (a
  LoRA, say) is refused with that hint.
  A Wan 2.2 fine-tune has a file per expert (high noise → `transformer`, low noise → `transformer_2`, about 28 GB each
  in bf16). Civitai publishes them as one version with two files or as two versions ("… HIGH" / "… LOW"); the app
  pairs them either way, and a fine-tune of one expert alone leaves the other the base model's. Official / ComfyUI
  names (Kijai's scaled fp8 included) are converted by diffusers' own converter; a file whose name doesn't say which
  expert it is is refused. Kijai's fp8 low-noise expert from Civitai (15 GB) imported in about 17 minutes (15 of them
  downloading) to 28.6 GB; its first clip spent about 40 s more reading it, and switching back to the base about 20 s.
  The Models window's chip follows it (*Server downloading … GB · … left*, *Converting*).
- **Using it:** a job with another model swaps only the transformer's weights in the warm GPU container (read from the
  Volume); the same model again costs nothing extra, and LoRAs still work on top. **Remove from <server>** deletes it
  from the Volume (`DELETE /v1/models/flux.1/<id>`).
- **API:** `POST /v1/models/import`, `GET /v1/models`, `DELETE /v1/models/{family}/{id}`; `/v1/info` lists them as
  `variants`, and the `flux1-dev`, `qwen-image` and Wan tasks take `model` (an id, or `base`). A Wan import's files
  each carry a `role` (`high` / `low`).

**LoRAs on cloud Wan 2.2:** the Wan nodes take *LoRA (Cloud GPU)* nodes too (**Add to Workflow › LoRA**), up to
four. Wan 2.2 A14B LoRAs mostly come as a pair, one file per expert: a pair from Civitai (one version with both files,
or "High" / "Low" versions of the same model) or from a Hugging Face repo with the two files is one library entry,
and each file goes to its own expert (`loraN` and `loraN_low`). A single file goes to both experts, or to the one its
name says (`…high_noise…`, `…LOW…`; `loraN_expert`). They work with the Lightning LoRAs (fast) and without. A
FishEye pair (2 × 153 MB) loaded in about 2 s on the H100; the clip took the same 10 s as without it. Some heavy style
LoRAs don't mix well with the 4-step Lightning mode at full strength (Technically Color gave streaky, blurred frames
there, on the GPU server and on the Mac alike): lower the strength, or turn **fast** off.
- The same entry can also be installed on this Mac (see the README: converted to 8-bit, about 12 GB).

**Qwen-Image (Cloud GPU):** the original 20B Qwen-Image (Apache 2.0), strong at text inside pictures and long
prompts. Its training sizes are 1328×1328 (the default), 1664×928, 1472×1140 and 1584×1056, and their portrait
versions.

- **fast** (on by default): the Lightning 8-step LoRA (lightx2v, V2.0), no guidance.
- **fast** off: the model card's 50 steps with true classifier-free guidance at 4 (`cfg`).
- The **negative** prompt needs guidance: it's used whenever `cfg` is above 1 (the full model's default), and
  ignored in fast mode unless you raise `cfg`, which doubles the time per step. (Wan's NAG trick needs a separate
  cross-attention; Qwen-Image's attention is joint.)
- On an H100 everything stays bf16; `AISTUDIO_GPU_QWEN=L40S` stores the transformer's weights in fp8 instead (48 GB).
- LoRAs and fine-tunes from the library work as for FLUX.

The Wan nodes' **negative** prompt ("blurry, static, distorted hands") works in both modes:

- **fast** on (no guidance): it's applied inside the model's cross-attention with Normalized Attention Guidance
  (NAG; Chen et al. 2025). The Lightning LoRAs are distilled without guidance, so classifier-free guidance wouldn't
  apply a negative prompt here. NAG costs a few percent per step.
- **fast** off, or `guidance` above 1: classifier-free guidance, as in Wan's own sampler. Each step runs the model
  twice.

### Music, sound and speech

| Node | Model | Licence | What it makes |
|---|---|---|---|
| ACE-Step 1.5 XL Music | ACE-Step 1.5 XL turbo DiT (4B) with its 4B planner | MIT | Songs with vocals from style tags and lyrics, or instrumentals; 48 kHz stereo, 10 s – 10 min. Same inputs as the local Text to Music node; `planner` off skips the language model. |
| Stable Audio 3 Sound | Stable Audio 3 medium (2B, distilled, 8 steps) | Stability AI Community: free for personal use and for organisations under $1M a year | Sound effects, ambience and instrumental beds; 44.1 kHz stereo, up to about six minutes. |
| Qwen3-TTS Speech | Qwen3-TTS 12Hz 1.7B Base, CustomVoice and VoiceDesign | Apache-2.0 | Speech in 10 languages, 24 kHz mono. |

The speech node picks its model from what you give it:

1. A clip connected to `reference` (Load Audio, or any audio node): **clones** that voice. A few seconds of clean
   speech is enough; the app sends up to 30 s, as mono. Typing what the clip says into `reference_text` makes the
   clone closer.
2. Otherwise, `voice` **designed**: a new voice made from the `description` ("a calm older man with a British
   accent").
3. Otherwise, one of nine **built-in voices**; the `description`, if any, sets the delivery ("excited, whispering").

Only clone voices you have the right to use.

While a job runs, the node shows what it's doing: *Starting GPU 0:41* (a cold container booting and loading weights),
*Sampling* with steps and live previews, *Downloading result*. When it finishes it shows the GPU time and an estimate of
the cost, e.g. *GPU 48.2 s + 95 s loading · ≈ $0.16* (loading appears only on a cold start). Cancel stops the GPU work
at the next step; the warm container stays for your next job until the idle window ends.

### What it costs

You pay Modal's per-second price while a GPU container is up: start-up and model loading on a cold start, the job
itself, then the idle window (default 2 min) in case another job follows. The node's estimate covers the first two.

- Wan with **fast** on (the default: Lightning LoRAs, 4 steps, no guidance) needs about a tenth of the GPU time of the
  full 40 steps; turn it off for the most detail.
- Back-to-back jobs reuse the warm container: no loading, no start-up.
- To trade money for latency, raise the idle window (`AISTUDIO_IDLE=600 modal deploy …`); to switch GPUs,
  `AISTUDIO_GPU_WAN=A100-80GB`, `AISTUDIO_GPU_FLUX=L4` (24 GB is too small for FLUX in bf16) or
  `AISTUDIO_GPU_AUDIO=A10` (24 GB: enough for speech or sound, tight for music).
Measured on Modal's H100 (2026-11-10), 832×480, 81 frames (5 s at 16 fps), fast on:

| | Wan 2.2 Text to Video | Wan 2.2 Image to Video |
|---|---|---|
| Job on a warm GPU (4 steps + VAE decode + encode) | 19 s ≈ **$0.021** | 24 s ≈ **$0.026** |
| Extra on a cold start (container + loading 63 GB + LoRAs) | ≈ 90–105 s ≈ $0.10–0.12 | ≈ 90 s ≈ $0.10 |
| Idle window afterwards (default 120 s) | ≈ $0.13 per session | |

FLUX.1 dev on Modal's L40S, 1024×1024, 28 steps: **15.8 s ≈ $0.009** per image on a warm GPU; a cold start adds about
30 s (21.5 s of it loading 34 GB) ≈ $0.016; the idle window afterwards ≈ $0.065.

Qwen-Image on Modal's H100 (bf16), 1328×1328: **6.2 s ≈ $0.007** per image in fast mode (Lightning, 8 steps) and
**38 s ≈ $0.042** with the full 50 steps and true CFG, on a warm GPU. A cold start adds about 60 s (51 s of it
loading 58 GB) ≈ $0.065; the idle window afterwards ≈ $0.13. A LoRA adds about 1 s the first time it's loaded. The
one-time download took about 3½ minutes. A Qwen-Image fine-tune from Civitai (Jib Mix Qwen v6, the 20 GB fp8 file)
imported in about 20 minutes, nearly all of it Civitai's download at ≈ 17 MB/s; it's stored as 41 GB in bf16. The
first job with it in a container spends about 100 s reading those weights from the Volume; after that, switching
between it and the base model takes about 10 s.

So a session that makes ten clips costs about 0.12 + 10 × 0.02 + 0.13 ≈ **$0.45**, and twenty FLUX images about
0.02 + 20 × 0.009 + 0.07 ≈ **$0.27**. A single clip after a break costs about $0.27, mostly start-up and idle. The
one-time downloads took 9 minutes for Wan (≈ 140 GB) and 3 minutes for FLUX (34 GB) on Modal CPUs.
`server/gpu/modal_check.py` runs a job straight on the workers and prints these numbers for your own settings.

Music, sound and speech on Modal's L40S (2026-11-10). The idle window afterwards costs about $0.065 for each model
family you used:

| | Job on a warm GPU | Extra on a cold start |
|---|---|---|
| Stable Audio 3, 20 s of sound (8 steps) | 2.0 s ≈ **$0.001** (8 s: 0.7 s) | ≈ 48 s (27 s loading 10 GB) ≈ $0.026 |
| ACE-Step 1.5 XL, 1-minute song with lyrics, planner on | 30 s ≈ **$0.017** | ≈ 84 s (69 s loading 38 GB) ≈ $0.045 |
| ACE-Step 1.5 XL, 2-minute instrumental, planner on | 46 s ≈ $0.025 (most of it the planner) | |
| ACE-Step 1.5 XL, 30 s, planner off | 2.2 s ≈ $0.0012 | |
| Qwen3-TTS, one or two sentences (5–6 s of speech) | 9–12 s ≈ **$0.005–0.006** | ≈ 29 s (14 s loading 14 GB) ≈ $0.016 |

Speech currently takes longer than the audio it makes (plain PyTorch attention, one sentence at a time). The one-time
downloads took about 3 minutes for music and speech together and 2 minutes for Stable Audio 3, including its
conversion.

### Language models (chats)

The same deployment runs language models for the app's **chats** (Chats in the rail, or ⌥⌘N). Each model runs in its
own container behind the front desk's `/v1/llm/…` routes (same proxy token; `GET /v1/llm/oai/v1/models` and
`POST /v1/llm/oai/v1/chat/completions` work with any OpenAI client):

- **Hugging Face, safetensors** (full precision, FP8, AWQ, GPTQ): vLLM (`LLM` class), split across GPUs with tensor
  and expert parallelism.
- **Hugging Face, GGUF** (a repo of GGUF files, or a link to one file; split files are fine): llama.cpp's server
  (`LLMLlamaCpp` class), layers split across GPUs. Look Up lists the repo's quantizations (Q4_K_M first) and reads
  the chosen file's header with HTTP Range requests, without downloading it. A `mmproj` file in the repo gives vision.
- **Civitai** (`civitai.com/models/<id>`, optionally `?modelVersionId=`): a GGUF file runs on llama.cpp; a zip of a
  Hugging Face model folder on vLLM (its `config.json` is read from inside the zip remotely, and only that folder is
  unpacked). A lone `.safetensors` file is refused: it has no config or tokenizer. Most Civitai downloads need the API
  key from the Models window (or the `civitai` Modal secret).

- **Add one:** Servers › Language Models › **+** (or a chat's model menu › Add a Model from a Link…). Paste
  `owner/name` or the model's URL and press **Look Up**: the server reads its config (no download) and lists the GPU
  options it fits on, cheapest first, with the longest context each allows. Gated models use the Hugging Face token
  from the Models window (sent with the request, used once, never stored).
- **Install:** `import_llm` (4 CPUs) downloads the files straight to the models volume: about 3–5 minutes per 100 GB.
- **GPUs:** L40S (1–2), A100 80 GB, H100, H200 or B200, on 1, 2, 4 or 8 of them. The pick is the cheapest that holds
  the weights plus the KV cache for the context you chose (vLLM gets 90% of the memory). Mixture-of-experts models
  split across GPUs with expert parallelism. Change it later in Servers › the model's settings (it
  takes effect at the next start; **Save and Restart** applies it now), or `PATCH /v1/llm/models/<id>`.
- **Warm while you chat:** the first message starts the container (the chat shows *Waiting for a GPU*, then *Loading
  weights 41%*). While a chat with that model is in front, the app is active and you used it in the last 10 minutes,
  it sends a keepalive every minute. The container stops `AISTUDIO_LLM_IDLE` seconds (default 180) after the last
  message or keepalive, and keepalives stop counting 30 minutes after the last message (`AISTUDIO_LLM_LEASE`). The
  power button in the chat (or Servers › model › Shut Down GPU) stops it at once.
- **Cost:** Modal bills the container's whole life, warm and idle, per GPU-second. A chat shows each reply's GPU time
  in dollars and the engine's running total; for example Qwen3.8-Flash-Next FP8 (180 GB) on 2× H200 is about
  $9.08 an hour while warm, so an hour of chatting plus the 3-minute idle tail costs about $9.50. Cold starts load the
  weights from the volume (a few minutes for 100+ GB) and are billed too.
- **Cold starts, timed:** each start is recorded phase by phase (waiting for a GPU, starting the container and the
  engine, loading weights, compiling, capturing CUDA graphs), ten per model, in the `<id>:coldstarts` entry of the
  `aistudio-gpu-llm` Dict. `GET /v1/llm/models/<id>/status` returns the last one and the typical one (`coldStart`),
  and `bootSeconds` while starting. The chat's engine chip says *Loading weights 41% · usually ~2 min*, its tooltip
  and the model's settings show the last start's breakdown, and the container log prints it.
- **Faster starts** (the model's settings, or `PATCH` with these fields):
  - `loadFormat: "runai"` (**Fast weight loading**): vLLM reads the weights with Run:ai's streamer, many reads at
    once (`AISTUDIO_RUNAI_CONCURRENCY`, default 16), instead of one shard at a time. vLLM models only. It's made
    for object storage and network file systems; on Modal's volume it was *slower*. Measured on 2026-11-11 with
    Qwen3.8-Flash-Next FP8 (173 GiB) on 2× H200: the default loader read the weights in 76–128 s (start 5 min 42 s
    – 6 min 48 s), the streamer in 165 s (start 7 min 1 s). Leave it off on Modal unless a measurement says
    otherwise.
  - `snapshot: true` (**GPU snapshot**, experimental): the model runs in the `LLMSnap` class. Its first start loads
    the weights, warms the engine up, puts it to sleep (`--enable-sleep-mode`: weights to CPU memory) and Modal
    saves a GPU memory snapshot; later starts restore it and wake the engine instead of loading and compiling. Only
    for vLLM models on one GPU (Modal's GPU snapshots are alpha and don't handle several GPUs), only on a deployed
    app, and it can take two or three starts before the snapshot is used. Each settings change makes a new
    snapshot. If a model fails to start with it, turn it off.
  - vLLM's compile, inductor and Triton caches, and FlashInfer's kernel builds, live on the models volume
    (`llm/.cache/`), so only the first start of a model on a GPU kind compiles from scratch. The vLLM image also
    has FlashInfer's prebuilt kernels (`flashinfer-cubin`, `flashinfer-jit-cache`, `AISTUDIO_FLASHINFER` /
    `AISTUDIO_FLASHINFER_CUDA` must match vLLM's FlashInfer and the CUDA image): without them, the first start of
    an MoE model builds FlashInfer's MoE kernels on the container's CPUs, which took over 25 minutes.
  - A start that isn't ready after `AISTUDIO_LLM_BOOT_TIMEOUT` (default 30 min) fails with the engine's last log
    lines, and **Stop** ends a start that's still loading.
- Deploy settings: `AISTUDIO_LLM_IDLE`, `AISTUDIO_VLLM` (the vLLM version, default 0.33.0), `AISTUDIO_CUDA_IMAGE`,
  `AISTUDIO_LLAMA_IMAGE` (llama.cpp's server image, default `ghcr.io/ggml-org/llama.cpp:server-cuda`),
  `AISTUDIO_LLM_MAX_GPUS` (default 8).

A GPU box serves the same routes (`python -m aistudio_gpu serve`): there the model runs on the box's own GPUs, one
model at a time, and the cost shows as $0.

### Check a deployment without spending GPU time

The front desk also runs three CPU test tasks (`test-video`, `test-image`, `test-audio`): a moving pattern or a tone,
with progress and a result. `aistudio-cli`'s GPU checks and the app's smoke test use them.

### Remove everything

```bash
modal app stop aistudio-gpu
modal volume delete aistudio-gpu-models   # the downloaded weights
```

### Upgrading a deployment from before the rename

The project used to be called MLXComfyUI, and its Modal app was `mlxcomfy-gpu`. The new app is `aistudio-gpu`, so its
URL changes from `https://<you>--mlxcomfy-gpu-api.modal.run` to `https://<you>--aistudio-gpu-api.modal.run`. Keep the
downloaded weights by renaming the Volumes before you deploy, then stop the old apps:

```bash
modal volume rename mlxcomfy-gpu-models aistudio-gpu-models
modal volume rename mlxcomfy-gpu-uploads aistudio-gpu-uploads
modal deploy server/gpu/modal_app.py
modal app stop mlxcomfy-gpu
modal app stop mlxcomfy-gpu-download
```

Then change the connection's URL in AIStudio › Servers. `MLXCOMFY_*` deploy settings (`MLXCOMFY_IDLE` and so on)
still work as their `AISTUDIO_*` names. The secrets keep their names (`huggingface`, `civitai`).

### Always-on GPU box (hourly)

The same GPU server runs on any Linux NVIDIA machine you rent by the hour or own (Ubuntu, Docker, NVIDIA container
toolkit), behind a small Caddy gate that requires `Authorization: Bearer <token>`:

```bash
scripts/bootstrap-gpu-server.sh aistudio-gpu --download wan22,flux   # FLUX needs HF_TOKEN in the environment
```

The box serves Wan and FLUX; the music and speech models pin older library versions and run on Modal, where each
has its own image. It listens on 127.0.0.1 by default (`--public` to publish it, `--domain your.domain` for HTTPS; over Tailscale use
`tailscale serve` as in section 2) and prints the URL and token for a **GPU server** connection. Here you pay while the
machine is on, busy or not. The `vllm` and `ollama` profiles start OpenAI-compatible servers for the chat nodes that
come in M23. Stop a profile with `--down`. Re-running it on a box set up before the rename (profile `mlxcomfy-gpu`,
data in `~/mlxcomfy-gpu`) moves the data folder to `~/aistudio-gpu`, stops the old containers and starts the new ones;
the token stays the same.

## 8. Operating it

| Task | Command |
|---|---|
| Update to the latest code | re-run `bootstrap-mac-server.sh` (same options) |
| Follow the log | `tail -f ~/Library/Logs/aistudio-server.log` |
| Restart | `launchctl kickstart -k gui/$(id -u)/com.aistudio.server` |
| Stop / start | `launchctl bootout gui/$(id -u)/com.aistudio.server` / `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.aistudio.server.plist` |
| Rotate the token | `rm ~/.config/aistudio/token`, restart, then `aistudio-cli serve --show-token` and update the app |
| Check speed from your laptop | `aistudio-cli bench-remote https://<mac>.<tailnet>.ts.net --token-file token.txt` |
| See or remove the Tailscale publishing | `tailscale serve status` / `tailscale serve --https=443 off` |
| Free the upload cache | dashboard › Clients & transfers › Purge |
| Save money on a cloud Mac | stop/release the host when you're done (the 24 h minimum still applies) |

The server keeps the output folders of the last 200 runs (`--keep-runs`) in
`~/Library/Application Support/AIStudio/server/runs`; the app has its own copies after downloading.

## 9. Security model

- Every API and WebSocket request needs the bearer token, compared in constant time. The dashboard trades the token
  for an `HttpOnly`, `SameSite=Strict` session cookie; changes from the dashboard also need a CSRF header, and its
  WebSocket must come from the same origin.
- A graph from the network can only read **uploaded inputs** and **installed models**, and only write into **its own
  run folder** (absolute paths and `..` are refused).
- The server sends a strict Content-Security-Policy, `X-Frame-Options: DENY` and `nosniff`.
- It does not do TLS itself: use `tailscale serve`, an SSH tunnel or an HTTPS reverse proxy (section 2). Behind a
  proxy that sets `X-Forwarded-Proto: https` the session cookies are `Secure`; a proxy that rewrites `Host` must pass
  the original in `X-Forwarded-Host` (Caddy and `tailscale serve` keep `Host` as it is).

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| App says "the server refused the token (401)" | Token mismatch: `aistudio-cli serve --show-token` on the server and paste it again in Servers › Edit. |
| "Connection refused" | Server not running (`tail` the log, `launchctl kickstart`), wrong port, or it listens on 127.0.0.1 and you didn't open the tunnel. |
| "Timed out" | Tailscale not up on one side, or a firewall in the way. `tailscale ping <server>`. |
| "…App Transport Security policy requires the use of a secure connection" | The URL is plain `http://` to a Tailscale or public address, which macOS blocks for apps. Use the `https://<mac>.<tailnet>.ts.net` URL from `--tailscale` (section 2), an SSH tunnel, or a LAN address. |
| `--tailscale` stops at "enable HTTPS" or the https URL doesn't answer | Turn on MagicDNS and HTTPS Certificates in the admin console (DNS page), then re-run; `tailscale serve status` shows what is published. The first request can take a few seconds while the certificate is issued. |
| Dashboard shows "No Metal device" / runs fail immediately | The service runs outside a logged-in session: enable automatic login and reboot, or log in via Screen Sharing. |
| "models not installed on this server: …" | Download or copy that folder (section 4); names must match the graph's Load Model folder. |
| Runs fail with out-of-memory or the Mac swaps | Lower the budget (dashboard › Settings), unload models, or use a bigger machine. |
| Uploads are slow | Run `bench-remote`; over Tailscale check you have a direct connection (`tailscale status` shows `direct`, not `relay`). |
| Dashboard is blank after sign-in | Hard-reload; check the browser console. The dashboard needs the same origin as the server (open it through the tunnel URL, not a different hostname). |
