# Running models on another machine

MLXComfyUI can send work to a server instead of running it on this Mac. There are two kinds of server, and they cost
very different amounts:

| Where the models run | What runs there | You pay | Status |
|---|---|---|---|
| **Another Mac you own** (studio Mac at home or in the office) | this app's own engine (`mlxcomfy-cli serve`): every model and node, same results as local | nothing extra | ready |
| **A rented cloud Mac** (AWS EC2 Mac, Scaleway, MacStadium) | the same server | by the hour, **minimum 24 hours per lease** (Apple's macOS license) | ready |
| **Serverless NVIDIA GPU** (Modal, RunPod Serverless, Replicate) | ComfyUI / vLLM containers, called through remote nodes | **per second of GPU time actually used**; idle costs nothing | M22 (Modal first) |
| **An always-on GPU box** (RunPod/Vast/Lambda pod, your own Linux PC) | ComfyUI / vLLM / Ollama behind a token gate (`scripts/bootstrap-gpu-server.sh`) | by the hour while it's on | server side ready; app nodes in M22 |

MLX only runs on Apple Silicon, so a Mac server runs *this app's* models (Qwen-Image, FLUX, LTX, Wan, ACE-Step…).
NVIDIA servers run their own model implementations (ComfyUI's FLUX, Wan…) and are reached through the API-agnostic
remote nodes coming in M22.

The rest of this guide sets up a **Mac server**: build it, keep it running, reach it safely, connect the app, and watch
it from the web dashboard.

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
   > app to `100.x.y.z` and `*.ts.net` (it allows LAN addresses, `.local` names and 127.0.0.1). `mlxcomfy-cli` is not
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
curl -fsSLO https://raw.githubusercontent.com/IIMrFreemanII/MLXComfyUI/main/scripts/bootstrap-mac-server.sh
bash bootstrap-mac-server.sh --tailscale
```

or, from a checkout: `scripts/bootstrap-mac-server.sh --source . --tailscale`.

What it does (and asks before the steps that need your password):

1. Checks for Apple Silicon and **full Xcode** (MLX compiles Metal shaders; the Command Line Tools are not enough) and
   installs the Metal toolchain component if it's missing.
2. Clones or updates the repository (default `~/MLXComfyUI`) and builds `mlxcomfy-cli` in Release.
3. Installs the binary with its resource bundles into `~/Library/Application Support/MLXComfyUI/server/bin`.
4. Creates the token in `~/.config/mlxcomfy/token` (256 random bits, readable only by you) unless one exists.
5. Installs a **LaunchAgent** (`com.mlxcomfy.server`): starts at login, restarts if it stops, logs to
   `~/Library/Logs/mlxcomfy-server.log`.
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
~/Library/Application\ Support/MLXComfyUI/server/bin/mlxcomfy-cli serve --show-token
```

## 4. Put models on it

Models live in the server's models folder (`~/Library/Application Support/MLXComfyUI/models`, the same place the app
uses). Graphs refer to models by **folder name**, so a server can run a graph made on your laptop as long as it has a
folder with that name.

- **Download on the server** (fastest from a data center):

  ```bash
  mlxcomfy-cli download                       # list the catalog
  mlxcomfy-cli download qwen-image-2.1-turbo  # by catalog id or folder name
  HF_TOKEN=hf_... mlxcomfy-cli download flux-1-dev   # gated repos need your Hugging Face token
  ```

  Downloads resume where they stopped and are checked against their SHA-256.
- **Copy from your Mac**:

  ```bash
  rsync -aP ~/Library/Application\ Support/MLXComfyUI/models/qwen-turbo-mlx you@cloud-mac:Library/Application\ Support/MLXComfyUI/models/
  ```

A run that needs a model the server doesn't have is refused right away with the missing folder names.

## 5. Connect the app

1. In MLXComfyUI open the **Servers** panel (the rack icon in the left rail) and click **+**.
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

If you want to pay **only for the seconds the GPU works**, use a serverless GPU provider rather than a rented
machine: the containers scale to zero when idle and you're billed per second while they run.

- **Modal** (supported first, in M22): a ComfyUI web endpoint deployed from `scripts/serverless/modal_comfy.py`
  with a volume for models and LoRAs.
- **RunPod Serverless** and **Replicate / Cog** follow.

Two things to know:
- **Cold starts.** The first request after idle starts a container and loads the model: from about 10 seconds to a
  few minutes for large models. Keeping one worker warm removes the wait but brings back idle cost.
- **Cloud Macs can't be serverless.** Apple's license forces 24-hour minimum leases, so a rented Mac is billed for
  whole days whether it works or not. For pay-per-use, use your own Mac (free) for MLX models and serverless NVIDIA
  GPUs for the rest.

### Always-on GPU box (hourly)

For a GPU machine you rent by the hour or own, `scripts/bootstrap-gpu-server.sh` (Ubuntu, Docker, NVIDIA container
toolkit) starts one of:

```bash
scripts/bootstrap-gpu-server.sh comfyui                       # ComfyUI from its official repo, on PyTorch's CUDA image
scripts/bootstrap-gpu-server.sh vllm --model Qwen/Qwen2.5-VL-7B-Instruct
scripts/bootstrap-gpu-server.sh ollama
```

Each runs behind a small Caddy gate that requires `Authorization: Bearer <token>` (ComfyUI and Ollama have no auth
of their own), listens on 127.0.0.1 by default (`--public` to publish it, `--domain your.domain` for HTTPS), and
prints the URL and token for the app's remote nodes (M22). Image versions are pinned at the top of the compose files
(`COMFYUI_REF`, `VLLM_TAG`, `OLLAMA_TAG`); they were chosen from the projects' release tags but not checked against the
container registries from here, so if a pull fails, set a current tag. Stop a profile with `--down`.

## 8. Operating it

| Task | Command |
|---|---|
| Update to the latest code | re-run `bootstrap-mac-server.sh` (same options) |
| Follow the log | `tail -f ~/Library/Logs/mlxcomfy-server.log` |
| Restart | `launchctl kickstart -k gui/$(id -u)/com.mlxcomfy.server` |
| Stop / start | `launchctl bootout gui/$(id -u)/com.mlxcomfy.server` / `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mlxcomfy.server.plist` |
| Rotate the token | `rm ~/.config/mlxcomfy/token`, restart, then `mlxcomfy-cli serve --show-token` and update the app |
| Check speed from your laptop | `mlxcomfy-cli bench-remote https://<mac>.<tailnet>.ts.net --token-file token.txt` |
| See or remove the Tailscale publishing | `tailscale serve status` / `tailscale serve --https=443 off` |
| Free the upload cache | dashboard › Clients & transfers › Purge |
| Save money on a cloud Mac | stop/release the host when you're done (the 24 h minimum still applies) |

The server keeps the output folders of the last 200 runs (`--keep-runs`) in
`~/Library/Application Support/MLXComfyUI/server/runs`; the app has its own copies after downloading.

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
| App says "the server refused the token (401)" | Token mismatch: `mlxcomfy-cli serve --show-token` on the server and paste it again in Servers › Edit. |
| "Connection refused" | Server not running (`tail` the log, `launchctl kickstart`), wrong port, or it listens on 127.0.0.1 and you didn't open the tunnel. |
| "Timed out" | Tailscale not up on one side, or a firewall in the way. `tailscale ping <server>`. |
| "…App Transport Security policy requires the use of a secure connection" | The URL is plain `http://` to a Tailscale or public address, which macOS blocks for apps. Use the `https://<mac>.<tailnet>.ts.net` URL from `--tailscale` (section 2), an SSH tunnel, or a LAN address. |
| `--tailscale` stops at "enable HTTPS" or the https URL doesn't answer | Turn on MagicDNS and HTTPS Certificates in the admin console (DNS page), then re-run; `tailscale serve status` shows what is published. The first request can take a few seconds while the certificate is issued. |
| Dashboard shows "No Metal device" / runs fail immediately | The service runs outside a logged-in session: enable automatic login and reboot, or log in via Screen Sharing. |
| "models not installed on this server: …" | Download or copy that folder (section 4); names must match the graph's Load Model folder. |
| Runs fail with out-of-memory or the Mac swaps | Lower the budget (dashboard › Settings), unload models, or use a bigger machine. |
| Uploads are slow | Run `bench-remote`; over Tailscale check you have a direct connection (`tailscale status` shows `direct`, not `relay`). |
| Dashboard is blank after sign-in | Hard-reload; check the browser console. The dashboard needs the same origin as the server (open it through the tunnel URL, not a different hostname). |
