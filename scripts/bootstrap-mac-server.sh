#!/bin/bash
# Sets up `aistudio-cli serve` on an Apple Silicon Mac (a cloud Mac, or your own) as a login service.
# Safe to re-run: it updates the checkout, rebuilds, and restarts the service.
#
#   scripts/bootstrap-mac-server.sh [options]
#
#   --source DIR      build from this checkout instead of cloning (default: clone into --dir)
#   --dir DIR         where to clone (default ~/AIStudio)
#   --repo URL        git repository (default the public AIStudio repo)
#   --branch NAME     branch or tag to build (default main)
#   --port N          port (default 8188)
#   --tailscale       publish it on your tailnet over https with `tailscale serve` (https://<mac>.<tailnet>.ts.net);
#                     the server itself stays on 127.0.0.1. The app needs https for tailnet addresses (macOS's App
#                     Transport Security blocks plain http to 100.x and *.ts.net).
#   --host ADDR       listen on ADDR (127.0.0.1, a LAN IP, or 0.0.0.0)
#   --models DIR      models folder (default the app's: ~/Library/Application Support/AIStudio/models)
#   --no-power        don't offer the no-sleep power settings
#   --yes             don't ask before each step
#   --dry-run         print what would happen, change nothing
#   --test            install under a temporary name and port, check it answers, then remove it again
set -euo pipefail

REPO="https://github.com/IIMrFreemanII/AIStudio.git"
BRANCH="main"
DIR="$HOME/AIStudio"
SOURCE=""
PORT=8188
PUBLIC_URL=""
HOST="127.0.0.1"
USE_TAILSCALE=0
MODELS=""
POWER=1
YES=0
DRY=0
TEST=0
LABEL="com.aistudio.server"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --tailscale) USE_TAILSCALE=1; shift ;;
    --models) MODELS="$2"; shift 2 ;;
    --no-power) POWER=0; shift ;;
    --yes) YES=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --test) TEST=1; YES=1; POWER=0; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

if [[ $TEST -eq 1 ]]; then
  LABEL="com.aistudio.server.test"
  [[ "$PORT" == 8188 ]] && PORT=18288
fi

SUPPORT="$HOME/Library/Application Support/AIStudio/server"
[[ $TEST -eq 1 ]] && SUPPORT="$(mktemp -d)/server"
BIN_DIR="$SUPPORT/bin"
TOKEN_FILE="$HOME/.config/aistudio/token"
[[ $TEST -eq 1 ]] && TOKEN_FILE="$SUPPORT/token"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/aistudio-server.log"
[[ $TEST -eq 1 ]] && LOG="$SUPPORT/server.log"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
run() {
  printf '    $ %s\n' "$*"
  [[ $DRY -eq 1 ]] || "$@"
}
confirm() {
  [[ $YES -eq 1 || $DRY -eq 1 ]] && return 0
  read -r -p "    $1 [Y/n] " a
  [[ -z "$a" || "$a" =~ ^[Yy] ]]
}
die() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# 0. A server set up before the project was renamed (MLXComfyUI → AIStudio): stop its service and move its folders,
#    token and checkout to the new names. The old folders are left as links, so old absolute paths still work.
migrate_legacy() {
  local old_plist="$HOME/Library/LaunchAgents/com.mlxcomfy.server.plist"
  local found=0
  if [[ -f "$old_plist" ]]; then
    found=1
    say "Stopping the service from before the rename (com.mlxcomfy.server)"
    run launchctl bootout "gui/$(id -u)/com.mlxcomfy.server" 2>/dev/null || true
    run rm -f "$old_plist"
  fi
  local base
  for base in "$HOME/Library/Application Support" "$HOME/Documents" "$HOME/Pictures"; do
    if [[ -d "$base/MLXComfyUI" && ! -L "$base/MLXComfyUI" && ! -e "$base/AIStudio" ]]; then
      found=1
      say "Moving $base/MLXComfyUI to $base/AIStudio"
      run mv "$base/MLXComfyUI" "$base/AIStudio"
      run ln -s "$base/AIStudio" "$base/MLXComfyUI"
    fi
  done
  if [[ -f "$HOME/.config/mlxcomfy/token" && ! -e "$HOME/.config/aistudio/token" ]]; then
    found=1
    say "Moving the token to ~/.config/aistudio/token"
    run mkdir -p "$HOME/.config/aistudio"
    run chmod 700 "$HOME/.config/aistudio"
    run mv "$HOME/.config/mlxcomfy/token" "$HOME/.config/aistudio/token"
  fi
  if [[ -z "$SOURCE" && "$DIR" == "$HOME/AIStudio" && -d "$HOME/MLXComfyUI/.git" && ! -e "$DIR" ]]; then
    found=1
    say "Moving the checkout ~/MLXComfyUI to ~/AIStudio"
    run mv "$HOME/MLXComfyUI" "$DIR"
  fi
  [[ $found -eq 1 ]] && note "(the old log ~/Library/Logs/mlxcomfy-server.log is no longer written; delete it when you like)"
  return 0
}
[[ $TEST -eq 1 ]] || migrate_legacy

# 1. The machine.
say "Checking this Mac"
[[ "$(uname -s)" == "Darwin" ]] || die "this script is for macOS (for Linux GPU servers see scripts/bootstrap-gpu-server.sh)"
[[ "$(uname -m)" == "arm64" ]] || die "MLX needs Apple Silicon (this is $(uname -m))"
note "$(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
DEV="$(xcode-select -p 2>/dev/null || true)"
if [[ "$DEV" != *Xcode*.app* ]]; then
  die "full Xcode is required (MLX compiles Metal shaders; the Command Line Tools are not enough).
    Install it from the App Store, or with: brew install xcodesorg/made/xcodes && xcodes install --latest
    then: sudo xcode-select -s /Applications/Xcode.app && sudo xcodebuild -license accept"
fi
note "Xcode: $DEV"
if ! xcrun --find metal >/dev/null 2>&1; then
  note "The Metal toolchain is missing; installing it (Xcode component)."
  run xcodebuild -downloadComponent MetalToolchain
fi

if [[ $USE_TAILSCALE -eq 1 ]]; then
  TS="$(command -v tailscale || echo /Applications/Tailscale.app/Contents/MacOS/Tailscale)"
  [[ -x "$TS" ]] || die "--tailscale: Tailscale is not installed (https://tailscale.com/download/mac), or use --host"
  "$TS" ip -4 >/dev/null 2>&1 || die "--tailscale: this Mac isn't on a tailnet yet; run 'tailscale up' first"
  # The MagicDNS name, e.g. studio.tail1234.ts.net (status reports it with a trailing dot).
  TS_NAME="$("$TS" status --json 2>/dev/null | plutil -extract Self.DNSName raw -o - - 2>/dev/null || true)"
  TS_NAME="${TS_NAME%.}"
  [[ -n "$TS_NAME" ]] || die "--tailscale: no MagicDNS name; turn on MagicDNS in the Tailscale admin console (DNS page)"
  [[ "$HOST" == "127.0.0.1" ]] || die "--tailscale serves through 127.0.0.1; drop --host"
  PUBLIC_URL="https://$TS_NAME"
  note "Tailscale name: $TS_NAME (served at $PUBLIC_URL)"
fi

# 2. The code.
if [[ -n "$SOURCE" ]]; then
  SRC="$(cd "$SOURCE" && pwd)"
  say "Using the checkout at $SRC"
else
  SRC="$DIR"
  say "Getting the code ($REPO, $BRANCH)"
  if [[ -d "$SRC/.git" ]]; then
    run git -C "$SRC" fetch --tags origin
    run git -C "$SRC" checkout "$BRANCH"
    run git -C "$SRC" pull --ff-only origin "$BRANCH" || note "(not a branch, or local changes: keeping the current tree)"
  else
    run git clone --branch "$BRANCH" "$REPO" "$SRC"
  fi
fi

# 3. Build (Release is fine for running; only the tests must use Debug).
say "Building aistudio-cli (Release; the first build takes several minutes)"
DERIVED="$SRC/.build/xcode"
if [[ $DRY -eq 0 ]]; then
  (cd "$SRC" && xcodebuild build -scheme aistudio-cli -destination 'platform=macOS' -derivedDataPath "$DERIVED" \
     -configuration Release -skipPackagePluginValidation -skipMacroValidation -quiet)
else
  printf '    $ (cd %s && xcodebuild build -scheme aistudio-cli -configuration Release …)\n' "$SRC"
fi
PRODUCTS="$DERIVED/Build/Products/Release"

say "Installing into $BIN_DIR"
run mkdir -p "$BIN_DIR"
if [[ $DRY -eq 0 ]]; then
  # The binary finds its resources (MLX's Metal library, tokenizer data, the dashboard) in bundles next to it.
  rm -rf "$BIN_DIR"/*.bundle
  cp "$PRODUCTS/aistudio-cli" "$BIN_DIR/"
  for b in "$PRODUCTS"/*.bundle; do cp -R "$b" "$BIN_DIR/"; done
  codesign --force --sign - "$BIN_DIR/aistudio-cli" >/dev/null 2>&1 || true
fi
CLI="$BIN_DIR/aistudio-cli"

# 4. Token.
say "Token"
if [[ $DRY -eq 1 ]]; then
  note "would create $TOKEN_FILE (256 random bits, mode 600) if it doesn't exist"
elif [[ -f "$TOKEN_FILE" ]]; then
  note "keeping the existing token in $TOKEN_FILE"
else
  "$CLI" serve --token-file "$TOKEN_FILE" --show-token >/dev/null
  note "created $TOKEN_FILE (show it with: $CLI serve --show-token)"
fi

# 5. Service.
say "Login service ($LABEL)"
ARGS=("$CLI" serve --host "$HOST" --port "$PORT" --token-file "$TOKEN_FILE")
[[ -n "$MODELS" ]] && ARGS+=(--models "$MODELS")
[[ $TEST -eq 1 ]] && ARGS+=(--data "$SUPPORT/data")
ARGS_XML=""
for a in "${ARGS[@]}"; do ARGS_XML+="    <string>$(printf '%s' "$a" | sed 's/&/\&amp;/g; s/</\&lt;/g')</string>"$'\n'; done
PLIST_BODY="<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
$ARGS_XML  </array>
  <key>WorkingDirectory</key><string>$SUPPORT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>"
note "listens on $HOST:$PORT, logs to $LOG"
if [[ $DRY -eq 1 ]]; then
  note "would write $PLIST:"; printf '%s\n' "$PLIST_BODY" | sed 's/^/      /'
else
  mkdir -p "$(dirname "$PLIST")" "$SUPPORT"
  printf '%s\n' "$PLIST_BODY" > "$PLIST"
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
fi

# 6. Self-check.
say "Checking the server answers"
if [[ $DRY -eq 0 ]]; then
  TOKEN="$(head -n1 "$TOKEN_FILE")"
  ok=0
  for _ in $(seq 1 60); do
    if curl -fsS -H "Authorization: Bearer $TOKEN" "http://$HOST:$PORT/v1/info" -o /tmp/aistudio-info.$$ 2>/dev/null; then ok=1; break; fi
    sleep 1
  done
  if [[ $ok -eq 1 ]]; then
    note "OK: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["hostName"], "·", d["chip"], "·", len(d["models"]), "models, server", d["serverVersion"])' /tmp/aistudio-info.$$ 2>/dev/null || echo answered)"
    rm -f /tmp/aistudio-info.$$
  else
    tail -20 "$LOG" 2>/dev/null || true
    [[ $TEST -eq 1 ]] && launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && rm -f "$PLIST"
    die "the server did not answer on $HOST:$PORT (log above)"
  fi
fi

# 6b. Publish on the tailnet: Tailscale terminates https with a real certificate and proxies to 127.0.0.1.
if [[ -n "$PUBLIC_URL" ]]; then
  say "Publishing on your tailnet with tailscale serve"
  if [[ $TEST -eq 1 ]]; then
    note "test mode: would run $TS serve --bg --https=443 http://127.0.0.1:$PORT (your tailnet config is left alone)"
  else
    note "The first time, Tailscale may ask you to enable HTTPS certificates for the tailnet: follow the link it prints."
    run "$TS" serve --bg --https=443 "http://127.0.0.1:$PORT" \
      || die "tailscale serve failed: enable MagicDNS and HTTPS certificates (admin console › DNS), then re-run"
    if [[ $DRY -eq 0 ]]; then
      ok=0
      # The first request fetches the certificate, which can take a little while.
      for _ in $(seq 1 30); do
        if curl -fsS -H "Authorization: Bearer $TOKEN" "$PUBLIC_URL/v1/health" -o /dev/null 2>/dev/null; then ok=1; break; fi
        sleep 2
      done
      if [[ $ok -eq 1 ]]; then note "OK: $PUBLIC_URL answers over https"
      else note "warning: $PUBLIC_URL doesn't answer yet; check '$TS serve status' and the HTTPS setting in the admin console"
      fi
    fi
  fi
fi

if [[ $TEST -eq 1 ]]; then
  say "Test mode: removing the temporary service again"
  run launchctl bootout "gui/$(id -u)/$LABEL"
  run rm -f "$PLIST"
  note "test install in $SUPPORT left for inspection (safe to delete)"
  exit 0
fi

# 7. Staying awake. A login service needs a logged-in GUI session (Metal is per-session), and a sleeping Mac drops it.
if [[ $POWER -eq 1 ]]; then
  say "Keeping the Mac awake"
  note "A headless Mac must not sleep, and must log in by itself after a reboot:"
  note "  • System Settings › Users & Groups › Automatically log in as: $(id -un)"
  note "  • System Settings › Energy: 'Prevent automatic sleeping', 'Start up automatically after a power failure'"
  if confirm "Set 'never sleep' now with pmset (needs your password)?"; then
    run sudo pmset -a sleep 0 disksleep 0 displaysleep 10 autorestart 1
  fi
fi

say "Done"
note "Token:       $CLI serve --show-token        (paste it into AIStudio › Servers)"
if [[ -n "$PUBLIC_URL" ]]; then
  note "Server URL:  $PUBLIC_URL               (add it in AIStudio › Servers; the dashboard is at / )"
  note "Unpublish:   $TS serve --https=443 off"
else
  note "Dashboard:   http://$HOST:$PORT/              (through your tunnel if this is 127.0.0.1)"
fi
note "Logs:        tail -f \"$LOG\""
note "Restart:     launchctl kickstart -k gui/$(id -u)/$LABEL"
note "Models:      $CLI download            (list)   ·   $CLI download qwen-image-2.1-turbo"
[[ "$HOST" == "127.0.0.1" && -z "$PUBLIC_URL" ]] && note "Reach it from your laptop with: ssh -N -L $PORT:127.0.0.1:$PORT $(id -un)@<this-mac>"
exit 0
