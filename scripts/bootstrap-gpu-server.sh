#!/bin/bash
# Runs a model server on a Linux NVIDIA GPU machine (RunPod/Vast/Lambda pod, or your own box) behind a bearer-token
# gate, for MLXComfyUI's remote nodes. Needs Docker with the NVIDIA container toolkit. (For pay-per-second GPUs use
# Modal instead: server/gpu/modal_app.py, see docs/remote-server.md.)
#
#   scripts/bootstrap-gpu-server.sh <profile> [options]
#
#   profiles:  mlxcomfy-gpu  MLXComfyUI's GPU server (Wan 2.2, FLUX.1 dev) — the app's "GPU server" connections
#              vllm          vLLM OpenAI-compatible server (--model <hf-repo> required; app nodes in M23)
#              ollama        Ollama (OpenAI-compatible at /v1; app nodes in M23)
#
#   --download LIST    mlxcomfy-gpu: download models first (wan22, flux, or wan22-t2v,wan22-i2v,flux1-dev;
#                      FLUX needs HF_TOKEN in the environment)
#   --model REPO       vllm: Hugging Face model id
#   --domain NAME      serve HTTPS for NAME (Caddy gets a certificate; port 443 must be reachable)
#   --port N           plain-HTTP port when no --domain (default 8188); use it over Tailscale or an SSH tunnel
#   --public           publish the plain-HTTP port on all interfaces (default: 127.0.0.1 only)
#   --data DIR         where models and outputs live on the host (default ~/mlxcomfy-gpu)
#   --dry-run          print what would happen, change nothing
#   --down             stop the profile's containers
set -euo pipefail

cd "$(dirname "$0")/gpu"
PROFILE="${1:-}"; shift || true
MODEL=""; DOMAIN=""; PORT=8188; PUBLIC=0; DATA_DIR="$HOME/mlxcomfy-gpu"; DRY=0; DOWN=0; DOWNLOAD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --download) DOWNLOAD="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --public) PUBLIC=1; shift ;;
    --data) DATA_DIR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --down) DOWN=1; shift ;;
    -h|--help) sed -n '2,23p' "../$(basename "$0")" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run() { printf '    $ %s\n' "$*"; [[ $DRY -eq 1 ]] || "$@"; }

case "$PROFILE" in
  mlxcomfy-gpu|ollama) ;;
  vllm) [[ -n "$MODEL" ]] || die "vllm needs --model <hugging face repo id>" ;;
  *) die "usage: $0 mlxcomfy-gpu|vllm|ollama [options] (see --help)" ;;
esac
COMPOSE="docker-compose.$PROFILE.yml"
ENV_FILE="$DATA_DIR/$PROFILE.env"

if [[ $DOWN -eq 1 ]]; then
  run docker compose --env-file "$ENV_FILE" -p "mlxcomfy-${PROFILE#mlxcomfy-}" -f "$COMPOSE" down
  exit 0
fi

say "Checking this machine"
if [[ $DRY -eq 0 ]]; then
  command -v docker >/dev/null || die "Docker is not installed (https://docs.docker.com/engine/install/)"
  docker compose version >/dev/null 2>&1 || die "the Docker Compose plugin is missing"
  command -v nvidia-smi >/dev/null || die "no NVIDIA driver (nvidia-smi not found)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
  docker info 2>/dev/null | grep -qi nvidia || note "warning: Docker doesn't list the NVIDIA runtime; install nvidia-container-toolkit"
else
  note "(dry run: skipping the Docker and GPU checks)"
fi

say "Token and settings ($ENV_FILE)"
run mkdir -p "$DATA_DIR"
if [[ -f "$ENV_FILE" ]] && grep -q '^API_TOKEN=' "$ENV_FILE"; then
  TOKEN="$(grep '^API_TOKEN=' "$ENV_FILE" | cut -d= -f2)"
  note "keeping the existing token"
else
  TOKEN="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
fi
if [[ -n "$DOMAIN" ]]; then SITE="$DOMAIN"; PUBLISH="443:443"; URL="https://$DOMAIN"
else
  SITE=":$PORT"
  if [[ $PUBLIC -eq 1 ]]; then PUBLISH="$PORT:$PORT"; else PUBLISH="127.0.0.1:$PORT:$PORT"; fi
  URL="http://<this-machine>:$PORT"
fi
ENV_BODY="API_TOKEN=$TOKEN
SITE_ADDRESS=$SITE
PUBLISH=$PUBLISH
DATA_DIR=$DATA_DIR
MODEL=$MODEL
HF_TOKEN=${HF_TOKEN:-}"
if [[ $DRY -eq 1 ]]; then
  note "would write $ENV_FILE (mode 600) with API_TOKEN=<new random token>, SITE_ADDRESS=$SITE"
else
  umask 077
  printf '%s\n' "$ENV_BODY" > "$ENV_FILE"
  [[ "$PROFILE" == mlxcomfy-gpu ]] && mkdir -p "$DATA_DIR"/{models,data}
fi
if [[ -n "$DOMAIN" ]]; then
  note "HTTPS for $DOMAIN: point its DNS here and open port 443 (Caddy gets the certificate over TLS-ALPN)."
fi

COMPOSE_CMD=(docker compose --env-file "$ENV_FILE" -p "mlxcomfy-${PROFILE#mlxcomfy-}" -f "$COMPOSE")
if [[ -n "$DOWNLOAD" ]]; then
  [[ "$PROFILE" == mlxcomfy-gpu ]] || die "--download is for the mlxcomfy-gpu profile"
  say "Downloading models ($DOWNLOAD) into $DATA_DIR/models"
  note "Wan 2.2 is ≈ 70 GB per task (bf16), FLUX.1 dev ≈ 34 GB; safe to re-run (finished models are skipped)."
  run "${COMPOSE_CMD[@]}" build app
  run "${COMPOSE_CMD[@]}" run --rm app python -m mlxcomfy_gpu download --models "$DOWNLOAD" --models-dir /models
fi

say "Starting $PROFILE"
run "${COMPOSE_CMD[@]}" up -d --build

if [[ $DRY -eq 0 ]]; then
  say "Checking the gate"
  CHECK="http://127.0.0.1:$PORT"; [[ -n "$DOMAIN" ]] && CHECK="https://$DOMAIN"
  code="$(curl -s -o /dev/null -w '%{http_code}' "$CHECK/" || true)"
  note "without the token: $code (expect 401)"
fi

say "Done"
note "URL:    $URL"
note "Token:  stored in $ENV_FILE (show it with: grep API_TOKEN $ENV_FILE)"
case "$PROFILE" in
  mlxcomfy-gpu)
    note "Add it in MLXComfyUI › Servers › + as a GPU server (the URL and the token)."
    note "Models: $0 mlxcomfy-gpu --download wan22,flux   (into $DATA_DIR/models)" ;;
  ollama) note "For the OpenAI-compatible nodes (M23). Pull a model: docker exec mlxcomfy-ollama-app-1 ollama pull llama3.2-vision" ;;
  vllm) note "For the OpenAI-compatible nodes (M23). Model: $MODEL (downloaded on first start into $DATA_DIR/huggingface)" ;;
esac
note "Stop:   $0 $PROFILE --down"
