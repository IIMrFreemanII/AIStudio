#!/bin/zsh
# Builds the server's web dashboard (web/dashboard, Svelte) into Sources/MLXComfyServer/Resources/dashboard.
# The output is committed, so building the server or bootstrapping a cloud Mac never needs Node.
# Usage: scripts/build-dashboard.sh [--check]   (--check: type-check and test only)
set -euo pipefail
cd "$(dirname "$0")/../web/dashboard"
if [[ ! -d node_modules ]]; then npm ci --no-audit --no-fund; fi
npx svelte-check --tsconfig ./tsconfig.json
npx vitest run
[[ "${1:-}" == "--check" ]] && exit 0
npm run build
