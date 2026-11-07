#!/bin/bash
# Mac mini: serve the built TanStack Start app on 127.0.0.1:3000 (launchd label com.flyrely.web).
# Build first (VITE_API_URL is baked in at build time): deploy/macmini/build-web.sh
set -euo pipefail
APP="${FLYRELY_APP_DIR:-$(cd "$(dirname "$0")/../../../flyrely-app" && pwd)}"
cd "$APP"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export HOST=127.0.0.1 PORT=3000 NODE_ENV=production
[ -f .output/server/index.mjs ] || { echo "no build: run deploy/macmini/build-web.sh" >&2; exit 1; }
exec node .output/server/index.mjs
