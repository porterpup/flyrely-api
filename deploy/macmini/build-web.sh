#!/bin/bash
# Mac mini: rebuild the web app for the public API URL. Supabase vars come from .env.local (not printed).
set -euo pipefail
APP="${FLYRELY_APP_DIR:-$(cd "$(dirname "$0")/../../../flyrely-app" && pwd)}"
cd "$APP"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
set -a; [ -f .env.local ] && . ./.env.local; set +a
export VITE_API_URL="${FLYRELY_PUBLIC_API_URL:-https://seans-mac-mini.tailc9e7f8.ts.net:8443}"

# Keep the previous build's hashed client chunks. Browsers with an open tab (or an installed PWA) still
# lazy-load chunks named by the OLD build; if they vanish the user hits "'text/html' is not a valid
# JavaScript MIME type" when navigating (e.g. add flight -> flight detail). Chunks are content-hashed, so
# old and new files never collide. Files under public/ are copied into the new .output/public and into
# Nitro's asset manifest, so they keep being served (public/_build is git-ignored in flyrely-app).
KEEP_DIR="$APP/public/_build/assets"
KEEP_DAYS="${FLYRELY_ASSET_KEEP_DAYS:-14}"
mkdir -p "$KEEP_DIR"
if [ -d .output/public/_build/assets ]; then
  cp -p -n .output/public/_build/assets/* "$KEEP_DIR"/ 2>/dev/null || true
fi
find "$KEEP_DIR" -type f -mtime +"$KEEP_DAYS" -delete 2>/dev/null || true

npm ci --no-audit --no-fund
npm run build
