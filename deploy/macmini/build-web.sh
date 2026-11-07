#!/bin/bash
# Mac mini: rebuild the web app for the public API URL. Supabase vars come from .env.local (not printed).
set -euo pipefail
APP="${FLYRELY_APP_DIR:-$(cd "$(dirname "$0")/../../../flyrely-app" && pwd)}"
cd "$APP"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
set -a; [ -f .env.local ] && . ./.env.local; set +a
export VITE_API_URL="${FLYRELY_PUBLIC_API_URL:-https://seans-mac-mini.tailc9e7f8.ts.net:8443}"
npm ci --no-audit --no-fund
npm run build
