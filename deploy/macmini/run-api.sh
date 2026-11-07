#!/bin/bash
# Mac mini: run the FlyRely API on 127.0.0.1:8000 (launchd label com.flyrely.api).
# Secrets stay in flyrely-api/.env; uvicorn reads it. Nothing secret is in the plist or here.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export FLYRELY_ALLOWED_ORIGINS="${FLYRELY_ALLOWED_ORIGINS:-https://seans-mac-mini.tailc9e7f8.ts.net}"
# Copilot stays off: never set FLYRELY_FEATURE_DISRUPTION_COPILOT here.
unset FLYRELY_FEATURE_DISRUPTION_COPILOT
exec venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env --no-access-log
