#!/bin/bash
# ============================================================
# FlyRely — Push to GitHub + Deploy to Railway
# Run this ONCE from your Mac terminal in the flyrely-api folder
# ============================================================
set -e

echo ""
echo "🚀 FlyRely Deploy Script"
echo "========================"
echo ""

# ── Step 1: Create GitHub repo ────────────────────────────────
echo "Step 1: Creating GitHub repo..."
if ! command -v gh &>/dev/null; then
  echo "  Installing GitHub CLI..."
  brew install gh
fi

# Log in if needed
gh auth status &>/dev/null || gh auth login

# Create repo (public so Railway can access it)
gh repo create flyrely-api \
  --public \
  --description "FlyRely flight delay prediction API" \
  --source=. \
  --remote=origin \
  --push 2>/dev/null || true

# If repo already exists, just push
git push -u origin master 2>/dev/null || git push -u origin main 2>/dev/null || true

REPO_URL=$(gh repo view --json url -q .url 2>/dev/null || echo "https://github.com/$(gh api user -q .login)/flyrely-api")
echo "  ✓ GitHub repo: $REPO_URL"
echo ""

# ── Step 2: Deploy to Railway ─────────────────────────────────
echo "Step 2: Deploying to Railway..."
if ! command -v railway &>/dev/null; then
  echo "  Installing Railway CLI..."
  brew install railway
fi

railway login

# Create project and deploy
railway init --name flyrely-api 2>/dev/null || true
railway link 2>/dev/null || true

# Set environment variables
echo "  Setting environment variables..."
railway variables set WEATHER_API_PROVIDER=tomorrow
railway variables set WEATHER_API_KEY=Hg4F1R0TRMgVKg3xMorsvg0CzIWXh0ox

# Deploy
railway up --detach

# Get the deployment URL
sleep 5
DEPLOY_URL=$(railway domain 2>/dev/null || echo "(check Railway dashboard)")

echo ""
echo "✅ Deployment complete!"
echo ""
echo "  API URL:      $DEPLOY_URL"
echo "  Health check: $DEPLOY_URL/health"
echo "  Test page:    open test-page.html in browser, set API URL to $DEPLOY_URL"
echo ""
