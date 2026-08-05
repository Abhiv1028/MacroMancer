#!/usr/bin/env bash
#
# Deploy Macromancer to Fly.io.
#
# Prereqs:
#   - Install flyctl:  https://fly.io/docs/hands-on/install-flyctl/
#   - Log in:          flyctl auth login
#   - (optional) export NUTRITIONIX_APP_ID / NUTRITIONIX_API_KEY to enable
#     restaurant search.
#
# Usage:  ./scripts/deploy.sh
set -euo pipefail

if ! command -v flyctl >/dev/null 2>&1; then
  echo "flyctl not found. Install it: https://fly.io/docs/hands-on/install-flyctl/" >&2
  exit 1
fi

APP_NAME="${FLY_APP:-macromancer}"

# 1) Create the app from the existing fly.toml (no deploy yet). Idempotent-ish:
#    ignore failure if the app already exists.
flyctl launch --no-deploy --copy-config --name "$APP_NAME" --region ord || true

# 2) Ensure the persistent volume exists (1 GB).
if ! flyctl volumes list --app "$APP_NAME" 2>/dev/null | grep -q "macromentor_data"; then
  flyctl volumes create macromentor_data --size 1 --region ord --app "$APP_NAME" --yes
fi

# 3) Set secrets if provided in the environment.
if [ -n "${NUTRITIONIX_APP_ID:-}" ] && [ -n "${NUTRITIONIX_API_KEY:-}" ]; then
  flyctl secrets set \
    NUTRITIONIX_APP_ID="$NUTRITIONIX_APP_ID" \
    NUTRITIONIX_API_KEY="$NUTRITIONIX_API_KEY" \
    --app "$APP_NAME"
else
  echo "ℹ️  NUTRITIONIX_APP_ID/API_KEY not set — restaurant search will return 503 until configured."
fi

# 4) Deploy and open.
flyctl deploy --app "$APP_NAME"
flyctl open --app "$APP_NAME"
