#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  BINANCE_FUTURES_API_KEY
  BINANCE_FUTURES_API_SECRET
  BINANCE_FUTURES_BASE_URL
  BINANCE_FUTURES_WS_URL
  ALLOW_PRODUCTION_TRADING
)

missing=()
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing required env vars: ${missing[*]}" >&2
  exit 1
fi

if [[ "${ALLOW_PRODUCTION_TRADING}" != "True" && "${ALLOW_PRODUCTION_TRADING}" != "true" ]]; then
  echo "ALLOW_PRODUCTION_TRADING must be True for production runs." >&2
  exit 1
fi

if [[ "${DRY_RUN:-}" == "True" || "${DRY_RUN:-}" == "true" ]]; then
  echo "DRY_RUN is enabled: orders will be simulated only." >&2
fi

exec python main.py
