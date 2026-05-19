#!/usr/bin/env python3
"""Check configuration and Binance API connectivity (no trades)."""

from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bot.config import reload_config
from bot.exchange.client import BinanceFuturesClient


def main() -> int:
    cfg = reload_config()
    errors = cfg.validate()
    print("=== Scalper setup check ===\n")
    print(f"Base URL:     {cfg.base_url}")
    print(f"Dry run:      {cfg.dry_run}")
    print(f"Live confirm: {cfg.confirm_live}")
    print(f"Symbols:      {', '.join(cfg.symbols)}")
    print(f"Leverage:     {cfg.leverage}x")
    print(f"API key set:  {'yes' if cfg.api_key else 'NO'}")

    if errors:
        print("\nConfiguration errors:")
        for e in errors:
            print(f"  - {e}")
        if not cfg.api_key:
            print("\nHow to create keys (2026):")
            print("  1. https://www.binance.com/en/binance-api → Create/Manage API Key")
            print("     (or Profile → API Management)")
            print("  2. Type: System generated (HMAC) — NOT Ed25519/RSA")
            print("  3. Permissions: Reading + USD-M Futures; NO Withdrawals")
            print("  See docs/API_SETUP.md for full guide")
        return 1

    client = BinanceFuturesClient(cfg)
    if not client.test_connection():
        return 1

    client.load_exchange_info()
    bal = client.get_balance()
    positions = client.get_positions()
    print(f"\nBalance:     {bal.available_balance:.2f} USDT available")
    print(f"Positions:   {len(positions)} open")
    print(f"Dual mode:   {client.is_dual_side()}")
    if cfg.telegram_enabled:
        ok = bool(cfg.telegram_token and cfg.telegram_chat_id)
        print(f"\nTelegram:     {'configured' if ok else 'ENABLED but missing TOKEN/CHAT_ID'}")
        if ok:
            print("  Commands: /status /balance /positions /history /stats")
    else:
        print("\nTelegram:     off (TELEGRAM_ENABLED=true)")

    if cfg.dry_run:
        if cfg.paper_balance > 0:
            print(f"Paper balance: {cfg.paper_balance:.2f} USDT (PAPER_BALANCE)")
        elif cfg.paper_use_real_balance_seed:
            print(f"Paper balance: ~{bal.available_balance:.2f} USDT (from exchange)")
        else:
            print(f"Paper balance: {cfg.paper_initial_balance:.2f} USDT (PAPER_INITIAL_BALANCE)")
        print("\nOK — paper mode: python main.py  (no real trades)")
    else:
        print("\nOK — ready to run: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
