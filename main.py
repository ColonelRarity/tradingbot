#!/usr/bin/env python3
"""
Binance USDT-M Futures scalping bot — entry point.

Setup:
  1. Copy .env.example → .env and fill API keys
  2. Set CONFIRM_LIVE_TRADING=true for real account
  3. python verify_setup.py
  4. python main.py
"""

from __future__ import annotations

import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bot.config import load_config, reload_config
from bot.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Futures scalping bot")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING")
    args = parser.parse_args()

    reload_config()
    cfg = load_config()
    level = args.log_level or cfg.log_level
    setup_logging(level)

    print("=" * 56)
    print("  Binance Futures Scalper v2")
    print(f"  API: {cfg.base_url}")
    if cfg.dry_run:
        print("  Mode: DRY RUN — paper wallet, NO real orders")
        if cfg.paper_balance > 0:
            print(f"  Paper balance: {cfg.paper_balance:.2f} USDT (PAPER_BALANCE)")
        elif cfg.paper_use_real_balance_seed:
            print("  Paper balance: from exchange (PAPER_USE_REAL_BALANCE)")
        else:
            print(f"  Paper balance: {cfg.paper_initial_balance:.2f} USDT (PAPER_INITIAL_BALANCE)")
    else:
        print(f"  Mode: LIVE (confirm={cfg.confirm_live})")
    if cfg.ml_enabled:
        print(f"  ML: {cfg.ml_decision_mode} (warmup<{cfg.ml_warmup_samples} labeled)")
    else:
        print("  ML: disabled")
    print(f"  Scan all: {cfg.scan_all_pairs}")
    if not cfg.champaign:
        print(f"  Strategy: {cfg.strategy_mode.upper()}")
    if cfg.paper_exploration_active and not cfg.champaign:
        print("  Paper exploration: ON (strict TA, cooldown, no daily loss cap)")
    if cfg.dry_run and cfg.paper_skip_daily_loss_limit:
        print("  Paper daily loss limit: OFF")
    elif cfg.dry_run:
        print("  Paper exploration: OFF (set PAPER_EXPLORATION=auto)")
    if cfg.champaign:
        print("  Strategy: CHAMPAIGN (high-vol, Fib SL/TP, hedge)")
    print("=" * 56)

    if cfg.champaign:
        from bot.champaign.engine import ChampaignEngine

        ChampaignEngine(cfg).start()
    else:
        from bot.engine import ScalpingEngine

        ScalpingEngine(cfg).start()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
