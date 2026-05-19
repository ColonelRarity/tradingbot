#!/usr/bin/env python3
"""Emergency: close all open futures positions and cancel orders."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bot.config import reload_config
from bot.exchange.client import BinanceFuturesClient, OrderSide
from bot.logging_setup import setup_logging


def main() -> None:
    setup_logging("INFO")
    cfg = reload_config()
    if cfg.validate():
        print("Fix config errors first (run verify_setup.py)")
        sys.exit(1)

    client = BinanceFuturesClient(cfg)
    positions = client.get_positions()
    if not positions:
        print("No open positions.")
        return

    for p in positions:
        print(f"Closing {p.symbol} {p.side} size={p.size}")
        client.cancel_all_orders(p.symbol)
        side = OrderSide.SELL if p.side == "LONG" else OrderSide.BUY
        ps = p.side if client.is_dual_side() else None
        client.market_order(p.symbol, side, p.size, position_side=ps, reduce_only=True)
    print("Done.")


if __name__ == "__main__":
    main()
