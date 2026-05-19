#!/usr/bin/env python3
"""Train ML model from journal + kline bootstrap."""

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
from bot.exchange.client import BinanceFuturesClient
from bot.learning.journal import TradeJournal
from bot.logging_setup import setup_logging
from bot.market.scanner import MarketScanner
from bot.ml.predictor import MLPredictor
from bot.ml.trainer import MLTrainer


def main() -> None:
    setup_logging("INFO")
    cfg = reload_config()
    client = BinanceFuturesClient(cfg)
    client.load_exchange_info()
    journal = TradeJournal()
    scanner = MarketScanner(client, cfg)
    scanner.rescan_if_due(force=True)
    symbols = scanner.universe[:20] or cfg.symbols

    trainer = MLTrainer(journal, client, cfg)
    result = trainer.train(bootstrap_symbols=symbols)
    print(result)

    if result.get("ok"):
        MLPredictor(cfg).reload()
        print("Model saved to", cfg.ml_model_path)


if __name__ == "__main__":
    main()
