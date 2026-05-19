"""Logging configuration."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging.

    DEBUG applies only to `bot.*` loggers. Binance connector / urllib3 stay at
    WARNING so DEBUG does not dump full HTTP bodies (klines JSON floods the console).
    """
    bot_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("bot").setLevel(bot_level)

    for noisy in ("urllib3", "websocket", "root", "binance", "binance.um_futures"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
