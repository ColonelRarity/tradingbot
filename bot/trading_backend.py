"""Factory: live vs paper (dry-run) execution."""

from __future__ import annotations

from typing import Union

from bot.config import Config
from bot.exchange.client import BinanceFuturesClient
from bot.paper.trader import PaperTrader
from bot.trader import Trader

TradingBackend = Union[Trader, PaperTrader]


def create_trading_backend(
    client: BinanceFuturesClient,
    config: Config,
    journal=None,
) -> TradingBackend:
    if config.dry_run:
        return PaperTrader(client, config, journal=journal)
    return Trader(client, config)
