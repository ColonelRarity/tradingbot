"""
Market universe: all liquid USDT-M perpetual pairs on Binance Futures.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Set

from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient

logger = logging.getLogger(__name__)


class MarketScanner:
    """Build and refresh the list of tradable USDT perpetual symbols."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        config: Optional[Config] = None,
    ):
        self.client = client
        self.cfg = config or load_config()
        self.universe: List[str] = []
        self._last_rescan = 0.0

    def rescan_if_due(self, force: bool = False) -> List[str]:
        now = time.time()
        if (
            not force
            and self.universe
            and (now - self._last_rescan) < self.cfg.market_rescan_sec
        ):
            return self.universe

        pairs = self.client.get_top_usdt_pairs(
            min_quote_volume_usdt=self.cfg.min_volume_24h_usdt,
            max_pairs=self.cfg.max_pairs_to_scan,
        )
        extra: Set[str] = set(self.cfg.extra_symbols)
        merged: List[str] = []
        seen: Set[str] = set()
        for s in list(extra) + pairs:
            if s not in seen and s in self.client.get_tradable_usdt_symbols():
                merged.append(s)
                seen.add(s)

        # Always keep symbols with open positions
        for p in self.client.get_positions():
            if p.symbol not in seen:
                merged.insert(0, p.symbol)
                seen.add(p.symbol)

        self.universe = merged
        self._last_rescan = now
        logger.info(
            "Market universe: %d pairs (min 24h vol %.0f USDT, cap %d)",
            len(self.universe),
            self.cfg.min_volume_24h_usdt,
            self.cfg.max_pairs_to_scan,
        )
        return self.universe

    def next_batch(self, offset: int, batch_size: int) -> tuple[List[str], int]:
        """Round-robin slice of universe for REST kline updates."""
        uni = self.universe
        if not uni:
            return [], 0
        n = len(uni)
        batch_size = min(batch_size, n)
        start = offset % n
        if start + batch_size <= n:
            batch = uni[start : start + batch_size]
        else:
            batch = uni[start:] + uni[: (start + batch_size) % n]
        next_offset = (start + batch_size) % n
        return batch, next_offset

    def top_for_websocket(self, limit: int) -> List[str]:
        """Highest-volume symbols for real-time kline stream."""
        return self.universe[:limit]
