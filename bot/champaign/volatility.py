"""High-volatility pair filter for Champaign mode."""

from __future__ import annotations

from typing import Dict, List, Tuple

from bot import indicators
from bot.exchange.client import Candle


def last_bar_range_pct(candles: List[Candle]) -> float:
    """Current 1m bar (high-low)/close — moves with each WS tick."""
    if not candles:
        return 0.0
    c = candles[-1]
    if c.close <= 0:
        return 0.0
    return ((c.high - c.low) / c.close) * 100.0


def atr_percent(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 0.0
    atr = indicators.atr(candles, period)
    price = candles[-1].close
    if price <= 0:
        return 0.0
    return (atr / price) * 100.0


def is_high_volatility(
    candles: List[Candle],
    min_atr_pct: float,
    period: int = 14,
) -> bool:
    return atr_percent(candles, period) >= min_atr_pct


def rank_by_volatility(
    symbol_candles: Dict[str, List[Candle]],
    min_atr_pct: float,
    period: int = 14,
    top_n: int | None = None,
) -> List[Tuple[str, float]]:
    """Return (symbol, atr_pct) sorted by volatility descending."""
    scored: List[Tuple[str, float]] = []
    for sym, candles in symbol_candles.items():
        pct = atr_percent(candles, period)
        if pct >= min_atr_pct:
            scored.append((sym, pct))
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None and top_n > 0:
        scored = scored[:top_n]
    return scored
