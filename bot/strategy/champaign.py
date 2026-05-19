"""
Champaign entry: high-vol pairs — LONG if momentum up, else SHORT (always in market).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bot.config import Config, load_config
from bot.exchange.client import Candle
from bot import indicators
from bot.strategy.scalper import Side


@dataclass
class ChampaignSignal:
    side: Side
    reason: str
    atr: float
    price: float
    rsi: float
    atr_pct: float


class ChampaignStrategy:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()

    def evaluate(self, candles: List[Candle]) -> ChampaignSignal:
        price = candles[-1].close if candles else 0.0
        need = max(self.cfg.slow_ema, self.cfg.rsi_period, 14) + 5
        if len(candles) < need:
            return ChampaignSignal(Side.NONE, "INSUFFICIENT_DATA", 0, price, 0, 0)

        c = indicators.closes(candles)
        fast = indicators.ema(c, self.cfg.fast_ema)
        slow = indicators.ema(c, self.cfg.slow_ema)
        rsi_arr = indicators.rsi(c, self.cfg.rsi_period)
        atr_val = indicators.atr(candles, self.cfg.atr_period)
        atr_pct = (atr_val / price * 100.0) if price > 0 else 0.0

        if len(fast) < 2 or len(rsi_arr) < 1:
            return ChampaignSignal(Side.NONE, "INDICATORS_NA", atr_val, price, 0, atr_pct)

        rsi_now = float(rsi_arr[-1])
        fast_now, slow_now = float(fast[-1]), float(slow[-1])

        bullish = (
            fast_now > slow_now
            and rsi_now >= self.cfg.champaign_bullish_rsi_min
        )
        if bullish:
            return ChampaignSignal(
                Side.LONG, "CHAMP_BULL", atr_val, price, rsi_now, atr_pct
            )
        return ChampaignSignal(
            Side.SHORT, "CHAMP_BEAR", atr_val, price, rsi_now, atr_pct
        )
