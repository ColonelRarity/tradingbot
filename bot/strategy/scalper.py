"""
Fast scalping strategy: EMA crossover + RSI band + volume confirmation on 1m.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from bot.config import Config, load_config
from bot.exchange.client import Candle
from bot import indicators

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class ScalpSignal:
    side: Side
    reason: str
    atr: float
    price: float
    rsi: float
    fast_ema: float
    slow_ema: float


class ScalperStrategy:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()

    def evaluate(self, candles: List[Candle]) -> ScalpSignal:
        price = candles[-1].close if candles else 0.0
        if len(candles) < max(self.cfg.slow_ema, self.cfg.rsi_period, self.cfg.volume_sma_period) + 5:
            return ScalpSignal(Side.NONE, "INSUFFICIENT_DATA", 0, price, 0, 0, 0)

        c = indicators.closes(candles)
        fast = indicators.ema(c, self.cfg.fast_ema)
        slow = indicators.ema(c, self.cfg.slow_ema)
        rsi_arr = indicators.rsi(c, self.cfg.rsi_period)
        atr_val = indicators.atr(candles, self.cfg.atr_period)

        if len(fast) < 3 or len(slow) < 3 or len(rsi_arr) < 2:
            return ScalpSignal(Side.NONE, "INDICATORS_NA", atr_val, price, 0, 0, 0)

        rsi_now = float(rsi_arr[-1])
        fast_now, fast_prev = float(fast[-1]), float(fast[-2])
        slow_now, slow_prev = float(slow[-1]), float(slow[-2])

        # Volume filter
        vols = [x.volume for x in candles[-self.cfg.volume_sma_period :]]
        avg_vol = sum(vols[:-1]) / max(len(vols) - 1, 1)
        vol_ratio = candles[-1].volume / avg_vol if avg_vol > 0 else 0

        if vol_ratio < self.cfg.min_volume_ratio:
            return ScalpSignal(Side.NONE, f"LOW_VOLUME:{vol_ratio:.2f}", atr_val, price, rsi_now, fast_now, slow_now)

        long_rsi = self.cfg.rsi_long_min <= rsi_now <= self.cfg.rsi_long_max
        short_rsi = self.cfg.rsi_short_min <= rsi_now <= self.cfg.rsi_short_max

        if self.cfg.strategy_mode == "relaxed":
            if fast_now > slow_now and long_rsi:
                return ScalpSignal(Side.LONG, "EMA_TREND_UP+RSI", atr_val, price, rsi_now, fast_now, slow_now)
            if fast_now < slow_now and short_rsi:
                return ScalpSignal(Side.SHORT, "EMA_TREND_DOWN+RSI", atr_val, price, rsi_now, fast_now, slow_now)
        else:
            bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
            bearish_cross = fast_prev >= slow_prev and fast_now < slow_now
            if bullish_cross and long_rsi:
                return ScalpSignal(Side.LONG, "EMA_CROSS_UP+RSI", atr_val, price, rsi_now, fast_now, slow_now)
            if bearish_cross and short_rsi:
                return ScalpSignal(Side.SHORT, "EMA_CROSS_DOWN+RSI", atr_val, price, rsi_now, fast_now, slow_now)

        return ScalpSignal(Side.NONE, "NO_SETUP", atr_val, price, rsi_now, fast_now, slow_now)
