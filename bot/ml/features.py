"""
Feature extraction for ML (numpy vector from OHLCV).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from bot.config import Config, load_config
from bot.exchange.client import Candle
from bot import indicators

FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_10",
    "ema_spread", "rsi_norm", "atr_pct", "vol_ratio",
    "body_ratio", "upper_wick", "lower_wick",
    "momentum_5", "momentum_10",
    "price_vs_high20", "price_vs_low20",
    "vol_trend", "range_pct",
    "bullish_cross", "bearish_cross",
    "hour_sin", "hour_cos",
]

FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(
    candles: Sequence[Candle],
    config: Optional[Config] = None,
) -> Optional[np.ndarray]:
    cfg = config or load_config()
    need = max(cfg.slow_ema, cfg.rsi_period, cfg.volume_sma_period, 20) + 5
    if len(candles) < need:
        return None

    c = indicators.closes(candles)
    price = float(c[-1])
    if price <= 0:
        return None

    fast = indicators.ema(c, cfg.fast_ema)
    slow = indicators.ema(c, cfg.slow_ema)
    rsi_arr = indicators.rsi(c, cfg.rsi_period)
    atr_val = indicators.atr(candles, cfg.atr_period)

    if len(fast) < 3 or len(slow) < 3 or len(rsi_arr) < 2:
        return None

    def ret(n: int) -> float:
        if len(c) <= n:
            return 0.0
        return (c[-1] - c[-1 - n]) / c[-1 - n]

    rsi_now = float(rsi_arr[-1])
    fast_now, fast_prev = float(fast[-1]), float(fast[-2])
    slow_now, slow_prev = float(slow[-1]), float(slow[-2])

    vols = [x.volume for x in candles[-cfg.volume_sma_period :]]
    avg_vol = sum(vols[:-1]) / max(len(vols) - 1, 1)
    vol_ratio = candles[-1].volume / avg_vol if avg_vol > 0 else 1.0

    last = candles[-1]
    body = abs(last.close - last.open)
    full_range = last.high - last.low
    body_ratio = body / full_range if full_range > 0 else 0.0
    upper_wick = (last.high - max(last.open, last.close)) / full_range if full_range > 0 else 0.0
    lower_wick = (min(last.open, last.close) - last.low) / full_range if full_range > 0 else 0.0

    highs = [x.high for x in candles[-20:]]
    lows = [x.low for x in candles[-20:]]
    h20, l20 = max(highs), min(lows)
    span = h20 - l20 if h20 > l20 else 1e-12

    vol_trend = (vols[-1] - vols[0]) / max(vols[0], 1e-12) if len(vols) > 1 else 0.0
    range_pct = full_range / price * 100

    bullish = 1.0 if fast_prev <= slow_prev and fast_now > slow_now else 0.0
    bearish = 1.0 if fast_prev >= slow_prev and fast_now < slow_now else 0.0

    hour = (last.open_time // 3600000) % 24
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    raw = np.array([
        ret(1) * 100,
        ret(5) * 100,
        ret(10) * 100,
        (fast_now - slow_now) / price * 100,
        rsi_now / 100.0,
        (atr_val / price * 100) if price else 0.0,
        min(vol_ratio, 5.0) / 5.0,
        body_ratio,
        upper_wick,
        lower_wick,
        ret(5) * 100,
        ret(10) * 100,
        (price - l20) / span,
        (h20 - price) / span,
        np.tanh(vol_trend),
        range_pct / 2.0,
        bullish,
        bearish,
        hour_sin,
        hour_cos,
    ], dtype=np.float32)

    return np.clip(raw, -5.0, 5.0)
