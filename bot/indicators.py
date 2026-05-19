"""Technical indicators for scalping."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from bot.exchange.client import Candle


def closes(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.close for c in candles], dtype=float)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.array([])
    alpha = 2.0 / (period + 1)
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period + 1:
        return np.array([])
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    out = np.empty(len(values))
    out[:] = np.nan
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    return out


def atr(candles: Sequence[Candle], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return 0.0
    return float(np.mean(trs[-period:]))
