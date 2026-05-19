"""Fibonacci extension levels for dynamic TP/SL."""

from __future__ import annotations

from typing import List, Sequence


DEFAULT_FIB: tuple[float, ...] = (0.382, 0.618, 1.0, 1.618)


def parse_fib_levels(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_FIB)
    out: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or list(DEFAULT_FIB)


def swing_range(candles, lookback: int = 20) -> tuple[float, float]:
    window = candles[-lookback:]
    return max(c.high for c in window), min(c.low for c in window)


def fib_target(
    entry: float,
    swing_high: float,
    swing_low: float,
    side: str,
    level: float,
) -> float:
    diff = max(swing_high - swing_low, entry * 1e-6)
    if side == "LONG":
        return entry + diff * level
    return entry - diff * level


def next_fib_step(levels: Sequence[float], current_index: int) -> tuple[int, float]:
    nxt = min(current_index + 1, len(levels) - 1)
    return nxt, levels[nxt]
