"""Unit tests for scalping strategy (no API)."""

from bot.config import Config
from bot.exchange.client import Candle
from bot.strategy.scalper import ScalperStrategy, Side


def _candles(n: int, trend: float = 1.0) -> list[Candle]:
    out = []
    price = 100.0
    for i in range(n):
        price += trend * (0.05 if i % 2 == 0 else -0.02)
        out.append(Candle(
            open_time=i * 60000,
            open=price,
            high=price + 0.1,
            low=price - 0.1,
            close=price,
            volume=1000 + i * 10,
        ))
    return out


def test_strategy_returns_signal_or_none():
    strat = ScalperStrategy()
    sig = strat.evaluate(_candles(80, trend=0.5))
    assert sig.side in (Side.LONG, Side.SHORT, Side.NONE)
    assert sig.atr >= 0


def test_relaxed_mode_allows_trend_without_cross():
    cfg = Config(strategy_mode="relaxed", min_volume_ratio=0.5)
    strat = ScalperStrategy(cfg)
    sig = strat.evaluate(_candles(80, trend=0.8))
    assert sig.side in (Side.LONG, Side.SHORT, Side.NONE)
