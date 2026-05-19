"""Champaign volatility ranking."""

from bot.champaign.volatility import atr_percent, rank_by_volatility
from bot.exchange.client import Candle


def _flat_candles(n: int, price: float = 100.0) -> list[Candle]:
    return [
        Candle(
            open_time=i * 60000,
            open=price,
            high=price + 0.01,
            low=price - 0.01,
            close=price,
            volume=1000,
        )
        for i in range(n)
    ]


def _volatile_candles(n: int) -> list[Candle]:
    out = []
    p = 10.0
    for i in range(n):
        p += 0.5 if i % 2 == 0 else -0.45
        out.append(
            Candle(
                open_time=i * 60000,
                open=p,
                high=p + 0.8,
                low=p - 0.8,
                close=p,
                volume=5000,
            )
        )
    return out


def test_rank_prefers_higher_atr():
    calm_atr = atr_percent(_flat_candles(50))
    hot_atr = atr_percent(_volatile_candles(50))
    ranked = rank_by_volatility(
        {
            "CALM": _flat_candles(50),
            "HOT": _volatile_candles(50),
        },
        min_atr_pct=0.0,
        top_n=2,
    )
    assert hot_atr > calm_atr
    assert ranked[0][0] == "HOT"
    assert len(ranked) == 2
    assert ranked[0][1] > ranked[1][1]
