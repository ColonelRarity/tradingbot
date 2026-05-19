"""Champaign strategy and fib helpers."""

from bot.champaign.fib import fib_target, parse_fib_levels
from bot.config import Config
from bot.exchange.client import Candle
from bot.strategy.champaign import ChampaignStrategy
from bot.strategy.scalper import Side


def _candles(n: int, up: bool = True) -> list[Candle]:
    out = []
    p = 10.0
    for i in range(n):
        p += 0.08 if up else -0.08
        out.append(
            Candle(
                open_time=i * 60000,
                open=p,
                high=p + 0.2,
                low=p - 0.1,
                close=p,
                volume=5000 + i * 50,
            )
        )
    return out


def test_champaign_picks_long_or_short():
    cfg = Config(champaign_bullish_rsi_min=48.0)
    strat = ChampaignStrategy(cfg)
    sig = strat.evaluate(_candles(80, up=True))
    assert sig.side in (Side.LONG, Side.SHORT)


def test_fib_long_target_above_entry():
    tp = fib_target(100.0, 110.0, 90.0, "LONG", 0.618)
    assert tp > 100.0


def test_parse_fib_levels():
    assert parse_fib_levels("0.382,0.618") == [0.382, 0.618]


def test_monitor_hits_sl_and_tp():
    from bot.champaign.monitor import ChampaignPositionMonitor
    from bot.champaign.state import ChampaignStack, LegState

    cfg = Config(
        champaign_hedge_enabled=False,
        champaign_breakeven_move_pct=99.0,
    )
    closed: list[str] = []

    def on_close(stack, mark, reason):
        closed.append(reason)

    mon = ChampaignPositionMonitor(
        cfg,
        [0.382, 0.618],
        on_open_hedge=lambda s, m: None,
        on_close_hedge=lambda s, m: None,
        on_close_main=on_close,
        on_update_orders=lambda s, sl, tp: None,
    )
    stack = ChampaignStack(
        symbol="TESTUSDT",
        main=LegState("LONG", 1.0, 100.0, 10.0),
        active_sl=99.0,
        active_tp=105.0,
    )
    mon.tick(stack, 98.5)
    assert closed == ["SL"]
    closed.clear()
    stack2 = ChampaignStack(
        symbol="T2",
        main=LegState("SHORT", 1.0, 100.0, 10.0),
        active_sl=101.0,
        active_tp=95.0,
    )
    mon.tick(stack2, 94.0)
    assert closed == ["TP"]
