"""Paper wallet SL/TP logic."""

from bot.paper.account import PaperPosition, check_exit


def test_long_tp_hit():
    pos = PaperPosition(
        symbol="BTCUSDT", side="LONG", quantity=0.01,
        entry_price=100, mark_at_entry=100, stop_price=98, tp_price=102,
        margin_usdt=2, notional_usdt=10,
    )
    assert check_exit(pos, 102.5) == "TP"
    assert check_exit(pos, 97.5) == "SL"
    assert check_exit(pos, 100) is None


def test_short_sl_hit():
    pos = PaperPosition(
        symbol="ETHUSDT", side="SHORT", quantity=1,
        entry_price=50, mark_at_entry=50, stop_price=52, tp_price=48,
        margin_usdt=10, notional_usdt=50,
    )
    assert check_exit(pos, 52.1) == "SL"
    assert check_exit(pos, 47.9) == "TP"
