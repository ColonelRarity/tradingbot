"""Profit-lock ladder math (net USDT, not price %)."""

import pytest

from bot.champaign.profit_lock import (
    highest_step_index,
    lock_profit_usdt_for_index,
    net_pnl_at_price,
    parse_profit_ladder,
    sl_price_for_locked_profit,
)
from bot.champaign.state import ChampaignStack, LegState
from bot.config import Config
from bot.paper.costs import PaperCostModel


def _long_stack(qty: float = 100.0, entry: float = 1.0, entry_costs: float = 0.04) -> ChampaignStack:
    return ChampaignStack(
        symbol="TESTUSDT",
        main=LegState("LONG", qty, entry, 10.0, entry_costs=entry_costs),
    )


def test_ladder_defaults():
    ladder = parse_profit_ladder("0,1,3,5,7,9,11,15,20", 5.0)
    assert ladder[:9] == [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0, 20.0]
    assert 25.0 in ladder and 455.0 in ladder


def test_lock_previous_rung():
    ladder = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]
    assert lock_profit_usdt_for_index(ladder, highest_step_index(1.6, ladder)) == 0.0
    assert lock_profit_usdt_for_index(ladder, highest_step_index(9.6, ladder)) == 7.0
    assert lock_profit_usdt_for_index(ladder, highest_step_index(3.0, ladder)) == 1.0


def test_sl_price_locks_net_profit_long():
    cfg = Config()
    costs = PaperCostModel(cfg)
    stack = _long_stack()
    ladder = parse_profit_ladder(cfg.champaign_profit_lock_steps, cfg.champaign_profit_lock_step_after_20)

    # ~+1.6 net at 1.02 → lock breakeven (+0)
    mark = 1.02
    net = net_pnl_at_price(stack, mark, costs)
    idx = highest_step_index(net, ladder)
    lock = lock_profit_usdt_for_index(ladder, idx)
    sl = sl_price_for_locked_profit(stack, lock, costs)
    assert lock == 0.0
    assert net_pnl_at_price(stack, sl, costs) == pytest.approx(0.0, abs=0.15)

    # Higher mark → lock +7 when float ~+9+
    stack2 = _long_stack()
    mark2 = 1.10
    net2 = net_pnl_at_price(stack2, mark2, costs)
    idx2 = highest_step_index(net2, ladder)
    lock2 = lock_profit_usdt_for_index(ladder, idx2)
    if net2 >= 9.0:
        assert lock2 == 7.0
        sl2 = sl_price_for_locked_profit(stack2, lock2, costs)
        assert net_pnl_at_price(stack2, sl2, costs) == pytest.approx(7.0, abs=0.25)
