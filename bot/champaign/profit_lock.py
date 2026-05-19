"""
Profit-lock ladder: SL prices that lock net USDT profit (after estimated exit costs).

Ladder default: 0, 1, 3, 5, 7, 9, 11, 15, 20, then +5 each (25, 30, 35, ...).
When floating PnL crosses step S, SL locks the previous step (e.g. at +9.6 → lock +7).
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bot.champaign.state import ChampaignStack
    from bot.paper.costs import PaperCostModel


def build_profit_ladder(
    steps_before_20: List[float] | None = None,
    step_after_20: float = 5.0,
    max_step: float = 5000.0,
) -> List[float]:
    base = steps_before_20 or [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0, 20.0]
    out = sorted(set(base))
    v = 20.0
    while v + step_after_20 <= max_step:
        v += step_after_20
        if v not in out:
            out.append(v)
    return sorted(out)


def parse_profit_ladder(raw: str, step_after_20: float = 5.0) -> List[float]:
    if not raw.strip():
        return build_profit_ladder(step_after_20=step_after_20)
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return build_profit_ladder(parts, step_after_20=step_after_20)


def highest_step_index(net_pnl_usdt: float, ladder: List[float]) -> Optional[int]:
    """Index of highest ladder rung reached by current net PnL."""
    if not ladder:
        return None
    best: Optional[int] = None
    for i, step in enumerate(ladder):
        if net_pnl_usdt >= step:
            best = i
    return best


def lock_profit_usdt_for_index(ladder: List[float], achieved_index: int) -> float:
    """Floor profit to lock when achieved_index rung is reached."""
    if achieved_index <= 0:
        return ladder[0] if ladder else 0.0
    return ladder[achieved_index - 1]


def gross_pnl_usdt(entry: float, mark: float, qty: float, side: str) -> float:
    if side == "LONG":
        return (mark - entry) * qty
    return (entry - mark) * qty


def net_pnl_at_price(
    stack: "ChampaignStack",
    price: float,
    costs: "PaperCostModel",
) -> float:
    m = stack.main
    gross = gross_pnl_usdt(m.entry_price, price, m.quantity, m.side)
    exit_fill = costs.simulate_exit_fill(price, m.quantity, m.side)
    return gross - m.entry_costs - exit_fill.total


def sl_price_for_locked_profit(
    stack: "ChampaignStack",
    target_net_usdt: float,
    costs: "PaperCostModel",
) -> float:
    """
    Solve for stop price where net PnL ≈ target_net_usdt (after entry + exit costs).
    """
    m = stack.main
    qty = m.quantity
    if qty <= 0 or m.entry_price <= 0:
        return m.entry_price

    if m.side == "LONG":
        guess = m.entry_price + (target_net_usdt + m.entry_costs) / qty
        for _ in range(12):
            net = net_pnl_at_price(stack, guess, costs)
            if abs(net - target_net_usdt) < 0.02:
                break
            guess += (target_net_usdt - net) / qty
        return max(guess, 0.0)

    guess = m.entry_price - (target_net_usdt + m.entry_costs) / qty
    for _ in range(12):
        net = net_pnl_at_price(stack, guess, costs)
        if abs(net - target_net_usdt) < 0.02:
            break
        guess -= (target_net_usdt - net) / qty
    return max(guess, 0.0)


def sl_improves(side: str, old_sl: float, new_sl: float) -> bool:
    if old_sl <= 0:
        return new_sl > 0
    if side == "LONG":
        return new_sl > old_sl
    return new_sl < old_sl
