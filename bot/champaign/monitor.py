"""
Real-time position monitor: profit-lock SL ladder, Fib TP, hedge.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from bot.champaign.fib import fib_target, swing_range
from bot.champaign.profit_lock import (
    highest_step_index,
    lock_profit_usdt_for_index,
    net_pnl_at_price,
    parse_profit_ladder,
    sl_improves,
    sl_price_for_locked_profit,
)
from bot.champaign.state import ChampaignStack
from bot.config import Config
from bot.exchange.client import Candle
from bot.paper.costs import PaperCostModel

logger = logging.getLogger(__name__)


class ChampaignPositionMonitor:
    def __init__(
        self,
        cfg: Config,
        fib_levels: list[float],
        on_open_hedge: Callable[[ChampaignStack, float], None],
        on_close_hedge: Callable[[ChampaignStack, float], None],
        on_close_main: Callable[[ChampaignStack, float, str], None],
        on_update_orders: Callable[[ChampaignStack, float, float], None],
    ):
        self.cfg = cfg
        self.fib_levels = fib_levels
        self._costs = PaperCostModel(cfg)
        self._ladder = parse_profit_ladder(
            cfg.champaign_profit_lock_steps,
            cfg.champaign_profit_lock_step_after_20,
        )
        self.on_open_hedge = on_open_hedge
        self.on_close_hedge = on_close_hedge
        self.on_close_main = on_close_main
        self.on_update_orders = on_update_orders

    def _main_gross_pnl(self, stack: ChampaignStack, mark: float) -> float:
        m = stack.main
        if m.side == "LONG":
            return (mark - m.entry_price) * m.quantity
        return (m.entry_price - mark) * m.quantity

    def _hedge_pnl_usdt(self, stack: ChampaignStack, mark: float) -> float:
        if not stack.hedge:
            return 0.0
        h = stack.hedge
        if h.side == "LONG":
            return (mark - h.entry_price) * h.quantity
        return (h.entry_price - mark) * h.quantity

    def _check_protective_hit(self, stack: ChampaignStack, mark: float) -> Optional[str]:
        sl, tp = stack.active_sl, stack.active_tp
        if stack.main.side == "LONG":
            if sl > 0 and mark <= sl:
                return "SL"
            if tp > 0 and mark >= tp:
                return "TP"
        else:
            if sl > 0 and mark >= sl:
                return "SL"
            if tp > 0 and mark <= tp:
                return "TP"
        return None

    def _apply_profit_lock(self, stack: ChampaignStack, mark: float) -> bool:
        """Raise SL to lock net USDT profit per ladder. Returns True if SL updated."""
        net = net_pnl_at_price(stack, mark, self._costs)
        if net > stack.peak_net_pnl_usdt:
            stack.peak_net_pnl_usdt = net

        idx = highest_step_index(net, self._ladder)
        if idx is None:
            return False

        lock_usdt = lock_profit_usdt_for_index(self._ladder, idx)
        if lock_usdt <= stack.locked_profit_usdt:
            return False

        new_sl = sl_price_for_locked_profit(stack, lock_usdt, self._costs)
        if not sl_improves(stack.main.side, stack.active_sl, new_sl):
            return False

        stack.locked_profit_usdt = lock_usdt
        stack.stage = "profit_lock"
        self.on_update_orders(stack, new_sl, stack.active_tp)
        logger.info(
            "[CHAMP] %s profit-lock +%.2f USDT (float=%.2f peak=%.2f) SL=%.6f",
            stack.symbol,
            lock_usdt,
            net,
            stack.peak_net_pnl_usdt,
            new_sl,
        )
        return True

    def tick(
        self,
        stack: ChampaignStack,
        mark: float,
        candles: Optional[list[Candle]] = None,
    ) -> None:
        if mark <= 0:
            return

        hit = self._check_protective_hit(stack, mark)
        if hit:
            self.on_close_main(stack, mark, hit)
            return

        net = net_pnl_at_price(stack, mark, self._costs)
        hedge_pnl = self._hedge_pnl_usdt(stack, mark)

        # Hedge rescue: main green, hedge red → close hedge, lock breakeven on main
        if stack.hedge and hedge_pnl < 0 and net > 0:
            logger.info("[CHAMP] %s hedge off — main net +%.2f", stack.symbol, net)
            self.on_close_hedge(stack, mark)
            if stack.locked_profit_usdt < 0:
                lock_usdt = self._ladder[0] if self._ladder else 0.0
                new_sl = sl_price_for_locked_profit(stack, lock_usdt, self._costs)
                if sl_improves(stack.main.side, stack.active_sl, new_sl):
                    stack.locked_profit_usdt = lock_usdt
                    stack.stage = "profit_lock"
                    self.on_update_orders(stack, new_sl, stack.active_tp)
            return

        # Deep loss (net, incl. costs) → hedge
        if (
            self.cfg.champaign_hedge_enabled
            and not stack.hedge
            and net <= -self.cfg.champaign_hedge_trigger_usdt
        ):
            logger.warning(
                "[CHAMP] %s hedge trigger net=%.2f (limit -%.0f)",
                stack.symbol,
                net,
                self.cfg.champaign_hedge_trigger_usdt,
            )
            self.on_open_hedge(stack, mark)
            stack.stage = "hedged"
            return

        self._apply_profit_lock(stack, mark)

        if candles and len(candles) >= 10:
            hi, lo = swing_range(candles)
            stack.swing_high, stack.swing_low = hi, lo
