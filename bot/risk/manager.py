"""
Position sizing and trade plan from account risk budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timezone, datetime
from typing import Optional

from bot.config import Config, load_config
from bot.strategy.scalper import Side

logger = logging.getLogger(__name__)


@dataclass
class TradePlan:
    quantity: float
    notional_usdt: float
    stop_price: float
    take_profit_price: float
    risk_usdt: float
    valid: bool
    reason: str


class RiskManager:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self._day = date.today()
        self._start_equity: Optional[float] = None
        self._halted = False

    def reset_day_if_needed(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self._start_equity = equity
            self._halted = False
            logger.info("Daily risk tracker reset. Equity=%.2f", equity)
        if self._start_equity is None:
            self._start_equity = equity

    def daily_loss_exceeded(self, equity: float) -> bool:
        if self.cfg.dry_run and self.cfg.paper_skip_daily_loss_limit:
            return False
        self.reset_day_if_needed(equity)
        if self._start_equity is None or self._start_equity <= 0:
            return False
        loss_pct = (self._start_equity - equity) / self._start_equity * 100
        if loss_pct >= self.cfg.daily_max_loss_pct:
            if not self._halted:
                logger.warning("Daily loss limit hit: %.2f%%", loss_pct)
            self._halted = True
            return True
        return self._halted

    def build_plan(
        self,
        side: Side,
        entry_price: float,
        atr_value: float,
        available_balance: float,
        leverage: int,
        min_qty: float,
        min_notional: float,
        *,
        max_positions: int | None = None,
    ) -> TradePlan:
        if side == Side.NONE or entry_price <= 0 or atr_value <= 0:
            return TradePlan(0, 0, 0, 0, 0, False, "NO_SIGNAL")

        sl_dist = atr_value * self.cfg.sl_atr_mult
        tp_dist = atr_value * self.cfg.tp_atr_mult
        if sl_dist > 0 and tp_dist / sl_dist < self.cfg.min_rr_ratio:
            tp_dist = sl_dist * self.cfg.min_rr_ratio

        if side == Side.LONG:
            stop = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            stop = entry_price + sl_dist
            tp = entry_price - tp_dist

        risk_usdt = available_balance * (self.cfg.risk_percent / 100.0)
        sl_pct = sl_dist / entry_price
        if sl_pct <= 0:
            return TradePlan(0, 0, stop, tp, 0, False, "INVALID_SL")

        notional = risk_usdt / sl_pct
        notional = min(notional, available_balance * leverage * 0.9)
        if self.cfg.dry_run:
            # Split margin across max_positions so one trade cannot lock the wallet
            slots = max(max_positions if max_positions is not None else self.cfg.max_positions, 1)
            per_slot = available_balance * leverage / slots * 0.92
            notional = min(notional, per_slot)
        notional = max(notional, self.cfg.min_notional_usdt, min_notional)

        qty = notional / entry_price
        if qty < min_qty:
            return TradePlan(0, 0, stop, tp, risk_usdt, False, "QTY_BELOW_MIN")

        return TradePlan(qty, notional, stop, tp, risk_usdt, True, "OK")
