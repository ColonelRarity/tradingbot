"""
Dry-run trader: same decisions as live, zero orders on the exchange.
Costs: taker fee, slippage, spread, pro-rated funding.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient
from bot.notify import send_message
from bot.paper.account import PaperPosition, PaperWallet, check_exit
from bot.paper.costs import PaperCostModel
from bot.paper.wallet_open import log_paper_wallet, open_paper_wallet, resolve_paper_target
from bot.risk.manager import TradePlan
from bot.strategy.scalper import Side

logger = logging.getLogger(__name__)


class BalanceView:
    def __init__(self, total: float, available: float):
        self.total_balance = total
        self.available_balance = available


class PaperTrader:
    def __init__(
        self,
        client: BinanceFuturesClient,
        config: Optional[Config] = None,
        journal=None,
    ):
        self.client = client
        self.cfg = config or load_config()
        self.journal = journal
        self.costs = PaperCostModel(self.cfg)
        self._funding_cache: Dict[str, tuple[float, float]] = {}
        self.on_position_closed: Optional[Callable[[str, str], None]] = None

        target = resolve_paper_target(self.cfg, self.client)
        self.wallet = open_paper_wallet(self.cfg, self.client)
        log_paper_wallet(self.wallet, target)

    def _funding_rate(self, symbol: str) -> float:
        now = time.time()
        cached = self._funding_cache.get(symbol)
        if cached and now - cached[1] < 120:
            return cached[0]
        rate = self.costs.funding_fallback
        if self.costs.use_live_funding:
            try:
                rate = self.client.get_funding_rate(symbol)
            except Exception:
                pass
        self._funding_cache[symbol] = (rate, now)
        return rate

    def _apply_costs_to_wallet(
        self,
        commission: float,
        slippage: float,
        spread: float,
        funding: float = 0.0,
    ) -> float:
        total = commission + slippage + spread + funding
        self.wallet.total_commission += commission
        self.wallet.total_slippage += slippage
        self.wallet.total_spread += spread
        self.wallet.total_funding += funding
        self.wallet.available -= total
        self.wallet.balance -= total
        return total

    def _accrue_funding(self, symbol: str, pos: PaperPosition, mark: float) -> None:
        now = time.time()
        elapsed = now - pos.last_funding_accrual
        if elapsed < 30:
            return
        rate = self._funding_rate(symbol)
        notional = pos.quantity * mark
        payment = self.costs.funding_payment(pos.side, notional, rate, elapsed)
        if abs(payment) < 1e-8:
            pos.last_funding_accrual = now
            return
        self._apply_costs_to_wallet(0, 0, 0, funding=payment)
        pos.funding_paid += payment
        pos.last_funding_accrual = now

    def get_balance(self) -> BalanceView:
        u = 0.0
        for sym, pos in self.wallet.positions.items():
            try:
                mark = self.client.get_mark_price(sym)
                u += (mark - pos.entry_price) * pos.quantity if pos.side == "LONG" else (pos.entry_price - mark) * pos.quantity
            except Exception:
                pass
        return BalanceView(
            total=self.wallet.balance + u,
            available=self.wallet.available,
        )

    def sync_positions(self) -> None:
        pass

    def count_positions(self) -> int:
        return len(self.wallet.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.wallet.positions

    def update_markets(self, prices: Dict[str, float]) -> None:
        for symbol in list(self.wallet.positions.keys()):
            mark = prices.get(symbol)
            if mark is None:
                continue
            pos = self.wallet.positions[symbol]
            self._accrue_funding(symbol, pos, mark)
            reason = check_exit(pos, mark)
            if reason:
                self._close_position(symbol, mark, reason)

    def open_scalp(
        self,
        symbol: str,
        side: Side,
        plan: TradePlan,
        mark_price: float,
        signal_reason: str = "",
        rsi: float = 0.0,
    ) -> bool:
        if not plan.valid or side == Side.NONE or self.has_position(symbol):
            return False

        fill = self.costs.simulate_entry_fill(mark_price, plan.quantity, side.value)
        notional = fill.fill_price * plan.quantity
        margin = notional / self.cfg.leverage
        open_costs = fill.total

        if margin + open_costs > self.wallet.available:
            logger.info(
                "[DRY RUN] SKIP %s — paper margin+costs %.2f > avail %.2f",
                symbol, margin + open_costs, self.wallet.available,
            )
            return False

        self.wallet.available -= margin
        self._apply_costs_to_wallet(fill.commission, fill.slippage_cost, fill.spread_cost)

        self.wallet.positions[symbol] = PaperPosition(
            symbol=symbol,
            side=side.value,
            quantity=plan.quantity,
            entry_price=fill.fill_price,
            mark_at_entry=mark_price,
            stop_price=plan.stop_price,
            tp_price=plan.take_profit_price,
            margin_usdt=margin,
            notional_usdt=notional,
            entry_reason=signal_reason,
            entry_costs=open_costs,
            last_funding_accrual=time.time(),
        )
        self.wallet.save()

        msg = (
            f"[DRY RUN] WOULD ENTER {side.value} {symbol}\n"
            f"  Live: відкрив би MARKET зараз.\n"
            f"  mark={mark_price:.6f} → fill={fill.fill_price:.6f}\n"
            f"  costs: comm={fill.commission:.4f} slip={fill.slippage_cost:.4f} "
            f"spread={fill.spread_cost:.4f} (total {open_costs:.4f})\n"
            f"  SL={plan.stop_price:.6f} TP={plan.take_profit_price:.6f}\n"
            f"  bal={self.wallet.balance:.2f} avail={self.wallet.available:.2f}"
        )
        logger.info(msg)
        return True

    def _close_position(self, symbol: str, mark: float, reason: str) -> None:
        pos = self.wallet.positions.get(symbol)
        if not pos:
            return

        self._accrue_funding(symbol, pos, mark)
        exit_fill = self.costs.simulate_exit_fill(mark, pos.quantity, pos.side)
        exit_costs = exit_fill.total

        if pos.side == "LONG":
            gross = (exit_fill.fill_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_fill.fill_price) * pos.quantity

        self._apply_costs_to_wallet(
            exit_fill.commission, exit_fill.slippage_cost, exit_fill.spread_cost
        )
        net = gross - exit_costs
        # entry costs already deducted at open; gross is vs fill prices

        self.wallet.available += pos.margin_usdt + gross
        self.wallet.balance += gross - exit_costs
        self.wallet.realized_pnl += gross - pos.entry_costs - exit_costs - pos.funding_paid
        self.wallet.closed_trades += 1
        if gross - exit_costs > 0:
            self.wallet.wins += 1

        del self.wallet.positions[symbol]
        self.wallet.save()

        all_in_net = gross - pos.entry_costs - exit_costs - pos.funding_paid
        if self.journal:
            self.journal.log_trade_close(symbol, exit_fill.fill_price, all_in_net)
        from bot.telegram.bot import push_trade

        push_trade(
            "EXIT",
            symbol,
            f"{reason} @ {exit_fill.fill_price:.6f}",
            pnl_usdt=all_in_net,
            config=self.cfg,
        )

        if self.on_position_closed:
            self.on_position_closed(symbol, reason)

        held = time.time() - pos.opened_at
        total_trade_costs = pos.entry_costs + exit_costs + pos.funding_paid
        logger.info(
            "[DRY RUN] WOULD EXIT %s (%s) | live: закрив би зараз\n"
            "  gross=%+.4f | exit costs=%.4f | entry costs=%.4f | funding=%.4f | "
            "all-in costs=%.4f\n"
            "  net≈%+.4f | held=%.0fs | bal=%.2f (cum costs %.4f)",
            symbol, reason, gross, exit_costs, pos.entry_costs, pos.funding_paid,
            total_trade_costs, gross - total_trade_costs, held,
            self.wallet.balance, self.wallet.total_costs,
        )

    def close_symbol(self, symbol: str, mark: float, reason: str = "MANUAL") -> None:
        if symbol in self.wallet.positions:
            self._close_position(symbol, mark, reason)

    def summary_line(self) -> str:
        w = self.wallet
        wr = (w.wins / w.closed_trades * 100) if w.closed_trades else 0
        pnl_vs_start = w.balance - w.initial_balance
        return (
            f"paper bal={w.balance:.2f} ({pnl_vs_start:+.2f} vs start) | "
            f"costs={w.total_costs:.4f} [fee {w.total_commission:.3f} "
            f"slip {w.total_slippage:.3f} spread {w.total_spread:.3f} "
            f"funding {w.total_funding:.3f}] | "
            f"trades={w.closed_trades} WR={wr:.0f}%"
        )
