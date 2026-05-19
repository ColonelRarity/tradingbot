"""
Open/close Champaign legs (paper simulation + live).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from bot.champaign.fib import fib_target, swing_range
from bot.champaign.live_sync import ensure_hedge_position_mode, reconcile_live_stacks
from bot.champaign.order_manager import ProtectiveOrderManager
from bot.champaign.state import ChampaignStack, LegState, load_stacks, save_stacks
from bot.config import Config
from bot.exchange.client import BinanceFuturesClient, OrderSide
from bot.paper.account import PaperWallet
from bot.paper.costs import PaperCostModel
from bot.paper.trader import BalanceView
from bot.risk.manager import RiskManager, TradePlan
from bot.strategy.champaign import ChampaignSignal
from bot.strategy.scalper import Side
from bot.telegram.bot import push_trade

logger = logging.getLogger(__name__)


class ChampaignExecutor:
    def __init__(
        self,
        client: BinanceFuturesClient,
        cfg: Config,
        journal=None,
    ):
        self.client = client
        self.cfg = cfg
        self.journal = journal
        self.dry_run = cfg.dry_run
        self._stacks_lock = threading.RLock()
        with self._stacks_lock:
            self.stacks: Dict[str, ChampaignStack] = load_stacks()
        self.orders = ProtectiveOrderManager(client, cfg, cfg.dry_run)
        self.risk = RiskManager(cfg)
        self.fib_levels = cfg.champaign_fib_levels
        self._costs = PaperCostModel(cfg)
        self._wallet: Optional[PaperWallet] = None
        if cfg.dry_run:
            self._init_paper_wallet()
        else:
            self._init_live()

    def _init_paper_wallet(self) -> None:
        from bot.paper.wallet_open import log_paper_wallet, open_paper_wallet, resolve_paper_target

        target = resolve_paper_target(self.cfg, self.client)
        self._wallet = open_paper_wallet(self.cfg, self.client)
        log_paper_wallet(self._wallet, target)

    def _init_live(self) -> None:
        ensure_hedge_position_mode(self.client, self.cfg)
        with self._stacks_lock:
            self.stacks = reconcile_live_stacks(self.client, self.cfg, self.stacks)
            n = len(self.stacks)
            keys = list(self.stacks)
        if n:
            self._persist()
            logger.info("[CHAMP] LIVE resumed %d stack(s): %s", n, ",".join(keys))

    def stacks_items(self) -> List[Tuple[str, ChampaignStack]]:
        with self._stacks_lock:
            return list(self.stacks.items())

    def stacks_keys(self) -> List[str]:
        with self._stacks_lock:
            return list(self.stacks.keys())

    def stacks_len(self) -> int:
        with self._stacks_lock:
            return len(self.stacks)

    def stacks_contains(self, symbol: str) -> bool:
        with self._stacks_lock:
            return symbol in self.stacks

    def clear_stacks(self) -> None:
        with self._stacks_lock:
            self.stacks.clear()

    def _persist(self) -> None:
        with self._stacks_lock:
            save_stacks(self.stacks)

    def _position_side(self, side: str) -> Optional[str]:
        if self.client.is_dual_side():
            return side
        return None

    def _estimate_entry_costs(self, entry: float, qty: float, side: str) -> float:
        fill = self._costs.simulate_entry_fill(entry, qty, side)
        return fill.total

    def get_balance(self) -> BalanceView:
        if self.dry_run and self._wallet:
            u = 0.0
            for sym, stack in self.stacks_items():
                try:
                    mark = self.client.get_mark_price(sym)
                    m = stack.main
                    if m.side == "LONG":
                        u += (mark - m.entry_price) * m.quantity
                    else:
                        u += (m.entry_price - mark) * m.quantity
                    if stack.hedge:
                        h = stack.hedge
                        if h.side == "LONG":
                            u += (mark - h.entry_price) * h.quantity
                        else:
                            u += (h.entry_price - mark) * h.quantity
                except Exception:
                    pass
            return BalanceView(
                total=self._wallet.balance + u,
                available=self._wallet.available,
            )
        bal = self.client.get_balance()
        return BalanceView(bal.total_balance, bal.available_balance)

    def count_positions(self) -> int:
        if self.dry_run:
            return self.stacks_len()
        return len([p for p in self.client.get_positions() if p.size > 0])

    def has_position(self, symbol: str) -> bool:
        if self.dry_run:
            return self.stacks_contains(symbol)
        return any(p.symbol == symbol and p.size > 0 for p in self.client.get_positions())

    def sync_positions(self) -> None:
        if not self.dry_run:
            with self._stacks_lock:
                self.stacks = reconcile_live_stacks(self.client, self.cfg, self.stacks)
            self._persist()

    def _close_side(self, side: str) -> OrderSide:
        return OrderSide.SELL if side == "LONG" else OrderSide.BUY

    def _open_side(self, side: str) -> OrderSide:
        return OrderSide.BUY if side == "LONG" else OrderSide.SELL

    def open_main(
        self,
        symbol: str,
        sig: ChampaignSignal,
        plan: TradePlan,
        mark: float,
        candles,
    ) -> bool:
        if not plan.valid or self.has_position(symbol):
            return False

        hi, lo = swing_range(candles)
        far_fib = self.fib_levels[-1] if self.fib_levels else 1.618

        if self.dry_run:
            fill = self._costs.simulate_entry_fill(mark, plan.quantity, sig.side.value)
            entry = fill.fill_price
            sl_dist = sig.atr * self.cfg.champaign_initial_sl_atr
            if sig.side == Side.LONG:
                sl = entry - sl_dist
            else:
                sl = entry + sl_dist
            tp_price = fib_target(entry, hi, lo, sig.side.value, far_fib)
            margin = fill.fill_price * plan.quantity / self.cfg.leverage
            cost = fill.total
            if margin + cost > self._wallet.available:
                logger.info("[CHAMP] SKIP %s — insufficient paper margin", symbol)
                return False
            self._wallet.available -= margin
            self._wallet.balance -= cost
            self._wallet.save()
            stack = ChampaignStack(
                symbol=symbol,
                main=LegState(
                    sig.side.value,
                    plan.quantity,
                    fill.fill_price,
                    margin,
                    entry_costs=cost,
                ),
                swing_high=hi,
                swing_low=lo,
                active_sl=sl,
                active_tp=tp_price,
            )
            with self._stacks_lock:
                self.stacks[symbol] = stack
            self._persist()
            self.orders.place_sl(stack, sl, plan.quantity, self._close_side(sig.side.value), None)
            self.orders.place_tp(stack, tp_price, plan.quantity, self._close_side(sig.side.value), None)
            logger.info(
                "[CHAMP] OPEN %s %s @ %.6f SL=%.6f TP=%.6f ATR%%=%.2f",
                symbol, sig.side.value, fill.fill_price, sl, tp_price, sig.atr_pct,
            )
            if self.journal:
                self.journal.log_trade_open(
                    symbol,
                    sig.side.value,
                    fill.fill_price,
                    plan.quantity,
                    meta={"strategy": "champaign", "atr_pct": sig.atr_pct},
                )
            push_trade(
                "OPEN",
                symbol,
                f"{sig.side.value} @ {fill.fill_price:.6f} SL={sl:.6f} TP={tp_price:.6f}",
                config=self.cfg,
            )
            return True

        # Live — market entry + algo SL/TP on exchange
        try:
            self.client.set_margin_type(symbol, self.cfg.margin_type)
            self.client.set_leverage(symbol, self.cfg.leverage)
            self.client.cancel_all_orders(symbol)
            pos_side = self._position_side(sig.side.value)
            self.client.market_order(
                symbol,
                self._open_side(sig.side.value),
                plan.quantity,
                position_side=pos_side,
            )
            pos = self.client.wait_for_position(symbol, sig.side.value)
            if not pos:
                logger.error("[CHAMP] LIVE %s — no position after market entry", symbol)
                return False
            entry = pos.entry_price
            qty = pos.size
            sl_dist = sig.atr * self.cfg.champaign_initial_sl_atr
            if sig.side == Side.LONG:
                sl = entry - sl_dist
            else:
                sl = entry + sl_dist
            tp_price = fib_target(entry, hi, lo, sig.side.value, far_fib)
            close = self._close_side(sig.side.value)
            entry_costs = self._estimate_entry_costs(entry, qty, sig.side.value)
            stack = ChampaignStack(
                symbol=symbol,
                main=LegState(
                    sig.side.value,
                    qty,
                    entry,
                    pos.entry_price * qty / max(self.cfg.leverage, 1),
                    entry_costs=entry_costs,
                ),
                swing_high=hi,
                swing_low=lo,
                active_sl=sl,
                active_tp=tp_price,
            )
            with self._stacks_lock:
                self.stacks[symbol] = stack
            self.orders.replace_sl_tp(stack, sl, tp_price, qty, close, pos_side)
            self._persist()
            logger.info(
                "[CHAMP] LIVE OPEN %s %s @ %.6f qty=%.4f SL=%.6f TP=%.6f",
                symbol,
                sig.side.value,
                entry,
                qty,
                sl,
                tp_price,
            )
            if self.journal:
                self.journal.log_trade_open(
                    symbol,
                    sig.side.value,
                    entry,
                    qty,
                    meta={"strategy": "champaign", "live": True, "atr_pct": sig.atr_pct},
                )
            push_trade(
                "OPEN",
                symbol,
                f"LIVE {sig.side.value} @ {entry:.6f} SL={sl:.6f} TP={tp_price:.6f}",
                config=self.cfg,
            )
            return True
        except Exception as e:
            logger.error("[CHAMP] open failed %s: %s", symbol, e)
            return False

    def open_hedge(self, stack: ChampaignStack, mark: float) -> None:
        if stack.hedge:
            return
        hedge_side = "SHORT" if stack.main.side == "LONG" else "LONG"
        qty = stack.main.quantity

        if self.dry_run:
            fill = self._costs.simulate_entry_fill(mark, qty, hedge_side)
            margin = fill.fill_price * qty / self.cfg.leverage
            if margin + fill.total > self._wallet.available:
                logger.warning("[CHAMP] hedge skipped %s — no margin", stack.symbol)
                return
            self._wallet.available -= margin
            self._wallet.balance -= fill.total
            stack.hedge = LegState(hedge_side, qty, fill.fill_price, margin)
            self._persist()
            logger.info("[CHAMP] HEDGE %s %s @ %.6f", stack.symbol, hedge_side, fill.fill_price)
            push_trade(
                "HEDGE",
                stack.symbol,
                f"{hedge_side} @ {fill.fill_price:.6f}",
                config=self.cfg,
            )
            return

        try:
            if not self.client.is_dual_side():
                logger.error(
                    "[CHAMP] hedge %s skipped — account is ONE-WAY; set POSITION_MODE=hedge",
                    stack.symbol,
                )
                return
            pos_side = self._position_side(hedge_side)
            self.client.market_order(
                stack.symbol, self._open_side(hedge_side), qty, position_side=pos_side
            )
            hpos = self.client.wait_for_position(stack.symbol, hedge_side, timeout_sec=3.0)
            entry = hpos.entry_price if hpos else mark
            stack.hedge = LegState(hedge_side, qty, entry, 0)
            self._persist()
            logger.info("[CHAMP] HEDGE live %s %s @ %.6f", stack.symbol, hedge_side, entry)
            push_trade(
                "HEDGE",
                stack.symbol,
                f"LIVE {hedge_side} @ {entry:.6f}",
                config=self.cfg,
            )
        except Exception as e:
            logger.error("[CHAMP] hedge failed %s: %s", stack.symbol, e)

    def close_hedge(self, stack: ChampaignStack, mark: float) -> None:
        if not stack.hedge:
            return
        h = stack.hedge
        if self.dry_run:
            fill = self._costs.simulate_exit_fill(mark, h.quantity, h.side)
            if h.side == "LONG":
                gross = (fill.fill_price - h.entry_price) * h.quantity
            else:
                gross = (h.entry_price - fill.fill_price) * h.quantity
            self._wallet.available += h.margin_usdt + gross - fill.total
            self._wallet.balance += gross - fill.total
            stack.hedge = None
            self._persist()
            logger.info("[CHAMP] hedge closed %s pnl≈%.4f", stack.symbol, gross)
            push_trade(
                "HEDGE_OFF",
                stack.symbol,
                f"hedge closed pnl≈{gross:.4f}",
                config=self.cfg,
            )
            return
        try:
            pos_side = self._position_side(h.side)
            self.client.market_order(
                stack.symbol,
                self._close_side(h.side),
                h.quantity,
                position_side=pos_side,
                reduce_only=not bool(pos_side),
            )
            stack.hedge = None
            self._persist()
            push_trade(
                "HEDGE_OFF",
                stack.symbol,
                "LIVE hedge closed",
                config=self.cfg,
            )
        except Exception as e:
            logger.error("[CHAMP] close hedge %s: %s", stack.symbol, e)

    def close_main(self, stack: ChampaignStack, mark: float, reason: str) -> None:
        sym = stack.symbol
        m = stack.main
        if mark <= 0:
            logger.error("[CHAMP] refuse close %s — invalid mark", sym)
            return
        if stack.hedge:
            self.close_hedge(stack, mark)

        if self.dry_run:
            fill = self._costs.simulate_exit_fill(mark, m.quantity, m.side)
            if m.side == "LONG":
                gross = (fill.fill_price - m.entry_price) * m.quantity
            else:
                gross = (m.entry_price - fill.fill_price) * m.quantity
            entry_costs = getattr(m, "entry_costs", 0.0)
            net = gross - entry_costs - fill.total
            self._wallet.available += m.margin_usdt + gross - fill.total
            self._wallet.balance += gross - fill.total
            self._wallet.realized_pnl += net
            self._wallet.closed_trades += 1
            if net > 0:
                self._wallet.wins += 1
            self._wallet.save()
            if self.journal:
                self.journal.log_trade_close(sym, fill.fill_price, net)
            with self._stacks_lock:
                self.stacks.pop(sym, None)
            self._persist()
            logger.info(
                "[CHAMP] EXIT %s (%s) qty=%.4f mark=%.6f net≈%.4f",
                sym, reason, m.quantity, mark, net,
            )
            push_trade(
                "EXIT",
                sym,
                f"{reason} @ {fill.fill_price:.6f}",
                pnl_usdt=net,
                config=self.cfg,
            )
            return

        try:
            self.client.cancel_all_orders(sym)
            pos_side = self._position_side(m.side)
            pos = next(
                (p for p in self.client.get_positions(sym) if p.side == m.side and p.size > 0),
                None,
            )
            qty = pos.size if pos else m.quantity
            if qty > 0:
                self.client.market_order(
                    sym,
                    self._close_side(m.side),
                    qty,
                    position_side=pos_side,
                    reduce_only=not bool(pos_side),
                )
            if self.journal:
                self.journal.log_trade_close(sym, mark, 0.0)
            with self._stacks_lock:
                self.stacks.pop(sym, None)
            self._persist()
            logger.info("[CHAMP] LIVE EXIT %s (%s) @ mark %.6f", sym, reason, mark)
            push_trade(
                "EXIT",
                sym,
                f"LIVE {reason} @ {mark:.6f}",
                config=self.cfg,
            )
        except Exception as e:
            logger.error("[CHAMP] close main %s: %s", sym, e)

    def update_orders(self, stack: ChampaignStack, sl: float, tp: float) -> None:
        qty = stack.main.quantity
        close = self._close_side(stack.main.side)
        pos_side = self._position_side(stack.main.side)
        tp_unchanged = abs(tp - stack.active_tp) < 1e-12
        if not self.dry_run and tp_unchanged and stack.active_tp > 0:
            self.orders.update_sl(stack, sl, qty, close, pos_side)
            stack.active_sl = sl
        else:
            stack.active_sl, stack.active_tp = sl, tp
            self.orders.replace_sl_tp(stack, sl, tp, qty, close, pos_side)
        self._persist()
