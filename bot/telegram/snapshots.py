"""Build BotSnapshot for Champaign / scalper engines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from bot.config import Config
from bot.paper.account import PaperWallet
from bot.telegram.formatters import BotSnapshot, PositionLine

if TYPE_CHECKING:
    from bot.champaign.executor import ChampaignExecutor
    from bot.champaign.state import ChampaignStack
    from bot.exchange.client import BinanceFuturesClient


def _leg_pnl(side: str, entry: float, mark: float, qty: float) -> float:
    if side == "LONG":
        return (mark - entry) * qty
    return (entry - mark) * qty


def _mark_for_stack(
    sym: str,
    stack: "ChampaignStack",
    prices: Dict[str, float],
    client: Optional["BinanceFuturesClient"],
) -> float:
    """Same source as ChampaignExecutor.get_balance (REST mark)."""
    if client:
        try:
            return client.get_mark_price(sym)
        except Exception:
            pass
    mark = prices.get(sym) or 0.0
    return mark if mark > 0 else stack.main.entry_price


def champaign_snapshot(
    cfg: Config,
    executor: "ChampaignExecutor",
    prices: Dict[str, float],
    client: Optional["BinanceFuturesClient"] = None,
) -> BotSnapshot:
    mode = "DRY RUN" if cfg.dry_run else ("TESTNET" if cfg.is_testnet() else "LIVE")
    wallet = executor._wallet if cfg.dry_run else None
    if wallet is None and cfg.dry_run:
        wallet = PaperWallet.load()

    positions: list[PositionLine] = []
    unrealized = 0.0
    for sym, stack in executor.stacks_items():
        mark = _mark_for_stack(sym, stack, prices, client)
        u = _leg_pnl(stack.main.side, stack.main.entry_price, mark, stack.main.quantity)
        if stack.hedge:
            h = stack.hedge
            u += _leg_pnl(h.side, h.entry_price, mark, h.quantity)
        unrealized += u
        positions.append(
            PositionLine(
                symbol=sym,
                side=stack.main.side,
                entry=stack.main.entry_price,
                mark=mark,
                sl=stack.active_sl,
                tp=stack.active_tp,
                stage=stack.stage,
                hedge=stack.hedge is not None,
                unrealized_usdt=u,
                locked_profit_usdt=stack.locked_profit_usdt,
            )
        )

    bal_view = executor.get_balance()
    # equity = wallet + uPnL (same as Heartbeat "bal=" in logs)
    equity = bal_view.total_balance
    if wallet:
        unrealized = equity - wallet.balance

    return BotSnapshot(
        mode=mode,
        strategy="CHAMPAIGN",
        balance=equity,
        available=bal_view.available_balance,
        initial_balance=wallet.initial_balance if wallet else 0.0,
        unrealized_usdt=unrealized,
        open_count=len(positions),
        max_positions=cfg.champaign_max_positions,
        positions=positions,
        paper_closed=wallet.closed_trades if wallet else 0,
        paper_wins=wallet.wins if wallet else 0,
        paper_costs=wallet.total_costs if wallet else 0.0,
    )


def scalper_snapshot(
    cfg: Config,
    trader,
    client: Optional["BinanceFuturesClient"] = None,
) -> BotSnapshot:
    from bot.paper.trader import PaperTrader

    mode = "DRY RUN" if cfg.dry_run else ("TESTNET" if cfg.is_testnet() else "LIVE")
    positions: list[PositionLine] = []
    unrealized = 0.0

    if isinstance(trader, PaperTrader):
        bal = trader.wallet.balance
        avail = trader.wallet.available
        wallet = trader.wallet
        for sym, pos in wallet.positions.items():
            mark = pos.entry_price
            if client:
                try:
                    mark = client.get_mark_price(sym)
                except Exception:
                    pass
            u = _leg_pnl(pos.side, pos.entry_price, mark, pos.quantity)
            unrealized += u
            positions.append(
                PositionLine(
                    symbol=sym,
                    side=pos.side,
                    entry=pos.entry_price,
                    mark=mark,
                    sl=pos.stop_price,
                    tp=pos.tp_price,
                    unrealized_usdt=u,
                )
            )
        bal = bal + unrealized
        max_pos = cfg.max_positions
    else:
        bal_obj = client.get_balance() if client else None
        bal = bal_obj.total_balance if bal_obj else 0.0
        avail = bal_obj.available_balance if bal_obj else 0.0
        wallet = None
        max_pos = cfg.max_positions
        if client:
            for p in client.get_positions():
                if p.size <= 0:
                    continue
                mark = p.mark_price or p.entry_price
                side = "LONG" if p.size > 0 else "SHORT"
                qty = abs(p.size)
                u = _leg_pnl(side, p.entry_price, mark, qty)
                unrealized += u
                positions.append(
                    PositionLine(
                        symbol=p.symbol,
                        side=side,
                        entry=p.entry_price,
                        mark=mark,
                        sl=0.0,
                        tp=0.0,
                        unrealized_usdt=u,
                    )
                )

    return BotSnapshot(
        mode=mode,
        strategy=f"SCALPER ({cfg.strategy_mode})",
        balance=bal,
        available=avail,
        initial_balance=wallet.initial_balance if wallet else 0.0,
        unrealized_usdt=unrealized,
        open_count=len(positions),
        max_positions=max_pos,
        positions=positions,
        paper_closed=wallet.closed_trades if wallet else 0,
        paper_wins=wallet.wins if wallet else 0,
        paper_costs=wallet.total_costs if wallet else 0.0,
    )
