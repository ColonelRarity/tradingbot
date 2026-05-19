"""
Reconcile Champaign stacks with live Binance positions and algo SL/TP orders.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from binance.error import ClientError

from bot.champaign.state import ChampaignStack, LegState, ProtectiveSlot
from bot.config import Config
from bot.exchange.client import BinanceFuturesClient, Order, Position

logger = logging.getLogger(__name__)


def _sl_tp_from_algos(
    algos: List[Order],
    main_side: str,
) -> Tuple[float, float, List[ProtectiveSlot]]:
    """Pick SL (STOP*) and TP (TAKE_PROFIT*) trigger prices from open algo orders."""
    sl, tp = 0.0, 0.0
    slots: List[ProtectiveSlot] = []
    for o in algos:
        ot = (o.order_type or "").upper()
        if "STOP" in ot and "TAKE_PROFIT" not in ot:
            sl = o.stop_price or sl
            slots.append(ProtectiveSlot("SL", sl, o.order_id))
        elif "TAKE_PROFIT" in ot:
            tp = o.stop_price or tp
            slots.append(ProtectiveSlot("TP", tp, o.order_id))
    return sl, tp, slots


def stack_from_position(
    pos: Position,
    algos: List[Order],
    cfg: Config,
) -> ChampaignStack:
    sl, tp, prot = _sl_tp_from_algos(algos, pos.side)
    est_cost = pos.entry_price * pos.size * cfg.paper_taker_fee
    return ChampaignStack(
        symbol=pos.symbol,
        main=LegState(
            pos.side,
            pos.size,
            pos.entry_price,
            margin_usdt=pos.entry_price * pos.size / max(cfg.leverage, 1),
            entry_costs=est_cost,
        ),
        active_sl=sl,
        active_tp=tp,
        protective=prot,
    )


def reconcile_live_stacks(
    client: BinanceFuturesClient,
    cfg: Config,
    stacks: Dict[str, ChampaignStack],
) -> Dict[str, ChampaignStack]:
    """
    Merge saved stacks with exchange: update qty/entry, adopt orphan positions,
  drop stacks whose main leg was closed on exchange.
    """
    positions = [p for p in client.get_positions() if p.size > 0]
    by_sym: Dict[str, List[Position]] = {}
    for p in positions:
        by_sym.setdefault(p.symbol, []).append(p)

    live_syms = set(by_sym)
    out = dict(stacks)

    for sym in list(out):
        if sym not in live_syms:
            logger.info("[CHAMP] live sync: %s closed on exchange — drop stack", sym)
            client.cancel_all_orders(sym)
            del out[sym]
            continue
        pos_list = by_sym[sym]
        stack = out[sym]
        main = next((p for p in pos_list if p.side == stack.main.side), None)
        if not main:
            logger.warning("[CHAMP] live sync: %s side mismatch — drop stack", sym)
            client.cancel_all_orders(sym)
            del out[sym]
            continue
        stack.main.quantity = main.size
        stack.main.entry_price = main.entry_price
        hedge_side = "SHORT" if stack.main.side == "LONG" else "LONG"
        hedge_pos = next((p for p in pos_list if p.side == hedge_side), None)
        if hedge_pos and stack.hedge:
            stack.hedge.quantity = hedge_pos.size
            stack.hedge.entry_price = hedge_pos.entry_price
        elif hedge_pos and not stack.hedge:
            stack.hedge = LegState(
                hedge_side,
                hedge_pos.size,
                hedge_pos.entry_price,
                margin_usdt=0,
            )
            stack.stage = "hedged"
        elif stack.hedge and not hedge_pos:
            stack.hedge = None
            if stack.stage == "hedged":
                stack.stage = "profit_lock" if stack.locked_profit_usdt >= 0 else "initial"

        algos = client.get_open_algo_orders(sym)
        sl, tp, prot = _sl_tp_from_algos(algos, stack.main.side)
        if sl > 0:
            stack.active_sl = sl
        if tp > 0:
            stack.active_tp = tp
        if prot:
            stack.protective = prot

    for sym, pos_list in by_sym.items():
        if sym in out:
            continue
        if len(pos_list) != 1:
            logger.warning(
                "[CHAMP] live sync: orphan %s has %d legs — adopt largest only",
                sym,
                len(pos_list),
            )
        pos = max(pos_list, key=lambda p: p.size)
        algos = client.get_open_algo_orders(sym)
        out[sym] = stack_from_position(pos, algos, cfg)
        logger.warning(
            "[CHAMP] live sync: adopted orphan %s %s qty=%.4f (no local stack)",
            sym,
            pos.side,
            pos.size,
        )

    return out


def ensure_hedge_position_mode(client: BinanceFuturesClient, cfg: Config) -> None:
    """Champaign hedge needs dual-side (hedge) mode on the account."""
    if not cfg.champaign_hedge_enabled:
        return
    if cfg.position_mode != "hedge":
        if cfg.champaign_hedge_enabled:
            logger.warning(
                "[CHAMP] Hedge enabled but POSITION_MODE=%s — "
                "set POSITION_MODE=hedge for mirror hedge on live",
                cfg.position_mode,
            )
        return
    if client.is_dual_side():
        return
    try:
        client.set_dual_side_position(True)
    except ClientError as e:
        logger.error(
            "[CHAMP] Cannot enable hedge mode (close all positions first): %s",
            e,
        )
