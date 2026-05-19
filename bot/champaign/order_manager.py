"""
Manage protective SL/TP on exchange (algo orders) or in paper state.
"""

from __future__ import annotations

import logging
from typing import Optional

from bot.champaign.state import ChampaignStack, ProtectiveSlot
from bot.config import Config
from bot.exchange.client import BinanceFuturesClient, OrderSide

logger = logging.getLogger(__name__)


class ProtectiveOrderManager:
    def __init__(self, client: BinanceFuturesClient, cfg: Config, dry_run: bool):
        self.client = client
        self.cfg = cfg
        self.dry_run = dry_run
        self._cap = cfg.champaign_max_protective_orders

    def _trim(self, stack: ChampaignStack) -> None:
        while len(stack.protective) > self._cap:
            removed = stack.protective.pop(0)
            if not self.dry_run and removed.order_id:
                self.client.cancel_algo_order(removed.order_id, stack.symbol)

    def _cancel_protective_kind(self, stack: ChampaignStack, kind: str) -> None:
        if self.dry_run:
            stack.protective = [p for p in stack.protective if p.kind != kind]
            return
        for slot in list(stack.protective):
            if slot.kind == kind and slot.order_id:
                self.client.cancel_algo_order(slot.order_id, stack.symbol)
        stack.protective = [p for p in stack.protective if p.kind != kind]

    def place_sl(
        self,
        stack: ChampaignStack,
        price: float,
        quantity: float,
        close_side: OrderSide,
        position_side: Optional[str],
        *,
        cancel_existing: bool = True,
    ) -> None:
        stack.active_sl = price
        oid = 0
        if not self.dry_run:
            if cancel_existing:
                self._cancel_protective_kind(stack, "SL")
            o = self.client.stop_loss(
                stack.symbol, close_side, price, quantity, position_side=position_side
            )
            oid = o.order_id
        stack.protective.append(ProtectiveSlot("SL", price, oid))
        self._trim(stack)
        logger.info("[CHAMP] %s SL @ %.6f%s", stack.symbol, price, "" if self.dry_run else " (algo)")

    def place_tp(
        self,
        stack: ChampaignStack,
        price: float,
        quantity: float,
        close_side: OrderSide,
        position_side: Optional[str],
        *,
        cancel_existing: bool = True,
    ) -> None:
        stack.active_tp = price
        oid = 0
        if not self.dry_run:
            if cancel_existing:
                self._cancel_protective_kind(stack, "TP")
            o = self.client.take_profit(
                stack.symbol, close_side, price, quantity, position_side=position_side
            )
            oid = o.order_id
        stack.protective.append(ProtectiveSlot("TP", price, oid))
        self._trim(stack)
        logger.info("[CHAMP] %s TP @ %.6f%s", stack.symbol, price, "" if self.dry_run else " (algo)")

    def replace_sl_tp(
        self,
        stack: ChampaignStack,
        sl: float,
        tp: float,
        quantity: float,
        close_side: OrderSide,
        position_side: Optional[str],
    ) -> None:
        if not self.dry_run:
            self.client.cancel_all_algo(stack.symbol)
            stack.protective.clear()
        self.place_sl(
            stack, sl, quantity, close_side, position_side, cancel_existing=False
        )
        self.place_tp(
            stack, tp, quantity, close_side, position_side, cancel_existing=False
        )

    def update_sl(
        self,
        stack: ChampaignStack,
        sl: float,
        quantity: float,
        close_side: OrderSide,
        position_side: Optional[str],
    ) -> None:
        """Profit-lock: replace SL algo only, keep TP on exchange."""
        self.place_sl(stack, sl, quantity, close_side, position_side, cancel_existing=True)
