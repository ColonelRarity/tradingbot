"""
Execute entries and protective orders on Binance Futures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient, OrderSide
from bot.notify import send_message
from bot.risk.manager import TradePlan
from bot.strategy.scalper import Side

logger = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_price: float
    tp_price: float
    opened_at: float = field(default_factory=time.time)


class Trader:
    def __init__(self, client: BinanceFuturesClient, config: Optional[Config] = None):
        self.client = client
        self.cfg = config or load_config()
        self.open_trades: Dict[str, OpenTrade] = {}

    def _setup_symbol(self, symbol: str) -> None:
        self.client.set_margin_type(symbol, self.cfg.margin_type)
        self.client.set_leverage(symbol, self.cfg.leverage)

    def sync_positions(self) -> None:
        """Align local state with exchange."""
        live = {p.symbol: p for p in self.client.get_positions()}
        for sym in list(self.open_trades):
            if sym not in live:
                del self.open_trades[sym]
        for sym, pos in live.items():
            if sym not in self.open_trades:
                self.open_trades[sym] = OpenTrade(
                    symbol=sym,
                    side=pos.side,
                    quantity=pos.size,
                    entry_price=pos.entry_price,
                    stop_price=0,
                    tp_price=0,
                )

    def count_positions(self) -> int:
        return len([p for p in self.client.get_positions() if p.size > 0])

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol and p.size > 0 for p in self.client.get_positions())

    def open_scalp(self, symbol: str, side: Side, plan: TradePlan, mark_price: float) -> bool:
        if not plan.valid or side == Side.NONE:
            logger.info("%s skip: %s", symbol, plan.reason)
            return False

        if self.has_position(symbol):
            logger.debug("%s already in position", symbol)
            return False

        self._setup_symbol(symbol)
        self.client.cancel_all_orders(symbol)

        order_side = OrderSide.BUY if side == Side.LONG else OrderSide.SELL
        pos_side = side.value if self.client.is_dual_side() else None

        try:
            entry = self.client.market_order(
                symbol, order_side, plan.quantity, position_side=pos_side
            )
            qty = entry.quantity or plan.quantity
            entry_px = mark_price

            close_side = OrderSide.SELL if side == Side.LONG else OrderSide.BUY
            self.client.stop_loss(symbol, close_side, plan.stop_price, qty, position_side=pos_side)
            self.client.take_profit(symbol, close_side, plan.take_profit_price, qty, position_side=pos_side)

            self.open_trades[symbol] = OpenTrade(
                symbol=symbol,
                side=side.value,
                quantity=qty,
                entry_price=entry_px,
                stop_price=plan.stop_price,
                tp_price=plan.take_profit_price,
            )

            msg = (
                f"<b>OPEN {side.value}</b> {symbol}\n"
                f"qty={qty:.6f} entry≈{entry_px:.4f}\n"
                f"SL={plan.stop_price:.4f} TP={plan.take_profit_price:.4f}"
            )
            logger.info(msg.replace("<b>", "").replace("</b>", ""))
            send_message(msg)
            return True
        except Exception as e:
            logger.error("Open failed %s: %s", symbol, e)
            send_message(f"❌ Open failed {symbol}: {e}")
            return False

    def close_symbol(self, symbol: str) -> None:
        pos = next((p for p in self.client.get_positions(symbol) if p.size > 0), None)
        if not pos:
            return
        self.client.cancel_all_orders(symbol)
        side = OrderSide.SELL if pos.side == "LONG" else OrderSide.BUY
        ps = pos.side if self.client.is_dual_side() else None
        self.client.market_order(symbol, side, pos.size, position_side=ps, reduce_only=True)
        self.open_trades.pop(symbol, None)
        logger.info("Closed %s", symbol)
        send_message(f"Closed {symbol}")
