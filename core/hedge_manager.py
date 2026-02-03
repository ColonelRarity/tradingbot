"""
Hedge (Mirror) Position Manager

Handles hedge positions for drawdown reduction:
- Opens opposite position when main position PnL <= -7 USDT
- Fixed size, no TP extension
- Closes immediately at +3 USDT PnL
- Strictly linked to parent position

CRITICAL RULES:
- Never pyramid uncontrollably
- Max 1 hedge per parent position
- Hedge must reduce net drawdown
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from threading import Lock

from exchange.binance_client import BinanceClient, get_binance_client, OrderSide
from core.order_manager import OrderManager, OrderResult
from config.settings import get_settings, HedgeConfig


logger = logging.getLogger(__name__)


@dataclass
class HedgePosition:
    """
    Tracked hedge (mirror) position.
    """
    hedge_id: str
    parent_position_id: str
    symbol: str
    
    # Position details
    side: str  # Opposite to parent: "LONG" or "SHORT"
    entry_price: float
    quantity: float
    size_usdt: float
    
    # Status
    is_open: bool = True
    pnl_usdt: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    
    # Orders
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None


class HedgeManager:
    """
    Hedge Position Manager.
    
    Opens mirror positions to reduce drawdown on losing trades:
    
    TRIGGER: Main position PnL <= -7 USDT
    SIZE: Fixed USDT amount
    CLOSE: At +3 USDT profit
    RULES:
      - Max 1 hedge per parent
      - No TP extension (strict target)
      - Must be linked to parent
      - Cooldown between attempts
    """
    
    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        order_manager: Optional[OrderManager] = None,
        config: Optional[HedgeConfig] = None
    ):
        """
        Initialize hedge manager.
        
        Args:
            client: Binance client
            order_manager: Order manager for placing orders
            config: Hedge configuration
        """
        self.client = client or get_binance_client()
        self.order_manager = order_manager or OrderManager()
        self.config = config or get_settings().hedge
        
        # Active hedges (hedge_id -> HedgePosition)
        self._hedges: Dict[str, HedgePosition] = {}
        self._hedges_lock = Lock()
        
        # Cooldown tracking (parent_position_id -> last_hedge_time)
        self._cooldowns: Dict[str, float] = {}
        
        # Statistics
        self._total_hedges = 0
        self._successful_hedges = 0
        self._total_hedge_pnl = 0.0
        
        logger.info(f"HedgeManager initialized: trigger={self.config.trigger_pnl_usdt}, "
                   f"target={self.config.close_profit_usdt}")
    
    # ==================== Hedge Evaluation ====================
    
    def should_open_hedge(
        self,
        parent_position_id: str,
        parent_pnl_usdt: float,
        parent_side: str
    ) -> bool:
        """
        Check if hedge should be opened for a parent position.
        
        Args:
            parent_position_id: Parent position ID
            parent_pnl_usdt: Current parent P&L in USDT
            parent_side: Parent position side
            
        Returns:
            True if hedge should be opened
        """
        # Check trigger
        if parent_pnl_usdt > self.config.trigger_pnl_usdt:
            return False
        
        # Check if already has hedge
        if self._has_active_hedge(parent_position_id):
            return False
        
        # Check max hedges
        existing = self._count_hedges_for_parent(parent_position_id)
        if existing >= self.config.max_hedges_per_position:
            return False
        
        # Check cooldown
        if not self._check_cooldown(parent_position_id):
            return False
        
        logger.info(f"Hedge trigger: parent {parent_position_id} PnL={parent_pnl_usdt:.2f}")
        return True
    
    # ==================== Hedge Operations ====================
    
    def open_hedge(
        self,
        parent_position_id: str,
        symbol: str,
        parent_side: str,
        current_price: float,
        parent_position_size_usdt: Optional[float] = None
    ) -> Optional[HedgePosition]:
        """
        Open hedge (mirror) position.
        
        Args:
            parent_position_id: Parent position ID
            symbol: Trading symbol
            parent_side: Parent position side (hedge is opposite)
            current_price: Current market price
            
        Returns:
            HedgePosition if successful, None otherwise
        """
        # Hedge side is opposite to parent
        hedge_side = "SHORT" if parent_side == "LONG" else "LONG"
        order_side = OrderSide.SELL if hedge_side == "SHORT" else OrderSide.BUY
        
        # Calculate hedge size: adaptive (30-50% of parent) or fixed fallback
        if parent_position_size_usdt and parent_position_size_usdt > 0:
            # Use adaptive sizing: percentage of parent position size
            hedge_size_usdt = parent_position_size_usdt * self.config.adaptive_size_percent
            # Ensure minimum and maximum bounds (30-50% of parent)
            min_hedge_size = parent_position_size_usdt * 0.30
            max_hedge_size = parent_position_size_usdt * 0.50
            hedge_size_usdt = max(min_hedge_size, min(hedge_size_usdt, max_hedge_size))
        else:
            # Fallback to fixed size if parent size not provided
            hedge_size_usdt = self.config.fixed_size_usdt
        
        # Calculate quantity from hedge size
        quantity = hedge_size_usdt / current_price
        
        # Generate hedge ID
        hedge_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Opening HEDGE: {hedge_side} {quantity:.6f} {symbol} "
                   f"(parent: {parent_position_id})")
        
        try:
            # Place market entry
            result = self.order_manager.place_entry_order(
                symbol=symbol,
                side=hedge_side,
                quantity=quantity,
                position_id=hedge_id
            )
            
            if not result.success:
                logger.error(f"Hedge entry failed: {result.error_message}")
                return None
            
            # Create hedge position
            hedge = HedgePosition(
                hedge_id=hedge_id,
                parent_position_id=parent_position_id,
                symbol=symbol,
                side=hedge_side,
                entry_price=current_price,
                quantity=quantity,
                size_usdt=hedge_size_usdt
            )
            
            # Place SL for hedge (wider SL to avoid premature stop)
            sl_distance = current_price * 0.02  # 2% SL for hedge
            if hedge_side == "LONG":
                sl_price = current_price - sl_distance
            else:
                sl_price = current_price + sl_distance
            
            sl_result = self.order_manager.place_stop_loss(
                symbol=symbol,
                side=hedge_side,
                stop_price=sl_price,
                quantity=quantity,
                position_id=hedge_id,
                is_hedge=True,
                current_price=current_price
            )
            
            if sl_result.success:
                hedge.sl_order_id = sl_result.order.order_id
            
            # Track hedge
            with self._hedges_lock:
                self._hedges[hedge_id] = hedge
            
            # Update cooldown
            self._cooldowns[parent_position_id] = time.time()
            
            self._total_hedges += 1
            
            logger.info(f"HEDGE opened: {hedge_id} {hedge_side} @ {current_price}")
            
            return hedge
            
        except Exception as e:
            logger.error(f"Failed to open hedge: {e}")
            return None
    
    def check_hedge_close(
        self,
        hedge_id: str,
        current_price: float
    ) -> bool:
        """
        Check if hedge should be closed at profit target or loss threshold.

        Args:
            hedge_id: Hedge position ID
            current_price: Current market price

        Returns:
            True if hedge was closed due to loss, False if closed due to profit or still open
        """
        with self._hedges_lock:
            if hedge_id not in self._hedges:
                return False
            hedge = self._hedges[hedge_id]
        
        if not hedge.is_open:
            return False
        
        # NOTE: Do NOT sync hedge side from exchange when parent position exists
        # Binance shows NET positions (parent + hedge combined), not individual positions
        # When there's a hedge, the exchange position side is the NET side, not the hedge side
        # We trust our tracked hedge side (which is always opposite to parent)
        # The hedge side is set correctly when opened and should not change
        
        # Calculate P&L
        if hedge.side == "LONG":
            pnl = (current_price - hedge.entry_price) * hedge.quantity
        else:
            pnl = (hedge.entry_price - current_price) * hedge.quantity
        
        hedge.pnl_usdt = pnl
        
        # Check profit target - NO TP EXTENSION
        if pnl >= self.config.close_profit_usdt:
            self._close_hedge(hedge_id, current_price, "TARGET_REACHED")
            return False  # Closed due to profit, not loss

        # CRITICAL: If hedge becomes loss >= -7 USDT, close it immediately
        # This means the original position is likely profitable and hedge is no longer needed
        # When hedge closes due to loss, the original position should also close
        if pnl <= -7.0:
            logger.info(f"HEDGE {hedge_id}: Loss reached -7 USDT (PnL={pnl:.2f}), closing hedge and triggering parent position closure")
            if self._close_hedge(hedge_id, current_price, "HEDGE_LOSS_CLOSE"):
                # Trigger parent position closure
                self._trigger_parent_closure(hedge.parent_position_id, current_price)
                return True  # Closed due to loss

        return False
    
    def _trigger_parent_closure(self, parent_position_id: str, current_price: float) -> None:
        """
        Trigger closure of parent position when hedge closes due to loss.

        This is called when hedge reaches -7 USDT loss, indicating the parent
        position is likely profitable and should be closed to lock in profits.
        """
        logger.info(f"Triggering closure of parent position {parent_position_id} due to hedge loss closure")

        # This will be handled by position_tracker when it checks hedge status
        # The position_tracker will detect that hedge is closed and should close parent

    def _close_hedge(
        self,
        hedge_id: str,
        current_price: float,
        reason: str
    ) -> bool:
        """
        Close hedge position.
        
        Args:
            hedge_id: Hedge ID
            current_price: Current price
            reason: Close reason
            
        Returns:
            True if closed successfully
        """
        with self._hedges_lock:
            if hedge_id not in self._hedges:
                return False
            hedge = self._hedges[hedge_id]
        
        if not hedge.is_open:
            return False
        
        try:
            # Check real position on exchange before closing
            # Binance shows NET position (original + hedge combined)
            exchange_positions = self.client.get_positions(hedge.symbol)
            exchange_pos = exchange_positions[0] if exchange_positions else None
            
            # If NET position is zero or very small, hedge is already effectively closed
            # (NET = 0 means original and hedge cancel each other out)
            if exchange_pos:
                net_size = abs(exchange_pos.size)
                # If NET position is very small (< 1% of hedge size), consider it closed
                if net_size < hedge.quantity * 0.01:
                    logger.info(f"HEDGE {hedge_id} already closed (NET position ≈ 0), marking as closed")
                    # Cancel any remaining orders
                    self.order_manager.cancel_all_for_position(hedge.symbol, hedge_id)
                    # Mark as closed without placing order
                    hedge.is_open = False
                    hedge.closed_at = time.time()
                    # Calculate final P&L
                    if hedge.side == "LONG":
                        final_pnl = (current_price - hedge.entry_price) * hedge.quantity
                    else:
                        final_pnl = (hedge.entry_price - current_price) * hedge.quantity
                    hedge.pnl_usdt = final_pnl
                    self._total_hedge_pnl += final_pnl
                    if final_pnl > 0:
                        self._successful_hedges += 1
                    logger.info(f"HEDGE closed: {hedge_id} | PnL={final_pnl:.2f} | Reason={reason} (NET=0)")
                    return True
            
            # Place market close order
            close_side = OrderSide.SELL if hedge.side == "LONG" else OrderSide.BUY
            
            order = self.client.place_market_order(
                symbol=hedge.symbol,
                side=close_side,
                quantity=hedge.quantity,
                reduce_only=True
            )
            
            # Cancel any remaining orders
            self.order_manager.cancel_all_for_position(hedge.symbol, hedge_id)
            
            # Update status
            hedge.is_open = False
            hedge.closed_at = time.time()
            
            # Calculate final P&L
            if hedge.side == "LONG":
                final_pnl = (current_price - hedge.entry_price) * hedge.quantity
            else:
                final_pnl = (hedge.entry_price - current_price) * hedge.quantity
            
            hedge.pnl_usdt = final_pnl
            self._total_hedge_pnl += final_pnl
            
            if final_pnl > 0:
                self._successful_hedges += 1
            
            logger.info(f"HEDGE closed: {hedge_id} | PnL={final_pnl:.2f} | Reason={reason}")
            
            return True
            
        except Exception as e:
            error_str = str(e)
            # Handle -2022 error: ReduceOnly Order is rejected
            # This happens when NET position doesn't allow closing hedge separately
            if "-2022" in error_str or "ReduceOnly" in error_str:
                logger.warning(f"Hedge {hedge_id} cannot be closed separately (NET position), marking as closed")
                # Cancel any remaining orders
                self.order_manager.cancel_all_for_position(hedge.symbol, hedge_id)
                # Mark as closed without placing order (hedge is effectively closed via NET position)
                hedge.is_open = False
                hedge.closed_at = time.time()
                # PnL is already calculated in check_hedge_close, use it
                final_pnl = hedge.pnl_usdt
                self._total_hedge_pnl += final_pnl
                if final_pnl > 0:
                    self._successful_hedges += 1
                logger.info(f"HEDGE closed: {hedge_id} | PnL={final_pnl:.2f} | Reason={reason} (NET)")
                return True
            # Handle -4131 error: PERCENT_PRICE filter violation
            # This happens when market price is too far from mark price
            elif "-4131" in error_str or "PERCENT_PRICE" in error_str:
                logger.warning(f"Hedge {hedge_id} cannot be closed via MARKET order (PERCENT_PRICE filter), trying STOP_MARKET with closePosition")
                # Try using STOP_MARKET with closePosition=True (closes entire position)
                # This is a workaround for PERCENT_PRICE filter
                try:
                    # Get mark price from exchange
                    exchange_positions = self.client.get_positions(hedge.symbol)
                    exchange_pos = exchange_positions[0] if exchange_positions else None
                    if not exchange_pos:
                        logger.error(f"Cannot get mark price for {hedge.symbol}")
                        return False
                    
                    mark_price = exchange_pos.mark_price
                    # Set trigger price very close to mark price (0.01% away)
                    if hedge.side == "LONG":
                        trigger_price = mark_price * 0.9999  # Just below mark for LONG
                    else:  # SHORT
                        trigger_price = mark_price * 1.0001  # Just above mark for SHORT
                    
                    # Place STOP_MARKET order with closePosition=True
                    # OrderSide is already imported at the top of the file
                    close_side = OrderSide.SELL if hedge.side == "LONG" else OrderSide.BUY
                    
                    order = self.client.place_stop_market(
                        symbol=hedge.symbol,
                        side=close_side,
                        stop_price=trigger_price,
                        close_position=True  # This closes the entire NET position
                    )
                    
                    logger.info(f"HEDGE {hedge_id} closed via STOP_MARKET with closePosition=True | Trigger: {trigger_price:.8f}")
                    
                    # Cancel any remaining orders
                    self.order_manager.cancel_all_for_position(hedge.symbol, hedge_id)
                    
                    # Mark as closed
                    hedge.is_open = False
                    hedge.closed_at = time.time()
                    final_pnl = hedge.pnl_usdt
                    self._total_hedge_pnl += final_pnl
                    if final_pnl > 0:
                        self._successful_hedges += 1
                    logger.info(f"HEDGE closed: {hedge_id} | PnL={final_pnl:.2f} | Reason={reason} (STOP_MARKET)")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to close hedge {hedge_id} via STOP_MARKET: {e2}")
                    # Mark as closed anyway to prevent retries
                    hedge.is_open = False
                    hedge.closed_at = time.time()
                    return False
            else:
                logger.error(f"Failed to close hedge {hedge_id}: {e}")
                return False
    
    def close_hedge_for_parent(
        self,
        parent_position_id: str,
        current_price: float
    ) -> int:
        """
        Close all hedges for a parent position.
        
        Called when parent position is closed.
        
        Args:
            parent_position_id: Parent position ID
            current_price: Current price for P&L calculation
            
        Returns:
            Number of hedges closed
        """
        closed = 0
        
        with self._hedges_lock:
            hedge_ids = [
                h.hedge_id for h in self._hedges.values()
                if h.parent_position_id == parent_position_id and h.is_open
            ]
        
        for hedge_id in hedge_ids:
            if self._close_hedge(hedge_id, current_price, "PARENT_CLOSED"):
                closed += 1
        
        return closed
    
    # ==================== Queries ====================
    
    def get_hedge_for_parent(self, parent_position_id: str) -> Optional[HedgePosition]:
        """Get active hedge for a parent position."""
        with self._hedges_lock:
            for hedge in self._hedges.values():
                if hedge.parent_position_id == parent_position_id and hedge.is_open:
                    return hedge
            return None
    
    def get_all_active_hedges(self) -> List[HedgePosition]:
        """Get all active hedges."""
        with self._hedges_lock:
            return [h for h in self._hedges.values() if h.is_open]
    
    def _has_active_hedge(self, parent_position_id: str) -> bool:
        """Check if parent has active hedge."""
        with self._hedges_lock:
            return any(
                h.parent_position_id == parent_position_id and h.is_open
                for h in self._hedges.values()
            )
    
    def _count_hedges_for_parent(self, parent_position_id: str) -> int:
        """Count total hedges (including closed) for parent."""
        with self._hedges_lock:
            return sum(
                1 for h in self._hedges.values()
                if h.parent_position_id == parent_position_id
            )
    
    def _check_cooldown(self, parent_position_id: str) -> bool:
        """Check if cooldown has passed."""
        last_hedge = self._cooldowns.get(parent_position_id, 0)
        return time.time() - last_hedge >= self.config.hedge_cooldown_sec
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """Get hedge statistics."""
        return {
            "total_hedges": self._total_hedges,
            "successful_hedges": self._successful_hedges,
            "success_rate": self._successful_hedges / max(1, self._total_hedges),
            "total_hedge_pnl": self._total_hedge_pnl,
            "avg_hedge_pnl": self._total_hedge_pnl / max(1, self._total_hedges),
            "active_hedges": len(self.get_all_active_hedges())
        }
