"""
Order Manager

Handles all order operations:
- Entry with MARKET orders
- Stop Loss with STOP_MARKET
- Take Profit with TAKE_PROFIT_MARKET
- Order cancellation and replacement
- Orphan order cleanup
- Idempotent operations

CRITICAL RULES:
- Only use MARKET, STOP_MARKET, TAKE_PROFIT_MARKET
- Cancel before replace
- Never leave orphan orders
- Idempotent operations
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

from exchange.binance_client import (
    BinanceClient, get_binance_client,
    Order, OrderSide
)
from config.settings import get_settings, OrderConfig


logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


@dataclass
class ManagedOrder:
    """
    Tracked order with management metadata.
    """
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    stop_price: float
    
    # Management metadata
    purpose: str  # "ENTRY", "SL", "TP", "HEDGE_ENTRY", "HEDGE_SL", "HEDGE_TP"
    parent_position_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Status
    is_active: bool = True


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order: Optional[ManagedOrder] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class OrderManager:
    """
    Order Management System.
    
    Handles all order operations with safety guarantees:
    - Idempotent operations
    - Cancel-before-replace
    - Orphan cleanup
    - Order tracking
    """
    
    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        config: Optional[OrderConfig] = None
    ):
        """
        Initialize order manager.
        
        Args:
            client: Binance client
            config: Order configuration
        """
        self.client = client or get_binance_client()
        self.config = config or get_settings().order
        
        # Active orders tracking (symbol -> order_id -> ManagedOrder)
        self._orders: Dict[str, Dict[int, ManagedOrder]] = {}
        self._orders_lock = Lock()
        
        # Order operation history for idempotency
        self._operation_log: Dict[str, float] = {}  # operation_key -> timestamp
        
        logger.info("OrderManager initialized")
    
    # ==================== Entry Orders ====================
    
    def place_entry_order(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        quantity: float,
        position_id: str
    ) -> OrderResult:
        """
        Place MARKET entry order.
        
        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            quantity: Order quantity
            position_id: Associated position ID
            
        Returns:
            OrderResult with execution details
        """
        order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL
        
        # CRITICAL: Check if there's an existing position in opposite direction
        # If there is, the new order might close it instead of opening new position
        try:
            existing_positions = self.client.get_positions(symbol)
            if existing_positions:
                existing_pos = existing_positions[0]
                existing_side = "LONG" if existing_pos.size > 0 else "SHORT"
                if existing_side != side:
                    logger.warning(f"[ENTRY] {symbol}: Existing {existing_side} position detected ({abs(existing_pos.size):.6f}) "
                                 f"before opening {side}. New order may close existing position instead of opening new one!")
        except Exception as e:
            logger.debug(f"[ENTRY] {symbol}: Could not check existing positions: {e}")
        
        logger.info(f"Placing ENTRY: {side} {quantity} {symbol} (order_side={order_side.value})")
        
        try:
            order = self.client.place_market_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity
            )
            
            # Validate order_id from API response
            if not order.order_id or order.order_id <= 0:
                error_msg = f"Invalid order_id {order.order_id} returned from API"
                logger.error(f"Entry order failed: {error_msg}")
                return OrderResult(
                    success=False,
                    error_code="INVALID_ORDER_ID",
                    error_message=error_msg
                )
            
            managed = ManagedOrder(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET.value,
                quantity=quantity,
                stop_price=0,
                purpose="ENTRY",
                parent_position_id=position_id
            )
            
            self._track_order(managed)
            
            logger.info(f"ENTRY filled: {order.order_id} @ {order.price}")
            
            return OrderResult(success=True, order=managed)
            
        except Exception as e:
            error_str = str(e)
            # Don't log validation errors as ERROR
            if "-4131" in error_str or "-4005" in error_str or "-2011" in error_str:
                logger.debug(f"Entry order validation failed: {error_str}")
            else:
                logger.warning(f"Entry order failed: {error_str}")
            return OrderResult(
                success=False,
                error_code="ENTRY_FAILED",
                error_message=error_str
            )
    
    # ==================== Stop Loss Orders ====================
    
    def place_stop_loss(
        self,
        symbol: str,
        side: str,  # Position side (SL will be opposite)
        stop_price: float,
        quantity: float,
        position_id: str,
        is_hedge: bool = False,
        current_price: Optional[float] = None,
        is_breakeven: bool = False
    ) -> OrderResult:
        """
        Place STOP_MARKET order for Stop Loss.
        
        CRITICAL: Cancels all existing SL orders for this position before placing new one.
        This prevents duplicate orders when get_open_orders fails.
        
        Args:
            symbol: Trading symbol
            side: Position side (SL will close this side)
            stop_price: SL trigger price
            quantity: Order quantity
            position_id: Associated position ID
            is_hedge: Whether this is for a hedge position
            current_price: Current market price for validation
            
        Returns:
            OrderResult
        """
        # SL order is opposite to position side
        order_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY
        purpose = "HEDGE_SL" if is_hedge else "SL"
        
        # CRITICAL: Cancel all existing SL orders for THIS SPECIFIC POSITION before placing new one
        # This prevents duplicate orders when get_open_orders fails
        # IMPORTANT: Only cancel orders for this position_id, not all orders for symbol
        # (because hedge positions on same symbol have different position_id)
        try:
            exchange_orders = self.client.get_open_orders(symbol)
            # First, try to find and cancel from exchange
            for order in exchange_orders:
                if (order.order_type == OrderType.STOP_MARKET.value and
                    order.side == order_side.value):
                    # Check if this order belongs to our position by checking internal tracker
                    # (we can't know position_id from exchange order, so check tracker)
                    with self._orders_lock:
                        if (symbol in self._orders and
                            order.order_id in self._orders[symbol] and
                            self._orders[symbol][order.order_id].parent_position_id == position_id):
                            # Found existing SL order for this position - cancel it
                            if order.order_id and order.order_id > 0:
                                logger.warning(f"Cancelling existing SL order {order.order_id} for position {position_id} before placing new one")
                                self._cancel_order(symbol, order.order_id)
        except Exception as e:
            # If get_open_orders fails, try to cancel from internal tracker
            logger.warning(f"Failed to get open orders for {symbol}, trying to cancel from tracker: {e}")
            with self._orders_lock:
                if symbol in self._orders:
                    for order_id, tracked_order in list(self._orders[symbol].items()):
                        if (tracked_order.is_active and
                            tracked_order.purpose == purpose and
                            tracked_order.parent_position_id == position_id):
                            logger.warning(f"Cancelling tracked SL order {order_id} before placing new one for {symbol}")
                            self._cancel_order(symbol, order_id)
        
        # Validate SL price if current_price provided
        if current_price is not None:
            # For breakeven SL, use smaller minimum distance (0.01% instead of 0.05%)
            # and allow SL to be closer to current price to protect profits
            min_distance_multiplier = 0.0001 if is_breakeven else 0.0005
            
            if side == "LONG":
                # SL must be below current price for LONG
                # For breakeven SL, allow SL to be very close to current price
                if not is_breakeven and stop_price >= current_price:
                    logger.warning(f"SL price {stop_price} >= current {current_price} for LONG, skipping")
                    return OrderResult(
                        success=False,
                        error_code="SL_TOO_CLOSE",
                        error_message=f"SL price {stop_price} >= current price {current_price}"
                    )
                # Check minimum distance (reduced for breakeven SL)
                min_distance = current_price * min_distance_multiplier
                if current_price - stop_price < min_distance:
                    if is_breakeven:
                        # For breakeven, try to set SL at minimum distance
                        stop_price = current_price - min_distance
                        logger.debug(f"Adjusting breakeven SL to minimum distance: {stop_price:.8f}")
                    else:
                        logger.warning(f"SL price too close to current price, skipping")
                        return OrderResult(
                            success=False,
                            error_code="SL_TOO_CLOSE",
                            error_message=f"SL too close to current price"
                        )
            else:  # SHORT
                # SL must be above current price for SHORT
                # For breakeven SL, allow SL to be very close to current price
                if not is_breakeven and stop_price <= current_price:
                    logger.warning(f"SL price {stop_price} <= current {current_price} for SHORT, skipping")
                    return OrderResult(
                        success=False,
                        error_code="SL_TOO_CLOSE",
                        error_message=f"SL price {stop_price} <= current price {current_price}"
                    )
                # Check minimum distance (reduced for breakeven SL)
                min_distance = current_price * min_distance_multiplier
                if stop_price - current_price < min_distance:
                    if is_breakeven:
                        # For breakeven, try to set SL at minimum distance
                        stop_price = current_price + min_distance
                        logger.debug(f"Adjusting breakeven SL to minimum distance: {stop_price:.8f}")
                    else:
                        logger.warning(f"SL price too close to current price, skipping")
                        return OrderResult(
                            success=False,
                            error_code="SL_TOO_CLOSE",
                            error_message=f"SL too close to current price"
                        )
        
        distance_info = ""
        if current_price is not None:
            if side == "LONG":
                distance = current_price - stop_price
                distance_pct = (distance / current_price) * 100
            else:  # SHORT
                distance = stop_price - current_price
                distance_pct = (distance / current_price) * 100
            distance_info = f" | Current: {current_price:.8f} | Distance: {distance:.8f} ({distance_pct:.2f}%)"
        
        # Check if quantity exceeds max_qty before placing order
        # If it does, use closePosition=true instead
        use_close_position = False
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            max_qty = symbol_info.get("max_qty", 1e15)
            if quantity > max_qty:
                logger.warning(f"Quantity {quantity} exceeds max_qty {max_qty} for {symbol}. Using closePosition=true instead.")
                use_close_position = True
        except Exception as e:
            logger.debug(f"Could not check max_qty for {symbol}: {e}, proceeding with quantity")
        
        logger.info(f"Placing SL: {order_side.value} {symbol} @ {stop_price} | Qty: {quantity:.2f}{distance_info} | closePosition={use_close_position}")
        
        try:
            if use_close_position:
                order = self.client.place_stop_market(
                    symbol=symbol,
                    side=order_side,
                    stop_price=stop_price,
                    close_position=True
                )
            else:
                order = self.client.place_stop_market(
                    symbol=symbol,
                    side=order_side,
                    stop_price=stop_price,
                    quantity=quantity
                )
            
            # Validate order_id from API response
            if not order.order_id or order.order_id <= 0:
                error_msg = f"Invalid order_id {order.order_id} returned from API for SL order"
                logger.error(f"SL order failed: {error_msg}")
                return OrderResult(
                    success=False,
                    error_code="INVALID_ORDER_ID",
                    error_message=error_msg
                )
            
            managed = ManagedOrder(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                symbol=symbol,
                side=order_side.value,
                order_type=OrderType.STOP_MARKET.value,
                quantity=quantity,
                stop_price=stop_price,
                purpose=purpose,
                parent_position_id=position_id
            )
            
            self._track_order(managed)
            
            logger.info(f"✅ SL placed successfully: OrderID={order.order_id} | {symbol} | {order_side.value} @ {stop_price}{distance_info}")
            
            return OrderResult(success=True, order=managed)
            
        except Exception as e:
            error_str = str(e)
            # -2021: Order would immediately trigger - treat as warning
            if "-2021" in error_str:
                logger.warning(f"SL order would immediately trigger: {stop_price}")
                return OrderResult(
                    success=False,
                    error_code="SL_IMMEDIATE_TRIGGER",
                    error_message="SL price too close to current price"
                )
            # -2019: Margin is insufficient - cannot update SL/TP
            if "-2019" in error_str or "Margin is insufficient" in error_str:
                logger.warning(f"SL order failed: Margin is insufficient. Cannot update SL for {symbol}")
                return OrderResult(
                    success=False,
                    error_code="MARGIN_INSUFFICIENT",
                    error_message="Margin is insufficient - cannot update SL"
                )
            # -2027: Exceeded maximum position at current leverage - cannot update SL/TP
            if "-2027" in error_str or "Exceeded the maximum allowable position" in error_str:
                logger.warning(f"SL order failed: Exceeded maximum position at current leverage for {symbol}")
                return OrderResult(
                    success=False,
                    error_code="MAX_POSITION_EXCEEDED",
                    error_message="Exceeded maximum position at current leverage - cannot update SL"
                )
            # -4045: Reach max stop order limit - need to clean up old orders
            if "-4045" in error_str or "Reach max stop order limit" in error_str:
                logger.warning(f"SL order failed: Max stop order limit reached for {symbol}. Attempting to clean up old orders...")
                # Try to cancel all stop orders for this symbol to free up space
                cleanup_count = self._cleanup_all_stop_orders(symbol)
                logger.warning(f"Cleaned up {cleanup_count} stop orders for {symbol} due to max limit. Retry may be needed.")
                return OrderResult(
                    success=False,
                    error_code="MAX_STOP_ORDERS_REACHED",
                    error_message=f"Max stop order limit reached. Cleaned up {cleanup_count} old orders. Retry may be needed."
                )
            # -4005: Quantity greater than max quantity - use closePosition=true instead
            if "-4005" in error_str or "Quantity greater than max quantity" in error_str:
                logger.warning(f"SL order failed: Quantity {quantity} exceeds max_qty for {symbol}. Retrying with closePosition=true...")
                # Retry with closePosition=true instead of quantity
                try:
                    order = self.client.place_stop_market(
                        symbol=symbol,
                        side=order_side,
                        stop_price=stop_price,
                        close_position=True
                    )
                    
                    # Validate order_id from API response
                    if not order.order_id or order.order_id <= 0:
                        error_msg = f"Invalid order_id {order.order_id} returned from API for SL order (closePosition)"
                        logger.error(f"SL order failed: {error_msg}")
                        return OrderResult(
                            success=False,
                            error_code="INVALID_ORDER_ID",
                            error_message=error_msg
                        )
                    
                    managed = ManagedOrder(
                        order_id=order.order_id,
                        client_order_id=order.client_order_id,
                        symbol=symbol,
                        side=order_side.value,
                        order_type=OrderType.STOP_MARKET.value,
                        quantity=quantity,  # Keep original quantity for tracking
                        stop_price=stop_price,
                        purpose=purpose,
                        parent_position_id=position_id
                    )
                    
                    self._track_order(managed)
                    
                    logger.info(f"✅ SL placed successfully with closePosition=true: OrderID={order.order_id} | {symbol} | {order_side.value} @ {stop_price}{distance_info}")
                    
                    return OrderResult(success=True, order=managed)
                except Exception as retry_error:
                    logger.error(f"SL order failed even with closePosition=true: {retry_error}")
                    return OrderResult(
                        success=False,
                        error_code="SL_QUANTITY_EXCEEDED",
                        error_message=f"Quantity exceeds max_qty and closePosition=true also failed: {str(retry_error)}"
                    )
            logger.error(f"SL order failed: {e}")
            return OrderResult(
                success=False,
                error_code="SL_FAILED",
                error_message=str(e)
            )
    
    def update_stop_loss(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        quantity: float,
        position_side: str,  # "LONG" or "SHORT" - REQUIRED
        is_hedge: bool = False,
        current_price: Optional[float] = None,
        is_breakeven: bool = False
    ) -> OrderResult:
        """
        Update Stop Loss (cancel + replace).
        
        PRIMARY SOURCE OF TRUTH: Exchange API - checks real orders on exchange.
        
        Args:
            symbol: Trading symbol
            position_id: Position ID
            new_stop_price: New SL price
            quantity: Order quantity
            position_side: Position side ("LONG" or "SHORT") - REQUIRED
            is_hedge: Whether this is for a hedge position
            current_price: Current market price for validation
            
        Returns:
            OrderResult
        """
        purpose = "HEDGE_SL" if is_hedge else "SL"
        
        # Get real orders from exchange (source of truth)
        exchange_orders = self.client.get_open_orders(symbol)
        
        # Find existing SL order on exchange by type
        # SL order type is STOP_MARKET
        # For LONG position: SL order side is SELL
        # For SHORT position: SL order side is BUY
        expected_sl_side = OrderSide.SELL if position_side == "LONG" else OrderSide.BUY
        existing_sl_order = None
        
        for order in exchange_orders:
            if (order.order_type == OrderType.STOP_MARKET.value and
                order.side == expected_sl_side.value):
                existing_sl_order = order
                break
        
        # Fallback: If get_open_orders returned empty (library bug), check internal tracker
        # This prevents creating duplicate orders when API fails
        if not existing_sl_order and len(exchange_orders) == 0:
            existing_tracked = self._find_order_by_purpose(symbol, position_id, purpose)
            if existing_tracked and existing_tracked.order_id and existing_tracked.order_id > 0:
                logger.warning(f"get_open_orders returned empty (library bug), but found SL in tracker (order_id={existing_tracked.order_id}). "
                             f"Attempting to cancel before creating new order.")
                # Try to cancel the tracked order before creating new one
                cancelled = self._cancel_order(symbol, existing_tracked.order_id)
                if cancelled:
                    logger.debug(f"Successfully cancelled tracked SL order {existing_tracked.order_id} before creating new one")
                else:
                    logger.warning(f"Failed to cancel tracked SL order {existing_tracked.order_id}, but proceeding with new order creation")
        
        # If found on exchange, check if update needed
        if existing_sl_order:
            # Idempotency check - don't update if price is same
            if abs(existing_sl_order.stop_price - new_stop_price) < 0.01:
                # Update tracking and return existing
                existing_tracked = self._find_order_by_purpose(symbol, position_id, purpose)
                if existing_tracked:
                    return OrderResult(success=True, order=existing_tracked)
            
            # Cancel existing SL order from exchange (source of truth)
            if existing_sl_order.order_id and existing_sl_order.order_id > 0:
                cancelled = self._cancel_order(symbol, existing_sl_order.order_id)
                if not cancelled:
                    logger.warning(f"Failed to cancel existing SL {existing_sl_order.order_id} from exchange")
                else:
                    # Mark as inactive in tracking if exists
                    with self._orders_lock:
                        if (symbol in self._orders and 
                            existing_sl_order.order_id in self._orders[symbol]):
                            self._orders[symbol][existing_sl_order.order_id].is_active = False
        
        return self.place_stop_loss(
            symbol=symbol,
            side=position_side,
            stop_price=new_stop_price,
            quantity=quantity,
            position_id=position_id,
            is_hedge=is_hedge,
            current_price=current_price,
            is_breakeven=is_breakeven
        )
    
    # ==================== Take Profit Orders ====================
    
    def place_take_profit(
        self,
        symbol: str,
        side: str,  # Position side (TP will be opposite)
        stop_price: float,
        quantity: float,
        position_id: str,
        is_hedge: bool = False,
        current_price: Optional[float] = None
    ) -> OrderResult:
        """
        Place TAKE_PROFIT_MARKET order.
        
        Args:
            symbol: Trading symbol
            side: Position side (TP will close this side)
            stop_price: TP trigger price
            quantity: Order quantity
            position_id: Associated position ID
            is_hedge: Whether this is for a hedge position
            current_price: Current market price for validation
            
        Returns:
            OrderResult
        """
        # TP order is opposite to position side
        order_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY
        purpose = "HEDGE_TP" if is_hedge else "TP"
        
        # CRITICAL: Cancel all existing TP orders for THIS SPECIFIC POSITION before placing new one
        # This prevents duplicate orders when get_open_orders returns empty (library bug)
        # IMPORTANT: Only cancel orders for this position_id, not all orders for symbol
        # (because hedge positions on same symbol have different position_id)
        try:
            exchange_orders = self.client.get_open_orders(symbol)
            # First, try to find and cancel from exchange
            for order in exchange_orders:
                if (order.order_type == OrderType.TAKE_PROFIT_MARKET.value and
                    order.side == order_side.value):
                    # Check if this order belongs to our position by checking internal tracker
                    # (we can't know position_id from exchange order, so check tracker)
                    with self._orders_lock:
                        if (symbol in self._orders and
                            order.order_id in self._orders[symbol] and
                            self._orders[symbol][order.order_id].parent_position_id == position_id):
                            # Found existing TP order for this position - cancel it
                            if order.order_id and order.order_id > 0:
                                logger.warning(f"Cancelling existing TP order {order.order_id} for position {position_id} before placing new one")
                                self._cancel_order(symbol, order.order_id)
        except Exception as e:
            # If get_open_orders fails, try to cancel from internal tracker
            logger.warning(f"Failed to get open orders for {symbol}, trying to cancel from tracker: {e}")
            with self._orders_lock:
                if symbol in self._orders:
                    for order_id, tracked_order in list(self._orders[symbol].items()):
                        if (tracked_order.is_active and
                            tracked_order.purpose == purpose and
                            tracked_order.parent_position_id == position_id):
                            logger.warning(f"Cancelling tracked TP order {order_id} for position {position_id} before placing new one")
                            self._cancel_order(symbol, order_id)
        
        distance_info = ""
        if current_price is not None:
            if side == "LONG":
                distance = stop_price - current_price
            else:  # SHORT
                distance = current_price - stop_price
            distance_pct = (distance / current_price) * 100 if current_price > 0 else 0
            distance_info = f" | Current: {current_price:.8f} | Distance: {distance:.8f} ({distance_pct:.2f}%)"
        
        # Validate TP price if current_price provided
        if current_price is not None:
            if side == "LONG":
                # TP must be above current price for LONG
                if stop_price <= current_price:
                    logger.warning(f"TP price {stop_price} <= current {current_price} for LONG, skipping")
                    return OrderResult(
                        success=False,
                        error_code="TP_TOO_CLOSE",
                        error_message=f"TP price {stop_price} <= current price {current_price}"
                    )
                # Check minimum distance (0.1% to avoid immediate trigger)
                min_distance = current_price * 0.001
                if stop_price - current_price < min_distance:
                    logger.warning(f"TP price too close to current price, skipping")
                    return OrderResult(
                        success=False,
                        error_code="TP_TOO_CLOSE",
                        error_message=f"TP too close to current price"
                    )
            else:  # SHORT
                # TP must be below current price for SHORT
                if stop_price >= current_price:
                    logger.warning(f"TP price {stop_price} >= current {current_price} for SHORT, skipping")
                    return OrderResult(
                        success=False,
                        error_code="TP_TOO_CLOSE",
                        error_message=f"TP price {stop_price} >= current price {current_price}"
                    )
                # Check minimum distance
                min_distance = current_price * 0.001
                if current_price - stop_price < min_distance:
                    logger.warning(f"TP price too close to current price, skipping")
                    return OrderResult(
                        success=False,
                        error_code="TP_TOO_CLOSE",
                        error_message=f"TP too close to current price"
                    )
        
        # Check if quantity exceeds max_qty before placing order
        # If it does, use closePosition=true instead
        use_close_position = False
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            max_qty = symbol_info.get("max_qty", 1e15)
            if quantity > max_qty:
                logger.warning(f"Quantity {quantity} exceeds max_qty {max_qty} for {symbol}. Using closePosition=true instead.")
                use_close_position = True
        except Exception as e:
            logger.debug(f"Could not check max_qty for {symbol}: {e}, proceeding with quantity")
        
        logger.info(f"Placing TP: {order_side.value} {symbol} @ {stop_price} | Qty: {quantity:.2f}{distance_info} | closePosition={use_close_position}")
        
        try:
            if use_close_position:
                order = self.client.place_take_profit_market(
                    symbol=symbol,
                    side=order_side,
                    stop_price=stop_price,
                    close_position=True
                )
            else:
                order = self.client.place_take_profit_market(
                    symbol=symbol,
                    side=order_side,
                    stop_price=stop_price,
                    quantity=quantity
                )
            
            # Validate order_id from API response
            if not order.order_id or order.order_id <= 0:
                error_msg = f"Invalid order_id {order.order_id} returned from API for TP order"
                logger.error(f"TP order failed: {error_msg}")
                return OrderResult(
                    success=False,
                    error_code="INVALID_ORDER_ID",
                    error_message=error_msg
                )
            
            managed = ManagedOrder(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                symbol=symbol,
                side=order_side.value,
                order_type=OrderType.TAKE_PROFIT_MARKET.value,
                quantity=quantity,
                stop_price=stop_price,
                purpose=purpose,
                parent_position_id=position_id
            )
            
            self._track_order(managed)
            
            logger.info(f"✅ TP placed successfully: OrderID={order.order_id} | {symbol} | {order_side.value} @ {stop_price}{distance_info}")
            
            return OrderResult(success=True, order=managed)
            
        except Exception as e:
            error_str = str(e)
            # -2021: Order would immediately trigger - treat as warning
            if "-2021" in error_str:
                logger.warning(f"TP order would immediately trigger: {stop_price}")
                return OrderResult(
                    success=False,
                    error_code="TP_IMMEDIATE_TRIGGER",
                    error_message="TP price too close to current price"
                )
            # -2019: Margin is insufficient - cannot update SL/TP
            if "-2019" in error_str or "Margin is insufficient" in error_str:
                logger.warning(f"TP order failed: Margin is insufficient. Cannot update TP for {symbol}")
                return OrderResult(
                    success=False,
                    error_code="MARGIN_INSUFFICIENT",
                    error_message="Margin is insufficient - cannot update TP"
                )
            # -2027: Exceeded maximum position at current leverage - cannot update SL/TP
            if "-2027" in error_str or "Exceeded the maximum allowable position" in error_str:
                logger.warning(f"TP order failed: Exceeded maximum position at current leverage for {symbol}")
                return OrderResult(
                    success=False,
                    error_code="MAX_POSITION_EXCEEDED",
                    error_message="Exceeded maximum position at current leverage - cannot update TP"
                )
            # -4045: Reach max stop order limit - need to clean up old orders
            if "-4045" in error_str or "Reach max stop order limit" in error_str:
                logger.warning(f"TP order failed: Max stop order limit reached for {symbol}. Attempting to clean up old orders...")
                # Try to cancel all stop orders for this symbol to free up space
                cleanup_count = self._cleanup_all_stop_orders(symbol)
                logger.warning(f"Cleaned up {cleanup_count} stop orders for {symbol} due to max limit. Retry may be needed.")
                return OrderResult(
                    success=False,
                    error_code="MAX_STOP_ORDERS_REACHED",
                    error_message=f"Max stop order limit reached. Cleaned up {cleanup_count} old orders. Retry may be needed."
                )
            # -4005: Quantity greater than max quantity - use closePosition=true instead
            if "-4005" in error_str or "Quantity greater than max quantity" in error_str:
                logger.warning(f"TP order failed: Quantity {quantity} exceeds max_qty for {symbol}. Retrying with closePosition=true...")
                # Retry with closePosition=true instead of quantity
                try:
                    order = self.client.place_take_profit_market(
                        symbol=symbol,
                        side=order_side,
                        stop_price=stop_price,
                        close_position=True
                    )
                    
                    # Validate order_id from API response
                    if not order.order_id or order.order_id <= 0:
                        error_msg = f"Invalid order_id {order.order_id} returned from API for TP order (closePosition)"
                        logger.error(f"TP order failed: {error_msg}")
                        return OrderResult(
                            success=False,
                            error_code="INVALID_ORDER_ID",
                            error_message=error_msg
                        )
                    
                    managed = ManagedOrder(
                        order_id=order.order_id,
                        client_order_id=order.client_order_id,
                        symbol=symbol,
                        side=order_side.value,
                        order_type=OrderType.TAKE_PROFIT_MARKET.value,
                        quantity=quantity,  # Keep original quantity for tracking
                        stop_price=stop_price,
                        purpose=purpose,
                        parent_position_id=position_id
                    )
                    
                    self._track_order(managed)
                    
                    logger.info(f"✅ TP placed successfully with closePosition=true: OrderID={order.order_id} | {symbol} | {order_side.value} @ {stop_price}{distance_info}")
                    
                    return OrderResult(success=True, order=managed)
                except Exception as retry_error:
                    logger.error(f"TP order failed even with closePosition=true: {retry_error}")
                    return OrderResult(
                        success=False,
                        error_code="TP_QUANTITY_EXCEEDED",
                        error_message=f"Quantity exceeds max_qty and closePosition=true also failed: {str(retry_error)}"
                    )
            logger.error(f"TP order failed: {e}")
            return OrderResult(
                success=False,
                error_code="TP_FAILED",
                error_message=str(e)
            )
    
    def update_take_profit(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        quantity: float,
        position_side: Optional[str] = None,  # "LONG" or "SHORT" - if None, will try to determine
        is_hedge: bool = False,
        current_price: Optional[float] = None
    ) -> OrderResult:
        """
        Update Take Profit (cancel + replace).
        
        PRIMARY SOURCE OF TRUTH: Exchange API - checks real orders on exchange.
        
        Args:
            symbol: Trading symbol
            position_id: Position ID
            new_stop_price: New TP price
            quantity: Order quantity
            position_side: Position side ("LONG" or "SHORT") - if None, will try to determine from tracked order
            is_hedge: Whether this is for a hedge position
            current_price: Current market price for validation
            
        Returns:
            OrderResult
        """
        purpose = "HEDGE_TP" if is_hedge else "TP"
        
        # Get real orders from exchange (source of truth)
        exchange_orders = self.client.get_open_orders(symbol)
        
        # Find existing TP order on exchange by type
        # TP order type is TAKE_PROFIT_MARKET
        # For LONG position: TP order side is SELL
        # For SHORT position: TP order side is BUY
        existing_tp_order = None
        
        # Try to determine position side if not provided
        if position_side is None:
            existing_tracked = self._find_order_by_purpose(symbol, position_id, purpose)
            if existing_tracked:
                position_side = "SHORT" if existing_tracked.side == "BUY" else "LONG"
            else:
                position_side = "LONG"  # Default fallback
        
        expected_tp_side = OrderSide.SELL if position_side == "LONG" else OrderSide.BUY
        
        for order in exchange_orders:
            if (order.order_type == OrderType.TAKE_PROFIT_MARKET.value and
                order.side == expected_tp_side.value):
                existing_tp_order = order
                break
        
        # Fallback: If get_open_orders returned empty (library bug), check internal tracker
        # This prevents creating duplicate orders when API fails
        if not existing_tp_order and len(exchange_orders) == 0:
            existing_tracked = self._find_order_by_purpose(symbol, position_id, purpose)
            if existing_tracked and existing_tracked.order_id and existing_tracked.order_id > 0:
                logger.warning(f"get_open_orders returned empty (library bug), but found TP in tracker (order_id={existing_tracked.order_id}). "
                             f"Attempting to cancel before creating new order.")
                # Try to cancel the tracked order before creating new one
                cancelled = self._cancel_order(symbol, existing_tracked.order_id)
                if cancelled:
                    logger.debug(f"Successfully cancelled tracked TP order {existing_tracked.order_id} before creating new one")
                else:
                    logger.warning(f"Failed to cancel tracked TP order {existing_tracked.order_id}, but proceeding with new order creation")
        
        # If found on exchange, check if update needed
        if existing_tp_order:
            # Idempotency check - don't update if price is same
            if abs(existing_tp_order.stop_price - new_stop_price) < 0.01:
                logger.debug(f"TP update skipped (idempotency): {symbol} | Price unchanged: {new_stop_price:.8f}")
                existing_tracked = self._find_order_by_purpose(symbol, position_id, purpose)
                if existing_tracked:
                    return OrderResult(success=True, order=existing_tracked)
            
            logger.info(f"Updating TP: {symbol} | Old: {existing_tp_order.stop_price:.8f} → New: {new_stop_price:.8f} | "
                       f"Cancel existing order {existing_tp_order.order_id}")
            
            # Cancel existing TP order from exchange (source of truth)
            if existing_tp_order.order_id and existing_tp_order.order_id > 0:
                cancelled = self._cancel_order(symbol, existing_tp_order.order_id)
                if not cancelled:
                    logger.warning(f"Failed to cancel existing TP {existing_tp_order.order_id} from exchange")
                else:
                    logger.debug(f"Existing TP order {existing_tp_order.order_id} cancelled successfully")
                    # Mark as inactive in tracking if exists
                    with self._orders_lock:
                        if (symbol in self._orders and 
                            existing_tp_order.order_id in self._orders[symbol]):
                            self._orders[symbol][existing_tp_order.order_id].is_active = False
        
        return self.place_take_profit(
            symbol=symbol,
            side=position_side,
            stop_price=new_stop_price,
            quantity=quantity,
            position_id=position_id,
            is_hedge=is_hedge,
            current_price=current_price
        )
    
    # ==================== Cancellation ====================
    
    def cancel_all_for_position(self, symbol: str, position_id: str) -> int:
        """
        Cancel all orders for a position.
        
        Args:
            symbol: Trading symbol
            position_id: Position ID
            
        Returns:
            Number of orders cancelled
        """
        with self._orders_lock:
            if symbol not in self._orders:
                return 0
            
            orders_to_cancel = [
                o for o in self._orders[symbol].values()
                if o.parent_position_id == position_id and o.is_active
            ]
        
        cancelled = 0
        for order in orders_to_cancel:
            # Skip orders with invalid order_id (0 or None)
            if not order.order_id or order.order_id <= 0:
                logger.warning(f"Skipping cancel for order with invalid order_id: {order.order_id} (position {position_id})")
                # Mark as inactive anyway
                order.is_active = False
                continue
            if self._cancel_order(symbol, order.order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders for position {position_id}")
        return cancelled
    
    def cancel_all_for_symbol(self, symbol: str) -> int:
        """
        Cancel all orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Number of orders cancelled
        """
        try:
            count = self.client.cancel_all_orders(symbol)
            
            # Clear tracking
            with self._orders_lock:
                if symbol in self._orders:
                    for o in self._orders[symbol].values():
                        o.is_active = False
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            return 0
    
    def _cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel specific order."""
        # Validate order_id before attempting cancellation
        if not order_id or order_id <= 0:
            logger.debug(f"Skipping cancel for invalid order_id: {order_id} (symbol: {symbol})")
            return False
        
        try:
            result = self.client.cancel_order(symbol, order_id)
            
            with self._orders_lock:
                if symbol in self._orders and order_id in self._orders[symbol]:
                    self._orders[symbol][order_id].is_active = False
            
            return result
            
        except Exception as e:
            error_str = str(e)
            # -2011: Unknown order sent - this is normal (order already filled/cancelled)
            if "-2011" in error_str:
                logger.debug(f"Order {order_id} not found (already filled/cancelled)")
                # Mark as inactive anyway
                with self._orders_lock:
                    if symbol in self._orders and order_id in self._orders[symbol]:
                        self._orders[symbol][order_id].is_active = False
                return True  # Treat as success
            # Handle "orderId is mandatory" error - indicates invalid order_id was passed
            if "orderId is mandatory" in error_str or "orderId is mandatory, but received empty" in error_str:
                logger.debug(f"Invalid order_id {order_id} for {symbol} - order may not exist")
                # Mark as inactive if tracked
                with self._orders_lock:
                    if symbol in self._orders and order_id in self._orders[symbol]:
                        self._orders[symbol][order_id].is_active = False
                return False
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False
    
    # ==================== Orphan Cleanup ====================
    
    def cleanup_orphan_orders(self, symbol: str, active_position_ids: List[str]) -> int:
        """
        Remove orders not linked to active positions.
        
        Args:
            symbol: Trading symbol
            active_position_ids: List of currently active position IDs
            
        Returns:
            Number of orphan orders cancelled
        """
        with self._orders_lock:
            if symbol not in self._orders:
                return 0
            
            orphans = [
                o for o in self._orders[symbol].values()
                if o.is_active and o.parent_position_id not in active_position_ids
            ]
        
        cancelled = 0
        for order in orphans:
            # Skip orders with invalid order_id (0 or None)
            if not order.order_id or order.order_id <= 0:
                logger.warning(f"Skipping cleanup for orphan order with invalid order_id: {order.order_id} (position {order.parent_position_id})")
                # Mark as inactive anyway
                order.is_active = False
                continue
            logger.warning(f"Cleaning orphan order: {order.order_id} (position {order.parent_position_id})")
            if self._cancel_order(symbol, order.order_id):
                cancelled += 1
        
        return cancelled
    
    def reconcile_with_exchange(self, symbol: str) -> Tuple[int, int]:
        """
        Reconcile local order tracking with exchange state.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            (orders_added, orders_removed)
        """
        try:
            exchange_orders = self.client.get_open_orders(symbol)
            # Filter valid order IDs (skip 0 or invalid)
            exchange_ids = {o.order_id for o in exchange_orders if o.order_id > 0}
            
            added = 0
            removed = 0
            
            with self._orders_lock:
                if symbol not in self._orders:
                    self._orders[symbol] = {}
                
                local_ids = set(self._orders[symbol].keys())
                
                # Mark filled/cancelled orders
                for order_id in local_ids - exchange_ids:
                    if order_id in self._orders[symbol] and self._orders[symbol][order_id].is_active:
                        self._orders[symbol][order_id].is_active = False
                        removed += 1
                
                # Add missing orders (from exchange)
                for order in exchange_orders:
                    if order.order_id > 0 and order.order_id not in local_ids:
                        managed = ManagedOrder(
                            order_id=order.order_id,
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=order.quantity,
                            stop_price=order.stop_price,
                            purpose="UNKNOWN",
                            parent_position_id=None
                        )
                        self._orders[symbol][order.order_id] = managed
                        added += 1
            
            if added > 0 or removed > 0:
                logger.debug(f"Reconciled {symbol}: +{added} -{removed}")
            return added, removed
            
        except Exception as e:
            logger.debug(f"Reconciliation note for {symbol}: {e}")
            return 0, 0
    
    # ==================== Order Cleanup ====================
    
    def _cleanup_all_stop_orders(self, symbol: str) -> int:
        """
        Clean up all stop orders (STOP_MARKET and TAKE_PROFIT_MARKET) for a symbol.
        
        This is used when max stop order limit is reached (-4045 error).
        Cancels all stop orders to free up space for new ones.
        
        CRITICAL: When get_open_orders() fails due to library bug, use cancel_all_orders()
        as fallback to cancel ALL orders (this works even when get_open_orders doesn't).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        cancelled_order_ids = set()
        
        # First, try to get orders from exchange
        try:
            exchange_orders = self.client.get_open_orders(symbol)
            
            for order in exchange_orders:
                # Cancel all STOP_MARKET and TAKE_PROFIT_MARKET orders
                if order.order_type in [OrderType.STOP_MARKET.value, OrderType.TAKE_PROFIT_MARKET.value]:
                    if order.order_id and order.order_id > 0:
                        if self._cancel_order(symbol, order.order_id):
                            cancelled += 1
                            cancelled_order_ids.add(order.order_id)
                            logger.debug(f"Cancelled {order.order_type} order {order.order_id} for {symbol} (cleanup due to max limit)")
        except Exception as e:
            # If get_open_orders fails (library bug), we'll use cancel_all_orders as fallback
            logger.warning(f"get_open_orders failed for cleanup (library bug), will use cancel_all_orders as fallback: {e}")
        
        # If we couldn't get orders from exchange (library bug), use cancel_all_orders as fallback
        # This cancels ALL orders for the symbol, which works even when get_open_orders doesn't
        # NOTE: This cancels ALL orders (including SL/TP for other positions), but since we usually
        # have only one position per symbol, this is acceptable. Orders will be restored on next update cycle.
        if cancelled == 0:
            try:
                # Use cancel_all_orders() which works even when get_open_orders() doesn't
                total_cancelled = self.client.cancel_all_orders(symbol)
                if total_cancelled > 0:
                    logger.warning(f"Used cancel_all_orders() to cancel {total_cancelled} orders for {symbol} (fallback due to library bug)")
                    cancelled = total_cancelled
                    # Mark all tracked orders as inactive
                    with self._orders_lock:
                        if symbol in self._orders:
                            for order_id, tracked_order in list(self._orders[symbol].items()):
                                if tracked_order.is_active:
                                    tracked_order.is_active = False
                                    logger.debug(f"Marked tracked order {order_id} as inactive (cancel_all_orders fallback)")
            except Exception as e:
                logger.error(f"Failed to cancel all orders for {symbol} (fallback): {e}")
        
        # Also cancel from internal tracker (in case exchange API didn't return all orders)
        # This ensures we cancel orders even when get_open_orders fails
        with self._orders_lock:
            if symbol in self._orders:
                for order_id, tracked_order in list(self._orders[symbol].items()):
                    # Skip if already cancelled
                    if order_id in cancelled_order_ids:
                        continue
                    
                    if (tracked_order.is_active and
                        tracked_order.order_type in [OrderType.STOP_MARKET.value, OrderType.TAKE_PROFIT_MARKET.value]):
                        if tracked_order.order_id and tracked_order.order_id > 0:
                            if self._cancel_order(symbol, tracked_order.order_id):
                                cancelled += 1
                                cancelled_order_ids.add(tracked_order.order_id)
                                logger.debug(f"Cancelled tracked {tracked_order.order_type} order {tracked_order.order_id} for {symbol} (cleanup due to max limit)")
                        # Mark as inactive even if cancel failed (order may already be filled/cancelled)
                        tracked_order.is_active = False
        
        if cancelled > 0:
            logger.info(f"Cleaned up {cancelled} stop orders for {symbol} due to max stop order limit")
        else:
            logger.warning(f"Could not clean up any orders for {symbol} - all cleanup methods failed")
        
        return cancelled
    
    # ==================== Order Tracking ====================
    
    def _track_order(self, order: ManagedOrder) -> None:
        """Add order to tracking."""
        # Validate order_id before tracking - don't track orders with invalid IDs
        if not order.order_id or order.order_id <= 0:
            logger.warning(f"Skipping tracking for order with invalid order_id: {order.order_id} (symbol: {order.symbol}, purpose: {order.purpose})")
            return
        
        with self._orders_lock:
            if order.symbol not in self._orders:
                self._orders[order.symbol] = {}
            self._orders[order.symbol][order.order_id] = order
    
    def _find_order_by_purpose(
        self,
        symbol: str,
        position_id: str,
        purpose: str
    ) -> Optional[ManagedOrder]:
        """Find active order by purpose."""
        with self._orders_lock:
            if symbol not in self._orders:
                return None
            
            for order in self._orders[symbol].values():
                # Only return orders with valid order_id
                if (order.is_active and
                    order.order_id and order.order_id > 0 and
                    order.parent_position_id == position_id and
                    order.purpose == purpose):
                    return order
            
            return None
    
    def get_active_orders(self, symbol: str) -> List[ManagedOrder]:
        """Get all active orders for symbol."""
        with self._orders_lock:
            if symbol not in self._orders:
                return []
            return [o for o in self._orders[symbol].values() if o.is_active]
    
    def get_orders_for_position(
        self,
        symbol: str,
        position_id: str
    ) -> List[ManagedOrder]:
        """Get all orders for a position."""
        with self._orders_lock:
            if symbol not in self._orders:
                return []
            return [
                o for o in self._orders[symbol].values()
                if o.parent_position_id == position_id
            ]
