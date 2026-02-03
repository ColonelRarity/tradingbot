"""
Position Tracker

Tracks all open positions and their lifecycle:
- Entry tracking
- P&L monitoring
- SL/TP management
- Position state reconciliation
- Pattern recording for learning

Provides single source of truth for position state.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock
from enum import Enum

from exchange.binance_client import BinanceClient, get_binance_client, Position
from core.market_data import MarketData
from core.risk_engine import RiskEngine, RiskCalculation
from core.order_manager import OrderManager
from core.hedge_manager import HedgeManager
from core.pattern_memory import PatternMemory, TradePattern
from core.feature_engineering import FeatureVector
from config.settings import get_settings


logger = logging.getLogger(__name__)


class PositionState(str, Enum):
    """Position lifecycle state."""
    PENDING = "PENDING"  # Entry order placed
    OPEN = "OPEN"  # Position is active
    CLOSING = "CLOSING"  # Close order placed
    CLOSED = "CLOSED"  # Position closed


@dataclass
class TrackedPosition:
    """
    Tracked trading position with full context.
    """
    position_id: str
    symbol: str
    
    # Position details
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    entry_time: float
    
    # Current state
    state: PositionState = PositionState.OPEN
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Stop Loss / Take Profit
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    
    # Risk calculation used
    risk_calc: Optional[RiskCalculation] = None
    
    # Entry features for learning
    entry_features: Optional[List[float]] = None
    entry_atr: float = 0.0
    entry_volatility_regime: float = 0.5
    
    # Exit information
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    realized_pnl: float = 0.0
    
    # Hedge reference
    has_hedge: bool = False
    hedge_id: Optional[str] = None
    
    # Pattern ID for learning
    pattern_id: Optional[str] = None


class PositionTracker:
    """
    Position Tracking System.
    
    Central manager for all position lifecycle:
    - Tracks entries and exits
    - Monitors P&L
    - Coordinates SL/TP updates
    - Records patterns for learning
    - Reconciles with exchange state
    """
    
    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        order_manager: Optional[OrderManager] = None,
        hedge_manager: Optional[HedgeManager] = None,
        risk_engine: Optional[RiskEngine] = None,
        pattern_memory: Optional[PatternMemory] = None,
        ml_inference=None,
        telegram_monitor=None
    ):
        """
        Initialize position tracker.
        
        Args:
            client: Binance client
            order_manager: Order manager
            hedge_manager: Hedge manager
            risk_engine: Risk engine
            pattern_memory: Pattern memory for learning
            ml_inference: ML inference engine for accuracy tracking
            telegram_monitor: Telegram monitor instance
        """
        self.client = client or get_binance_client()
        self.order_manager = order_manager or OrderManager()
        self.hedge_manager = hedge_manager or HedgeManager()
        self.risk_engine = risk_engine or RiskEngine()
        self.pattern_memory = pattern_memory or PatternMemory()
        self.ml_inference = ml_inference
        self._telegram_monitor = telegram_monitor
        
        # Position tracking (position_id -> TrackedPosition)
        self._positions: Dict[str, TrackedPosition] = {}
        self._positions_lock = Lock()
        
        # Symbol -> position_id mapping for quick lookup
        self._symbol_positions: Dict[str, List[str]] = {}
        
        # Cooldown tracking: symbol -> last_close_time
        self._symbol_cooldowns: Dict[str, float] = {}
        self._cooldown_duration = 300.0  # 5 minutes cooldown after closing
        
        # Problematic symbols blacklist: symbol -> timestamp when marked
        # Symbols in blacklist will be skipped when opening new positions
        self._problematic_symbols: Dict[str, float] = {}
        # Use shorter blacklist duration for testnet (1 hour instead of 24)
        import os
        if "testnet" in os.getenv("EXCHANGE_BASE_URL", "").lower() or "testnet" in "https://testnet.binancefuture.com":
            self._problematic_symbol_duration = 3600.0  # 1 hour for testnet
        else:
            self._problematic_symbol_duration = 86400.0  # 24 hours for production
        
        # API error tracking for automatic blacklisting: symbol -> error_count
        self._api_error_counts: Dict[str, int] = {}
        self._max_api_errors_before_blacklist = 3  # Mark as problematic after 3 consecutive errors
        
        # Aggressive monitoring: positions that need urgent breakeven SL check
        self._urgent_breakeven_check: Dict[str, float] = {}  # position_id -> last_check_time
        
        # Cooldown for positions with margin/leverage errors: position_id -> timestamp
        # Prevents repeated attempts to update SL/TP when margin is insufficient or position limit exceeded
        self._sl_update_cooldown: Dict[str, float] = {}  # position_id -> cooldown_until_timestamp
        self._sl_update_cooldown_duration = 300.0  # 5 minutes cooldown after margin/leverage errors
        
        # Statistics
        self._total_positions = 0
        self._winning_positions = 0
        self._total_pnl = 0.0
        
        logger.info("PositionTracker initialized")
    
    # ==================== Position Entry ====================
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        risk_calc: RiskCalculation,
        features: Optional[FeatureVector] = None
    ) -> Optional[TrackedPosition]:
        """
        Open and track a new position.
        
        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            quantity: Position quantity
            entry_price: Entry price
            risk_calc: Risk calculation used
            features: Entry features for learning
            
        Returns:
            TrackedPosition if successful
        """
        position_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Opening position: {position_id} {side} {quantity} {symbol} @ {entry_price}")
        
        # Place entry order
        entry_result = self.order_manager.place_entry_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            position_id=position_id
        )
        
        if not entry_result.success:
            error_msg = entry_result.error_message or "Unknown error"
            # Handle -4005 error (max_qty exceeded) - mark symbol as problematic
            if "-4005" in error_msg or "max_qty" in error_msg.lower() or "Quantity greater than max" in error_msg:
                logger.warning(f"Position entry failed for {symbol}: {error_msg}")
                # Mark symbol as problematic - it cannot be traded with current position sizing
                self.mark_symbol_as_problematic(symbol, reason="max_qty exceeded - symbol not tradeable with current position sizing")
                return None
            # Don't log validation errors as ERROR for other cases
            elif "-4131" in error_msg:
                logger.debug(f"Position entry validation failed: {error_msg}")
            else:
                logger.warning(f"Position entry failed: {error_msg}")
            return None
        
        # Create tracked position
        position = TrackedPosition(
            position_id=position_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=time.time(),
            state=PositionState.OPEN,
            sl_price=risk_calc.stop_loss_price,
            tp_price=risk_calc.take_profit_price,
            risk_calc=risk_calc,
            entry_features=features.features.tolist() if features else None,
            entry_atr=features.raw_values.get("atr_norm", 0) if features else 0,
            entry_volatility_regime=features.raw_values.get("volatility_regime", 0.5) if features else 0.5
        )
        
        # Place SL order
        sl_distance = abs(entry_price - risk_calc.stop_loss_price)
        sl_percent = (sl_distance / entry_price) * 100
        logger.info(f"[{symbol}] PLACING INITIAL SL: {side} | Entry: {entry_price:.8f} | SL: {risk_calc.stop_loss_price:.8f} | "
                   f"Distance: {sl_distance:.8f} ({sl_percent:.2f}%) | Qty: {quantity:.2f}")
        
        sl_result = self.order_manager.place_stop_loss(
            symbol=symbol,
            side=side,
            stop_price=risk_calc.stop_loss_price,
            quantity=quantity,
            position_id=position_id
        )
        
        if sl_result.success:
            position.sl_order_id = sl_result.order.order_id
            logger.info(f"[{symbol}] ✅ INITIAL SL PLACED: OrderID={sl_result.order.order_id} | "
                       f"Price={risk_calc.stop_loss_price:.8f} | Distance={sl_percent:.2f}% | "
                       f"Qty={quantity:.6f} | Side={side}")
        else:
            logger.warning(f"[{symbol}] ❌ INITIAL SL PLACEMENT FAILED: {sl_result.error_message} | "
                          f"Price={risk_calc.stop_loss_price:.8f} | Distance={sl_percent:.2f}% | Side={side}")
        
        # Place TP order
        tp_distance = abs(risk_calc.take_profit_price - entry_price)
        tp_percent = (tp_distance / entry_price) * 100
        rr_ratio = tp_percent / sl_percent if sl_percent > 0 else 0
        logger.info(f"[{symbol}] PLACING INITIAL TP: {side} | Entry: {entry_price:.8f} | TP: {risk_calc.take_profit_price:.8f} | "
                   f"Distance: {tp_distance:.8f} ({tp_percent:.2f}%) | R:R={rr_ratio:.2f} | Qty: {quantity:.2f}")
        
        tp_result = self.order_manager.place_take_profit(
            symbol=symbol,
            side=side,
            stop_price=risk_calc.take_profit_price,
            quantity=quantity,
            position_id=position_id
        )
        
        if tp_result.success:
            position.tp_order_id = tp_result.order.order_id
            logger.info(f"[{symbol}] ✅ INITIAL TP PLACED: OrderID={tp_result.order.order_id} | "
                       f"Price={risk_calc.take_profit_price:.8f} | Distance={tp_percent:.2f}% | R:R={rr_ratio:.2f}")
        else:
            logger.warning(f"[{symbol}] ❌ INITIAL TP PLACEMENT FAILED: {tp_result.error_message}")
        
        # Store pattern for learning
        pattern_id = str(uuid.uuid4())[:12]
        position.pattern_id = pattern_id
        
        if features:
            pattern = TradePattern(
                pattern_id=pattern_id,
                timestamp=int(time.time() * 1000),
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                entry_features=features.features.tolist(),
                exit_price=0,
                pnl_usdt=0,
                pnl_percent=0,
                duration_sec=0,
                exit_reason="",
                is_profitable=False,
                atr_at_entry=position.entry_atr,
                volatility_regime=position.entry_volatility_regime
            )
            self.pattern_memory.store_pattern(pattern)
        
        # Track position
        with self._positions_lock:
            self._positions[position_id] = position
            if symbol not in self._symbol_positions:
                self._symbol_positions[symbol] = []
            self._symbol_positions[symbol].append(position_id)
        
        self._total_positions += 1
        
        # CRITICAL: Verify actual position on exchange immediately after opening
        # This catches cases where the order was filled with wrong side or other issues
        try:
            time.sleep(0.5)  # Small delay to ensure position is registered on exchange
            exchange_positions = self.client.get_positions(symbol)
            if exchange_positions:
                exchange_pos = exchange_positions[0]
                actual_side = "LONG" if exchange_pos.size > 0 else "SHORT"
                actual_size = abs(exchange_pos.size)
                actual_entry = exchange_pos.entry_price
                
                logger.info(f"[VERIFY] {symbol}: Position {position_id} opened | "
                          f"Expected: {side} {quantity:.6f} @ {entry_price:.8f} | "
                          f"Exchange: {actual_side} {actual_size:.6f} @ {actual_entry:.8f} | "
                          f"PnL: {exchange_pos.unrealized_pnl:.2f} USDT")
                
                # Check if side matches - CRITICAL ERROR if not
                if actual_side != side:
                    logger.error(f"[CRITICAL] {symbol}: Position {position_id} SIDE MISMATCH! "
                               f"Expected {side}, but exchange shows {actual_side}. "
                               f"This is a SERIOUS ERROR - position opened with wrong side!")
                    logger.error(f"[CRITICAL] {symbol}: This could happen if:")
                    logger.error(f"[CRITICAL] {symbol}: 1. Existing opposite position was closed instead of opening new")
                    logger.error(f"[CRITICAL] {symbol}: 2. API error caused wrong order execution")
                    logger.error(f"[CRITICAL] {symbol}: 3. Hedge position exists and NET side is different")
                    
                    # Check if hedge exists (might explain the mismatch)
                    hedge = self.hedge_manager.get_hedge_for_parent(position_id)
                    if hedge and hedge.is_open:
                        logger.warning(f"[CRITICAL] {symbol}: Hedge {hedge.hedge_id} exists - NET side {actual_side} may be correct")
                        position.has_hedge = True
                        position.hedge_id = hedge.hedge_id
                    else:
                        # No hedge - this is a real error, update to match exchange
                        logger.error(f"[CRITICAL] {symbol}: No hedge found - correcting position to match exchange!")
                        old_side = position.side
                        position.side = actual_side
                        position.quantity = actual_size
                        position.entry_price = actual_entry
                        
                        logger.error(f"[CRITICAL] {symbol}: Position {position_id} corrected from {old_side} to {actual_side}")
                
                # Check if size matches (within 5% tolerance for rounding)
                size_diff = abs(actual_size - quantity) / quantity if quantity > 0 else 0
                if size_diff > 0.05:  # More than 5% difference
                    logger.warning(f"[VERIFY] {symbol}: Position {position_id} size mismatch! "
                                 f"Expected {quantity:.6f}, exchange shows {actual_size:.6f} (diff: {size_diff*100:.2f}%)")
                    # Update quantity to match exchange
                    position.quantity = actual_size
            else:
                logger.warning(f"[VERIFY] {symbol}: Position {position_id} opened but no position found on exchange!")
        except Exception as e:
            logger.warning(f"[VERIFY] {symbol}: Failed to verify position {position_id} on exchange: {e}")
        
        logger.info(f"Position opened: {position_id} | SL={risk_calc.stop_loss_price:.2f} | TP={risk_calc.take_profit_price:.2f}")
        
        return position
    
    # ==================== Position Updates ====================
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        atr: float,
        volatility_regime: float
    ) -> Optional[TrackedPosition]:
        """
        Update position state with current market data.
        
        Handles:
        - P&L calculation
        - SL/TP updates via RiskEngine
        - Hedge evaluation
        
        Args:
            position_id: Position ID
            current_price: Current market price
            atr: Current ATR
            volatility_regime: Current volatility regime
            
        Returns:
            Updated position or None
        """
        with self._positions_lock:
            if position_id not in self._positions:
                return None
            position = self._positions[position_id]
        
        if position.state != PositionState.OPEN:
            return position
        
        # Update current price
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.side == "LONG":
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity
        
        position.unrealized_pnl = pnl
        
        # === Verify and restore missing SL/TP orders ===
        self._ensure_sl_tp_orders(position, current_price)
        
        # === SL Update ===
        # PROACTIVE: Mark profitable positions for urgent breakeven check BEFORE calculating update
        # This ensures breakeven SL is set even if calculate_sl_update doesn't return is_update_needed
        if pnl >= self.risk_engine.sl_config.breakeven_trigger_usdt:
            # Check if current SL is at breakeven or better
            min_profit_usdt = self.risk_engine.sl_config.breakeven_trigger_usdt
            if position.quantity > 0:
                if position.side == "LONG":
                    required_breakeven_sl = position.entry_price + (min_profit_usdt / position.quantity)
                    sl_is_profitable = position.sl_price >= required_breakeven_sl - 0.0001
                else:  # SHORT
                    required_breakeven_sl = position.entry_price - (min_profit_usdt / position.quantity)
                    sl_is_profitable = position.sl_price <= required_breakeven_sl + 0.0001
                
                # If SL is not at breakeven, mark for urgent check
                if not sl_is_profitable:
                    self._urgent_breakeven_check[position_id] = time.time()
        
        sl_update = self.risk_engine.calculate_sl_update(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            current_price=current_price,
            current_sl=position.sl_price,
            unrealized_pnl=pnl,
            atr=atr,
            quantity=position.quantity  # Pass quantity for USDT profit calculation
        )
        
        if sl_update.is_update_needed:
            old_sl = position.sl_price
            new_sl = sl_update.new_sl_price
            sl_change = new_sl - old_sl if position.side == "LONG" else old_sl - new_sl
            sl_change_percent = (sl_change / old_sl) * 100 if old_sl > 0 else 0
            
            logger.info(f"[{position.symbol}] UPDATING SL: {position.side} | "
                       f"Old SL: {old_sl:.8f} → New SL: {new_sl:.8f} | "
                       f"Change: {sl_change:.8f} ({sl_change_percent:+.2f}%) | "
                       f"Current: {current_price:.8f} | PnL: {pnl:.2f} USDT | "
                       f"Reason: {sl_update.reason}")
            
            # Check cooldown for margin/leverage errors before attempting update
            if position_id in self._sl_update_cooldown:
                cooldown_until = self._sl_update_cooldown[position_id]
                if time.time() < cooldown_until:
                    # Still in cooldown, skip update
                    remaining = cooldown_until - time.time()
                    logger.debug(f"[{position.symbol}] SL update skipped due to cooldown ({remaining/60:.1f} min remaining)")
                    return
            
            result = self.order_manager.update_stop_loss(
                symbol=position.symbol,
                position_id=position_id,
                new_stop_price=sl_update.new_sl_price,
                quantity=position.quantity,
                position_side=position.side,  # Pass position side explicitly
                current_price=current_price,
                is_breakeven=sl_update.is_breakeven  # Allow smaller minimum distance for breakeven SL
            )
            
            if result.success:
                position.sl_price = sl_update.new_sl_price
                position.sl_order_id = result.order.order_id
                logger.info(f"[{position.symbol}] ✅ SL UPDATED: OrderID={result.order.order_id} | "
                           f"Price={new_sl:.8f} | Change={sl_change_percent:+.2f}% | "
                           f"Current={current_price:.8f} | PnL={pnl:.2f} USDT | Reason={sl_update.reason} | "
                           f"Breakeven={sl_update.is_breakeven} | Trailing={sl_update.is_trailing}")
                # Remove from urgent check if breakeven was set successfully (aggressive check no longer needed)
                # Trailing SL will continue to work in normal update cycle (every 10 seconds)
                if sl_update.is_breakeven and position_id in self._urgent_breakeven_check:
                    del self._urgent_breakeven_check[position_id]
                # Clear cooldown if SL was successfully updated
                if position_id in self._sl_update_cooldown:
                    del self._sl_update_cooldown[position_id]
                logger.debug(f"[SL] {position.symbol}: Breakeven SL set for {position_id}, removed from urgent check (trailing will continue in normal mode)")
            else:
                error_code = result.error_code if hasattr(result, 'error_code') else ""
                # Check if error is due to margin/leverage issues
                if error_code in ["MARGIN_INSUFFICIENT", "MAX_POSITION_EXCEEDED"]:
                    # For breakeven SL, use shorter cooldown (1 minute instead of 5)
                    # For regular SL updates, use normal cooldown (5 minutes)
                    if sl_update.is_breakeven:
                        cooldown_duration = 60.0  # 1 minute for breakeven SL
                    else:
                        cooldown_duration = self._sl_update_cooldown_duration  # 5 minutes for regular SL
                    
                    self._sl_update_cooldown[position_id] = time.time() + cooldown_duration
                    logger.warning(f"[{position.symbol}] ❌ SL UPDATE FAILED: {error_code} | "
                                  f"Cannot update {'breakeven' if sl_update.is_breakeven else 'SL'} due to margin/leverage constraints. "
                                  f"Cooldown set for {cooldown_duration/60:.1f} minutes. | "
                                  f"Old SL: {old_sl:.8f} | Attempted: {new_sl:.8f} | Current: {current_price:.8f} | "
                                  f"PnL: {pnl:.2f} USDT")
                    # For breakeven SL, keep in urgent check to retry faster
                    # For regular SL, remove from urgent check temporarily
                    if not sl_update.is_breakeven and position_id in self._urgent_breakeven_check:
                        del self._urgent_breakeven_check[position_id]
                # Check if error is due to quantity exceeding max_qty
                elif error_code in ["SL_QUANTITY_EXCEEDED", "TP_QUANTITY_EXCEEDED"]:
                    # Use shorter cooldown (2 minutes) for quantity exceeded - fallback to closePosition should work
                    self._sl_update_cooldown[position_id] = time.time() + 120.0  # 2 minutes
                    logger.warning(f"[{position.symbol}] ❌ SL UPDATE FAILED: Quantity exceeds max_qty. "
                                  f"Fallback to closePosition should work. Cooldown set for 2 minutes. | "
                                  f"Old SL: {old_sl:.8f} | Attempted: {new_sl:.8f} | Current: {current_price:.8f} | "
                                  f"PnL: {pnl:.2f} USDT | Qty: {position.quantity:.6f}")
                    # Keep in urgent check for breakeven SL to retry after cooldown
                    if not sl_update.is_breakeven and position_id in self._urgent_breakeven_check:
                        del self._urgent_breakeven_check[position_id]
                # Check if error is due to max stop order limit
                elif error_code == "MAX_STOP_ORDERS_REACHED":
                    # Use shorter cooldown (2 minutes) for max stop order limit - cleanup was attempted
                    self._sl_update_cooldown[position_id] = time.time() + 120.0  # 2 minutes
                    logger.warning(f"[{position.symbol}] ❌ SL UPDATE FAILED: Max stop order limit reached. "
                                  f"Cleanup attempted. Cooldown set for 2 minutes. | "
                                  f"Old SL: {old_sl:.8f} | Attempted: {new_sl:.8f} | Current: {current_price:.8f} | "
                                  f"PnL: {pnl:.2f} USDT")
                    # For breakeven SL, keep in urgent check to retry after cooldown
                    if sl_update.is_breakeven:
                        # Keep in urgent check but respect cooldown
                        pass
                    else:
                        # For regular SL, remove from urgent check temporarily
                        if position_id in self._urgent_breakeven_check:
                            del self._urgent_breakeven_check[position_id]
                else:
                    logger.warning(f"[{position.symbol}] ❌ SL UPDATE FAILED: {result.error_message} | "
                                  f"Old SL: {old_sl:.8f} | Attempted: {new_sl:.8f} | Current: {current_price:.8f} | "
                                  f"PnL: {pnl:.2f} USDT | Side={position.side} | Qty={position.quantity:.6f}")
                    # If update failed and position is profitable, mark for urgent checking
                    if pnl >= self.risk_engine.sl_config.breakeven_trigger_usdt:
                        self._urgent_breakeven_check[position_id] = time.time()
                        logger.warning(f"[SL] {position.symbol}: SL update failed for profitable position {position_id} (PnL={pnl:.2f}), marking for urgent check")
        else:
            # Even if no update needed from calculate_sl_update, check if we need breakeven
            # This handles cases where calculate_sl_update returns NO_CHANGE but position should have breakeven
            if pnl >= self.risk_engine.sl_config.breakeven_trigger_usdt:
                min_profit_usdt = self.risk_engine.sl_config.breakeven_trigger_usdt
                if position.quantity > 0:
                    if position.side == "LONG":
                        required_breakeven_sl = position.entry_price + (min_profit_usdt / position.quantity)
                        sl_is_profitable = position.sl_price >= required_breakeven_sl - 0.0001
                    else:  # SHORT
                        required_breakeven_sl = position.entry_price - (min_profit_usdt / position.quantity)
                        sl_is_profitable = position.sl_price <= required_breakeven_sl + 0.0001
                    
                    if not sl_is_profitable:
                        # Position is profitable but SL is not at breakeven - mark for urgent check
                        self._urgent_breakeven_check[position_id] = time.time()
                        logger.debug(f"Position {position_id} profitable (PnL={pnl:.2f}) but SL not at breakeven, marking for urgent check")
        
        # === TP Update ===
        tp_update = self.risk_engine.calculate_tp_update(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            current_price=current_price,
            current_tp=position.tp_price,
            atr=atr,
            volatility_regime=volatility_regime
        )
        
        if tp_update.is_update_needed:
            old_tp = position.tp_price
            new_tp = tp_update.new_tp_price
            tp_change = new_tp - old_tp if position.side == "LONG" else old_tp - new_tp
            tp_change_percent = (tp_change / old_tp) * 100 if old_tp > 0 else 0
            
            logger.info(f"[{position.symbol}] UPDATING TP: {position.side} | "
                       f"Old TP: {old_tp:.8f} → New TP: {new_tp:.8f} | "
                       f"Change: {tp_change:.8f} ({tp_change_percent:+.2f}%) | "
                       f"Current: {current_price:.8f} | PnL: {pnl:.2f} USDT | "
                       f"Reason: {tp_update.reason}")
            
            # Check cooldown for margin/leverage errors before attempting update
            if position_id in self._sl_update_cooldown:
                cooldown_until = self._sl_update_cooldown[position_id]
                if time.time() < cooldown_until:
                    # Still in cooldown, skip update
                    remaining = cooldown_until - time.time()
                    logger.debug(f"[{position.symbol}] TP update skipped due to cooldown ({remaining/60:.1f} min remaining)")
                    return
            
            result = self.order_manager.update_take_profit(
                symbol=position.symbol,
                position_id=position_id,
                new_stop_price=tp_update.new_tp_price,
                quantity=position.quantity,
                position_side=position.side,  # Pass position side explicitly
                current_price=current_price
            )
            
            if result.success:
                position.tp_price = tp_update.new_tp_price
                position.tp_order_id = result.order.order_id
                logger.info(f"[{position.symbol}] ✅ TP UPDATED: OrderID={result.order.order_id} | "
                           f"Price={new_tp:.8f} | Change={tp_change_percent:+.2f}% | Reason={tp_update.reason}")
                # Clear cooldown if TP was successfully updated
                if position_id in self._sl_update_cooldown:
                    del self._sl_update_cooldown[position_id]
            else:
                error_code = result.error_code if hasattr(result, 'error_code') else ""
                # Check if error is due to margin/leverage issues
                if error_code in ["MARGIN_INSUFFICIENT", "MAX_POSITION_EXCEEDED"]:
                    # Set cooldown to prevent repeated attempts (shared with SL cooldown)
                    self._sl_update_cooldown[position_id] = time.time() + self._sl_update_cooldown_duration
                    logger.warning(f"[{position.symbol}] ❌ TP UPDATE FAILED: {error_code} | "
                                  f"Cannot update TP due to margin/leverage constraints. "
                                  f"Cooldown set for {self._sl_update_cooldown_duration/60:.1f} minutes. | "
                                  f"Old TP: {old_tp:.8f} | Attempted: {new_tp:.8f} | PnL: {pnl:.2f} USDT")
                # Check if error is due to max stop order limit
                elif error_code == "MAX_STOP_ORDERS_REACHED":
                    # Use shorter cooldown (2 minutes) for max stop order limit - cleanup was attempted
                    self._sl_update_cooldown[position_id] = time.time() + 120.0  # 2 minutes
                    logger.warning(f"[{position.symbol}] ❌ TP UPDATE FAILED: Max stop order limit reached. "
                                  f"Cleanup attempted. Cooldown set for 2 minutes. | "
                                  f"Old TP: {old_tp:.8f} | Attempted: {new_tp:.8f} | PnL: {pnl:.2f} USDT")
                # Check if error is due to quantity exceeding max_qty
                elif error_code in ["TP_QUANTITY_EXCEEDED"]:
                    # Use shorter cooldown (2 minutes) for quantity exceeded - fallback to closePosition should work
                    self._sl_update_cooldown[position_id] = time.time() + 120.0  # 2 minutes
                    logger.warning(f"[{position.symbol}] ❌ TP UPDATE FAILED: Quantity exceeds max_qty. "
                                  f"Fallback to closePosition should work. Cooldown set for 2 minutes. | "
                                  f"Old TP: {old_tp:.8f} | Attempted: {new_tp:.8f} | PnL: {pnl:.2f} USDT | Qty: {position.quantity:.6f}")
                else:
                    logger.warning(f"[{position.symbol}] ❌ TP UPDATE FAILED: {result.error_message} | "
                                  f"Old TP: {old_tp:.8f} | Attempted: {new_tp:.8f} | PnL: {pnl:.2f} USDT")
        
        # === Hedge Evaluation ===
        if not position.has_hedge:
            if self.hedge_manager.should_open_hedge(position_id, pnl, position.side):
                # Calculate parent position size in USDT for adaptive hedge sizing
                parent_position_size_usdt = position.quantity * position.entry_price
                
                hedge = self.hedge_manager.open_hedge(
                    parent_position_id=position_id,
                    symbol=position.symbol,
                    parent_side=position.side,
                    current_price=current_price,
                    parent_position_size_usdt=parent_position_size_usdt
                )
                
                if hedge:
                    position.has_hedge = True
                    position.hedge_id = hedge.hedge_id
        
                # Check hedge close and sync status
                if position.has_hedge and position.hedge_id:
                    # Check if hedge is still open
                    hedge = self.hedge_manager.get_hedge_for_parent(position_id)
                    if hedge and hedge.is_open:
                        # Hedge is open in our tracking, check if it should be closed
                        closed_by_loss = self.hedge_manager.check_hedge_close(position.hedge_id, current_price)

                        # Verify hedge still exists on exchange (re-check after potential close)
                        hedge = self.hedge_manager.get_hedge_for_parent(position_id)
                        if not hedge or not hedge.is_open:
                            # Hedge was closed by check_hedge_close, update position flag
                            logger.info(f"[UPDATE] {position.symbol}: Hedge {position.hedge_id} was closed, updating position {position_id} has_hedge=False")
                            position.has_hedge = False
                            position.hedge_id = None

                            # CRITICAL: If hedge was closed due to loss (-7 USDT), close the parent position immediately
                            if closed_by_loss:
                                logger.info(f"[HEDGE_LOSS_CLOSE] {position.symbol}: Hedge closed due to -7 USDT loss, closing parent position {position_id}")
                                self.close_position(position_id, current_price, "HEDGE_LOSS_CLOSE")
                                return position
                    else:
                        # Hedge is closed or doesn't exist, update position flag
                        if position.has_hedge:
                            logger.info(f"[UPDATE] {position.symbol}: Hedge {position.hedge_id} is closed (not in active hedges), updating position {position_id} has_hedge=False")
                            position.has_hedge = False
                            position.hedge_id = None
        
        return position
    
    def _ensure_sl_tp_orders(self, position: TrackedPosition, current_price: float) -> None:
        """
        Verify that SL/TP orders exist on exchange, and place them if missing.
        
        Args:
            position: Tracked position
            current_price: Current market price
        """
        # Check if SL order exists on exchange
        exchange_orders = self.client.get_open_orders(position.symbol)
        
        # If get_open_orders returns empty list, it might be due to library bug
        # In this case, we can't verify if orders exist, so we skip restoration to avoid duplicates
        # NOTE: Library bug with get_open_orders is a known issue and should NOT be counted as API error
        # because it's not a real API failure - it's a bug in the binance-futures-connector library
        if len(exchange_orders) == 0:
            # Check if we have tracked orders that should exist
            # We know we should have SL/TP orders if position has order IDs set
            has_tracked_orders = (position.sl_order_id is not None) or (position.tp_order_id is not None)
            
            # If we have tracked order IDs but get_open_orders returned empty, it's likely a library bug
            # Do NOT record this as API error - it's a known library bug, not a real API failure
            if has_tracked_orders:
                # Skip restoration when we can't verify orders exist (to avoid duplicates)
                # Orders will be checked again on next update cycle
                # This is a known library bug, not a real API error, so we don't blacklist the symbol
                logger.debug(f"[{position.symbol}] get_open_orders returned empty (library bug), skipping SL/TP restoration to avoid duplicates")
                return
        
        sl_exists = False
        tp_exists = False
        
        if position.sl_order_id:
            sl_exists = any(o.order_id == position.sl_order_id for o in exchange_orders)
        
        if position.tp_order_id:
            tp_exists = any(o.order_id == position.tp_order_id for o in exchange_orders)
        
        # Reset error count on successful API call (if we got orders back)
        if len(exchange_orders) > 0:
            self._reset_api_error_count(position.symbol)
        
        # Restore missing SL order
        if not sl_exists and position.sl_price > 0:
            logger.warning(f"[{position.symbol}] SL order missing on exchange (OrderID={position.sl_order_id or 'None'}), restoring...")
            
            # Adjust SL price if it's too close to current price
            adjusted_sl_price = position.sl_price
            if current_price > 0:
                if position.side == "LONG":
                    # For LONG, SL must be below current price
                    if adjusted_sl_price >= current_price:
                        # Adjust SL to be 0.1% below current price
                        adjusted_sl_price = current_price * 0.999
                        logger.warning(f"[{position.symbol}] SL price {position.sl_price} >= current {current_price} for LONG, adjusting to {adjusted_sl_price:.8f}")
                    else:
                        # Check minimum distance (0.05%)
                        min_distance = current_price * 0.0005
                        if current_price - adjusted_sl_price < min_distance:
                            adjusted_sl_price = current_price - min_distance
                            logger.warning(f"[{position.symbol}] SL too close to current, adjusting to {adjusted_sl_price:.8f}")
                else:  # SHORT
                    # For SHORT, SL must be above current price
                    if adjusted_sl_price <= current_price:
                        # Adjust SL to be 0.1% above current price
                        adjusted_sl_price = current_price * 1.001
                        logger.warning(f"[{position.symbol}] SL price {position.sl_price} <= current {current_price} for SHORT, adjusting to {adjusted_sl_price:.8f}")
                    else:
                        # Check minimum distance (0.05%)
                        min_distance = current_price * 0.0005
                        if adjusted_sl_price - current_price < min_distance:
                            adjusted_sl_price = current_price + min_distance
                            logger.warning(f"[{position.symbol}] SL too close to current, adjusting to {adjusted_sl_price:.8f}")
            
            sl_result = self.order_manager.place_stop_loss(
                symbol=position.symbol,
                side=position.side,
                stop_price=adjusted_sl_price,
                quantity=position.quantity,
                position_id=position.position_id,
                current_price=current_price
            )
            if sl_result.success:
                position.sl_price = adjusted_sl_price  # Update tracked SL price
                position.sl_order_id = sl_result.order.order_id
                logger.info(f"[{position.symbol}] ✅ SL order restored: OrderID={sl_result.order.order_id} | Price={adjusted_sl_price:.8f}")
            else:
                logger.error(f"[{position.symbol}] ❌ Failed to restore SL order: {sl_result.error_message}")
        
        # Restore missing TP order
        if not tp_exists and position.tp_price > 0:
            logger.warning(f"[{position.symbol}] TP order missing on exchange (OrderID={position.tp_order_id or 'None'}), restoring...")
            
            # Adjust TP price if it's too close to current price
            adjusted_tp_price = position.tp_price
            if current_price > 0:
                if position.side == "LONG":
                    # For LONG, TP must be above current price
                    if adjusted_tp_price <= current_price:
                        # Adjust TP to be 0.1% above current price
                        adjusted_tp_price = current_price * 1.001
                        logger.warning(f"[{position.symbol}] TP price {position.tp_price} <= current {current_price} for LONG, adjusting to {adjusted_tp_price:.8f}")
                    else:
                        # Check minimum distance (0.1%)
                        min_distance = current_price * 0.001
                        if adjusted_tp_price - current_price < min_distance:
                            adjusted_tp_price = current_price + min_distance
                            logger.warning(f"[{position.symbol}] TP too close to current, adjusting to {adjusted_tp_price:.8f}")
                else:  # SHORT
                    # For SHORT, TP must be below current price
                    if adjusted_tp_price >= current_price:
                        # Adjust TP to be 0.1% below current price
                        adjusted_tp_price = current_price * 0.999
                        logger.warning(f"[{position.symbol}] TP price {position.tp_price} >= current {current_price} for SHORT, adjusting to {adjusted_tp_price:.8f}")
                    else:
                        # Check minimum distance (0.1%)
                        min_distance = current_price * 0.001
                        if current_price - adjusted_tp_price < min_distance:
                            adjusted_tp_price = current_price - min_distance
                            logger.warning(f"[{position.symbol}] TP too close to current, adjusting to {adjusted_tp_price:.8f}")
            
            tp_result = self.order_manager.place_take_profit(
                symbol=position.symbol,
                side=position.side,
                stop_price=adjusted_tp_price,
                quantity=position.quantity,
                position_id=position.position_id,
                current_price=current_price
            )
            if tp_result.success:
                position.tp_price = adjusted_tp_price  # Update tracked TP price
                position.tp_order_id = tp_result.order.order_id
                logger.info(f"[{position.symbol}] ✅ TP order restored: OrderID={tp_result.order.order_id} | Price={adjusted_tp_price:.8f}")
            else:
                logger.error(f"[{position.symbol}] ❌ Failed to restore TP order: {tp_result.error_message}")
    
    # ==================== Position Exit ====================
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        force_full_close: bool = True
    ) -> Optional[TrackedPosition]:
        """
        Close a position and record outcome.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_reason: Reason for exit ("TP", "SL", "MANUAL")
            
        Returns:
            Closed position or None
        """
        with self._positions_lock:
            if position_id not in self._positions:
                return None
            position = self._positions[position_id]
        
        if position.state != PositionState.OPEN:
            return position
        
        # Cancel remaining orders
        self.order_manager.cancel_all_for_position(position.symbol, position_id)

        # Close any hedge
        if position.has_hedge:
            self.hedge_manager.close_hedge_for_parent(position_id, exit_price)

        # CRITICAL: Ensure FULL position closure using MARKET order with reduceOnly=True
        # This guarantees complete position closure regardless of other modes
        try:
            logger.info(f"[FULL_CLOSE] {position.symbol}: Ensuring complete position closure for {position_id}")

            # Get current position from exchange to verify closure
            exchange_positions = self.client.get_positions(position.symbol)
            exchange_pos = exchange_positions[0] if exchange_positions else None

            if exchange_pos and abs(exchange_pos.size) > 0.0001:
                # Position still exists on exchange, force close it
                close_side = "SELL" if exchange_pos.size > 0 else "BUY"
                close_quantity = abs(exchange_pos.size)

                # Use MARKET order with reduceOnly=True for guaranteed full closure
                order = self.client.place_market_order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=close_quantity,
                    reduce_only=True
                )

                logger.info(f"[FULL_CLOSE] {position.symbol}: MARKET close order placed - {close_side} {close_quantity:.6f}")
            else:
                logger.info(f"[FULL_CLOSE] {position.symbol}: Position already closed on exchange")

        except Exception as e:
            logger.error(f"[FULL_CLOSE] {position.symbol}: Failed to ensure full closure: {e}")
            # Continue with normal closure flow even if forced close fails
        
        # Calculate realized P&L
        if position.side == "LONG":
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
            pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100
        
        # Update position
        position.state = PositionState.CLOSED
        position.exit_price = exit_price
        position.exit_time = time.time()
        position.exit_reason = exit_reason
        position.realized_pnl = realized_pnl
        
        # Record in risk engine
        self.risk_engine.record_trade(realized_pnl)
        
        # Set cooldown for symbol (prevent immediate reopening)
        self._symbol_cooldowns[position.symbol] = time.time()
        logger.debug(f"Cooldown set for {position.symbol} until {time.time() + self._cooldown_duration:.0f}")
        
        # Update pattern for learning
        if position.pattern_id:
            duration = int(position.exit_time - position.entry_time)
            self.pattern_memory.update_pattern_outcome(
                pattern_id=position.pattern_id,
                exit_price=exit_price,
                pnl_usdt=realized_pnl,
                pnl_percent=pnl_percent,
                duration_sec=duration,
                exit_reason=exit_reason
            )
        
        # Update statistics
        self._total_pnl += realized_pnl
        if realized_pnl > 0:
            self._winning_positions += 1
        
        # Record prediction outcome for ML accuracy tracking
        # Prediction is correct if position was profitable (PnL > 0)
        # This means we correctly predicted the direction and price movement
        if self.ml_inference and position.pattern_id:
            was_correct = realized_pnl > 0
            self.ml_inference.record_outcome(was_correct)
            logger.debug(
                f"Recorded prediction outcome: {position_id} | "
                f"Correct={was_correct} | PnL={realized_pnl:.2f}"
            )
        
        logger.info(
            f"Position closed: {position_id} | "
            f"PnL={realized_pnl:.2f} USDT ({pnl_percent:.2f}%) | "
            f"Reason={exit_reason}"
        )
        
        # Notify Telegram if available (will be set from main)
        if hasattr(self, '_telegram_monitor') and self._telegram_monitor:
            try:
                self._telegram_monitor.send_position_closed(
                    symbol=position.symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    pnl_percent=pnl_percent,
                    reason=exit_reason
                )
            except Exception:
                pass
        
        return position
    
    # ==================== Reconciliation ====================
    
    def reconcile_with_exchange(self, symbol: str) -> Tuple[int, int]:
        """
        Reconcile tracked positions with exchange state.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            (positions_added, positions_removed)
        """
        try:
            # Get exchange positions
            exchange_positions = self.client.get_positions(symbol)
            exchange_has_position = len(exchange_positions) > 0
            
            with self._positions_lock:
                tracked_ids = self._symbol_positions.get(symbol, [])
                open_tracked = [
                    self._positions[pid]
                    for pid in tracked_ids
                    if pid in self._positions and
                    self._positions[pid].state == PositionState.OPEN
                ]
            
            added = 0
            removed = 0
            
            # If exchange has position but we don't track it
            if exchange_has_position and not open_tracked:
                # Exchange has position we didn't track - likely manual trade
                exchange_pos = exchange_positions[0]
                logger.warning(f"[RECONCILE] {symbol}: Untracked position found on exchange: "
                             f"side={exchange_pos.side}, size={exchange_pos.size:.6f}, "
                             f"entry={exchange_pos.entry_price:.8f}, PnL={exchange_pos.unrealized_pnl:.2f}")
                # Could add tracking here if needed
                added = 1
            
            # If we track position but exchange doesn't have it
            if not exchange_has_position and open_tracked:
                for pos in open_tracked:
                    # Position was closed externally (filled SL/TP or manual)
                    logger.info(f"[RECONCILE] {pos.symbol}: Position {pos.position_id} closed externally "
                              f"(tracked but not on exchange). Entry={pos.entry_price:.8f}, "
                              f"current={pos.current_price:.8f}, PnL={pos.unrealized_pnl:.2f}")
                    
                    # Try to determine exit price and reason
                    exit_price = pos.current_price
                    
                    # Check if SL or TP was hit
                    if pos.side == "LONG":
                        if exit_price <= pos.sl_price:
                            exit_reason = "SL"
                        elif exit_price >= pos.tp_price:
                            exit_reason = "TP"
                        else:
                            exit_reason = "EXTERNAL"
                    else:
                        if exit_price >= pos.sl_price:
                            exit_reason = "SL"
                        elif exit_price <= pos.tp_price:
                            exit_reason = "TP"
                        else:
                            exit_reason = "EXTERNAL"
                    
                    logger.info(f"[RECONCILE] {pos.symbol}: Closing position {pos.position_id}, reason={exit_reason}")
                    self.close_position(pos.position_id, exit_price, exit_reason)
                    removed += 1
            
            # Sync hedge status: verify hedge exists on exchange using NET position
            for pos in open_tracked:
                if pos.has_hedge and pos.hedge_id:
                    hedge = self.hedge_manager.get_hedge_for_parent(pos.position_id)
                    if not hedge or not hedge.is_open:
                        # Hedge is closed in our tracking, update position flag
                        logger.info(f"[RECONCILE] Hedge {pos.hedge_id} is closed (not in active hedges), updating position {pos.position_id} has_hedge=False")
                        pos.has_hedge = False
                        pos.hedge_id = None
                    else:
                        # Verify hedge exists on exchange by checking NET position
                        # NET position = parent + hedge combined
                        # If NET size ≈ parent quantity, hedge doesn't exist or is closed
                        if exchange_has_position and exchange_positions:
                            exchange_pos = exchange_positions[0]
                            net_size = abs(exchange_pos.size)
                            parent_size = abs(pos.quantity)
                            
                            # Hedge should reduce NET size: NET ≈ parent - hedge
                            # If NET ≈ parent, hedge is closed or doesn't exist
                            size_diff = abs(net_size - parent_size)
                            hedge_size_threshold = parent_size * 0.3  # Hedge is typically 30-50% of parent
                            
                            if size_diff < hedge_size_threshold:
                                # NET size ≈ parent size, hedge doesn't exist on exchange
                                logger.warning(f"[RECONCILE] {pos.symbol}: NET size ({net_size:.6f}) ≈ parent size ({parent_size:.6f}), "
                                             f"hedge {pos.hedge_id} doesn't exist on exchange. Marking hedge as closed.")
                                # Mark hedge as closed
                                hedge.is_open = False
                                pos.has_hedge = False
                                pos.hedge_id = None
                            else:
                                logger.debug(f"[RECONCILE] {pos.symbol}: Hedge {pos.hedge_id} verified. NET={net_size:.6f}, parent={parent_size:.6f}, diff={size_diff:.6f}")
            
            # Reconcile orders
            self.order_manager.reconcile_with_exchange(symbol)
            
            return added, removed
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            return 0, 0
    
    # ==================== Queries ====================
    
    def get_position(self, position_id: str) -> Optional[TrackedPosition]:
        """Get position by ID."""
        with self._positions_lock:
            return self._positions.get(position_id)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[TrackedPosition]:
        """Get all open positions, optionally filtered by symbol."""
        with self._positions_lock:
            positions = [
                p for p in self._positions.values()
                if p.state == PositionState.OPEN
            ]
            
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            
            return positions
    
    def get_position_for_symbol(self, symbol: str) -> Optional[TrackedPosition]:
        """Get open position for symbol (assumes max 1 per symbol)."""
        positions = self.get_open_positions(symbol)
        return positions[0] if positions else None
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has open position."""
        return self.get_position_for_symbol(symbol) is not None
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period after closing a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is in cooldown (should not open new position)
        """
        if symbol not in self._symbol_cooldowns:
            return False
        
        last_close_time = self._symbol_cooldowns[symbol]
        elapsed = time.time() - last_close_time
        
        if elapsed >= self._cooldown_duration:
            # Cooldown expired, remove from tracking
            del self._symbol_cooldowns[symbol]
            return False
        
        return True
    
    def is_problematic_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is in problematic symbols blacklist.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is problematic (should not open new position or perform detailed updates)
        """
        if symbol not in self._problematic_symbols:
            return False
        
        marked_time = self._problematic_symbols[symbol]
        elapsed = time.time() - marked_time
        
        if elapsed >= self._problematic_symbol_duration:
            # Blacklist expired, remove from tracking
            del self._problematic_symbols[symbol]
            if symbol in self._api_error_counts:
                del self._api_error_counts[symbol]
            logger.info(f"Symbol {symbol} removed from problematic list (blacklist expired after {elapsed/3600:.1f} hours)")
            print(f"DEBUG: Symbol {symbol} removed from blacklist (expired)", flush=True)
            return False
        
        # Symbol is still in blacklist
        remaining_time = self._problematic_symbol_duration - elapsed
        logger.debug(f"Symbol {symbol} is problematic (blacklist expires in {remaining_time/3600:.1f} hours)")
        return True
    
    def mark_symbol_as_problematic(self, symbol: str, reason: str = "API errors") -> None:
        """
        Mark symbol as problematic (add to blacklist).
        
        Args:
            symbol: Trading symbol
            reason: Reason for marking as problematic
        """
        self._problematic_symbols[symbol] = time.time()
        duration_hours = self._problematic_symbol_duration/3600
        logger.warning(f"Symbol {symbol} marked as PROBLEMATIC (reason: {reason}). "
                      f"Will be skipped for {duration_hours:.1f} hours.")
        print(f"DEBUG: Symbol {symbol} added to blacklist (reason: {reason}, duration: {duration_hours:.1f}h)", flush=True)
        
        # Reset error count
        if symbol in self._api_error_counts:
            del self._api_error_counts[symbol]
    
    def _record_api_error(self, symbol: str) -> None:
        """
        Record API error for symbol. Automatically blacklist if threshold reached.
        
        Args:
            symbol: Trading symbol
        """
        self._api_error_counts[symbol] = self._api_error_counts.get(symbol, 0) + 1
        error_count = self._api_error_counts[symbol]
        
        logger.debug(f"API error recorded for {symbol} (count: {error_count}/{self._max_api_errors_before_blacklist})")
        
        if error_count >= self._max_api_errors_before_blacklist:
            self.mark_symbol_as_problematic(symbol, f"API errors ({error_count} consecutive)")
    
    def _reset_api_error_count(self, symbol: str) -> None:
        """Reset API error count for symbol (on successful API call)."""
        if symbol in self._api_error_counts:
            del self._api_error_counts[symbol]
    
    def has_hedge(self, symbol: str) -> bool:
        """Check if symbol has active hedge."""
        position = self.get_position_for_symbol(symbol)
        return position is not None and position.has_hedge
    
    def check_and_enforce_breakeven_sl_urgent(self, position_id: str, current_price: float, atr: float) -> bool:
        """
        Aggressively check and enforce breakeven SL for profitable positions.
        
        Called once per second until breakeven SL is properly set.
        
        Args:
            position_id: Position ID
            current_price: Current market price
            atr: Current ATR
            
        Returns:
            True if breakeven SL is now properly set, False if still needs checking
        """
        # Check cooldown for margin/leverage errors
        if position_id in self._sl_update_cooldown:
            cooldown_until = self._sl_update_cooldown[position_id]
            if time.time() < cooldown_until:
                # Still in cooldown, skip update
                remaining = cooldown_until - time.time()
                logger.debug(f"SL update cooldown active for {position_id} ({remaining/60:.1f} min remaining)")
                return False  # Still needs checking after cooldown
            else:
                # Cooldown expired, remove it
                del self._sl_update_cooldown[position_id]
                logger.debug(f"SL update cooldown expired for {position_id}, will retry")
        
        position = self.get_position(position_id)
        if not position or position.state != PositionState.OPEN:
            # Position no longer exists, remove from urgent check
            if position_id in self._urgent_breakeven_check:
                del self._urgent_breakeven_check[position_id]
            if position_id in self._sl_update_cooldown:
                del self._sl_update_cooldown[position_id]
            return True
        
        # Calculate current PnL
        if position.side == "LONG":
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity
        
        # Only enforce if position is profitable
        if pnl < self.risk_engine.sl_config.breakeven_trigger_usdt:
            # Position no longer profitable, remove from urgent check
            if position_id in self._urgent_breakeven_check:
                del self._urgent_breakeven_check[position_id]
            return True
        
        # Verify SL order exists on exchange
        exchange_orders = self.client.get_open_orders(position.symbol)
        sl_orders = [o for o in exchange_orders if o.order_type == "STOP_MARKET" and o.order_id == position.sl_order_id]
        
        # Calculate required breakeven SL price
        min_profit_usdt = self.risk_engine.sl_config.breakeven_trigger_usdt
        if position.quantity > 0:
            if position.side == "LONG":
                required_breakeven_sl = position.entry_price + (min_profit_usdt / position.quantity)
            else:  # SHORT
                required_breakeven_sl = position.entry_price - (min_profit_usdt / position.quantity)
            
            # Check if current SL is at breakeven or better
            sl_is_profitable = False
            if position.side == "LONG":
                # For LONG, SL must be >= breakeven (higher is better)
                sl_is_profitable = position.sl_price >= required_breakeven_sl - 0.0001
            else:  # SHORT
                # For SHORT, SL must be <= breakeven (lower is better)
                sl_is_profitable = position.sl_price <= required_breakeven_sl + 0.0001
            
            # If SL is not profitable, or order doesn't exist, force update
            if not sl_is_profitable or not sl_orders:
                logger.warning(f"URGENT: Position {position_id} profitable (PnL={pnl:.2f}) but SL not in breakeven. "
                             f"Current SL: {position.sl_price:.8f}, Required: {required_breakeven_sl:.8f}. "
                             f"Force updating...")
                
                # Force breakeven SL update
                # Mark as breakeven to allow smaller minimum distance
                result = self.order_manager.update_stop_loss(
                    symbol=position.symbol,
                    position_id=position_id,
                    new_stop_price=required_breakeven_sl,
                    quantity=position.quantity,
                    position_side=position.side,
                    current_price=current_price,
                    is_breakeven=True  # Allow smaller minimum distance for breakeven SL
                )
                
                if result.success:
                    position.sl_price = required_breakeven_sl
                    position.sl_order_id = result.order.order_id
                    logger.info(f"URGENT: Breakeven SL enforced for {position_id}: {required_breakeven_sl:.8f}")
                    # Remove from urgent check - breakeven is set, trailing will continue in normal mode (every 10s)
                    if position_id in self._urgent_breakeven_check:
                        del self._urgent_breakeven_check[position_id]
                        logger.debug(f"Breakeven SL set for {position_id}, removed from urgent check (trailing will continue in normal mode)")
                    return True
                else:
                    # Check if error is due to margin/leverage issues
                    error_code = result.error_code if hasattr(result, 'error_code') else ""
                    if error_code in ["MARGIN_INSUFFICIENT", "MAX_POSITION_EXCEEDED"]:
                        # For breakeven SL, use shorter cooldown (1 minute instead of 5)
                        # This allows faster retry for critical breakeven protection
                        breakeven_cooldown = 60.0  # 1 minute for breakeven SL
                        self._sl_update_cooldown[position_id] = time.time() + breakeven_cooldown
                        logger.warning(f"URGENT: Cannot update breakeven SL for {position_id} due to {error_code}. "
                                     f"Cooldown set for {breakeven_cooldown/60:.1f} minutes (reduced for breakeven). "
                                     f"Error: {result.error_message}")
                        # Keep in urgent check - will retry after shorter cooldown
                        # Don't remove from urgent check for breakeven SL
                        return False  # Still needs checking after cooldown
                    else:
                        logger.error(f"URGENT: Failed to enforce breakeven SL for {position_id}: {result.error_message}")
                        return False  # Still needs checking
            else:
                # SL is properly set, remove from urgent check
                if position_id in self._urgent_breakeven_check:
                    del self._urgent_breakeven_check[position_id]
                return True
        
        return True
    
    def get_active_position_ids(self, symbol: str) -> List[str]:
        """Get IDs of all active positions (including hedges) for symbol."""
        ids = []
        
        with self._positions_lock:
            for pos in self._positions.values():
                if pos.symbol == symbol and pos.state == PositionState.OPEN:
                    ids.append(pos.position_id)
                    if pos.hedge_id:
                        ids.append(pos.hedge_id)
        
        return ids
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """Get position tracking statistics."""
        with self._positions_lock:
            open_count = sum(1 for p in self._positions.values() if p.state == PositionState.OPEN)
            total_unrealized = sum(
                p.unrealized_pnl
                for p in self._positions.values()
                if p.state == PositionState.OPEN
            )
        
        return {
            "total_positions": self._total_positions,
            "open_positions": open_count,
            "closed_positions": self._total_positions - open_count,
            "winning_positions": self._winning_positions,
            "win_rate": self._winning_positions / max(1, self._total_positions - open_count),
            "total_realized_pnl": self._total_pnl,
            "total_unrealized_pnl": total_unrealized,
            "hedge_stats": self.hedge_manager.get_stats(),
            "daily_stats": self.risk_engine.get_daily_stats()
        }
