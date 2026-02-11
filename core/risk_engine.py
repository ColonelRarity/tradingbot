"""
Risk Engine

Manages all risk-related calculations:
- Position sizing based on risk budget
- Stop Loss calculation and updates
- Take Profit calculation using Fibonacci
- Daily P&L tracking
- Risk limit enforcement

CRITICAL RULES:
- SL moves to breakeven ASAP
- Worst-case outcome: small profit most of the time
- TP uses Fibonacci logic (second step from entry)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from core.market_data import MarketSnapshot
from exchange.binance_client import get_binance_client
from config.settings import (
    get_settings,
    RiskConfig,
    StopLossConfig,
    TakeProfitConfig,
    HedgeConfig
)


logger = logging.getLogger(__name__)


@dataclass
class RiskCalculation:
    """Result of risk calculation for a trade."""
    
    # Position sizing
    position_size_usdt: float
    position_size_qty: float
    
    # Stop Loss
    stop_loss_price: float
    stop_loss_percent: float
    max_loss_usdt: float
    
    # Take Profit
    take_profit_price: float
    take_profit_percent: float
    expected_profit_usdt: float
    
    # Risk/Reward
    risk_reward_ratio: float
    
    # Validation
    is_valid: bool
    reason: str


@dataclass
class StopLossUpdate:
    """Stop Loss update calculation."""
    
    new_sl_price: float
    previous_sl_price: float
    is_update_needed: bool
    is_breakeven: bool
    is_trailing: bool
    reason: str


@dataclass
class TakeProfitUpdate:
    """Take Profit update calculation."""
    
    new_tp_price: float
    previous_tp_price: float
    is_update_needed: bool
    fib_level_used: int
    reason: str


class DailyTracker:
    """Track daily P&L and trade counts."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset daily stats (call at UTC midnight)."""
        self.date = datetime.now(timezone.utc).date()
        self.total_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.trade_count: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.max_drawdown: float = 0.0
        self.peak_pnl: float = 0.0
    
    def check_date_change(self) -> bool:
        """Check if date changed and reset if needed."""
        today = datetime.now(timezone.utc).date()
        if today != self.date:
            self.reset()
            return True
        return False
    
    def add_trade(self, pnl: float) -> None:
        """Record a closed trade."""
        self.check_date_change()
        
        self.trade_count += 1
        self.realized_pnl += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Track drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        
        drawdown = self.peak_pnl - self.total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update with current unrealized P&L."""
        self.check_date_change()
        self.unrealized_pnl = unrealized_pnl
        self.total_pnl = self.realized_pnl + unrealized_pnl


class RiskEngine:
    """
    Risk Management Engine.
    
    Handles all risk calculations:
    - Position sizing
    - Stop Loss (with breakeven priority)
    - Take Profit (Fibonacci-based)
    - Daily limits
    """
    
    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        sl_config: Optional[StopLossConfig] = None,
        tp_config: Optional[TakeProfitConfig] = None,
        hedge_config: Optional[HedgeConfig] = None
    ):
        """
        Initialize risk engine.
        
        Args:
            risk_config: Risk configuration
            sl_config: Stop Loss configuration
            tp_config: Take Profit configuration
            hedge_config: Hedge configuration (for SL calculation)
        """
        settings = get_settings()
        self.risk_config = risk_config or settings.risk
        self.sl_config = sl_config or settings.stop_loss
        self.tp_config = tp_config or settings.take_profit
        self.hedge_config = hedge_config or settings.hedge
        
        # Daily tracking
        self.daily = DailyTracker()
        
        # Last SL/TP update times per symbol
        self._last_sl_update: Dict[str, float] = {}
        self._last_tp_update: Dict[str, float] = {}
        
        # Throttle daily limit warnings (once per 60 seconds)
        self._last_daily_limit_warning: float = 0.0
        self._daily_limit_warning_interval: float = 60.0
        
        logger.info("RiskEngine initialized")
    
    # ==================== Position Sizing ====================
    
    def calculate_position_size(
        self,
        available_balance: float,
        current_price: float,
        atr: float,
        side: str,  # "LONG" or "SHORT"
        leverage: int = 1,
        symbol: Optional[str] = None  # Symbol for max_qty check
    ) -> RiskCalculation:
        """
        Calculate risk-adjusted position size.
        
        Args:
            available_balance: Available USDT balance
            current_price: Current market price
            atr: Current ATR value
            side: Position side
            leverage: Leverage multiplier
            
        Returns:
            RiskCalculation with position size and SL/TP
        """
        # Check daily limits
        if not self._check_daily_limits():
            return RiskCalculation(
                position_size_usdt=0,
                position_size_qty=0,
                stop_loss_price=0,
                stop_loss_percent=0,
                max_loss_usdt=0,
                take_profit_price=0,
                take_profit_percent=0,
                expected_profit_usdt=0,
                risk_reward_ratio=0,
                is_valid=False,
                reason="DAILY_LIMIT_REACHED"
            )
        
        # Max position from balance
        max_position_usdt = available_balance * self.risk_config.max_position_percent

        # For small balances (< 1000 USDT), use much smaller position sizes to avoid poor RR ratios
        if available_balance < 1000:
            # Use smaller percentage for small accounts to get better RR ratios
            max_position_usdt = available_balance * 0.15  # Reduced from 0.3 to 0.15

        # Set minimum position size for testnet/small accounts
        min_position_size = 30.0  # Reduced minimum position size
        position_usdt = max(max_position_usdt, min_position_size)

        # Cap at available balance (can't use more than we have)
        position_usdt = min(position_usdt, available_balance * 0.8)  # Reduced from 0.9 to 0.8  # Leave 10% buffer

        # Fixed SL distance of ±10 USDT as requested
        # Since we want 10 USDT stop loss, the distance depends on position size
        # We'll calculate it after we have the position size
        sl_distance = (10.0 / position_usdt) * current_price if position_usdt > 0 else current_price * 0.01
        sl_percent = (sl_distance / current_price) * 100
        
        # Validate current price (must be positive and reasonable)
        MIN_PRICE = 0.0001  # Minimum reasonable price (adjust if needed)
        if current_price <= 0 or current_price < MIN_PRICE:
            logger.warning(f"Invalid current_price ({current_price}) for position sizing, skipping")
            return RiskCalculation(
                position_size_usdt=0,
                position_size_qty=0,
                stop_loss_price=0,
                stop_loss_percent=0,
                max_loss_usdt=0,
                take_profit_price=0,
                take_profit_percent=0,
                expected_profit_usdt=0,
                risk_reward_ratio=0,
                is_valid=False,
                reason="INVALID_PRICE"
            )
        
        position_qty = position_usdt / current_price
        
        # CRITICAL: Check max_qty BEFORE using quantity to prevent -4005 errors
        # Get max_qty from exchange info and adjust position_size_usdt if needed
        if symbol:
            try:
                client = get_binance_client()
                symbol_info = client.get_symbol_info(symbol)
                max_qty = symbol_info.get("max_qty", 1e15)
                
                # If calculated quantity exceeds max_qty, reduce position_size_usdt proportionally
                if position_qty > max_qty:
                    # Reduce position_size_usdt to fit within max_qty
                    position_usdt = max_qty * current_price
                    position_qty = max_qty
                    logger.debug(f"Quantity {position_qty:.2f} exceeds max_qty {max_qty:.2f} for {symbol}, "
                               f"reducing position_size_usdt to {position_usdt:.2f} USDT")
            except Exception as e:
                # If we can't get symbol info, continue with calculated quantity
                # _format_quantity will handle it later, but we log a warning
                logger.debug(f"Could not check max_qty for {symbol}: {e}, continuing with calculated quantity")
        
        # Note: Final quantity validation is handled by _format_quantity in binance_client.py
        # which uses the real max_qty from exchange info and handles step_size rounding.
        
        # With fixed 10 USDT SL, ensure it's wide enough for hedge to trigger at -7 USDT
        # Since hedge triggers at -7 USDT and SL is at -10 USDT, this should be fine
        # No adjustment needed as 10 USDT > 7 USDT required for hedge
        
        # Calculate SL price
        if side == "LONG":
            sl_price = current_price - sl_distance
        else:
            sl_price = current_price + sl_distance
        
        # Calculate actual max loss at SL
        actual_max_loss = position_usdt * sl_percent / 100
        
        # Calculate TP using Fibonacci
        tp_price = self._calculate_fibonacci_tp(
            current_price, atr, side, self.tp_config.default_fib_step
        )
        
        if side == "LONG":
            tp_percent = (tp_price - current_price) / current_price * 100
        else:
            tp_percent = (current_price - tp_price) / current_price * 100
        
        expected_profit = position_usdt * tp_percent / 100
        
        # Risk/Reward ratio
        rr_ratio = tp_percent / sl_percent if sl_percent > 0 else 0
        
        # Debug logging for small balances
        if available_balance < 1000:
            logger.debug(f"Small balance position sizing: balance={available_balance:.2f}, "
                        f"position_usdt={position_usdt:.2f}, sl_percent={sl_percent:.4f}%, "
                        f"tp_percent={tp_percent:.4f}%, rr_ratio={rr_ratio:.2f}")

        # Validate
        is_valid = True
        reason = "OK"

        if position_usdt < 10:  # Minimum position
            is_valid = False
            reason = "POSITION_TOO_SMALL"
        elif rr_ratio < 1.0:
            # For small positions with small balances, relax RR ratio requirement
            min_rr_required = 1.0
            if available_balance < 1000 and position_usdt < 200:
                min_rr_required = 0.005  # Ultra relaxed for small testnet positions
            if rr_ratio < min_rr_required:
                is_valid = False
                reason = f"POOR_RR_RATIO:{rr_ratio:.2f} (min: {min_rr_required:.2f})"
        elif tp_percent < self.risk_config.min_tp_percent:
            # For small positions, relax TP requirement if SL is very tight
            min_tp_required = self.risk_config.min_tp_percent
            if position_usdt < 100:  # Small positions get relaxed TP requirements
                min_tp_required = max(0.01, self.risk_config.min_tp_percent * 0.2)  # Very relaxed: at least 0.01%
            if tp_percent < min_tp_required:
                is_valid = False
                reason = f"TP_TOO_CLOSE:{tp_percent:.2f}% (min: {min_tp_required:.2f}%)"
        
        return RiskCalculation(
            position_size_usdt=position_usdt,
            position_size_qty=position_qty,
            stop_loss_price=sl_price,
            stop_loss_percent=sl_percent,
            max_loss_usdt=actual_max_loss,
            take_profit_price=tp_price,
            take_profit_percent=tp_percent,
            expected_profit_usdt=expected_profit,
            risk_reward_ratio=rr_ratio,
            is_valid=is_valid,
            reason=reason
        )
    
    # ==================== Stop Loss Logic ====================
    
    def calculate_sl_update(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        unrealized_pnl: float,
        atr: float,
        quantity: float  # Position quantity for USDT profit calculation
    ) -> StopLossUpdate:
        """
        Calculate Stop Loss update.
        
        PRIMARY GOAL: Move SL to breakeven as fast as possible.
        WORST-CASE: Small profit most of the time.
        
        Args:
            symbol: Trading symbol
            side: Position side
            entry_price: Entry price
            current_price: Current market price
            current_sl: Current SL price
            unrealized_pnl: Current unrealized P&L
            atr: Current ATR
            
        Returns:
            StopLossUpdate with new SL details
        """
        # Always calculate if update is needed
        now = time.time()
        last_update = self._last_sl_update.get(symbol, 0)
        interval_passed = (now - last_update) >= self.sl_config.recalc_interval_sec
        
        new_sl = current_sl
        is_breakeven = False
        is_trailing = False
        reason = "NO_CHANGE"
        
        # Calculate profit percent
        if side == "LONG":
            profit_percent = (current_price - entry_price) / entry_price * 100
            # Check if in profit
            if profit_percent > 0:
                # BREAKEVEN: Guarantee minimum profit (10 USDT as requested)
                # When position reaches trigger profit, set SL to guarantee minimum profit
                if unrealized_pnl >= self.sl_config.breakeven_trigger_usdt or \
                   profit_percent >= self.sl_config.breakeven_trigger_percent:

                    # Calculate SL price that guarantees minimum profit in USDT
                    # For LONG: PnL = (exit_price - entry_price) * quantity
                    # We want: min_profit_usdt = (sl_price - entry_price) * quantity
                    # Therefore: sl_price = entry_price + (min_profit_usdt / quantity)
                    min_profit_usdt = self.sl_config.breakeven_trigger_usdt  # Use trigger as minimum profit
                    if quantity > 0:
                        breakeven_sl = entry_price + (min_profit_usdt / quantity)

                        # CRITICAL: SL can NEVER move lower than it was before (only higher for LONG)
                        # This prevents widening losses
                        if breakeven_sl > current_sl:
                            new_sl = breakeven_sl
                            is_breakeven = True
                            reason = f"BREAKEVEN_TRIGGERED:PnL={unrealized_pnl:.2f},min_profit={min_profit_usdt:.2f}USDT"
                
                # TRAILING: Move SL closer to current price as profit increases
                # But always guarantee minimum profit (10 USDT)
                if unrealized_pnl >= self.sl_config.trail_activation_profit_usdt:
                    trail_distance = atr * self.sl_config.trail_distance_atr_mult
                    trailing_sl = current_price - trail_distance

                    # Calculate minimum SL that guarantees min profit
                    min_profit_usdt = self.sl_config.breakeven_trigger_usdt
                    if quantity > 0:
                        min_sl_for_profit = entry_price + (min_profit_usdt / quantity)
                        # Use the better of trailing SL or minimum profit SL (higher is better for LONG)
                        trailing_sl = max(trailing_sl, min_sl_for_profit)

                    # CRITICAL: Trailing SL can NEVER move lower than current SL (only higher for LONG)
                    # This ensures trailing only locks in more profit, never widens losses
                    if trailing_sl > new_sl:
                        new_sl = trailing_sl
                        is_trailing = True
                        reason = f"TRAILING_SL:distance={trail_distance:.4f}"
        
        else:  # SHORT
            profit_percent = (entry_price - current_price) / entry_price * 100

            if profit_percent > 0:
                # BREAKEVEN: Guarantee minimum profit (10 USDT as requested)
                # When position reaches trigger profit, set SL to guarantee minimum profit
                if unrealized_pnl >= self.sl_config.breakeven_trigger_usdt or \
                   profit_percent >= self.sl_config.breakeven_trigger_percent:

                    # Calculate SL price that guarantees minimum profit in USDT
                    # For SHORT: PnL = (entry_price - exit_price) * quantity
                    # We want: min_profit_usdt = (entry_price - sl_price) * quantity
                    # Therefore: sl_price = entry_price - (min_profit_usdt / quantity)
                    min_profit_usdt = self.sl_config.breakeven_trigger_usdt  # Use trigger as minimum profit
                    if quantity > 0:
                        breakeven_sl = entry_price - (min_profit_usdt / quantity)

                        # CRITICAL: SL can NEVER move higher than it was before (only lower for SHORT)
                        # This prevents widening losses
                        if breakeven_sl < current_sl:
                            new_sl = breakeven_sl
                            is_breakeven = True
                            reason = f"BREAKEVEN_TRIGGERED:PnL={unrealized_pnl:.2f},min_profit={min_profit_usdt:.2f}USDT"
                
                # TRAILING: Move SL closer to current price as profit increases
                # But always guarantee minimum profit (10 USDT)
                if unrealized_pnl >= self.sl_config.trail_activation_profit_usdt:
                    trail_distance = atr * self.sl_config.trail_distance_atr_mult
                    trailing_sl = current_price + trail_distance

                    # Calculate minimum SL that guarantees min profit
                    min_profit_usdt = self.sl_config.breakeven_trigger_usdt
                    if quantity > 0:
                        min_sl_for_profit = entry_price - (min_profit_usdt / quantity)
                        # Use the better of trailing SL or minimum profit SL (lower is better for SHORT)
                        trailing_sl = min(trailing_sl, min_sl_for_profit)

                    # CRITICAL: Trailing SL can NEVER move higher than current SL (only lower for SHORT)
                    # This ensures trailing only locks in more profit, never widens losses
                    if trailing_sl < new_sl:
                        new_sl = trailing_sl
                        is_trailing = True
                        reason = f"TRAILING_SL:distance={trail_distance:.4f}"
        
        # Check if update needed
        is_update_needed = abs(new_sl - current_sl) > 0.0001
        
        # Only enforce interval for non-critical updates (if no breakeven/trailing)
        if is_update_needed and not is_breakeven and not is_trailing and not interval_passed:
            return StopLossUpdate(
                new_sl_price=current_sl,
                previous_sl_price=current_sl,
                is_update_needed=False,
                is_breakeven=False,
                is_trailing=False,
                reason="UPDATE_COOLDOWN"
            )
        
        if is_update_needed:
            self._last_sl_update[symbol] = now
        
        return StopLossUpdate(
            new_sl_price=new_sl,
            previous_sl_price=current_sl,
            is_update_needed=is_update_needed,
            is_breakeven=is_breakeven,
            is_trailing=is_trailing,
            reason=reason
        )
    
    # ==================== Take Profit Logic ====================
    
    def calculate_tp_update(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_tp: float,
        atr: float,
        volatility_regime: float
    ) -> TakeProfitUpdate:
        """
        Calculate Take Profit update using Fibonacci logic.
        
        TP adapts to volatility regime:
        - Low volatility: Use closer Fibonacci level
        - High volatility: Use further Fibonacci level
        
        Args:
            symbol: Trading symbol
            side: Position side
            entry_price: Entry price
            current_price: Current market price
            current_tp: Current TP price
            atr: Current ATR
            volatility_regime: Volatility level (0-1)
            
        Returns:
            TakeProfitUpdate with new TP details
        """
        # Check update interval
        now = time.time()
        last_update = self._last_tp_update.get(symbol, 0)
        if now - last_update < self.tp_config.recalc_interval_sec:
            return TakeProfitUpdate(
                new_tp_price=current_tp,
                previous_tp_price=current_tp,
                is_update_needed=False,
                fib_level_used=self.tp_config.default_fib_step,
                reason="UPDATE_COOLDOWN"
            )
        
        # Select Fibonacci level based on volatility
        if volatility_regime < self.tp_config.volatility_threshold_low:
            fib_step = self.tp_config.low_volatility_fib_step
        elif volatility_regime > self.tp_config.volatility_threshold_high:
            fib_step = self.tp_config.high_volatility_fib_step
        else:
            fib_step = self.tp_config.default_fib_step
        
        # Calculate new TP
        new_tp = self._calculate_fibonacci_tp(entry_price, atr, side, fib_step)
        
        # Validate TP distance from current price
        if side == "LONG":
            distance_percent = (new_tp - current_price) / current_price * 100
        else:
            distance_percent = (current_price - new_tp) / current_price * 100
        
        reason = f"FIB_LEVEL_{fib_step}"
        
        # TP must not be too close to current price
        if distance_percent < self.tp_config.min_distance_from_current_price_percent:
            # Adjust TP further
            min_distance = current_price * self.tp_config.min_distance_from_current_price_percent / 100
            if side == "LONG":
                new_tp = current_price + min_distance
            else:
                new_tp = current_price - min_distance
            reason = "ADJUSTED_MIN_DISTANCE"
        
        # Cap TP at max distance
        max_distance = entry_price * self.tp_config.max_distance_percent / 100
        if side == "LONG":
            max_tp = entry_price + max_distance
            if new_tp > max_tp:
                new_tp = max_tp
                reason = "CAPPED_MAX_DISTANCE"
        else:
            min_tp = entry_price - max_distance
            if new_tp < min_tp:
                new_tp = min_tp
                reason = "CAPPED_MAX_DISTANCE"
        
        is_update_needed = abs(new_tp - current_tp) / current_tp > 0.001  # 0.1% change threshold
        
        if is_update_needed:
            self._last_tp_update[symbol] = now
        
        return TakeProfitUpdate(
            new_tp_price=new_tp,
            previous_tp_price=current_tp,
            is_update_needed=is_update_needed,
            fib_level_used=fib_step,
            reason=reason
        )
    
    def _calculate_fibonacci_tp(
        self,
        entry_price: float,
        atr: float,
        side: str,
        fib_step: int
    ) -> float:
        """
        Calculate TP using Fibonacci extension.
        
        Args:
            entry_price: Entry price
            atr: ATR for range calculation
            side: Position side
            fib_step: Fibonacci level index (0-5)
            
        Returns:
            Take profit price
        """
        fib_levels = self.tp_config.fib_levels
        
        # Ensure valid step
        fib_step = max(0, min(fib_step, len(fib_levels) - 1))
        fib_level = fib_levels[fib_step]
        
        # Calculate range using ATR
        tp_range = atr * self.risk_config.atr_tp_multiplier * (1 + fib_level)
        
        if side == "LONG":
            tp_price = entry_price + tp_range
        else:
            tp_price = entry_price - tp_range
        
        return tp_price
    
    # ==================== Daily Limits ====================
    
    def _check_daily_limits(self) -> bool:
        """Check if daily limits allow new trades."""
        self.daily.check_date_change()
        
        now = time.time()
        should_warn = (now - self._last_daily_limit_warning) >= self._daily_limit_warning_interval
        
        # Check trade count (MAX_TRADES_PER_DAY <= 0 means unlimited)
        if self.risk_config.max_trades_per_day > 0 and self.daily.trade_count >= self.risk_config.max_trades_per_day:
            if should_warn:
                logger.warning(f"Daily trade limit reached: {self.daily.trade_count}")
                self._last_daily_limit_warning = now
            return False
        
        # Check max loss
        if self.daily.realized_pnl <= -self.risk_config.daily_max_loss_usdt:
            if should_warn:
                logger.warning(f"Daily loss limit reached: {self.daily.realized_pnl:.2f}")
                self._last_daily_limit_warning = now
            return False
        
        # Check maximum unrealized loss (prevents opening new positions when drawdown is high)
        if self.daily.unrealized_pnl <= -self.risk_config.max_unrealized_loss_usdt:
            if should_warn:
                logger.warning(f"Maximum unrealized loss reached: {self.daily.unrealized_pnl:.2f} USDT")
                self._last_daily_limit_warning = now
            return False
        
        return True
    
    def record_trade(self, pnl: float) -> None:
        """Record a closed trade for daily tracking."""
        self.daily.add_trade(pnl)
        
        logger.info(
            f"Trade recorded: PnL={pnl:.2f} | "
            f"Daily: {self.daily.realized_pnl:.2f} | "
            f"Trades: {self.daily.trade_count}"
        )
    
    def get_daily_stats(self) -> Dict:
        """Get daily statistics."""
        self.daily.check_date_change()
        
        max_trades = self.risk_config.max_trades_per_day
        if max_trades > 0:
            remaining_trades = max(0, max_trades - self.daily.trade_count)
        else:
            remaining_trades = -1

        return {
            "date": str(self.daily.date),
            "total_pnl": self.daily.total_pnl,
            "realized_pnl": self.daily.realized_pnl,
            "trade_count": self.daily.trade_count,
            "winning_trades": self.daily.winning_trades,
            "losing_trades": self.daily.losing_trades,
            "win_rate": self.daily.winning_trades / max(1, self.daily.trade_count),
            "max_drawdown": self.daily.max_drawdown,
            "remaining_trades": remaining_trades,
            "remaining_loss_budget": self.risk_config.daily_max_loss_usdt + self.daily.realized_pnl
        }
