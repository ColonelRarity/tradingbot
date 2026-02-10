"""
Self-Learning Binance USDT-M Futures Trading Bot

Multi-pair trading system that:
- Scans all USDT pairs every 12 hours
- Opens positions on best opportunities
- Monitors and manages existing positions continuously
- Learns from trade outcomes

TESTNET ONLY - No production trading.
"""

from __future__ import annotations

import os
import sys
import signal
import time
import logging
import threading
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Setup logging first
from utils.logging_config import init_logging, get_logger

# Core components
from config.settings import get_settings, Settings
from exchange.binance_client import BinanceClient, get_binance_client
from core.market_data import MarketData, get_market_data
from core.feature_engineering import FeatureEngine
from core.pattern_memory import PatternMemory
from core.signal_engine import SignalEngine, SignalDirection
from core.risk_engine import RiskEngine
from core.order_manager import OrderManager
from core.hedge_manager import HedgeManager
from core.position_tracker import PositionTracker, PositionState

# ML components
from ml.model import create_model
from ml.trainer import ModelTrainer
from ml.inference import ModelInference

# Telegram monitor
from telegram_monitor import create_telegram_monitor, TelegramMonitor


# Initialize logger
logger = get_logger(__name__)


class MultiPairTradingBot:
    """
    Multi-Pair Self-Learning Futures Trading Bot.
    
    Features:
    - Scans top USDT pairs by volume every 12 hours
    - Evaluates signals across multiple pairs
    - Opens positions on best opportunities
    - Monitors existing positions every 10 seconds
    - Learns from all trade outcomes
    """
    
    def __init__(self):
        """Initialize multi-pair trading bot."""
        self.settings = get_settings()
        
        # Validate settings
        errors = self.settings.validate()
        if errors:
            for err in errors:
                logger.error(f"Config error: {err}")
            raise ValueError("Configuration validation failed")
        
        logger.info("Initializing Multi-Pair Trading Bot")
        logger.info(f"Testnet URL: {self.settings.exchange.base_url}")
        
        # Initialize core components
        self._init_components()
        
        # Active symbols being traded
        self.active_symbols: List[str] = []
        self.opportunities_cache: Dict[str, float] = {}  # symbol -> confidence, for sorting
        self.market_data_cache: Dict[str, MarketData] = {}
        
        # State
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Timing
        self._last_market_scan = 0
        self._last_position_update = 0
        self._last_model_retrain = 0
        self._last_urgent_breakeven_check = 0
        self._last_signal_check = 0
        self._last_symbol_refresh = 0
        
        logger.info("Multi-Pair Trading Bot initialized")
    
    def _init_components(self) -> None:
        """Initialize all trading components."""
        # Exchange client
        self.client = get_binance_client()
        
        # Feature engineering
        self.feature_engine = FeatureEngine()
        
        # Pattern memory
        self.pattern_memory = PatternMemory()
        
        # ML components
        self.ml_inference = ModelInference()
        self.ml_trainer = ModelTrainer(pattern_memory=self.pattern_memory)
        
        # Signal engine
        self.signal_engine = SignalEngine(
            feature_engine=self.feature_engine,
            ml_inference=self.ml_inference
        )
        
        # Risk engine
        self.risk_engine = RiskEngine()
        
        # Order manager
        self.order_manager = OrderManager(client=self.client)
        
        # Hedge manager
        self.hedge_manager = HedgeManager(
            client=self.client,
            order_manager=self.order_manager
        )
        
        # Telegram monitor (create first)
        self.telegram_monitor = create_telegram_monitor()
        
        # Position tracker (with telegram monitor and ml_inference)
        self.position_tracker = PositionTracker(
            client=self.client,
            order_manager=self.order_manager,
            hedge_manager=self.hedge_manager,
            risk_engine=self.risk_engine,
            pattern_memory=self.pattern_memory,
            ml_inference=self.ml_inference,
            telegram_monitor=self.telegram_monitor
        )
    
    def start(self) -> None:
        """Start the multi-pair trading bot."""
        print("=" * 60, flush=True)
        print("Starting Multi-Pair Self-Learning Trading Bot", flush=True)
        print("Mode: TESTNET ONLY", flush=True)
        print(f"Scan interval: {self.settings.market_data.market_scan_interval/3600:.1f} hours", flush=True)
        print(f"Max pairs: {self.settings.market_data.max_pairs_to_scan}", flush=True)
        print("=" * 60, flush=True)
        
        # Test connection
        print("Testing connection...", flush=True)
        if not self.client.test_connection():
            logger.error("Failed to connect to exchange")
            return
        print("Connection OK!", flush=True)
        
        # Load exchange info
        print("Loading exchange info...", flush=True)
        self.client.load_exchange_info()
        print("Exchange info loaded!", flush=True)
        
        # Load ML model
        print("Loading ML model...", flush=True)
        self.ml_trainer.load_checkpoint()
        self.ml_inference.update_model(self.ml_trainer.model)
        print("ML model loaded!", flush=True)
        
        # Initial market scan
        print("\nPerforming initial market scan...", flush=True)
        self._scan_market()
        
        # Start main loop
        print("\nStarting main trading loop...", flush=True)
        self._running = True
        self._run_loop()
    
    def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")
        self._running = False
        self._shutdown_event.set()
    
    def _refresh_active_symbols(self) -> None:
        """
        Quickly refresh active symbols list every 5 minutes.
        Adds new opportunities and removes stale ones.
        """
        try:
            print("DEBUG: Refreshing active symbols list...", flush=True)

            # Get current top symbols (lighter version of market scan)
            client = self.client
            symbols = client.get_top_usdt_pairs(
                min_volume=self.settings.market_data.min_volume_24h,
                max_pairs=min(50, self.settings.market_data.max_pairs_to_scan // 2)  # Check fewer symbols
            )

            new_opportunities = []
            current_active = set(self.active_symbols)

            # Quick check for new opportunities
            for symbol in symbols[:20]:  # Check top 20 only
                try:
                    # Skip if already active
                    if symbol in current_active:
                        continue

                    # Quick market data check
                    if symbol not in self.market_data_cache:
                        self.market_data_cache[symbol] = MarketData(symbol, self.client)
                        if not self.market_data_cache[symbol].initialize():
                            continue

                    md = self.market_data_cache[symbol]
                    if not md._initialized:
                        continue

                    # Quick update
                    md.update()
                    snapshot = md.get_snapshot()
                    if not snapshot:
                        continue

                    # Generate signal
                    signal = self.signal_engine.generate_signal(
                        market_data=md,
                        has_open_position=False,
                        has_hedge=False
                    )

                    # Add if signal is good enough
                    if signal.is_valid and signal.direction != SignalDirection.NONE and signal.confidence > 0.2:
                        new_opportunities.append((symbol, signal.confidence))
                        print(f"DEBUG: Found new opportunity: {symbol} conf={signal.confidence:.3f}", flush=True)

                except Exception as e:
                    continue

            # Add top new opportunities to active list
            if new_opportunities:
                # Sort by confidence
                new_opportunities.sort(key=lambda x: x[1], reverse=True)

                # Add top 3 new opportunities
                added_count = 0
                for symbol, confidence in new_opportunities[:3]:
                    if symbol not in self.active_symbols:
                        self.active_symbols.append(symbol)
                        self.opportunities_cache[symbol] = confidence
                        added_count += 1
                        print(f"DEBUG: Added {symbol} to active symbols", flush=True)

                if added_count > 0:
                    # Re-sort active symbols by confidence
                    self.active_symbols.sort(key=lambda s: self.opportunities_cache.get(s, 0.0), reverse=True)

            # Remove stale symbols (low confidence, no recent activity)
            symbols_to_remove = []
            for symbol in self.active_symbols[:]:  # Copy to avoid modification during iteration
                confidence = self.opportunities_cache.get(symbol, 0.0)
                # Remove if confidence is very low and not in top opportunities
                if confidence < 0.1 and symbol not in [s for s, _ in new_opportunities[:5]]:
                    symbols_to_remove.append(symbol)

            for symbol in symbols_to_remove:
                self.active_symbols.remove(symbol)
                if symbol in self.opportunities_cache:
                    del self.opportunities_cache[symbol]
                print(f"DEBUG: Removed stale symbol: {symbol}", flush=True)

            print(f"DEBUG: Active symbols refreshed: {len(self.active_symbols)} total", flush=True)
            self._last_symbol_refresh = time.time()

        except Exception as e:
            print(f"DEBUG: Error refreshing symbols: {e}", flush=True)

    def _scan_market(self) -> None:
        """
        Scan market for trading opportunities using parallel workers.
        
        Gets top USDT pairs and initializes market data for each.
        """
        print("\n" + "=" * 60, flush=True)
        print("MARKET SCAN", flush=True)
        print("=" * 60, flush=True)
        
        # Get top pairs by volume
        symbols = self.client.get_top_usdt_pairs(
            min_volume=self.settings.market_data.min_volume_24h,
            max_pairs=self.settings.market_data.max_pairs_to_scan
        )
        
        print(f"Analyzing {len(symbols)} pairs with parallel workers...", flush=True)
        
        # Analyze symbols in parallel
        opportunities = []
        completed = 0
        
        def scan_symbol(symbol: str) -> Optional[Dict]:
            """Scan single symbol and return opportunity if found."""
            try:
                # Get or create market data
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = MarketData(symbol, self.client)
                
                md = self.market_data_cache[symbol]
                
                # Initialize if needed
                if not md._initialized:
                    if not md.initialize():
                        return None
                else:
                    md.update()
                
                snapshot = md.get_snapshot()
                if not snapshot:
                    return None
                
                # Check if we already have position
                has_position = self.position_tracker.has_open_position(symbol)
                has_hedge = self.position_tracker.has_hedge(symbol)
                
                # Generate signal
                signal = self.signal_engine.generate_signal(
                    market_data=md,
                    has_open_position=has_position,
                    has_hedge=has_hedge
                )
                
                if signal.is_valid and signal.direction != SignalDirection.NONE:
                    # Setup trading params
                    self._setup_trading_params(symbol)
                    
                    return {
                        "symbol": symbol,
                        "signal": signal,
                        "snapshot": snapshot,
                        "confidence": signal.confidence
                    }
                
                return None
                
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel scanning
        max_workers = min(10, len(symbols))  # Max 10 parallel workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(scan_symbol, sym): sym for sym in symbols}
            
            for future in as_completed(future_to_symbol):
                completed += 1
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result()
                    if result:
                        opportunities.append(result)
                        direction = result['signal'].direction.value
                        conf = result['confidence']
                        print(f"  [{completed}/{len(symbols)}] {symbol}: {direction} (conf: {conf:.3f})", flush=True)
                
                except Exception as e:
                    logger.debug(f"Future error for {symbol}: {e}")
                
                # Progress indicator
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{len(symbols)}...", flush=True)
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        
        print(f"\nFound {len(opportunities)} opportunities", flush=True)
        
        # Show top opportunities
        if opportunities:
            print("\nTop opportunities:", flush=True)
            for opp in opportunities[:10]:
                direction = opp['signal'].direction.value
                conf = opp['confidence']
                print(f"  {opp['symbol']}: {direction} (conf: {conf:.3f})", flush=True)
        
        # Cache confidence for sorting
        self.opportunities_cache = {opp["symbol"]: opp["confidence"] for opp in opportunities}
        
        # CRITICAL: Re-verify confidence for top opportunities before adding to active_symbols
        # This ensures we only try to open positions in symbols that still have good signals
        top_opportunities = opportunities[:20]  # Top 20 from scan
        verified_top_symbols = []
        
        logger.info(f"Re-verifying confidence for top {len(top_opportunities)} opportunities...")
        for opp in top_opportunities:
            symbol = opp["symbol"]
            cached_confidence = opp["confidence"]
            
            # Skip if already has open position
            open_positions = self.position_tracker.get_open_positions()
            if any(pos.symbol == symbol for pos in open_positions):
                continue
            
            # Re-check signal and confidence
            try:
                if symbol in self.market_data_cache:
                    md = self.market_data_cache[symbol]
                    md.update()
                    snapshot = md.get_snapshot()
                    
                    if snapshot:
                        signal = self.signal_engine.generate_signal(
                            market_data=md,
                            has_open_position=False,
                            has_hedge=False
                        )
                        
                        # Only add if signal is valid and confidence is still good
                        # Use a threshold to ensure we only take good opportunities
                        min_confidence_threshold = 0.15  # Minimum confidence to open position
                        if signal.is_valid and signal.confidence >= min_confidence_threshold:
                            verified_top_symbols.append(symbol)
                            # Update cached confidence with fresh value
                            self.opportunities_cache[symbol] = signal.confidence
                            logger.debug(f"Verified {symbol}: confidence={signal.confidence:.3f} (cached: {cached_confidence:.3f})")
                        else:
                            logger.debug(f"Skipping {symbol}: signal invalid or confidence too low ({signal.confidence:.3f})")
                    else:
                        logger.debug(f"Skipping {symbol}: no snapshot available")
                else:
                    logger.debug(f"Skipping {symbol}: not in market_data_cache")
            except Exception as e:
                logger.warning(f"Error re-verifying {symbol}: {e}")
                # On error, skip this symbol (better safe than sorry)
                continue
        
        # Update active symbols list with verified top opportunities
        self.active_symbols = verified_top_symbols
        
        # Add symbols with existing positions (with low confidence to prioritize new opportunities)
        for pos in self.position_tracker.get_open_positions():
            if pos.symbol not in self.active_symbols:
                self.active_symbols.append(pos.symbol)
                # Use low confidence so existing positions don't interfere with new opportunity selection
                if pos.symbol not in self.opportunities_cache:
                    self.opportunities_cache[pos.symbol] = 0.0
        
        # Sort active_symbols by confidence (highest first) for priority opening
        self.active_symbols.sort(key=lambda s: self.opportunities_cache.get(s, 0.0), reverse=True)
        
        logger.info(f"Active symbols after verification: {len(self.active_symbols)} (top opportunities: {len(verified_top_symbols)})")
        
        print(f"\nActive symbols: {len(self.active_symbols)}", flush=True)
        print("=" * 60 + "\n", flush=True)
        
        self._last_market_scan = time.time()
    
    def _calculate_adaptive_leverage(self, confidence: float) -> int:
        """
        Calculate adaptive leverage based on confidence.
        
        Uses linear interpolation between min and max leverage:
        - confidence 0.0 → min_leverage (5x)
        - confidence 1.0 → max_leverage (default, 20x)
        
        Args:
            confidence: Signal confidence (0-1)
            
        Returns:
            Leverage value (5-20x)
        """
        settings = get_settings()
        max_leverage = settings.exchange.default_leverage
        min_leverage = 5  # Minimum leverage for low confidence
        
        # Linear interpolation: leverage = min + (max - min) * confidence
        leverage = min_leverage + (max_leverage - min_leverage) * confidence
        
        # Round to nearest integer and ensure within bounds
        leverage = max(min_leverage, min(max_leverage, int(round(leverage))))
        
        return leverage
    
    def _setup_trading_params(self, symbol: str, leverage: Optional[int] = None) -> None:
        """
        Setup leverage and margin type for symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage to set (uses default if None)
        """
        try:
            if leverage is None:
                leverage = get_settings().exchange.default_leverage
            self.client.set_leverage(symbol, leverage)
            self.client.set_margin_type(symbol, "CROSSED")
        except Exception:
            pass  # Ignore errors (already set, etc.)
    
    def _run_loop(self) -> None:
        """
        Main trading loop.

        Every iteration:
        1. Update existing positions (10s)
        2. Check for new signals on active pairs
        3. Market scan every 12 hours
        4. Model retraining every hour
        """
        logger.info("Main trading loop started")
        iteration = 0

        while self._running:
            try:
                iteration += 1
                now = time.time()

                # Debug log every 10 iterations
                if iteration % 10 == 0:
                    logger.debug(f"Main loop iteration {iteration}, active_symbols: {len(self.active_symbols)}, time: {now:.0f}")

                # Safety check - if we've been running too long without progress, log it
                if iteration > 100 and (iteration % 50) == 0:
                    positions = self.position_tracker.get_open_positions()
                    logger.info(f"Status check: {len(positions)} open positions, balance: {self.client.get_account_balance().available_balance:.2f} USDT")
                
                # === Symbol Refresh (every 5 minutes) ===
                if now - self._last_symbol_refresh >= 300:  # 5 minutes
                    print("DEBUG: Starting symbol refresh...", flush=True)
                    self._refresh_active_symbols()
                    self._last_symbol_refresh = now

                # === Market Scan (every 12 hours) ===
                if now - self._last_market_scan >= self.settings.market_data.market_scan_interval:
                    self._scan_market()
                
                # === Position Management (every 10s) ===
                if now - self._last_position_update >= self.settings.stop_loss.recalc_interval_sec:
                    logger.debug("Starting position management...")
                    start_time = time.time()
                    self._manage_all_positions()
                    position_mgmt_duration = time.time() - start_time
                    logger.debug(f"Position management completed in {position_mgmt_duration:.2f}s")
                    self._last_position_update = now

                # === Aggressive Breakeven SL Check (every 1s for urgent positions) ===
                if now - self._last_urgent_breakeven_check >= 1.0:
                    self._check_urgent_breakeven_sl()
                    self._last_urgent_breakeven_check = now
                
                # === Check Signals on Active Symbols ===
                if now - self._last_signal_check >= 30.0:
                    logger.debug(f"Checking signals for {len(self.active_symbols)} active symbols")
                    start_time = time.time()
                    try:
                        self._check_signals_multi()
                        signal_check_duration = time.time() - start_time
                        logger.debug(f"Signal check completed in {signal_check_duration:.2f}s")
                    except Exception as e:
                        logger.error(f"Signal check failed: {e}")
                    self._last_signal_check = now

                # === Symbol Refresh (every 5 minutes) ===
                if now - self._last_symbol_refresh >= 300:  # 5 minutes
                    logger.debug("Starting symbol refresh...")
                    self._refresh_active_symbols()
                    self._last_symbol_refresh = now

                # === Market Scan (every 12 hours) ===
                if now - self._last_market_scan >= self.settings.market_data.market_scan_interval:
                    self._scan_market()

                # === Model Retraining (every hour) ===
                if now - self._last_model_retrain >= 3600:
                    self._retrain_model()
                    self._last_model_retrain = now

                # === Status Log (every 60 iterations) ===
                if iteration % 60 == 0:
                    self._log_status()
                    self._send_telegram_status()

                # Sleep
                time.sleep(self.settings.market_data.ticker_interval)
                
            except KeyboardInterrupt:
                print("DEBUG: Keyboard interrupt received", flush=True)
                logger.info("Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                print(f"DEBUG: Error in main loop: {e}", flush=True)
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        print(f"DEBUG: Main loop exited after {iteration} iterations", flush=True)
    
    def _manage_all_positions(self) -> None:
        """Manage all open positions across all symbols."""
        positions = self.position_tracker.get_open_positions()
        
        if not positions:
            return
        
        for position in positions:
            try:
                # Handle problematic symbols with repeated errors
                # As per user requirements: if position has multiple errors, try to close with minimal loss
                if self.position_tracker.is_problematic_symbol(position.symbol):
                    # Track consecutive errors for this position
                    error_key = f"{position.symbol}_{position.position_id}"
                    consecutive_errors = getattr(position, '_consecutive_errors', 0)
                    position._consecutive_errors = consecutive_errors + 1

                    # If position has 3+ consecutive errors, try emergency close
                    if consecutive_errors >= 3:
                        logger.warning(f"[EMERGENCY_CLOSE] {position.symbol}: Position {position.position_id} has {consecutive_errors} consecutive errors, attempting emergency close")
                        try:
                            # Get current price for emergency close
                            if position.symbol in self.market_data_cache:
                                md = self.market_data_cache[position.symbol]
                                md.update()
                                snapshot = md.get_snapshot()
                                current_price = snapshot.current_price if snapshot else position.current_price

                            # Emergency close with minimal loss allowed (as per user requirements)
                            self.position_tracker.close_position(position.position_id, current_price, "EMERGENCY_CLOSE_DUE_TO_ERRORS")
                            continue
                        except Exception as close_error:
                            logger.error(f"[EMERGENCY_CLOSE_FAILED] {position.symbol}: Failed to emergency close {position.position_id}: {close_error}")
                            # Continue with basic P&L update only

                # Also check if position has accumulated too many API errors
                position_error_count = getattr(position, '_api_errors', 0)
                if position_error_count >= 5:  # If position itself has 5+ errors
                    logger.warning(f"[POSITION_ERROR_CLOSE] {position.symbol}: Position {position.position_id} has {position_error_count} API errors, closing")
                    try:
                        current_price = position.current_price
                        self.position_tracker.close_position(position.position_id, current_price, "POSITION_API_ERRORS")
                        continue
                    except Exception as close_error:
                        logger.error(f"[POSITION_CLOSE_FAILED] {position.symbol}: Failed to close error-prone position {position.position_id}: {close_error}")

                    # Still update basic P&L for existing positions, but skip SL/TP updates to avoid errors
                    logger.debug(f"[SKIP] {position.symbol}: Position {position.position_id} on problematic symbol ({consecutive_errors} errors), skipping detailed updates")
                    # Still update basic P&L
                    if position.symbol in self.market_data_cache:
                        md = self.market_data_cache[position.symbol]
                        md.update()
                        snapshot = md.get_snapshot()
                        if snapshot:
                            if position.side == "LONG":
                                pnl = (snapshot.current_price - position.entry_price) * position.quantity
                            else:
                                pnl = (position.entry_price - snapshot.current_price) * position.quantity
                            position.unrealized_pnl = pnl
                    # Increment position error counter
                    if not hasattr(position, '_api_errors'):
                        position._api_errors = 0
                    position._api_errors += 1
                    continue

                # Get market data for this symbol
                if position.symbol not in self.market_data_cache:
                    self.market_data_cache[position.symbol] = MarketData(
                        position.symbol, self.client
                    )
                    self.market_data_cache[position.symbol].initialize()
                
                md = self.market_data_cache[position.symbol]
                md.update()
                snapshot = md.get_snapshot()
                
                if not snapshot:
                    continue
                
                # Get exchange position for verification and logging
                exchange_positions = self.client.get_positions(position.symbol)
                exchange_pos = exchange_positions[0] if exchange_positions else None
                
                # CRITICAL: Always verify position side from exchange (source of truth)
                # This catches cases where position was opened with wrong side or changed externally
                if exchange_pos:
                    actual_side = "LONG" if exchange_pos.size > 0 else "SHORT"
                    actual_size = abs(exchange_pos.size)
                    actual_entry = exchange_pos.entry_price
                    
                    # First, verify if hedge actually exists by checking hedge manager
                    hedge = self.position_tracker.hedge_manager.get_hedge_for_parent(position.position_id)
                    hedge_exists = hedge is not None and hedge.is_open
                    
                    # Also check by comparing NET size with parent size
                    net_size = actual_size
                    parent_size = abs(position.quantity)
                    size_diff = abs(net_size - parent_size)
                    hedge_size_threshold = parent_size * 0.3  # Hedge is typically 30-50% of parent
                    hedge_exists_by_size = size_diff >= hedge_size_threshold
                    
                    # Log position details for debugging
                    logger.debug(f"[SYNC] {position.symbol}: Position {position.position_id} | "
                               f"Tracked: {position.side} {parent_size:.6f} @ {position.entry_price:.8f} | "
                               f"Exchange: {actual_side} {net_size:.6f} @ {actual_entry:.8f} | "
                               f"Hedge: {hedge_exists} (by_size: {hedge_exists_by_size})")
                    
                    # CRITICAL: If side doesn't match and no hedge, this is a serious error
                    if position.side != actual_side:
                        if position.has_hedge and (hedge_exists or hedge_exists_by_size):
                            # Hedge exists - NET side can be different, this is expected
                            logger.debug(f"[SYNC] {position.symbol}: Hedge exists (hedge_id={position.hedge_id}), "
                                       f"NET side {actual_side} != tracked side {position.side} (expected)")
                        else:
                            # No hedge but side mismatch - CRITICAL ERROR
                            logger.error(f"[CRITICAL] {position.symbol}: Position {position.position_id} SIDE MISMATCH! "
                                       f"Tracked: {position.side}, Exchange: {actual_side}. "
                                       f"This is a serious error - correcting to match exchange!")
                            
                            # Update to match exchange (source of truth)
                            old_side = position.side
                            position.side = actual_side
                            position.quantity = actual_size
                            position.entry_price = actual_entry
                            
                            logger.warning(f"[CRITICAL] {position.symbol}: Corrected position {position.position_id} "
                                         f"from {old_side} to {actual_side} to match exchange")
                            
                            # If has_hedge flag was set but hedge doesn't exist, clear it
                            if position.has_hedge:
                                logger.warning(f"[CRITICAL] {position.symbol}: Clearing has_hedge flag for {position.position_id} "
                                             f"(hedge doesn't exist but side mismatch occurred)")
                                position.has_hedge = False
                                position.hedge_id = None
                    else:
                        # Side matches - verify hedge status
                        if position.has_hedge and not (hedge_exists or hedge_exists_by_size):
                            # has_hedge flag is set but hedge doesn't exist
                            logger.info(f"[SYNC] {position.symbol}: Hedge {position.hedge_id} doesn't exist "
                                      f"(NET={net_size:.6f}, parent={parent_size:.6f}), updating position.")
                            position.has_hedge = False
                            position.hedge_id = None
                        elif not position.has_hedge and (hedge_exists or hedge_exists_by_size):
                            # Hedge exists but flag not set - update flag
                            logger.info(f"[SYNC] {position.symbol}: Hedge exists but flag not set, updating.")
                            if hedge_exists:
                                position.has_hedge = True
                                position.hedge_id = hedge.hedge_id
                
                # Log NET position info for hedge verification
                if position.has_hedge and exchange_pos:
                    net_size = abs(exchange_pos.size)
                    parent_size = abs(position.quantity)
                    logger.debug(f"[POSITION] {position.symbol}: Parent size={parent_size:.6f}, NET size={net_size:.6f}, "
                               f"hedge_id={position.hedge_id}, has_hedge={position.has_hedge}")
                
                # Calculate PnL
                if position.side == "LONG":
                    pnl = (snapshot.current_price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - snapshot.current_price) * position.quantity

                position.unrealized_pnl = pnl

                # Reset position error counter on successful update
                if hasattr(position, '_api_errors'):
                    position._api_errors = 0

                # Mark hedge positions in output
                position_label = position.side
                if position.has_hedge:
                    # Verify hedge actually exists before showing [HEDGE] label
                    hedge = self.hedge_manager.get_hedge_for_parent(position.position_id)
                    if hedge and hedge.is_open:
                        position_label = f"{position.side}[HEDGE]"
                    else:
                        # Hedge flag is set but hedge doesn't exist - clear the flag
                        logger.warning(f"[DISPLAY] {position.symbol}: Position has has_hedge=True but hedge doesn't exist. Clearing flag.")
                        position.has_hedge = False
                        position.hedge_id = None
                        position_label = position.side
                
                print(f"[{position.symbol}] {position_label} | "
                      f"Entry: {position.entry_price:.8f} | "
                      f"Now: {snapshot.current_price:.8f} | "
                      f"PnL: {pnl:.2f} USDT", flush=True)
                
                # Update position (SL/TP management)
                self.position_tracker.update_position(
                    position_id=position.position_id,
                    current_price=snapshot.current_price,
                    atr=snapshot.atr,
                    volatility_regime=snapshot.volatility
                )
                
                # Check hedge for this parent position
                if position.has_hedge and position.hedge_id:
                    self.hedge_manager.check_hedge_close(
                        position.hedge_id,
                        snapshot.current_price
                    )
                
                # Reconcile with exchange
                self.position_tracker.reconcile_with_exchange(position.symbol)
                
            except Exception as e:
                import traceback
                logger.error(f"Error managing position {position.position_id}: {e}\n{traceback.format_exc()}")
        
        # Update unrealized PnL for daily limit checking
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        self.risk_engine.daily.update_unrealized(total_unrealized)
    
    def _check_urgent_breakeven_sl(self) -> None:
        """
        Aggressively check and enforce breakeven SL for profitable positions.
        
        Called once per second to ensure profitable positions have breakeven SL set.
        """
        # Get list of positions that need urgent checking
        urgent_positions = list(self.position_tracker._urgent_breakeven_check.keys())
        
        if not urgent_positions:
            return
        
        for position_id in urgent_positions[:]:  # Copy list to avoid modification during iteration
            try:
                position = self.position_tracker.get_position(position_id)
                if not position or position.state != PositionState.OPEN:
                    continue
                
                # Get market data
                if position.symbol not in self.market_data_cache:
                    continue
                
                md = self.market_data_cache[position.symbol]
                md.update()
                snapshot = md.get_snapshot()
                
                if not snapshot:
                    continue
                
                # Check and enforce breakeven SL
                self.position_tracker.check_and_enforce_breakeven_sl_urgent(
                    position_id=position_id,
                    current_price=snapshot.current_price,
                    atr=snapshot.atr
                )
                
            except Exception as e:
                logger.error(f"Error in urgent breakeven check for {position_id}: {e}")
        
        # Check all active hedges separately (in case parent was closed or hedge wasn't linked properly)
        active_hedges = self.hedge_manager.get_all_active_hedges()
        for hedge in active_hedges:
            try:
                # Get market data for hedge symbol
                if hedge.symbol not in self.market_data_cache:
                    self.market_data_cache[hedge.symbol] = MarketData(
                        hedge.symbol, self.client
                    )
                    self.market_data_cache[hedge.symbol].initialize()
                
                md = self.market_data_cache[hedge.symbol]
                md.update()
                snapshot = md.get_snapshot()
                
                if snapshot:
                    # Check if hedge should be closed
                    self.hedge_manager.check_hedge_close(
                        hedge.hedge_id,
                        snapshot.current_price
                    )
            except Exception as e:
                logger.debug(f"Failed to check hedge {hedge.hedge_id}: {e}")
    
    def _check_signals_multi(self) -> None:
        """Check signals across all active symbols."""
        logger.debug(f"Starting signal check for {len(self.active_symbols)} symbols")
        try:
            # HARD CHECK: Get real positions from exchange first
            exchange_positions = self.client.get_positions()
            real_position_count = len([p for p in exchange_positions if abs(p.size) > 0.0001])
            logger.debug(f"Got {real_position_count} positions from exchange")
        except Exception as e:
            logger.warning(f"Failed to get positions from exchange: {e}, skipping signal check")
            return

        max_positions = self.settings.risk.max_concurrent_positions

        # HARD LIMIT: Don't even check signals if at limit
        # For small balances, reduce max concurrent positions
        balance = self.client.get_account_balance()
        if balance.available_balance < 1000:
            # With small balance, limit to 3 max concurrent positions
            max_positions = min(max_positions, 3)

        if real_position_count >= max_positions:
            return

        # Also check tracked positions
        tracked_positions = self.position_tracker.get_open_positions()
        tracked_count = len(tracked_positions)

        # Use the maximum to be safe
        current_count = max(real_position_count, tracked_count)

        if current_count >= max_positions:
            return
        
        # Check each active symbol (limit to prevent overwhelming API)
        max_symbols_per_check = 5  # Process max 5 symbols per check
        symbols_to_check = self.active_symbols[:max_symbols_per_check]
        # Log problematic symbols
        problematic_symbols = [s for s in symbols_to_check if self.position_tracker.is_problematic_symbol(s)]
        if problematic_symbols:
            print(f"DEBUG: PROBLEMATIC SYMBOLS in batch: {problematic_symbols}", flush=True)

        print(f"DEBUG: Processing {len(symbols_to_check)}/{len(self.active_symbols)} symbols: {symbols_to_check}", flush=True)

        for i, symbol in enumerate(symbols_to_check):
            logger.debug(f"[{i+1}/{len(symbols_to_check)}] Checking {symbol}")
            # RE-CHECK before each iteration
            try:
                exchange_positions = self.client.get_positions()
                real_count = len([p for p in exchange_positions if abs(p.size) > 0.0001])
            except Exception as e:
                logger.debug(f"{symbol} - ERROR: Failed to get positions: {e}")
                continue

            if real_count >= max_positions:
                logger.debug(f"{symbol} - SKIPPED: Max positions reached ({real_count}/{max_positions})")
                break

            # Skip if already have position for this symbol
            if any(p.symbol == symbol for p in exchange_positions if abs(p.size) > 0.0001):
                logger.debug(f"{symbol} - SKIPPED: Already have exchange position")
                continue

            if self.position_tracker.has_open_position(symbol):
                logger.debug(f"{symbol} - SKIPPED: Tracker shows open position")
                continue

            # Check cooldown (prevent immediate reopening after closing)
            if self.position_tracker.is_in_cooldown(symbol):
                logger.debug(f"{symbol} - SKIPPED: In cooldown period")
                continue

            # Check problematic symbols blacklist
            if self.position_tracker.is_problematic_symbol(symbol):
                logger.debug(f"{symbol} - SKIPPED: Problematic symbol")
                continue
            
            try:
                # Get market data with error handling - skip if not in cache
                if symbol not in self.market_data_cache:
                    logger.debug(f"{symbol} - SKIPPED: Not in market data cache")
                    continue

                md = self.market_data_cache[symbol]

                # Quick check if market data is initialized
                if not hasattr(md, '_initialized') or not md._initialized:
                    logger.debug(f"{symbol} - SKIPPED: Market data not initialized")
                    continue

                # Update market data with timeout protection
                try:
                    md.update()
                    # Reset API error count on successful market data update
                    self.position_tracker._reset_api_error_count(symbol)
                except Exception as e:
                    logger.debug(f"{symbol} - ERROR: Failed to update market data: {e}")
                    # Record API error for this symbol
                    self.position_tracker._record_api_error(symbol)

                    # More aggressive error handling - mark as problematic after 2 errors
                    error_count = self.position_tracker._api_error_counts.get(symbol, 0)
                    if error_count >= 2:
                        logger.warning(f"{symbol} - MARKING AS PROBLEMATIC after {error_count} errors")
                        self.position_tracker.mark_symbol_as_problematic(symbol, f"Market data errors ({error_count})")

                    continue

                snapshot = md.get_snapshot()

                if not snapshot:
                    logger.debug(f"{symbol} - SKIPPED: No market snapshot available")
                    continue

                # Generate signal
                try:
                    signal = self.signal_engine.generate_signal(
                        market_data=md,
                        has_open_position=False,
                        has_hedge=False
                    )
                    logger.debug(f"{symbol} - Signal: {signal.direction} conf={signal.confidence:.3f}")
                except Exception as e:
                    logger.debug(f"{symbol} - ERROR: Failed to generate signal: {e}")
                    continue

                if not signal.is_valid or signal.direction == SignalDirection.NONE:
                    logger.debug(f"{symbol} - SKIPPED: Invalid signal or no direction")
                    continue

                # FINAL CHECK before opening
                try:
                    exchange_positions = self.client.get_positions()
                    final_count = len([p for p in exchange_positions if abs(p.size) > 0.0001])
                except Exception as e:
                    logger.debug(f"{symbol} - ERROR: Failed to get final position count: {e}")
                    continue

                if final_count >= max_positions:
                    logger.debug(f"{symbol} - SKIPPED: Position limit reached ({final_count}/{max_positions})")
                    break

                # Calculate position size
                try:
                    balance = self.client.get_account_balance()
                except Exception as e:
                    logger.debug(f"{symbol} - ERROR: Failed to get account balance: {e}")
                    continue

                # Calculate adaptive leverage based on confidence
                leverage = self._calculate_adaptive_leverage(signal.confidence)

                risk_calc = self.risk_engine.calculate_position_size(
                    available_balance=balance.available_balance,
                    current_price=snapshot.current_price,
                    atr=snapshot.atr,
                    side=signal.direction.value,
                    leverage=leverage,
                    symbol=symbol
                )

                if not risk_calc.is_valid:
                    logger.debug(f"{symbol} - SKIPPED: Risk calculation invalid - {risk_calc.reason}")
                    continue

                logger.info(f"{symbol} - Opening position: {signal.direction} {risk_calc.position_size_usdt:.2f} USDT")
                # Open position
                print(f"\n{'='*60}", flush=True)
                print(f">>> OPENING {signal.direction.value} on {symbol} <<<", flush=True)
                print(f"    Size: {risk_calc.position_size_usdt:.2f} USDT", flush=True)
                print(f"    Entry: {snapshot.current_price:.2f}", flush=True)
                print(f"    SL: {risk_calc.stop_loss_price:.2f}", flush=True)
                print(f"    TP: {risk_calc.take_profit_price:.2f}", flush=True)
                print(f"    Confidence: {signal.confidence:.3f}", flush=True)
                print(f"    Leverage: {leverage}x", flush=True)
                print(f"{'='*60}\n", flush=True)

                print(f"DEBUG: {symbol} - Setting up trading parameters...", flush=True)
                # Setup trading params with adaptive leverage
                try:
                    self._setup_trading_params(symbol, leverage=leverage)
                except Exception as e:
                    print(f"DEBUG: {symbol} - ERROR setting up trading parameters: {e}", flush=True)
                    # Record API error for this symbol
                    self.position_tracker._record_api_error(symbol)
                    continue

                print(f"DEBUG: {symbol} - Calling position_tracker.open_position...", flush=True)
                try:
                    position = self.position_tracker.open_position(
                        symbol=symbol,
                        side=signal.direction.value,
                        quantity=risk_calc.position_size_qty,
                        entry_price=snapshot.current_price,
                        risk_calc=risk_calc,
                        features=signal.features
                    )
                except Exception as e:
                    print(f"DEBUG: {symbol} - ERROR opening position: {e}", flush=True)
                    # Record API error for this symbol
                    self.position_tracker._record_api_error(symbol)

                    # Mark as problematic after position opening errors
                    error_count = self.position_tracker._api_error_counts.get(symbol, 0)
                    if error_count >= 1:  # Be more strict for position opening errors
                        print(f"DEBUG: {symbol} - MARKING AS PROBLEMATIC due to position opening error", flush=True)
                        self.position_tracker.mark_symbol_as_problematic(symbol, "Position opening failed")

                    continue

                if position:
                    print(f"DEBUG: {symbol} - SUCCESS: Position opened!", flush=True)
                    # Reset API error count on successful operation
                    self.position_tracker._reset_api_error_count(symbol)
                    logger.info(f"Position opened: {symbol} {signal.direction.value}")
                    # Notify Telegram
                    if hasattr(self, 'telegram_monitor') and self.telegram_monitor:
                        self.telegram_monitor.send_position_opened(
                            symbol=symbol,
                            side=signal.direction.value,
                            entry_price=snapshot.current_price,
                            quantity=risk_calc.position_size_qty,
                            size_usdt=risk_calc.position_size_usdt,
                            sl_price=risk_calc.stop_loss_price,
                            tp_price=risk_calc.take_profit_price,
                            confidence=signal.confidence
                        )
                
            except Exception as e:
                logger.error(f"Error checking signal for {symbol}: {e}")
    
    def _retrain_model(self) -> None:
        """Retrain ML model with new data."""
        pattern_count = self.pattern_memory.get_pattern_count()
        
        if pattern_count < self.settings.ml.min_samples_for_training:
            return
        
        logger.info("Starting model retraining...")
        
        try:
            metrics = self.ml_trainer.train(epochs=3)
            
            if metrics:
                last = metrics[-1]
                logger.info(f"Retraining: acc={last.accuracy:.4f}, f1={last.f1:.4f}")
            
            self.ml_inference.update_model(self.ml_trainer.model)
            self.pattern_memory.decay_weights()
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _log_status(self) -> None:
        """Log current bot status."""
        stats = self.position_tracker.get_stats()
        daily = self.risk_engine.get_daily_stats()
        
        print(f"\n[STATUS] Positions: {stats['open_positions']} | "
              f"Daily PnL: {daily['realized_pnl']:.2f} USDT | "
              f"Win Rate: {stats['win_rate']:.1%} | "
              f"Active Pairs: {len(self.active_symbols)}", flush=True)
    
    def _send_telegram_status(self) -> None:
        """Send status to Telegram."""
        if not self.telegram_monitor:
            return
        
        try:
            positions = self.position_tracker.get_open_positions()
            orders = []
            for symbol in set(p.symbol for p in positions):
                orders.extend(self.order_manager.get_active_orders(symbol))
            
            daily_stats = self.risk_engine.get_daily_stats()
            model_stats = self.ml_inference.get_stats()
            training_stats = self.ml_trainer.get_training_stats()
            
            balance = self.client.get_account_balance()
            
            self.telegram_monitor.send_status(
                positions=positions,
                orders=orders,
                daily_stats=daily_stats,
                model_stats=model_stats,
                training_stats=training_stats,
                balance=balance.total_balance,
                available_balance=balance.available_balance
            )
        except Exception as e:
            logger.debug(f"Failed to send Telegram status: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Pair Self-Learning Trading Bot")
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to trade (disables multi-pair mode)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Initialize logging first (before settings to set LOG_LEVEL)
    os.environ["LOG_LEVEL"] = args.log_level
    init_logging()
    
    settings = get_settings()
    mode = "TESTNET" if "testnet" in settings.exchange.base_url.lower() else "PRODUCTION"
    print("=" * 60, flush=True)
    print("Self-Learning Binance USDT-M Futures Trading Bot", flush=True)
    print(f"{mode} MODE", flush=True)
    print("=" * 60, flush=True)
    
    # Create and start bot
    bot = MultiPairTradingBot()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot.start()
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)
    
    print("Bot stopped", flush=True)


if __name__ == "__main__":
    main()
