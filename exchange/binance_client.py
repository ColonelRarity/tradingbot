"""
Binance USDT-M Futures Testnet Client

Uses binance-futures-connector for Testnet API ONLY.
No Demo API, no python-binance Client.

Responsibilities:
- Connection management
- Rate limiting
- Order execution (MARKET, STOP_MARKET, TAKE_PROFIT_MARKET)
- Position queries
- Account balance
- Candle data fetching
"""

from __future__ import annotations

import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from collections import deque

import requests
from binance.um_futures import UMFutures
from binance.error import ClientError, ServerError, ParameterRequiredError

from config.settings import get_settings, ExchangeConfig


logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    """Position side for hedge mode."""
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class AccountBalance:
    """Account balance information."""
    total_balance: float
    available_balance: float
    unrealized_pnl: float
    margin_balance: float


@dataclass
class Position:
    """Open position information."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float  # Positive for LONG, negative for SHORT
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float
    margin_type: str


@dataclass
class Order:
    """Order information."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    status: str
    price: float
    stop_price: float
    quantity: float
    executed_qty: float
    time: int


@dataclass
class Candle:
    """OHLCV candle data."""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int


class RateLimiter:
    """
    Thread-safe rate limiter for Binance API.
    
    Limits:
    - 1200 requests per minute (we use 1000 conservatively)
    - 50 orders per 10 seconds
    - 10 orders per second
    """
    
    def __init__(self, config: ExchangeConfig):
        self.max_requests_per_minute = config.max_requests_per_minute
        self.max_orders_per_10sec = config.max_orders_per_10sec
        self.max_orders_per_second = config.max_orders_per_second
        
        self._request_times: deque = deque()
        self._order_times_10s: deque = deque()
        self._order_times_1s: deque = deque()
        self._lock = Lock()
        
        self._total_requests = 0
        self._total_waited = 0.0
    
    def wait_if_needed(self, is_order: bool = False) -> float:
        """Wait if rate limits would be exceeded. Returns wait time."""
        with self._lock:
            now = time.time()
            wait_time = 0.0
            
            # Clean old requests (older than 1 minute)
            while self._request_times and (now - self._request_times[0]) > 60:
                self._request_times.popleft()
            
            # Check per-minute limit
            if len(self._request_times) >= self.max_requests_per_minute:
                oldest = self._request_times[0]
                wait_time = max(wait_time, 60 - (now - oldest) + 0.1)
            
            if is_order:
                # Clean old orders
                while self._order_times_10s and (now - self._order_times_10s[0]) > 10:
                    self._order_times_10s.popleft()
                while self._order_times_1s and (now - self._order_times_1s[0]) > 1:
                    self._order_times_1s.popleft()
                
                # Check order limits
                if len(self._order_times_10s) >= self.max_orders_per_10sec:
                    oldest = self._order_times_10s[0]
                    wait_time = max(wait_time, 10 - (now - oldest) + 0.1)
                
                if len(self._order_times_1s) >= self.max_orders_per_second:
                    oldest = self._order_times_1s[0]
                    wait_time = max(wait_time, 1 - (now - oldest) + 0.1)
            
            if wait_time > 0:
                self._total_waited += wait_time
                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                now = time.time()
            
            # Record this request
            self._request_times.append(now)
            self._total_requests += 1
            
            if is_order:
                self._order_times_10s.append(now)
                self._order_times_1s.append(now)
            
            return wait_time


class BinanceClient:
    """
    Binance USDT-M Futures Client.
    
    Defaults to Testnet; production requires explicit opt-in.
    Uses binance-futures-connector UMFutures.
    """
    
    def __init__(self, config: Optional[ExchangeConfig] = None):
        """
        Initialize Binance client for Testnet.
        
        Args:
            config: Exchange configuration (uses global settings if None)
        """
        self.config = config or get_settings().exchange
        
        # Validate production opt-in
        if "testnet" not in self.config.base_url.lower() and not self.config.allow_production:
            raise ValueError(
                "Production URL set but ALLOW_PRODUCTION_TRADING is not enabled."
            )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.config)
        
        # Initialize UMFutures client
        self._client = UMFutures(
            key=self.config.api_key,
            secret=self.config.api_secret,
            base_url=self.config.base_url
        )
        
        # Cache for symbol info
        self._symbol_info: Dict[str, Dict] = {}
        self._exchange_info_loaded = False
        self._dry_run_order_id = 10_000_000
        self._dry_run_orders: Dict[int, Order] = {}
        
        mode = "Testnet" if "testnet" in self.config.base_url.lower() else "Production"
        dry_run_label = " DRY_RUN" if self.config.dry_run else ""
        logger.info(f"BinanceClient initialized for {mode}{dry_run_label}: {self.config.base_url}")

    def _next_dry_run_order_id(self) -> int:
        """Generate a unique order ID for dry-run mode."""
        self._dry_run_order_id += 1
        return self._dry_run_order_id

    def _record_dry_run_order(self, order: Order) -> None:
        """Store simulated order for dry-run open order queries."""
        self._dry_run_orders[order.order_id] = order

    def _clear_dry_run_order(self, order_id: int) -> None:
        """Remove simulated order."""
        if order_id in self._dry_run_orders:
            del self._dry_run_orders[order_id]
    
    def _api_call(self, func, *args, is_order: bool = False, **kwargs) -> Any:
        """
        Execute API call with rate limiting and error handling.
        
        Args:
            func: API function to call
            *args: Positional arguments
            is_order: Whether this is an order-related call
            **kwargs: Keyword arguments
            
        Returns:
            API response
            
        Raises:
            ClientError: API client error
            ServerError: API server error
        """
        self.rate_limiter.wait_if_needed(is_order=is_order)
        
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            # Network/DNS error - retry once after delay
            logger.warning(f"Connection error (network/DNS): {e}. Retrying once after 1 second...")
            time.sleep(1.0)  # Wait 1 second before retry
            try:
                return func(*args, **kwargs)
            except Exception as retry_error:
                logger.error(f"Connection error retry failed: {retry_error}")
                raise
        except requests.exceptions.Timeout as e:
            # Request timeout - retry once
            logger.warning(f"Request timeout: {e}. Retrying once after 0.5 seconds...")
            time.sleep(0.5)
            try:
                return func(*args, **kwargs)
            except Exception as retry_error:
                logger.error(f"Timeout retry failed: {retry_error}")
                raise
        except ClientError as e:
            # Don't log expected errors
            if e.error_code not in [-4046, -2011, -2021, -4028]:  # -4046: margin already set, -2011: order not found, -2021: immediate trigger, -4028: leverage not valid (handled in set_leverage)
                # Log but don't treat as ERROR for some cases
                if e.error_code in [-4131, -4005]:  # PERCENT_PRICE, Quantity too large
                    logger.warning(f"API ClientError: {e.error_code} - {e.error_message}")
                else:
                    logger.error(f"API ClientError: {e.error_code} - {e.error_message}")
            raise
        except ServerError as e:
            logger.error(f"API ServerError: {e.status_code} - {e.message}")
            raise
    
    # ==================== Account Methods ====================
    
    def get_account_balance(self) -> AccountBalance:
        """
        Get account balance information.
        
        Returns:
            AccountBalance with total, available, unrealized PnL, margin
        """
        response = self._api_call(self._client.account)
        
        # Find USDT balance
        usdt_balance = None
        for asset in response.get("assets", []):
            if asset["asset"] == "USDT":
                usdt_balance = asset
                break
        
        if not usdt_balance:
            raise ValueError("USDT balance not found in account")
        
        return AccountBalance(
            total_balance=float(usdt_balance["walletBalance"]),
            available_balance=float(usdt_balance["availableBalance"]),
            unrealized_pnl=float(usdt_balance["unrealizedProfit"]),
            margin_balance=float(usdt_balance["marginBalance"])
        )
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            List of Position objects (only non-zero positions)
        """
        response = self._api_call(self._client.get_position_risk, symbol=symbol)
        
        positions = []
        for pos in response:
            size = float(pos["positionAmt"])
            if size == 0:
                continue  # Skip zero positions
            
            positions.append(Position(
                symbol=pos["symbol"],
                side="LONG" if size > 0 else "SHORT",
                size=size,
                entry_price=float(pos.get("entryPrice", 0)),
                mark_price=float(pos.get("markPrice", 0)),
                unrealized_pnl=float(pos.get("unRealizedProfit", 0)),
                leverage=int(pos.get("leverage", 1)),
                liquidation_price=float(pos.get("liquidationPrice", 0)),
                margin_type=pos.get("marginType", "CROSSED")
            ))
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position or None if no position
        """
        positions = self.get_positions(symbol)
        return positions[0] if positions else None
    
    # ==================== Order Methods ====================
    
    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False
    ) -> Order:
        """
        Place MARKET order for entry.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            reduce_only: Whether this is reduce-only
            
        Returns:
            Order object with execution details
        """
        # Format and validate quantity (this should already handle max_qty)
        formatted_qty = self._format_quantity(symbol, quantity)
        qty_float = float(formatted_qty)
        
        # Double-check max quantity (should already be handled in _format_quantity, but verify)
        info = self.get_symbol_info(symbol)
        max_qty = info.get("max_qty", 1e15)
        if qty_float > max_qty:
            # If still exceeds, use max_qty and reformat
            formatted_qty = self._format_quantity(symbol, max_qty)
            qty_float = float(formatted_qty)
            logger.warning(f"Quantity capped to max_qty for {symbol}: {formatted_qty} (requested: {quantity})")
        
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": "MARKET",
            "quantity": formatted_qty,
        }
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        logger.debug(f"Placing MARKET order: {side.value} {formatted_qty} {symbol}")

        if self.config.dry_run:
            price = 0.0
            try:
                price = self.get_ticker_price(symbol)
            except Exception:
                pass
            order = Order(
                order_id=self._next_dry_run_order_id(),
                client_order_id=f"dry_run_{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                side=side.value,
                order_type="MARKET",
                status="FILLED",
                price=price,
                stop_price=0.0,
                quantity=float(formatted_qty),
                executed_qty=float(formatted_qty),
                time=int(time.time() * 1000)
            )
            logger.info(f"[DRY_RUN] MARKET {side.value} {formatted_qty} {symbol} @ {price}")
            return order
        
        try:
            response = self._api_call(
                self._client.new_order,
                is_order=True,
                **params
            )
            
            return self._parse_order(response)
        except ClientError as e:
            # Handle specific errors gracefully
            if e.error_code == -4131:  # PERCENT_PRICE filter
                logger.warning(f"PERCENT_PRICE filter violation for {symbol}")
                raise
            elif e.error_code == -4005:  # Quantity too large
                # Quantity exceeds max_qty - cannot place order
                # This should not happen if _format_quantity works correctly
                # But if it does, the symbol likely has very restrictive limits
                logger.error(f"Cannot place order for {symbol}: quantity {formatted_qty} exceeds max_qty. "
                           f"This symbol may not be tradeable with current position sizing.")
                raise
            raise
    
    def place_stop_market(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = False
    ) -> Order:
        """
        Place STOP_MARKET order for Stop Loss.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            stop_price: Trigger price for stop
            quantity: Order quantity (optional if close_position=True)
            close_position: Whether to close entire position
            
        Returns:
            Order object
        """
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": "STOP_MARKET",
            "stopPrice": self._format_price(symbol, stop_price),
            "workingType": "MARK_PRICE",
        }
        
        if close_position:
            params["closePosition"] = "true"
        elif quantity:
            params["quantity"] = self._format_quantity(symbol, quantity)
        else:
            raise ValueError("Either quantity or close_position must be specified")
        
        logger.info(f"Placing STOP_MARKET: {side.value} @ {stop_price} {symbol}")

        if self.config.dry_run:
            qty_value = float(self._format_quantity(symbol, quantity)) if quantity else 0.0
            order = Order(
                order_id=self._next_dry_run_order_id(),
                client_order_id=f"dry_run_{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                side=side.value,
                order_type="STOP_MARKET",
                status="NEW",
                price=0.0,
                stop_price=stop_price,
                quantity=qty_value,
                executed_qty=0.0,
                time=int(time.time() * 1000)
            )
            self._record_dry_run_order(order)
            logger.info(f"[DRY_RUN] STOP_MARKET {side.value} {symbol} @ {stop_price} qty={qty_value} closePosition={close_position}")
            return order
        
        response = self._api_call(
            self._client.new_order,
            is_order=True,
            **params
        )
        
        return self._parse_order(response)
    
    def place_take_profit_market(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = False
    ) -> Order:
        """
        Place TAKE_PROFIT_MARKET order for Take Profit.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            stop_price: Trigger price for take profit
            quantity: Order quantity (optional if close_position=True)
            close_position: Whether to close entire position
            
        Returns:
            Order object
        """
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self._format_price(symbol, stop_price),
            "workingType": "MARK_PRICE",
        }
        
        if close_position:
            params["closePosition"] = "true"
        elif quantity:
            params["quantity"] = self._format_quantity(symbol, quantity)
        else:
            raise ValueError("Either quantity or close_position must be specified")
        
        logger.info(f"Placing TAKE_PROFIT_MARKET: {side.value} @ {stop_price} {symbol}")

        if self.config.dry_run:
            qty_value = float(self._format_quantity(symbol, quantity)) if quantity else 0.0
            order = Order(
                order_id=self._next_dry_run_order_id(),
                client_order_id=f"dry_run_{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                side=side.value,
                order_type="TAKE_PROFIT_MARKET",
                status="NEW",
                price=0.0,
                stop_price=stop_price,
                quantity=qty_value,
                executed_qty=0.0,
                time=int(time.time() * 1000)
            )
            self._record_dry_run_order(order)
            logger.info(f"[DRY_RUN] TAKE_PROFIT_MARKET {side.value} {symbol} @ {stop_price} qty={qty_value} closePosition={close_position}")
            return order
        
        response = self._api_call(
            self._client.new_order,
            is_order=True,
            **params
        )
        
        return self._parse_order(response)
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """
        Cancel specific order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        # Validate order_id before making API call
        if not order_id or order_id <= 0:
            logger.debug(f"Skipping cancel for invalid order_id: {order_id} (symbol: {symbol})")
            return False
        
        if self.config.dry_run:
            self._clear_dry_run_order(order_id)
            logger.info(f"[DRY_RUN] Cancelled order {order_id} for {symbol}")
            return True

        try:
            self._api_call(
                self._client.cancel_order,
                is_order=True,
                symbol=symbol,
                orderId=order_id
            )
            logger.debug(f"Cancelled order {order_id} for {symbol}")
            return True
        except ClientError as e:
            error_msg = str(e.error_message) if hasattr(e, 'error_message') else str(e)
            # Order not found - this is normal (already filled/cancelled)
            if e.error_code == -2011:
                return True  # Don't log - it's expected
            # Handle "orderId is mandatory" error - indicates invalid order_id
            if "orderId is mandatory" in error_msg or "orderId is mandatory, but received empty" in error_msg:
                logger.debug(f"Invalid order_id {order_id} for {symbol} - skipping cancellation")
                return False
            raise
        except Exception as e:
            error_str = str(e)
            # Handle "orderId is mandatory" error from library validation
            if "orderId is mandatory" in error_str or "orderId is mandatory, but received empty" in error_str:
                logger.debug(f"Invalid order_id {order_id} for {symbol} - skipping cancellation (library validation)")
                return False
            raise
    
    def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Number of orders cancelled
        """
        if self.config.dry_run:
            to_remove = [oid for oid, order in self._dry_run_orders.items() if order.symbol == symbol]
            for oid in to_remove:
                self._clear_dry_run_order(oid)
            logger.info(f"[DRY_RUN] Cancelled {len(to_remove)} orders for {symbol}")
            return len(to_remove)

        try:
            response = self._api_call(
                self._client.cancel_open_orders,
                is_order=True,
                symbol=symbol
            )
            count = len(response) if isinstance(response, list) else 0
            logger.info(f"Cancelled {count} orders for {symbol}")
            return count
        except ClientError as e:
            if e.error_code == -2011:  # No orders to cancel
                return 0
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            List of open Order objects
        """
        if self.config.dry_run:
            if symbol:
                return [order for order in self._dry_run_orders.values() if order.symbol == symbol]
            return list(self._dry_run_orders.values())

        try:
            if symbol:
                response = self._api_call(self._client.get_open_orders, symbol=symbol)
            else:
                response = self._api_call(self._client.get_open_orders)
        except ParameterRequiredError as e:
            # Handle library bug where it incorrectly requires orderId
            # This can happen with some versions of binance-futures-connector
            if "orderId" in str(e):
                logger.warning(f"Library requires orderId parameter (library bug), returning empty list for {symbol or 'all symbols'}")
                return []
            raise
        except Exception as e:
            # Handle any other errors gracefully
            error_str = str(e)
            if "orderId is mandatory" in error_str or "orderId is mandatory, but received empty" in error_str:
                logger.warning(f"API requires orderId parameter (unexpected), returning empty list for {symbol or 'all symbols'}")
                return []
            raise
        
        orders = []
        for o in response:
            # Skip orders without orderId
            if not o.get("orderId"):
                continue
            try:
                parsed = self._parse_order(o)
                if parsed.order_id > 0:  # Only valid orders
                    orders.append(parsed)
            except Exception:
                continue
        
        return orders
    
    # ==================== Market Data Methods ====================
    
    def get_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500
    ) -> List[Candle]:
        """
        Get historical candles.
        
        Args:
            symbol: Trading symbol
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles (max 1500)
            
        Returns:
            List of Candle objects (oldest first)
        """
        response = self._api_call(
            self._client.klines,
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        candles = []
        for k in response:
            candles.append(Candle(
                open_time=k[0],
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=k[6],
                quote_volume=float(k[7]),
                trades=int(k[8])
            ))
        
        return candles
    
    def get_ticker_price(self, symbol: str) -> float:
        """
        Get current ticker price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current mark price
        """
        response = self._api_call(self._client.mark_price, symbol=symbol)
        return float(response["markPrice"])
    
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24h ticker statistics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with price change, volume, etc.
        """
        response = self._api_call(self._client.ticker_24hr_price_change, symbol=symbol)
        return response
    
    def get_all_tickers_24h(self) -> List[Dict[str, Any]]:
        """
        Get 24h ticker statistics for all symbols.
        
        Returns:
            List of ticker dicts
        """
        response = self._api_call(self._client.ticker_24hr_price_change)
        return response
    
    def get_top_usdt_pairs(
        self,
        min_volume: float = 10_000_000,
        max_pairs: int = 50
    ) -> List[str]:
        """
        Get top USDT futures pairs by 24h volume.
        
        Args:
            min_volume: Minimum 24h quote volume
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of symbol names sorted by volume (highest first)
        """
        try:
            tickers = self.get_all_tickers_24h()
            
            # Filter USDT pairs with sufficient volume
            usdt_pairs = []
            for t in tickers:
                symbol = t.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                
                volume = float(t.get("quoteVolume", 0))
                if volume < min_volume:
                    continue
                
                usdt_pairs.append({
                    "symbol": symbol,
                    "volume": volume,
                    "price_change": float(t.get("priceChangePercent", 0))
                })
            
            # Sort by volume (highest first)
            usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
            
            # Return top N symbols
            symbols = [p["symbol"] for p in usdt_pairs[:max_pairs]]
            
            logger.info(f"Found {len(symbols)} USDT pairs with volume > {min_volume:,.0f}")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get top pairs: {e}")
            return ["BTCUSDT", "ETHUSDT"]  # Fallback
    
    # ==================== Exchange Info Methods ====================
    
    def load_exchange_info(self) -> None:
        """Load and cache exchange info for all symbols."""
        if self._exchange_info_loaded:
            return
        
        response = self._api_call(self._client.exchange_info)
        
        for symbol_info in response.get("symbols", []):
            sym = symbol_info["symbol"]
            self._symbol_info[sym] = {
                "price_precision": symbol_info["pricePrecision"],
                "quantity_precision": symbol_info["quantityPrecision"],
                "min_qty": None,
                "step_size": None,
                "tick_size": None,
            }
            
            for f in symbol_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    self._symbol_info[sym]["min_qty"] = float(f["minQty"])
                    self._symbol_info[sym]["step_size"] = float(f["stepSize"])
                    self._symbol_info[sym]["max_qty"] = float(f.get("maxQty", 1e15))
                elif f["filterType"] == "PRICE_FILTER":
                    self._symbol_info[sym]["tick_size"] = float(f["tickSize"])
                elif f["filterType"] == "PERCENT_PRICE":
                    self._symbol_info[sym]["percent_price_multiplier_up"] = float(f.get("multiplierUp", 5.0))
                    self._symbol_info[sym]["percent_price_multiplier_down"] = float(f.get("multiplierDown", 0.2))
        
        self._exchange_info_loaded = True
        logger.info(f"Loaded exchange info for {len(self._symbol_info)} symbols")
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol trading info.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with precision, min_qty, step_size, tick_size
        """
        self.load_exchange_info()
        
        if symbol not in self._symbol_info:
            raise ValueError(f"Symbol {symbol} not found in exchange info")
        
        return self._symbol_info[symbol]
    
    # ==================== Leverage Methods ====================
    
    def set_leverage(self, symbol: str, leverage: int) -> int:
        """
        Set leverage for symbol with fallback for invalid values.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage value (1-125)
            
        Returns:
            Confirmed leverage value
        """
        # Try requested leverage first
        try:
            response = self._api_call(
                self._client.change_leverage,
                symbol=symbol,
                leverage=leverage
            )
            confirmed = int(response.get("leverage", leverage))
            logger.debug(f"Set leverage for {symbol}: {confirmed}x")
            return confirmed
        except ClientError as e:
            # If leverage is not valid, try common fallback values
            if e.error_code == -4028:  # Leverage not valid
                logger.warning(f"Leverage {leverage}x not valid for {symbol}, trying fallback values...")
                # Try common leverage values in descending order
                fallback_leverages = [10, 5, 3, 2, 1]
                for fallback in fallback_leverages:
                    if fallback >= leverage:
                        continue  # Skip if fallback is >= requested (would be same error)
                    try:
                        response = self._api_call(
                            self._client.change_leverage,
                            symbol=symbol,
                            leverage=fallback,
                            is_order=False  # Not an order, just config
                        )
                        confirmed = int(response.get("leverage", fallback))
                        logger.warning(f"Set leverage for {symbol} to {confirmed}x (fallback, requested {leverage}x)")
                        return confirmed
                    except ClientError:
                        continue  # Try next fallback
                
                # If all fallbacks failed, log and re-raise
                logger.error(f"Could not set any leverage for {symbol} (requested {leverage}x)")
                # Record API error for this symbol
                if hasattr(self, '_position_tracker') and self._position_tracker:
                    self._position_tracker._record_api_error(symbol)
                raise
            else:
                # Other errors - re-raise
                raise
    
    def set_margin_type(self, symbol: str, margin_type: Literal["ISOLATED", "CROSSED"]) -> bool:
        """
        Set margin type for symbol.
        
        Args:
            symbol: Trading symbol
            margin_type: ISOLATED or CROSSED
            
        Returns:
            True if successful
        """
        try:
            self._api_call(
                self._client.change_margin_type,
                symbol=symbol,
                marginType=margin_type
            )
            logger.info(f"Set margin type for {symbol}: {margin_type}")
            return True
        except ClientError as e:
            if e.error_code == -4046:  # Already set - this is fine, not an error
                # Don't log - it's expected
                return True
            raise
    
    # ==================== Helper Methods ====================
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol precision."""
        info = self.get_symbol_info(symbol)
        precision = info["price_precision"]
        return f"{price:.{precision}f}"
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol precision and limits."""
        info = self.get_symbol_info(symbol)
        precision = info["quantity_precision"]
        max_qty = info.get("max_qty", 1e15)
        step_size = info.get("step_size", 10 ** -precision)
        
        # CRITICAL: Cap quantity to max_qty FIRST (before any rounding)
        if quantity > max_qty:
            logger.debug(f"Capping quantity {quantity} to max_qty {max_qty} for {symbol}")
            quantity = max_qty
        
        # Round DOWN to step size
        if step_size and step_size > 0:
            # Calculate how many steps fit (always round DOWN using floor division)
            steps = int(quantity / step_size)  # int() truncates toward zero (rounds down for positive numbers)
            quantity = steps * step_size
            
            # Final safety check: ensure we didn't exceed max_qty after rounding
            # Use a small safety margin to account for floating point precision
            safety_margin = step_size * 0.1  # 10% of step size as safety margin
            if quantity > max_qty - safety_margin:
                # Round down one more step to be safe
                steps = max(0, int((max_qty - safety_margin) / step_size))
                quantity = steps * step_size
            
            # Absolute final check: if still exceeds, force to max valid step
            if quantity > max_qty:
                steps = max(0, int(max_qty / step_size))
                quantity = steps * step_size
        else:
            # No step_size - just cap to max_qty
            quantity = min(quantity, max_qty)
        
        # Ensure quantity is non-negative
        quantity = max(0.0, quantity)
        
        # Format and verify: parse back to ensure formatted value doesn't exceed max_qty
        formatted = f"{quantity:.{precision}f}"
        parsed_qty = float(formatted)
        
        # If formatted value exceeds max_qty, round down one more step
        if parsed_qty > max_qty and step_size and step_size > 0:
            # Round down one more step
            steps = max(0, int((max_qty - step_size) / step_size))
            quantity = steps * step_size
            formatted = f"{quantity:.{precision}f}"
        
        return formatted
    
    def validate_order_price(self, symbol: str, price: float) -> bool:
        """
        Validate order price against PERCENT_PRICE filter.
        
        Args:
            symbol: Trading symbol
            price: Order price to validate
            
        Returns:
            True if price is within PERCENT_PRICE limits
        """
        try:
            info = self.get_symbol_info(symbol)
            mark_price = self.get_ticker_price(symbol)
            
            multiplier_up = info.get("percent_price_multiplier_up", 5.0)
            multiplier_down = info.get("percent_price_multiplier_down", 0.2)
            
            max_price = mark_price * multiplier_up
            min_price = mark_price * multiplier_down
            
            return min_price <= price <= max_price
        except Exception:
            return True  # If we can't validate, allow it
    
    def _parse_order(self, response: Dict) -> Order:
        """Parse API response into Order object."""
        return Order(
            order_id=response.get("orderId", 0),
            client_order_id=response.get("clientOrderId", ""),
            symbol=response.get("symbol", ""),
            side=response.get("side", ""),
            order_type=response.get("type", ""),
            status=response.get("status", ""),
            price=float(response.get("price", 0)),
            stop_price=float(response.get("stopPrice", 0)),
            quantity=float(response.get("origQty", 0)),
            executed_qty=float(response.get("executedQty", 0)),
            time=response.get("time", 0)
        )
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if connection successful
        """
        try:
            self._api_call(self._client.ping)
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Global client instance
_client: Optional[BinanceClient] = None


def get_binance_client() -> BinanceClient:
    """Get global Binance client instance (singleton)."""
    global _client
    if _client is None:
        _client = BinanceClient()
    return _client


def reset_binance_client() -> None:
    """Reset global client (for testing)."""
    global _client
    _client = None
