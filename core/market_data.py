"""
Market Data Manager

Responsibilities:
- Fetch and cache 1m candles for ML inference
- Provide 10s ticker updates for trade management
- Store volatility, ATR, momentum, candle patterns, micro-trend
- Thread-safe data access
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from exchange.binance_client import BinanceClient, Candle, get_binance_client
from config.settings import get_settings, MarketDataConfig


logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """
    Point-in-time market snapshot with all derived metrics.
    
    This is the primary input for ML inference and trading decisions.
    """
    timestamp: int  # Unix timestamp ms
    symbol: str
    
    # Price data
    current_price: float
    bid: float = 0.0
    ask: float = 0.0
    
    # OHLCV (latest candle)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    
    # Volatility metrics
    atr: float = 0.0
    atr_percent: float = 0.0
    volatility: float = 0.0  # Standard deviation of returns
    
    # Momentum metrics
    momentum: float = 0.0  # Price change over momentum period
    momentum_percent: float = 0.0
    roc: float = 0.0  # Rate of change
    
    # Candle patterns
    body_percent: float = 0.0  # Body as % of range
    upper_wick_percent: float = 0.0
    lower_wick_percent: float = 0.0
    is_bullish: bool = True
    
    # Micro-trend (short-term direction)
    micro_trend: int = 0  # -1=down, 0=neutral, 1=up
    micro_trend_strength: float = 0.0
    
    # Additional context
    spread_percent: float = 0.0
    volume_ratio: float = 1.0  # Current vs average volume


@dataclass
class CandleBuffer:
    """Thread-safe circular buffer for candles."""
    
    max_size: int = 500
    _candles: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        self._candles = deque(maxlen=self.max_size)
        self._lock = threading.Lock()
    
    def add(self, candle: Candle) -> None:
        """Add candle to buffer."""
        with self._lock:
            # Avoid duplicates
            if self._candles and self._candles[-1].open_time == candle.open_time:
                self._candles[-1] = candle  # Update existing
            else:
                self._candles.append(candle)
    
    def add_many(self, candles: List[Candle]) -> None:
        """Add multiple candles."""
        with self._lock:
            for c in candles:
                if self._candles and self._candles[-1].open_time == c.open_time:
                    self._candles[-1] = c
                else:
                    self._candles.append(c)
    
    def get_latest(self, n: int = 1) -> List[Candle]:
        """Get latest n candles."""
        with self._lock:
            if n >= len(self._candles):
                return list(self._candles)
            return list(self._candles)[-n:]
    
    def get_all(self) -> List[Candle]:
        """Get all candles."""
        with self._lock:
            return list(self._candles)
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._candles)


class MarketData:
    """
    Market Data Manager for trading decisions.
    
    Collects and processes:
    - 1m candles for ML inference
    - Real-time ticker for trade management
    - Derived metrics (ATR, volatility, momentum, etc.)
    """
    
    def __init__(
        self,
        symbol: str,
        client: Optional[BinanceClient] = None,
        config: Optional[MarketDataConfig] = None
    ):
        """
        Initialize market data manager.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            client: Binance client instance
            config: Market data configuration
        """
        self.symbol = symbol
        self.client = client or get_binance_client()
        self.config = config or get_settings().market_data
        
        # Candle buffers for different intervals
        self._candle_buffers: Dict[str, CandleBuffer] = {
            "1m": CandleBuffer(max_size=self.config.lookback_candles),
        }
        
        # Add MTF buffers
        for interval in self.config.mtf_intervals:
            self._candle_buffers[interval] = CandleBuffer(max_size=100)
        
        # Latest snapshot
        self._latest_snapshot: Optional[MarketSnapshot] = None
        self._snapshot_lock = threading.Lock()
        
        # Price tracking for micro-trend
        self._recent_prices: deque = deque(maxlen=10)
        
        # Initialization flag
        self._initialized = False
        
        logger.info(f"MarketData initialized for {symbol}")
    
    def initialize(self) -> bool:
        """
        Load historical data and initialize metrics.
        
        Returns:
            True if successful
        """
        try:
            # Load 1m candles
            candles = self.client.get_candles(
                self.symbol,
                interval="1m",
                limit=self.config.lookback_candles
            )
            self._candle_buffers["1m"].add_many(candles)
            
            # Load MTF candles
            for interval in self.config.mtf_intervals:
                mtf_candles = self.client.get_candles(
                    self.symbol,
                    interval=interval,
                    limit=100
                )
                self._candle_buffers[interval].add_many(mtf_candles)
            
            # Create initial snapshot
            self._update_snapshot()
            
            self._initialized = True
            logger.info(f"MarketData initialized with {len(self._candle_buffers['1m'])} candles")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize market data: {e}")
            return False
    
    def update(self) -> MarketSnapshot:
        """
        Update market data with latest candle and ticker.
        
        Returns:
            Latest MarketSnapshot
        """
        try:
            # Get latest candles (last 2 to ensure we have current)
            candles = self.client.get_candles(self.symbol, interval="1m", limit=2)
            self._candle_buffers["1m"].add_many(candles)
            
            # Get current price
            current_price = self.client.get_ticker_price(self.symbol)
            self._recent_prices.append(current_price)
            
            # Update snapshot
            return self._update_snapshot(current_price)
            
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
            return self._latest_snapshot
    
    def get_snapshot(self) -> Optional[MarketSnapshot]:
        """Get latest market snapshot."""
        with self._snapshot_lock:
            return self._latest_snapshot
    
    def get_candles(self, interval: str = "1m", n: int = 100) -> List[Candle]:
        """
        Get candles for specific interval.
        
        Args:
            interval: Candle interval
            n: Number of candles
            
        Returns:
            List of candles (oldest first)
        """
        if interval not in self._candle_buffers:
            return []
        return self._candle_buffers[interval].get_latest(n)
    
    def get_ohlcv_arrays(self, n: int = 100) -> Dict[str, np.ndarray]:
        """
        Get OHLCV data as numpy arrays for ML.
        
        Args:
            n: Number of candles
            
        Returns:
            Dict with 'open', 'high', 'low', 'close', 'volume' arrays
        """
        candles = self.get_candles("1m", n)
        
        if not candles:
            return {}
        
        return {
            "open": np.array([c.open for c in candles]),
            "high": np.array([c.high for c in candles]),
            "low": np.array([c.low for c in candles]),
            "close": np.array([c.close for c in candles]),
            "volume": np.array([c.volume for c in candles]),
            "timestamp": np.array([c.open_time for c in candles]),
        }
    
    def _update_snapshot(self, current_price: Optional[float] = None) -> MarketSnapshot:
        """
        Create updated market snapshot with all metrics.
        
        Args:
            current_price: Current ticker price (if available)
            
        Returns:
            Updated MarketSnapshot
        """
        candles = self._candle_buffers["1m"].get_all()
        
        if not candles:
            return None
        
        latest = candles[-1]
        price = current_price or latest.close
        
        # Extract arrays for calculations
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes, self.config.atr_period)
        atr_percent = (atr / price * 100) if price > 0 else 0
        
        # Calculate volatility (std of returns)
        returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0])
        volatility = np.std(returns) * 100 if len(returns) > 0 else 0
        
        # Calculate momentum
        momentum_period = min(self.config.momentum_period, len(closes) - 1)
        if momentum_period > 0:
            momentum = closes[-1] - closes[-momentum_period - 1]
            momentum_percent = momentum / closes[-momentum_period - 1] * 100
            roc = momentum_percent
        else:
            momentum = 0
            momentum_percent = 0
            roc = 0
        
        # Candle patterns
        candle_range = latest.high - latest.low
        if candle_range > 0:
            body = abs(latest.close - latest.open)
            body_percent = body / candle_range * 100
            
            if latest.close >= latest.open:  # Bullish
                upper_wick = latest.high - latest.close
                lower_wick = latest.open - latest.low
            else:  # Bearish
                upper_wick = latest.high - latest.open
                lower_wick = latest.close - latest.low
            
            upper_wick_percent = upper_wick / candle_range * 100
            lower_wick_percent = lower_wick / candle_range * 100
        else:
            body_percent = 0
            upper_wick_percent = 0
            lower_wick_percent = 0
        
        is_bullish = latest.close >= latest.open
        
        # Micro-trend from recent prices
        micro_trend, micro_trend_strength = self._calculate_micro_trend()
        
        # Volume ratio
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        snapshot = MarketSnapshot(
            timestamp=int(time.time() * 1000),
            symbol=self.symbol,
            current_price=price,
            open=latest.open,
            high=latest.high,
            low=latest.low,
            close=latest.close,
            volume=latest.volume,
            atr=atr,
            atr_percent=atr_percent,
            volatility=volatility,
            momentum=momentum,
            momentum_percent=momentum_percent,
            roc=roc,
            body_percent=body_percent,
            upper_wick_percent=upper_wick_percent,
            lower_wick_percent=lower_wick_percent,
            is_bullish=is_bullish,
            micro_trend=micro_trend,
            micro_trend_strength=micro_trend_strength,
            volume_ratio=volume_ratio,
        )
        
        with self._snapshot_lock:
            self._latest_snapshot = snapshot
        
        return snapshot
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> float:
        """
        Calculate Average True Range.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(closes) < 2:
            return 0.0
        
        # True Range components
        tr1 = highs[1:] - lows[1:]  # High - Low
        tr2 = np.abs(highs[1:] - closes[:-1])  # |High - PrevClose|
        tr3 = np.abs(lows[1:] - closes[:-1])  # |Low - PrevClose|
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR as simple moving average of TR
        if len(true_range) >= period:
            atr = np.mean(true_range[-period:])
        else:
            atr = np.mean(true_range)
        
        return float(atr)
    
    def _calculate_micro_trend(self) -> tuple[int, float]:
        """
        Calculate micro-trend from recent price updates.
        
        Returns:
            (direction: -1/0/1, strength: 0-1)
        """
        if len(self._recent_prices) < 3:
            return 0, 0.0
        
        prices = list(self._recent_prices)
        
        # Count up and down moves
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        total_moves = up_moves + down_moves
        
        if total_moves == 0:
            return 0, 0.0
        
        if up_moves > down_moves:
            direction = 1
            strength = (up_moves - down_moves) / total_moves
        elif down_moves > up_moves:
            direction = -1
            strength = (down_moves - up_moves) / total_moves
        else:
            direction = 0
            strength = 0.0
        
        return direction, strength


# Global instance
_market_data: Dict[str, MarketData] = {}
_market_data_lock = threading.Lock()


def get_market_data(symbol: str) -> MarketData:
    """
    Get or create MarketData instance for symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        MarketData instance
    """
    with _market_data_lock:
        if symbol not in _market_data:
            _market_data[symbol] = MarketData(symbol)
        return _market_data[symbol]
