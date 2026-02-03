"""
Feature Engineering Module

Extracts ML-ready features from market data:
- Rolling window of engineered features
- Volatility regime detection
- Momentum indicators
- Candle body/wick analysis
- Micro-trend direction

Output is normalized feature vectors for model input.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from core.market_data import MarketData, MarketSnapshot
from config.settings import get_settings, MLConfig


logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """
    Engineered feature vector for ML model input.
    
    Contains normalized features ready for inference.
    """
    timestamp: int
    symbol: str
    
    # Feature array (normalized)
    features: np.ndarray
    
    # Feature names (for debugging/logging)
    feature_names: List[str]
    
    # Raw values (for interpretability)
    raw_values: Dict[str, float]
    
    # Validity flag
    is_valid: bool = True


class FeatureEngine:
    """
    Feature Engineering for ML Models.
    
    Transforms raw market data into ML-ready features:
    - Price-based features (returns, momentum, etc.)
    - Volatility features (ATR, std, range)
    - Volume features (volume ratio, OBV direction)
    - Candle pattern features (body, wicks)
    - Trend features (micro-trend, higher TF alignment)
    
    All features are normalized to [-1, 1] or [0, 1] range.
    """
    
    # Feature names for the model input
    FEATURE_NAMES = [
        # Returns features (5)
        "return_1", "return_5", "return_10", "return_20", "cumulative_return_20",
        
        # Momentum features (4)
        "momentum_5", "momentum_10", "roc_5", "roc_10",
        
        # Volatility features (5)
        "atr_norm", "volatility_norm", "range_norm", "bb_position", "volatility_regime",
        
        # Volume features (3)
        "volume_ratio", "volume_trend", "volume_momentum",
        
        # Candle features (5)
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction", "candle_strength",
        
        # Trend features (4)
        "micro_trend", "micro_trend_strength", "price_vs_sma10", "price_vs_sma20",
        
        # Pattern features (4)
        "higher_high", "lower_low", "consecutive_direction", "reversal_signal",
    ]
    
    def __init__(self, config: Optional[MLConfig] = None):
        """
        Initialize feature engine.
        
        Args:
            config: ML configuration
        """
        self.config = config or get_settings().ml
        self.feature_window = self.config.feature_window
        
        # Normalization parameters (will be updated during training)
        self._norm_params: Dict[str, Tuple[float, float]] = {}
        
        logger.info(f"FeatureEngine initialized with window={self.feature_window}")
    
    def extract_features(self, market_data: MarketData) -> Optional[FeatureVector]:
        """
        Extract features from market data.
        
        Args:
            market_data: MarketData instance with loaded candles
            
        Returns:
            FeatureVector or None if insufficient data
        """
        ohlcv = market_data.get_ohlcv_arrays(n=self.feature_window + 20)
        
        if not ohlcv or len(ohlcv.get("close", [])) < self.feature_window:
            logger.warning("Insufficient data for feature extraction")
            return None
        
        snapshot = market_data.get_snapshot()
        if not snapshot:
            return None
        
        try:
            features, raw_values = self._compute_features(ohlcv, snapshot)
            
            return FeatureVector(
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                features=features,
                feature_names=self.FEATURE_NAMES,
                raw_values=raw_values,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def extract_features_from_arrays(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract features directly from OHLCV arrays.
        
        Used for batch processing during training.
        
        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            
        Returns:
            Feature array or None
        """
        if len(closes) < self.feature_window:
            return None
        
        ohlcv = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }
        
        # Create dummy snapshot for feature extraction
        class DummySnapshot:
            def __init__(self, close, atr, vol, momentum):
                self.current_price = close
                self.atr = atr
                self.atr_percent = atr / close * 100 if close > 0 else 0
                self.volatility = vol
                self.momentum = momentum
                self.micro_trend = 0
                self.micro_trend_strength = 0
                self.body_percent = 50
                self.upper_wick_percent = 25
                self.lower_wick_percent = 25
                self.is_bullish = True
                self.volume_ratio = 1.0
        
        # Calculate basic metrics for snapshot
        atr = self._calculate_atr(highs, lows, closes, 14)
        vol = np.std(np.diff(closes) / closes[:-1]) * 100 if len(closes) > 1 else 0
        momentum = closes[-1] - closes[-10] if len(closes) >= 10 else 0
        
        snapshot = DummySnapshot(closes[-1], atr, vol, momentum)
        
        features, _ = self._compute_features(ohlcv, snapshot)
        return features
    
    def _compute_features(
        self,
        ohlcv: Dict[str, np.ndarray],
        snapshot: MarketSnapshot
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute all features from OHLCV data.
        
        Args:
            ohlcv: Dict with open, high, low, close, volume arrays
            snapshot: Current market snapshot
            
        Returns:
            (normalized_features, raw_values)
        """
        closes = ohlcv["close"]
        highs = ohlcv["high"]
        lows = ohlcv["low"]
        opens = ohlcv["open"]
        volumes = ohlcv["volume"]
        
        raw = {}
        
        # === Returns Features ===
        returns = np.diff(closes) / closes[:-1]
        raw["return_1"] = returns[-1] if len(returns) > 0 else 0
        raw["return_5"] = np.sum(returns[-5:]) if len(returns) >= 5 else 0
        raw["return_10"] = np.sum(returns[-10:]) if len(returns) >= 10 else 0
        raw["return_20"] = np.sum(returns[-20:]) if len(returns) >= 20 else 0
        raw["cumulative_return_20"] = (closes[-1] / closes[-21] - 1) if len(closes) > 20 else 0
        
        # === Momentum Features ===
        raw["momentum_5"] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0
        raw["momentum_10"] = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 else 0
        raw["roc_5"] = raw["momentum_5"] * 100
        raw["roc_10"] = raw["momentum_10"] * 100
        
        # === Volatility Features ===
        atr = self._calculate_atr(highs, lows, closes, 14)
        raw["atr_norm"] = atr / closes[-1] if closes[-1] > 0 else 0
        
        vol_std = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        raw["volatility_norm"] = vol_std
        
        ranges = (highs - lows) / closes
        raw["range_norm"] = np.mean(ranges[-10:]) if len(ranges) >= 10 else np.mean(ranges)
        
        # Bollinger Band position (0=lower, 0.5=middle, 1=upper)
        sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
        std20 = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
        if std20 > 0:
            bb_pos = (closes[-1] - (sma20 - 2*std20)) / (4*std20)
            raw["bb_position"] = np.clip(bb_pos, 0, 1)
        else:
            raw["bb_position"] = 0.5
        
        # Volatility regime: low=0, normal=0.5, high=1
        vol_percentile = self._volatility_percentile(vol_std, returns)
        raw["volatility_regime"] = vol_percentile
        
        # === Volume Features ===
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        raw["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1
        
        vol_sma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.mean(volumes)
        vol_sma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        raw["volume_trend"] = (vol_sma5 / vol_sma20 - 1) if vol_sma20 > 0 else 0
        
        if len(volumes) >= 10:
            volume_base = volumes[-10:-1]
            # Avoid division by zero - replace zeros with small value
            volume_base = np.where(volume_base == 0, 1e-10, volume_base)
            vol_changes = np.diff(volumes[-10:]) / volume_base
        else:
            vol_changes = np.array([0])
        raw["volume_momentum"] = np.mean(vol_changes)
        
        # === Candle Features ===
        latest_range = highs[-1] - lows[-1]
        if latest_range > 0:
            body = abs(closes[-1] - opens[-1])
            raw["body_ratio"] = body / latest_range
            
            if closes[-1] >= opens[-1]:
                raw["upper_wick_ratio"] = (highs[-1] - closes[-1]) / latest_range
                raw["lower_wick_ratio"] = (opens[-1] - lows[-1]) / latest_range
            else:
                raw["upper_wick_ratio"] = (highs[-1] - opens[-1]) / latest_range
                raw["lower_wick_ratio"] = (closes[-1] - lows[-1]) / latest_range
        else:
            raw["body_ratio"] = 0
            raw["upper_wick_ratio"] = 0
            raw["lower_wick_ratio"] = 0
        
        raw["candle_direction"] = 1 if closes[-1] >= opens[-1] else -1
        raw["candle_strength"] = raw["body_ratio"] * raw["candle_direction"]
        
        # === Trend Features ===
        raw["micro_trend"] = snapshot.micro_trend
        raw["micro_trend_strength"] = snapshot.micro_trend_strength
        
        sma10 = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
        raw["price_vs_sma10"] = (closes[-1] / sma10 - 1) if sma10 > 0 else 0
        raw["price_vs_sma20"] = (closes[-1] / sma20 - 1) if sma20 > 0 else 0
        
        # === Pattern Features ===
        raw["higher_high"] = 1 if highs[-1] > highs[-2] else 0
        raw["lower_low"] = 1 if lows[-1] < lows[-2] else 0
        
        # Consecutive same direction candles
        directions = np.sign(closes[-5:] - opens[-5:]) if len(closes) >= 5 else [0]
        consecutive = 0
        for i in range(len(directions) - 1, 0, -1):
            if directions[i] == directions[i-1]:
                consecutive += 1
            else:
                break
        raw["consecutive_direction"] = consecutive / 4  # Normalize to [0, 1]
        
        # Reversal signal (long lower wick after downtrend or vice versa)
        if raw["momentum_5"] < -0.01 and raw["lower_wick_ratio"] > 0.5:
            raw["reversal_signal"] = 1
        elif raw["momentum_5"] > 0.01 and raw["upper_wick_ratio"] > 0.5:
            raw["reversal_signal"] = -1
        else:
            raw["reversal_signal"] = 0
        
        # === Normalize Features ===
        features = np.array([
            self._normalize(raw["return_1"], -0.05, 0.05),
            self._normalize(raw["return_5"], -0.1, 0.1),
            self._normalize(raw["return_10"], -0.15, 0.15),
            self._normalize(raw["return_20"], -0.2, 0.2),
            self._normalize(raw["cumulative_return_20"], -0.3, 0.3),
            
            self._normalize(raw["momentum_5"], -0.1, 0.1),
            self._normalize(raw["momentum_10"], -0.15, 0.15),
            self._normalize(raw["roc_5"], -10, 10),
            self._normalize(raw["roc_10"], -15, 15),
            
            self._normalize(raw["atr_norm"], 0, 0.05),
            self._normalize(raw["volatility_norm"], 0, 0.05),
            self._normalize(raw["range_norm"], 0, 0.05),
            raw["bb_position"],  # Already [0, 1]
            raw["volatility_regime"],  # Already [0, 1]
            
            self._normalize(raw["volume_ratio"], 0, 5),
            self._normalize(raw["volume_trend"], -1, 1),
            self._normalize(raw["volume_momentum"], -1, 1),
            
            raw["body_ratio"],  # Already [0, 1]
            raw["upper_wick_ratio"],  # Already [0, 1]
            raw["lower_wick_ratio"],  # Already [0, 1]
            self._normalize(raw["candle_direction"], -1, 1),
            self._normalize(raw["candle_strength"], -1, 1),
            
            self._normalize(raw["micro_trend"], -1, 1),
            raw["micro_trend_strength"],  # Already [0, 1]
            self._normalize(raw["price_vs_sma10"], -0.1, 0.1),
            self._normalize(raw["price_vs_sma20"], -0.15, 0.15),
            
            raw["higher_high"],  # Already {0, 1}
            raw["lower_low"],  # Already {0, 1}
            raw["consecutive_direction"],  # Already [0, 1]
            self._normalize(raw["reversal_signal"], -1, 1),
        ], dtype=np.float32)
        
        return features, raw
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value to [-1, 1] range.
        
        Args:
            value: Raw value
            min_val: Expected minimum
            max_val: Expected maximum
            
        Returns:
            Normalized value clipped to [-1, 1]
        """
        if max_val == min_val:
            return 0.0
        
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return float(np.clip(normalized, -1, 1))
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> float:
        """Calculate Average True Range."""
        if len(closes) < 2:
            return 0.0
        
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        if len(true_range) >= period:
            return float(np.mean(true_range[-period:]))
        return float(np.mean(true_range))
    
    def _volatility_percentile(
        self,
        current_vol: float,
        returns: np.ndarray
    ) -> float:
        """
        Calculate volatility percentile (regime).
        
        Returns:
            0-1 value (0=low vol, 0.5=normal, 1=high vol)
        """
        if len(returns) < 20:
            return 0.5
        
        # Rolling volatility
        window = 10
        rolling_vols = []
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            rolling_vols.append(vol)
        
        if not rolling_vols:
            return 0.5
        
        # Percentile of current vol
        percentile = sum(1 for v in rolling_vols if v <= current_vol) / len(rolling_vols)
        return float(percentile)
    
    @property
    def feature_dim(self) -> int:
        """Number of features in the output vector."""
        return len(self.FEATURE_NAMES)
