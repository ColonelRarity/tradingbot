"""
Signal Engine

Generates trading signals using ML model inference:
- Evaluates entry conditions
- Calculates confidence scores
- Determines position direction
- Validates against risk constraints

Entry Requirements:
- confidence > threshold
- volatility within acceptable band
- no conflicting open hedge
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from core.market_data import MarketData, MarketSnapshot
from core.feature_engineering import FeatureEngine, FeatureVector
from ml.inference import ModelInference
from config.settings import get_settings, MLConfig, RiskConfig


logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Trading signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class TradingSignal:
    """
    Trading signal output from signal engine.
    
    Contains:
    - Direction (LONG/SHORT/NONE)
    - Confidence score (0-1)
    - Expected move range
    - Reason codes for decision
    """
    timestamp: int
    symbol: str
    
    # Signal direction
    direction: SignalDirection
    
    # Confidence metrics
    confidence: float  # 0-1, higher = more confident
    expected_move_percent: float  # Expected price movement
    expected_move_range: Tuple[float, float]  # (min, max) expected range
    
    # Decision reasoning
    reason_codes: list[str]
    is_valid: bool
    
    # Feature context (for pattern storage)
    features: Optional[FeatureVector] = None
    
    # Market context
    volatility_regime: float = 0.5  # 0=low, 0.5=normal, 1=high
    atr_percent: float = 0.0


class SignalEngine:
    """
    Signal Generation Engine.
    
    Combines ML model inference with rule-based filters to generate
    trading signals. Ensures signals meet all entry requirements:
    
    1. ML confidence > threshold
    2. Volatility within acceptable band
    3. No conflicting conditions
    """
    
    def __init__(
        self,
        feature_engine: Optional[FeatureEngine] = None,
        ml_inference: Optional[ModelInference] = None,
        ml_config: Optional[MLConfig] = None,
        risk_config: Optional[RiskConfig] = None
    ):
        """
        Initialize signal engine.
        
        Args:
            feature_engine: Feature extraction engine
            ml_inference: ML model inference
            ml_config: ML configuration
            risk_config: Risk configuration
        """
        settings = get_settings()
        self.ml_config = ml_config or settings.ml
        self.risk_config = risk_config or settings.risk
        
        self.feature_engine = feature_engine or FeatureEngine()
        self.ml_inference = ml_inference or ModelInference()
        
        # Signal history for analysis
        self._signal_history: list[TradingSignal] = []
        self._max_history = 1000
        
        logger.info("SignalEngine initialized")
    
    def generate_signal(
        self,
        market_data: MarketData,
        has_open_position: bool = False,
        has_hedge: bool = False
    ) -> TradingSignal:
        """
        Generate trading signal from market data.
        
        Args:
            market_data: MarketData instance with current data
            has_open_position: Whether there's an open position
            has_hedge: Whether there's an open hedge position
            
        Returns:
            TradingSignal with direction and confidence
        """
        timestamp = int(time.time() * 1000)
        symbol = market_data.symbol
        reason_codes = []
        
        # Get market snapshot
        snapshot = market_data.get_snapshot()
        if not snapshot:
            return self._no_signal(timestamp, symbol, ["NO_MARKET_DATA"])
        
        # Extract features
        features = self.feature_engine.extract_features(market_data)
        if not features or not features.is_valid:
            return self._no_signal(timestamp, symbol, ["FEATURE_EXTRACTION_FAILED"])
        
        # === Volatility Check ===
        volatility_valid, vol_reason = self._check_volatility(snapshot)
        if not volatility_valid:
            reason_codes.append(vol_reason)
            return self._no_signal(timestamp, symbol, reason_codes, features, snapshot)
        
        # === ML Model Inference ===
        try:
            ml_result = self.ml_inference.predict(features.features)
        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            return self._no_signal(timestamp, symbol, ["ML_INFERENCE_FAILED"], features, snapshot)
        
        # Extract ML outputs
        direction_prob = ml_result.get("direction_prob", 0.5)  # P(LONG)
        confidence = ml_result.get("confidence", 0.0)
        expected_move = ml_result.get("expected_move", 0.0)
        
        # === Confidence Check ===
        if confidence < self.ml_config.min_confidence_for_entry:
            reason_codes.append(f"LOW_CONFIDENCE:{confidence:.3f}")
            return self._no_signal(timestamp, symbol, reason_codes, features, snapshot)
        
        # === Determine Direction (very relaxed for testnet learning) ===
        if direction_prob > 0.505:  # Almost any bias toward LONG
            direction = SignalDirection.LONG
            reason_codes.append(f"LONG_PROB:{direction_prob:.3f}")
        elif direction_prob < 0.495:  # Almost any bias toward SHORT
            direction = SignalDirection.SHORT
            reason_codes.append(f"SHORT_PROB:{1-direction_prob:.3f}")
        else:
            reason_codes.append("NEUTRAL_DIRECTION")
            return self._no_signal(timestamp, symbol, reason_codes, features, snapshot)
        
        # === Conflict Check ===
        if has_hedge:
            reason_codes.append("HEDGE_CONFLICT")
            return self._no_signal(timestamp, symbol, reason_codes, features, snapshot)
        
        # === Additional Filters ===
        
        # Micro-trend alignment
        if direction == SignalDirection.LONG and snapshot.micro_trend < 0:
            if snapshot.micro_trend_strength > 0.5:
                reason_codes.append("MICRO_TREND_CONFLICT")
                # Still allow but reduce confidence
                confidence *= 0.8
        elif direction == SignalDirection.SHORT and snapshot.micro_trend > 0:
            if snapshot.micro_trend_strength > 0.5:
                reason_codes.append("MICRO_TREND_CONFLICT")
                confidence *= 0.8
        
        # Volume confirmation
        if snapshot.volume_ratio < 0.5:
            reason_codes.append("LOW_VOLUME")
            confidence *= 0.9
        elif snapshot.volume_ratio > 2.0:
            reason_codes.append("HIGH_VOLUME_SPIKE")
            # Volume spike can be good but also risky
        
        # Final confidence check after adjustments
        if confidence < self.ml_config.min_confidence_for_entry:
            reason_codes.append(f"ADJUSTED_CONFIDENCE_LOW:{confidence:.3f}")
            return self._no_signal(timestamp, symbol, reason_codes, features, snapshot)
        
        # === Calculate Expected Move Range ===
        atr = snapshot.atr
        if direction == SignalDirection.LONG:
            expected_min = snapshot.current_price + atr * 0.5
            expected_max = snapshot.current_price + atr * 2.0
        else:
            expected_min = snapshot.current_price - atr * 2.0
            expected_max = snapshot.current_price - atr * 0.5
        
        reason_codes.append(f"CONFIDENCE:{confidence:.3f}")
        reason_codes.append(f"EXPECTED_MOVE:{expected_move:.4f}")
        
        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            expected_move_percent=expected_move * 100,
            expected_move_range=(expected_min, expected_max),
            reason_codes=reason_codes,
            is_valid=True,
            features=features,
            volatility_regime=features.raw_values.get("volatility_regime", 0.5),
            atr_percent=snapshot.atr_percent
        )
        
        # Store in history
        self._add_to_history(signal)
        
        logger.info(
            f"Signal: {direction.value} {symbol} | "
            f"Confidence: {confidence:.3f} | "
            f"Reasons: {', '.join(reason_codes[:3])}"
        )
        
        return signal
    
    def _check_volatility(self, snapshot: MarketSnapshot) -> Tuple[bool, str]:
        """
        Check if volatility is within acceptable band.
        
        Args:
            snapshot: Current market snapshot
            
        Returns:
            (is_valid, reason_code)
        """
        vol_percent = snapshot.atr_percent
        
        if vol_percent < self.risk_config.min_volatility_percent:
            return False, f"VOLATILITY_TOO_LOW:{vol_percent:.3f}"
        
        if vol_percent > self.risk_config.max_volatility_percent:
            return False, f"VOLATILITY_TOO_HIGH:{vol_percent:.3f}"
        
        return True, "VOLATILITY_OK"
    
    def _no_signal(
        self,
        timestamp: int,
        symbol: str,
        reason_codes: list[str],
        features: Optional[FeatureVector] = None,
        snapshot: Optional[MarketSnapshot] = None
    ) -> TradingSignal:
        """Create a no-signal result."""
        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=SignalDirection.NONE,
            confidence=0.0,
            expected_move_percent=0.0,
            expected_move_range=(0.0, 0.0),
            reason_codes=reason_codes,
            is_valid=False,
            features=features,
            volatility_regime=snapshot.volatility if snapshot else 0.5,
            atr_percent=snapshot.atr_percent if snapshot else 0.0
        )
    
    def _add_to_history(self, signal: TradingSignal) -> None:
        """Add signal to history, maintaining max size."""
        self._signal_history.append(signal)
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]
    
    def get_signal_stats(self) -> dict:
        """Get statistics about recent signals."""
        if not self._signal_history:
            return {"total": 0}
        
        total = len(self._signal_history)
        valid = sum(1 for s in self._signal_history if s.is_valid)
        longs = sum(1 for s in self._signal_history if s.direction == SignalDirection.LONG)
        shorts = sum(1 for s in self._signal_history if s.direction == SignalDirection.SHORT)
        
        avg_confidence = sum(s.confidence for s in self._signal_history if s.is_valid)
        avg_confidence = avg_confidence / valid if valid > 0 else 0
        
        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "longs": longs,
            "shorts": shorts,
            "avg_confidence": avg_confidence,
            "valid_rate": valid / total if total > 0 else 0
        }
