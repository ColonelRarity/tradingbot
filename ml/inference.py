"""
Model Inference

Handles ML model inference for trading signals:
- Load trained model
- Make predictions from features
- Confidence calibration
- Prediction caching
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Optional
import numpy as np

import torch

from ml.model import TradingModel, create_model
from core.feature_engineering import FeatureEngine
from config.settings import get_settings, MLConfig


logger = logging.getLogger(__name__)


class ModelInference:
    """
    ML Model Inference Engine.
    
    Handles:
    - Model loading
    - Feature-to-prediction conversion
    - Confidence calibration
    - Prediction caching
    """
    
    def __init__(
        self,
        model: Optional[TradingModel] = None,
        config: Optional[MLConfig] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded model (will load from disk if None)
            config: ML configuration
        """
        self.config = config or get_settings().ml
        
        # Feature dimension
        feature_engine = FeatureEngine()
        self.feature_dim = feature_engine.feature_dim
        
        # Model
        self.model = model
        self._model_loaded = model is not None
        
        # Prediction cache (for avoiding repeated predictions)
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 5.0  # Cache TTL in seconds
        self._last_cache_clean = time.time()
        
        # Calibration statistics
        self._prediction_count = 0
        self._correct_predictions = 0
        self._outcome_count = 0  # Count of recorded outcomes (positions that were opened and closed)
        
        logger.info("ModelInference initialized")
    
    def load_model(self) -> bool:
        """
        Load model from disk.
        
        Returns:
            True if loaded successfully
        """
        model_path = os.path.join(
            self.config.model_save_path,
            f"{self.config.model_type.value}_model.pt"
        )
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, using untrained model")
            self.model = create_model(self.feature_dim)
            self._model_loaded = True
            return False
        
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Create model
            self.model = create_model(self.feature_dim)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            
            self._model_loaded = True
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = create_model(self.feature_dim)
            self._model_loaded = True
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Make prediction from features.
        
        Args:
            features: Feature array (feature_dim,)
            
        Returns:
            Dict with:
            - direction_prob: P(LONG) probability (0-1)
            - expected_move: Expected price movement
            - confidence: Prediction confidence (0-1)
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()
        
        # Check cache
        cache_key = self._get_cache_key(features)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Make prediction
        try:
            self.model.eval()
            
            with torch.no_grad():
                x = torch.FloatTensor(features)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                outputs = self.model(x)
                
                # Extract predictions
                direction_prob = outputs["direction_logits"].sigmoid().item()
                expected_move = outputs["expected_move"].item()
                raw_confidence = outputs["confidence"].item()
                
                # Calibrate confidence
                confidence = self._calibrate_confidence(
                    raw_confidence,
                    direction_prob
                )
            
            result = {
                "direction_prob": direction_prob,
                "expected_move": expected_move,
                "confidence": confidence,
                "raw_confidence": raw_confidence
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            self._prediction_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "direction_prob": 0.5,
                "expected_move": 0.0,
                "confidence": 0.0,
                "raw_confidence": 0.0
            }
    
    def _calibrate_confidence(
        self,
        raw_confidence: float,
        direction_prob: float
    ) -> float:
        """
        Calibrate confidence score.
        
        Reduces confidence when:
        - Direction probability is near 0.5 (uncertain)
        - Model hasn't been validated yet
        
        Args:
            raw_confidence: Raw confidence from model
            direction_prob: Direction probability
            
        Returns:
            Calibrated confidence
        """
        # Direction certainty (0 when prob is 0.5, 1 when prob is 0 or 1)
        direction_certainty = abs(direction_prob - 0.5) * 2
        
        # Combine with raw confidence
        calibrated = raw_confidence * direction_certainty
        
        # Scale to 0-1
        calibrated = min(1.0, max(0.0, calibrated))
        
        return calibrated
    
    def _get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        # Use hash of feature values (rounded for stability)
        rounded = np.round(features, 4)
        return str(hash(rounded.tobytes()))
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached prediction if not expired."""
        self._clean_cache()
        
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                return entry["result"]
        
        return None
    
    def _cache_result(self, key: str, result: Dict) -> None:
        """Cache prediction result."""
        self._cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        
        # Clean every 60 seconds
        if now - self._last_cache_clean < 60:
            return
        
        expired = [
            k for k, v in self._cache.items()
            if now - v["timestamp"] > self._cache_ttl
        ]
        
        for k in expired:
            del self._cache[k]
        
        self._last_cache_clean = now
    
    def record_outcome(self, was_correct: bool) -> None:
        """
        Record prediction outcome for calibration.
        
        Called when a position is closed to track if the prediction was correct.
        
        Args:
            was_correct: Whether prediction was correct (position was profitable)
        """
        self._outcome_count += 1
        if was_correct:
            self._correct_predictions += 1
    
    def get_stats(self) -> Dict:
        """Get inference statistics."""
        # Accuracy: correct predictions / total outcomes (positions that closed)
        # Use outcome_count if available (more accurate), otherwise fall back to prediction_count
        if self._outcome_count > 0:
            accuracy = self._correct_predictions / self._outcome_count
        elif self._prediction_count > 0:
            # Fallback: use prediction_count if no outcomes recorded yet
            accuracy = self._correct_predictions / self._prediction_count
        else:
            accuracy = 0.0
        
        return {
            "prediction_count": self._prediction_count,
            "correct_predictions": self._correct_predictions,
            "outcome_count": self._outcome_count,
            "accuracy": accuracy,
            "cache_size": len(self._cache),
            "model_loaded": self._model_loaded
        }
    
    def update_model(self, model: TradingModel) -> None:
        """
        Update model after retraining.
        
        Args:
            model: New trained model
        """
        self.model = model
        self.model.eval()
        
        # Clear cache when model changes
        self._cache.clear()
        
        logger.info("Model updated for inference")
