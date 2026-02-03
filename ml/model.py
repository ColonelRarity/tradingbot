"""
ML Model Definitions

Contains MLP and LSTM model architectures for trading predictions:
- Direction prediction (LONG/SHORT probability)
- Expected move prediction
- Confidence scoring

Models are configurable via settings.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import get_settings, MLConfig, ModelType


logger = logging.getLogger(__name__)


class TradingModel(ABC, nn.Module):
    """
    Abstract base class for trading models.
    
    All models must implement:
    - forward() for inference
    - predict() for getting trading outputs
    """
    
    def __init__(self, input_dim: int, config: Optional[MLConfig] = None):
        super().__init__()
        self.input_dim = input_dim
        self.config = config or get_settings().ml
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, seq, features)
            
        Returns:
            Dict with model outputs
        """
        pass
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Make prediction from numpy features.
        
        Args:
            features: Feature array
            
        Returns:
            Dict with:
            - direction_prob: P(LONG) probability
            - expected_move: Expected price movement
            - confidence: Prediction confidence
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensor
            if features.ndim == 1:
                x = torch.FloatTensor(features).unsqueeze(0)
            else:
                x = torch.FloatTensor(features)
            
            outputs = self.forward(x)
            
            # Extract predictions
            direction_prob = outputs["direction_logits"].sigmoid().item()
            expected_move = outputs["expected_move"].item()
            confidence = outputs["confidence"].item()
        
        return {
            "direction_prob": direction_prob,
            "expected_move": expected_move,
            "confidence": confidence
        }
    
    def save(self, path: str) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load model weights."""
        try:
            self.load_state_dict(torch.load(path, weights_only=True))
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            return False


class MLPModel(TradingModel):
    """
    Multi-Layer Perceptron for trading predictions.
    
    Architecture:
    - Input normalization
    - Hidden layers with ReLU + Dropout
    - Multi-head output:
      - Direction (logit for LONG probability)
      - Expected move (regression)
      - Confidence (0-1 via sigmoid)
    """
    
    def __init__(self, input_dim: int, config: Optional[MLConfig] = None):
        super().__init__(input_dim, config)
        
        hidden_layers = self.config.mlp_hidden_layers
        dropout = self.config.mlp_dropout
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # Output heads
        self.direction_head = nn.Linear(prev_dim, 1)  # Logit for P(LONG)
        self.move_head = nn.Linear(prev_dim, 1)  # Expected move
        self.confidence_head = nn.Linear(prev_dim, 1)  # Confidence
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"MLPModel created: input={input_dim}, hidden={hidden_layers}")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features)
            
        Returns:
            Dict with direction_logits, expected_move, confidence
        """
        # Handle single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Hidden layers
        h = self.hidden(x)
        
        # Output heads
        direction_logits = self.direction_head(h)
        expected_move = self.move_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        
        return {
            "direction_logits": direction_logits,
            "expected_move": expected_move,
            "confidence": confidence
        }


class LSTMModel(TradingModel):
    """
    LSTM model for sequential trading predictions.
    
    Architecture:
    - LSTM layers for temporal patterns
    - Attention mechanism for important time steps
    - Multi-head output
    """
    
    def __init__(self, input_dim: int, config: Optional[MLConfig] = None):
        super().__init__(input_dim, config)
        
        hidden_size = self.config.lstm_hidden_size
        num_layers = self.config.lstm_num_layers
        dropout = self.config.lstm_dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output heads
        self.direction_head = nn.Linear(hidden_size, 1)
        self.move_head = nn.Linear(hidden_size, 1)
        self.confidence_head = nn.Linear(hidden_size, 1)
        
        logger.info(f"LSTMModel created: input={input_dim}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            Dict with direction_logits, expected_move, confidence
        """
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq, hidden)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)
        
        # Output heads
        direction_logits = self.direction_head(context)
        expected_move = self.move_head(context)
        confidence = torch.sigmoid(self.confidence_head(context))
        
        return {
            "direction_logits": direction_logits,
            "expected_move": expected_move,
            "confidence": confidence
        }


def create_model(input_dim: int, model_type: Optional[ModelType] = None) -> TradingModel:
    """
    Factory function to create appropriate model.
    
    Args:
        input_dim: Number of input features
        model_type: Model type (uses config if None)
        
    Returns:
        TradingModel instance
    """
    config = get_settings().ml
    model_type = model_type or config.model_type
    
    if model_type == ModelType.MLP:
        return MLPModel(input_dim, config)
    elif model_type == ModelType.LSTM:
        return LSTMModel(input_dim, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class EnsembleModel(TradingModel):
    """
    Ensemble of multiple models for robust predictions.
    
    Combines MLP and LSTM predictions with confidence weighting.
    """
    
    def __init__(self, input_dim: int, config: Optional[MLConfig] = None):
        super().__init__(input_dim, config)
        
        self.mlp = MLPModel(input_dim, config)
        self.lstm = LSTMModel(input_dim, config)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining both models.
        """
        mlp_out = self.mlp(x)
        lstm_out = self.lstm(x)
        
        # Normalize weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted combination
        direction_logits = (
            weights[0] * mlp_out["direction_logits"] +
            weights[1] * lstm_out["direction_logits"]
        )
        
        expected_move = (
            weights[0] * mlp_out["expected_move"] +
            weights[1] * lstm_out["expected_move"]
        )
        
        confidence = (
            weights[0] * mlp_out["confidence"] +
            weights[1] * lstm_out["confidence"]
        )
        
        return {
            "direction_logits": direction_logits,
            "expected_move": expected_move,
            "confidence": confidence
        }
