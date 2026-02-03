"""
Model Trainer

Handles ML model training:
- Initial training from historical data
- Incremental retraining on closed positions
- Sample weighting (recent > historical)
- Model validation and metrics
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ml.model import TradingModel, create_model
from core.pattern_memory import PatternMemory
from core.feature_engineering import FeatureEngine
from config.settings import get_settings, MLConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics from one epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array (N, feature_dim)
            labels: Label array (N,) with 1=profitable, 0=not
            weights: Sample weights (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(labels))
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.weights[idx]


class ModelTrainer:
    """
    ML Model Trainer.
    
    Handles:
    - Initial training from pattern memory
    - Incremental retraining on new data
    - Sample weighting (recent patterns weighted higher)
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: Optional[TradingModel] = None,
        pattern_memory: Optional[PatternMemory] = None,
        config: Optional[MLConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train (creates new if None)
            pattern_memory: Pattern memory for training data
            config: ML configuration
        """
        self.config = config or get_settings().ml
        self.pattern_memory = pattern_memory or PatternMemory()
        
        # Feature dimension from FeatureEngine
        feature_engine = FeatureEngine()
        self.feature_dim = feature_engine.feature_dim
        
        # Create or use provided model
        self.model = model or create_model(self.feature_dim)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Loss functions
        self.direction_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.move_loss = nn.MSELoss(reduction='none')
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history: List[TrainingMetrics] = []
        
        # Model save path
        self.model_path = os.path.join(
            self.config.model_save_path,
            f"{self.config.model_type.value}_model.pt"
        )
        
        logger.info(f"ModelTrainer initialized: feature_dim={self.feature_dim}")
    
    def train(
        self,
        epochs: Optional[int] = None,
        validation_split: float = 0.2
    ) -> List[TrainingMetrics]:
        """
        Train model on data from pattern memory.
        
        Args:
            epochs: Number of epochs (uses config if None)
            validation_split: Fraction for validation
            
        Returns:
            List of TrainingMetrics per epoch
        """
        epochs = epochs or self.config.epochs_per_retrain
        
        # Get training data
        features, labels, weights = self.pattern_memory.get_training_data(
            min_samples=self.config.min_samples_for_training
        )
        
        if len(features) == 0:
            logger.warning("No training data available")
            return []
        
        logger.info(f"Training with {len(features)} samples")
        
        # Split data
        n_val = int(len(features) * validation_split)
        indices = np.random.permutation(len(features))
        
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        train_dataset = TradingDataset(
            features[train_idx],
            labels[train_idx],
            weights[train_idx]
        )
        
        val_dataset = TradingDataset(
            features[val_idx],
            labels[val_idx],
            weights[val_idx]
        )
        
        # Adjust batch_size to handle small datasets (BatchNorm requires batch_size > 1)
        effective_batch_size = min(self.config.batch_size, max(2, len(train_dataset)))
        val_batch_size = min(self.config.batch_size, max(2, len(val_dataset)))
        
        # Create weighted sampler for training
        sampler = WeightedRandomSampler(
            weights=train_dataset.weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False
        )
        
        # Training loop
        metrics_history = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            metrics = TrainingMetrics(
                epoch=self.current_epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                accuracy=val_metrics["accuracy"],
                precision=val_metrics["precision"],
                recall=val_metrics["recall"],
                f1=val_metrics["f1"]
            )
            
            metrics_history.append(metrics)
            self.training_history.append(metrics)
            self.current_epoch += 1
            
            logger.info(
                f"Epoch {metrics.epoch}: "
                f"train_loss={metrics.train_loss:.4f}, "
                f"val_loss={metrics.val_loss:.4f}, "
                f"acc={metrics.accuracy:.4f}, "
                f"f1={metrics.f1:.4f}"
            )
            
            # Save best model
            if metrics.val_loss < self.best_val_loss:
                self.best_val_loss = metrics.val_loss
                self._save_checkpoint()
        
        return metrics_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for features, labels, weights in train_loader:
            # BatchNorm requires batch_size > 1, skip if batch_size=1
            if features.size(0) == 1:
                logger.warning("Skipping batch with size=1 (BatchNorm requirement)")
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate weighted loss
            dir_loss = self.direction_loss(
                outputs["direction_logits"].squeeze(),
                labels
            )
            
            # Weight the loss
            weighted_loss = (dir_loss * weights).mean()
            
            # Backward pass
            weighted_loss.backward()
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for features, labels, weights in val_loader:
                # BatchNorm requires batch_size > 1, skip if batch_size=1
                if features.size(0) == 1:
                    logger.warning("Skipping validation batch with size=1 (BatchNorm requirement)")
                    continue
                
                outputs = self.model(features)
                
                # Loss
                dir_loss = self.direction_loss(
                    outputs["direction_logits"].squeeze(),
                    labels
                )
                total_loss += dir_loss.mean().item()
                n_batches += 1
                
                # Predictions
                probs = outputs["direction_logits"].sigmoid().squeeze()
                preds = (probs > 0.5).float()
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Precision, Recall, F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        
        return {
            "loss": total_loss / max(1, n_batches),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def incremental_train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Incremental training on new data.
        
        Called after positions close to learn from outcomes.
        
        Args:
            features: New feature samples
            labels: New labels
            weights: Sample weights
            
        Returns:
            Training loss
        """
        if len(features) == 0:
            return 0.0
        
        self.model.train()
        
        # Convert to tensors
        features_t = torch.FloatTensor(features)
        labels_t = torch.FloatTensor(labels)
        weights_t = torch.FloatTensor(weights) if weights is not None else torch.ones(len(labels))
        
        # Apply recent data weight
        weights_t = weights_t * self.config.weight_recent_data
        
        self.optimizer.zero_grad()
        
        # Forward
        outputs = self.model(features_t)
        
        # Weighted loss
        dir_loss = self.direction_loss(
            outputs["direction_logits"].squeeze(),
            labels_t
        )
        weighted_loss = (dir_loss * weights_t).mean()
        
        # Backward
        weighted_loss.backward()
        self.optimizer.step()
        
        loss = weighted_loss.item()
        logger.debug(f"Incremental training: loss={loss:.4f}")
        
        return loss
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
        
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model_type": self.config.model_type.value,
                "feature_dim": self.feature_dim
            }
        }
        
        torch.save(checkpoint, self.model_path)
        logger.info(f"Checkpoint saved: {self.model_path}")
    
    def load_checkpoint(self) -> bool:
        """Load model checkpoint if exists."""
        if not os.path.exists(self.model_path):
            logger.info("No checkpoint found, using fresh model")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, weights_only=False)
            
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.current_epoch = checkpoint.get("epoch", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            
            logger.info(f"Checkpoint loaded: epoch={self.current_epoch}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        if not self.training_history:
            return {"epochs_trained": 0}
        
        recent = self.training_history[-1]
        
        return {
            "epochs_trained": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "latest_train_loss": recent.train_loss,
            "latest_val_loss": recent.val_loss,
            "latest_accuracy": recent.accuracy,
            "latest_f1": recent.f1
        }
