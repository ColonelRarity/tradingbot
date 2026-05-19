"""ML inference + decision gate (FILTER / FULL)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from bot.config import Config, load_config
from bot.ml.features import FEATURE_DIM, extract_features
from bot.ml.model import ScalpMLP, load_model
from bot.strategy.scalper import ScalpSignal, Side

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    p_long: float
    p_short: float
    confidence: float
    edge: float
    model_loaded: bool


class MLPredictor:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self.model: Optional[ScalpMLP] = None
        self.meta: dict = {}
        self._load()

    def _load(self) -> None:
        path = Path(self.cfg.ml_model_path)
        if not path.exists():
            logger.info("ML model not found at %s — warmup / bootstrap required", path)
            return
        try:
            self.model, self.meta = load_model(path)
            logger.info(
                "ML model loaded (%s samples, acc=%.3f)",
                self.meta.get("n_samples", "?"),
                self.meta.get("val_accuracy", 0),
            )
        except Exception as e:
            logger.warning("ML load failed: %s", e)
            self.model = None

    def reload(self) -> None:
        self._load()

    def predict_features(self, features: np.ndarray) -> MLPrediction:
        if self.model is None:
            return MLPrediction(0.5, 0.5, 0.0, 0.0, False)

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            logit = self.model(x)
            p_long = torch.sigmoid(logit).item()

        p_short = 1.0 - p_long
        confidence = abs(p_long - 0.5) * 2.0
        edge = abs(p_long - p_short)
        return MLPrediction(p_long, p_short, confidence, edge, True)

    def predict_candles(self, candles) -> Optional[MLPrediction]:
        feat = extract_features(candles, self.cfg)
        if feat is None:
            return None
        return self.predict_features(feat)

    def in_warmup(self, labeled_count: int) -> bool:
        return labeled_count < self.cfg.ml_warmup_samples

    def gate_signal(
        self,
        ta: ScalpSignal,
        pred: MLPrediction,
        labeled_count: int,
    ) -> tuple[ScalpSignal, str]:
        """
        Apply ML_DECISION_MODE to TA signal.
        Returns (possibly modified signal, reason_suffix).
        """
        mode = self.cfg.ml_decision_mode.upper()
        if not self.cfg.ml_enabled or mode == "OFF":
            return ta, ""

        if self.in_warmup(labeled_count):
            return ta, "ML_WARMUP"

        if not pred.model_loaded:
            return ta, "ML_NO_MODEL"

        min_conf = self.cfg.ml_min_confidence
        min_edge = self.cfg.ml_min_edge
        short_conf = max(0.0, min_conf - self.cfg.ml_short_conf_offset)

        if mode == "FILTER":
            if ta.side == Side.NONE:
                return ta, ""
            if ta.side == Side.LONG:
                ok = pred.p_long >= min_conf and pred.edge >= min_edge
                block = f"ML_BLOCK_LONG p={pred.p_long:.3f}"
            else:
                ok = pred.p_short >= short_conf and pred.edge >= min_edge
                block = f"ML_BLOCK_SHORT p={pred.p_short:.3f}"
            if ok:
                return ta, f"ML_OK p_long={pred.p_long:.3f} conf={pred.confidence:.3f}"
            return ScalpSignal(
                Side.NONE, block, ta.atr, ta.price, ta.rsi, ta.fast_ema, ta.slow_ema
            ), block

        if mode == "FULL":
            if pred.confidence < min_conf or pred.edge < min_edge:
                return ScalpSignal(
                    Side.NONE,
                    f"ML_LOW_EDGE c={pred.confidence:.3f}",
                    ta.atr, ta.price, ta.rsi, ta.fast_ema, ta.slow_ema,
                ), "ML_FULL_SKIP"
            if pred.p_long > pred.p_short:
                side = Side.LONG
            else:
                side = Side.SHORT
            if self.cfg.ml_require_ta_signal and ta.side != side and ta.side != Side.NONE:
                return ScalpSignal(
                    Side.NONE, "ML_TA_DISAGREE", ta.atr, ta.price, ta.rsi, ta.fast_ema, ta.slow_ema
                ), "ML_TA_DISAGREE"
            return ScalpSignal(
                side,
                f"ML_FULL_{side.value} p={pred.p_long:.3f}",
                ta.atr, ta.price, ta.rsi, ta.fast_ema, ta.slow_ema,
            ), "ML_FULL"

        return ta, ""
