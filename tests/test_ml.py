"""ML features and gate."""

import numpy as np

from bot.config import Config
from bot.exchange.client import Candle
from bot.ml.features import FEATURE_DIM, extract_features
from bot.ml.predictor import MLPredictor
from bot.strategy.scalper import ScalperStrategy, Side


def _candles(n: int = 80, price: float = 100.0) -> list[Candle]:
    out = []
    for i in range(n):
        p = price + i * 0.05
        out.append(Candle(
            open_time=1_700_000_000_000 + i * 60_000,
            open=p, high=p + 0.2, low=p - 0.2, close=p + 0.05,
            volume=1000 + i * 10,
        ))
    return out


def test_feature_dim():
    f = extract_features(_candles())
    assert f is not None
    assert f.shape == (FEATURE_DIM,)


def test_ml_gate_warmup_passes_ta():
    cfg = Config()
    cfg.ml_enabled = True
    cfg.ml_warmup_samples = 9999
    ml = MLPredictor(cfg)
    strat = ScalperStrategy(cfg)
    candles = _candles()
    ta = strat.evaluate(candles)
    pred = ml.predict_features(np.zeros(FEATURE_DIM, dtype=np.float32))
    out, _ = ml.gate_signal(ta, pred, labeled_count=0)
    assert out.side == ta.side or ta.side == Side.NONE
