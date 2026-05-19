"""
Train ScalpMLP from journal labels + optional kline bootstrap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient, Candle
from bot.learning.journal import TradeJournal
from bot.ml.features import FEATURE_DIM, extract_features
from bot.ml.model import ScalpMLP, save_model

logger = logging.getLogger(__name__)


def bootstrap_from_klines(
    client: BinanceFuturesClient,
    symbols: List[str],
    config: Optional[Config] = None,
    max_samples: int = 1500,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = config or load_config()
    xs: List[np.ndarray] = []
    ys: List[int] = []

    for symbol in symbols:
        try:
            candles = client.get_klines(symbol, cfg.interval, limit=cfg.kline_limit)
        except Exception as e:
            logger.debug("Bootstrap skip %s: %s", symbol, e)
            continue

        horizon = cfg.ml_label_horizon
        min_pct = cfg.ml_bootstrap_min_move_pct

        for i in range(50, len(candles) - horizon):
            window = candles[: i + 1]
            future = candles[i + 1 : i + 1 + horizon]
            if len(future) < horizon:
                continue
            entry = window[-1].close
            exit_p = future[-1].close
            if entry <= 0:
                continue
            chg_pct = (exit_p - entry) / entry * 100
            if chg_pct >= min_pct:
                label = 1
            elif chg_pct <= -min_pct:
                label = 0
            else:
                continue
            feat = extract_features(window, cfg)
            if feat is None:
                continue
            xs.append(feat)
            ys.append(label)
            if len(xs) >= max_samples:
                break
        if len(xs) >= max_samples:
            break

    if not xs:
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)
    return np.stack(xs), np.array(ys, dtype=np.float32)


class MLTrainer:
    def __init__(
        self,
        journal: TradeJournal,
        client: Optional[BinanceFuturesClient] = None,
        config: Optional[Config] = None,
    ):
        self.journal = journal
        self.client = client
        self.cfg = config or load_config()

    def _load_labeled_from_journal(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.journal.get_labeled_ml_samples()
        if not rows:
            return np.zeros((0, FEATURE_DIM)), np.zeros(0)
        xs = np.stack([np.array(r["features"], dtype=np.float32) for r in rows])
        ys = np.array([r["label"] for r in rows], dtype=np.float32)
        return xs, ys

    def train(self, bootstrap_symbols: Optional[List[str]] = None) -> dict:
        xj, yj = self._load_labeled_from_journal()
        xb = yb = np.zeros(0)

        if len(yj) < self.cfg.ml_min_samples and self.client and bootstrap_symbols:
            logger.info("ML bootstrap from klines (%d journal samples)...", len(yj))
            xb, yb = bootstrap_from_klines(self.client, bootstrap_symbols, self.cfg)

        if len(yj) == 0 and len(yb) == 0:
            return {"ok": False, "reason": "no_data"}

        if len(yj) > 0 and len(yb) > 0:
            x = np.concatenate([xj, xb], axis=0)
            y = np.concatenate([yj, yb], axis=0)
        elif len(yj) > 0:
            x, y = xj, yj
        else:
            x, y = xb, yb

        n = len(y)
        if n < max(20, self.cfg.ml_min_samples // 4):
            return {"ok": False, "reason": f"too_few_samples:{n}"}

        # Shuffle split
        idx = np.random.permutation(n)
        x, y = x[idx], y[idx]
        split = int(n * 0.85)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]

        device = "cpu"
        model = ScalpMLP(FEATURE_DIM).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.ml_learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        ds = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        loader = DataLoader(ds, batch_size=min(32, len(x_train)), shuffle=True)

        model.train()
        for epoch in range(self.cfg.ml_epochs):
            total_loss = 0.0
            for bx, by in loader:
                opt.zero_grad()
                logits = model(bx)
                loss = loss_fn(logits, by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.debug("ML epoch %d loss=%.4f", epoch + 1, total_loss / max(len(loader), 1))

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(x_val))
            val_prob = torch.sigmoid(val_logits)
            val_pred = (val_prob >= 0.5).float()
            acc = (val_pred == torch.FloatTensor(y_val)).float().mean().item() if len(y_val) else 0.0

        path = Path(self.cfg.ml_model_path)
        meta = {
            "n_samples": int(n),
            "n_journal": int(len(yj)),
            "n_bootstrap": int(len(yb)),
            "val_accuracy": float(acc),
            "trained_at": time.time(),
        }
        save_model(model, path, meta)
        logger.info("ML trained: n=%d acc=%.3f → %s", n, acc, path)
        return {"ok": True, **meta}
