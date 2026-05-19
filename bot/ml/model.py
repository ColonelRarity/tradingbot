"""PyTorch MLP — P(profitable LONG direction)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from bot.ml.features import FEATURE_DIM


class ScalpMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_model(
    model: ScalpMLP,
    path: Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": FEATURE_DIM,
        "meta": meta or {},
    }, path)


def load_model(path: Path, device: str = "cpu") -> tuple[ScalpMLP, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dim = int(ckpt.get("input_dim", FEATURE_DIM))
    model = ScalpMLP(input_dim=dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})
