"""
Virtual futures wallet for dry-run (paper) trading.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/paper_wallet.json")


@dataclass
class PaperPosition:
    symbol: str
    side: str  # LONG | SHORT
    quantity: float
    entry_price: float  # after slippage/spread
    mark_at_entry: float
    stop_price: float
    tp_price: float
    margin_usdt: float
    notional_usdt: float
    opened_at: float = field(default_factory=time.time)
    entry_reason: str = ""
    entry_costs: float = 0.0  # commission+slip+spread at open
    funding_paid: float = 0.0
    last_funding_accrual: float = field(default_factory=time.time)


@dataclass
class PaperWallet:
    initial_balance: float
    balance: float
    available: float
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_spread: float = 0.0
    total_funding: float = 0.0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    closed_trades: int = 0
    wins: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def total_costs(self) -> float:
        return self.total_commission + self.total_slippage + self.total_spread + self.total_funding

    def equity(self) -> float:
        return self.balance

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "available": self.available,
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_spread": self.total_spread,
            "total_funding": self.total_funding,
            "total_fees": self.total_commission,
            "closed_trades": self.closed_trades,
            "wins": self.wins,
            "started_at": self.started_at,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> Optional["PaperWallet"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            positions = {}
            for k, v in data.get("positions", {}).items():
                v.setdefault("mark_at_entry", v.get("entry_price", 0))
                v.setdefault("entry_costs", 0.0)
                v.setdefault("funding_paid", 0.0)
                v.setdefault("last_funding_accrual", v.get("opened_at", time.time()))
                positions[k] = PaperPosition(**v)
            return cls(
                initial_balance=float(data["initial_balance"]),
                balance=float(data["balance"]),
                available=float(data["available"]),
                realized_pnl=float(data.get("realized_pnl", 0)),
                total_commission=float(
                    data.get("total_commission", data.get("total_fees", 0))
                ),
                total_slippage=float(data.get("total_slippage", 0)),
                total_spread=float(data.get("total_spread", 0)),
                total_funding=float(data.get("total_funding", 0)),
                closed_trades=int(data.get("closed_trades", 0)),
                wins=int(data.get("wins", 0)),
                started_at=float(data.get("started_at", time.time())),
                positions=positions,
            )
        except Exception as e:
            logger.warning("Could not load paper wallet: %s", e)
            return None

    @classmethod
    def create_fresh(cls, initial: float) -> "PaperWallet":
        return cls(
            initial_balance=initial,
            balance=initial,
            available=initial,
        )


def unrealized_pnl(pos: PaperPosition, mark: float) -> float:
    if pos.side == "LONG":
        return (mark - pos.entry_price) * pos.quantity
    return (pos.entry_price - mark) * pos.quantity


def check_exit(pos: PaperPosition, mark: float) -> Optional[str]:
    if pos.side == "LONG":
        if mark <= pos.stop_price:
            return "SL"
        if mark >= pos.tp_price:
            return "TP"
    else:
        if mark >= pos.stop_price:
            return "SL"
        if mark <= pos.tp_price:
            return "TP"
    return None
