"""Runtime state for Champaign positions (main + optional hedge)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

STATE_PATH = Path("data/champaign_state.json")


@dataclass
class ProtectiveSlot:
    kind: str  # SL | TP
    price: float
    order_id: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class LegState:
    side: str
    quantity: float
    entry_price: float
    margin_usdt: float = 0.0
    entry_costs: float = 0.0
    opened_at: float = field(default_factory=time.time)


@dataclass
class ChampaignStack:
    symbol: str
    main: LegState
    hedge: Optional[LegState] = None
    swing_high: float = 0.0
    swing_low: float = 0.0
    fib_index: int = 0
    stage: str = "initial"  # initial | profit_lock | hedged
    protective: List[ProtectiveSlot] = field(default_factory=list)
    active_sl: float = 0.0
    active_tp: float = 0.0
    locked_profit_usdt: float = -1.0  # highest net-profit floor locked via SL (-1 = none yet)
    peak_net_pnl_usdt: float = 0.0

    def protective_count(self) -> int:
        return len(self.protective)


def save_stacks(stacks: Dict[str, ChampaignStack], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: asdict(v) for k, v in stacks.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_stacks(path: Path = STATE_PATH) -> Dict[str, ChampaignStack]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, ChampaignStack] = {}
        for sym, d in raw.items():
            main = LegState(**d["main"])
            hedge = LegState(**d["hedge"]) if d.get("hedge") else None
            prot = [ProtectiveSlot(**p) for p in d.get("protective", [])]
            out[sym] = ChampaignStack(
                symbol=sym,
                main=main,
                hedge=hedge,
                swing_high=float(d.get("swing_high", 0)),
                swing_low=float(d.get("swing_low", 0)),
                fib_index=int(d.get("fib_index", 0)),
                stage=str(d.get("stage", "initial")),
                protective=prot,
                active_sl=float(d.get("active_sl", 0)),
                active_tp=float(d.get("active_tp", 0)),
                locked_profit_usdt=float(d.get("locked_profit_usdt", -1)),
                peak_net_pnl_usdt=float(d.get("peak_net_pnl_usdt", 0)),
            )
        return out
    except Exception:
        return {}
