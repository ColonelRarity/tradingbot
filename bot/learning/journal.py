"""
Trade journal + ML sample store (SQLite).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/journal.db")


class TradeJournal:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    reason TEXT,
                    rsi REAL,
                    traded INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl_usdt REAL,
                    opened_at REAL NOT NULL,
                    closed_at REAL,
                    meta TEXT,
                    ml_sample_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS ml_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    ta_side TEXT NOT NULL,
                    features TEXT NOT NULL,
                    p_long REAL,
                    ml_confidence REAL,
                    label INTEGER,
                    trade_id INTEGER,
                    pnl_usdt REAL
                );
                """
            )
            try:
                c.execute("ALTER TABLE trades ADD COLUMN ml_sample_id INTEGER")
            except sqlite3.OperationalError:
                pass

    def log_signal(
        self,
        symbol: str,
        side: str,
        reason: str,
        rsi: float,
        traded: bool,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO signals (ts, symbol, side, reason, rsi, traded) VALUES (?,?,?,?,?,?)",
                (time.time(), symbol, side, reason, rsi, 1 if traded else 0),
            )

    def log_ml_sample(
        self,
        symbol: str,
        ta_side: str,
        features: np.ndarray,
        p_long: float,
        ml_confidence: float,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO ml_samples
                   (ts, symbol, ta_side, features, p_long, ml_confidence)
                   VALUES (?,?,?,?,?,?)""",
                (
                    time.time(),
                    symbol,
                    ta_side,
                    json.dumps(features.tolist()),
                    p_long,
                    ml_confidence,
                ),
            )
            return int(cur.lastrowid)

    def link_trade_to_sample(self, trade_id: int, sample_id: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE trades SET ml_sample_id=? WHERE id=?",
                (sample_id, trade_id),
            )

    def label_sample(self, sample_id: int, label: int, pnl_usdt: float = 0.0) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE ml_samples SET label=?, pnl_usdt=? WHERE id=?",
                (label, pnl_usdt, sample_id),
            )

    def label_sample_for_closed_trade(self, symbol: str, pnl_usdt: float) -> None:
        with self._conn() as c:
            row = c.execute(
                """SELECT t.ml_sample_id FROM trades t
                   WHERE t.symbol=? AND t.closed_at IS NOT NULL
                   ORDER BY t.closed_at DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            if not row or row["ml_sample_id"] is None:
                row2 = c.execute(
                    """SELECT id FROM ml_samples
                       WHERE symbol=? AND label IS NULL
                       ORDER BY ts DESC LIMIT 1""",
                    (symbol,),
                ).fetchone()
                if not row2:
                    return
                sid = row2["id"]
            else:
                sid = row["ml_sample_id"]
            label = 1 if pnl_usdt > 0 else 0
            c.execute(
                "UPDATE ml_samples SET label=?, pnl_usdt=? WHERE id=?",
                (label, pnl_usdt, sid),
            )

    def count_labeled_samples(self) -> int:
        with self._conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM ml_samples WHERE label IS NOT NULL"
            ).fetchone()[0]

    def get_labeled_ml_samples(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT features, label FROM ml_samples WHERE label IS NOT NULL"
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "features": json.loads(r["features"]),
                "label": int(r["label"]),
            })
        return out

    def log_trade_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        meta: Optional[Dict[str, Any]] = None,
        ml_sample_id: Optional[int] = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO trades (symbol, side, entry_price, quantity, opened_at, meta, ml_sample_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    symbol,
                    side,
                    entry_price,
                    quantity,
                    time.time(),
                    json.dumps(meta or {}),
                    ml_sample_id,
                ),
            )
            return int(cur.lastrowid)

    def log_trade_close(
        self,
        symbol: str,
        exit_price: float,
        pnl_usdt: float,
    ) -> None:
        with self._conn() as c:
            row = c.execute(
                """SELECT id, ml_sample_id FROM trades WHERE symbol=? AND closed_at IS NULL
                   ORDER BY opened_at DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            if not row:
                return
            c.execute(
                """UPDATE trades SET exit_price=?, pnl_usdt=?, closed_at=? WHERE id=?""",
                (exit_price, pnl_usdt, time.time(), row["id"]),
            )
        self.label_sample_for_closed_trade(symbol, pnl_usdt)

    def recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT symbol, side, entry_price, exit_price, quantity, pnl_usdt,
                          opened_at, closed_at
                   FROM trades
                   ORDER BY COALESCE(closed_at, opened_at) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            closed = c.execute(
                "SELECT COUNT(*) n, SUM(pnl_usdt) pnl FROM trades WHERE closed_at IS NOT NULL"
            ).fetchone()
            wins = c.execute(
                "SELECT COUNT(*) FROM trades WHERE closed_at IS NOT NULL AND pnl_usdt > 0"
            ).fetchone()[0]
            n = closed["n"] or 0
            ml_labeled = c.execute(
                "SELECT COUNT(*) FROM ml_samples WHERE label IS NOT NULL"
            ).fetchone()[0]
            return {
                "closed_trades": n,
                "total_pnl": closed["pnl"] or 0.0,
                "win_rate": (wins / n) if n else 0.0,
                "signals_logged": c.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
                "ml_labeled_samples": ml_labeled,
            }
