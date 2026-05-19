"""Append-only activity feed for Telegram /history."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/activity.db")


def log_activity(
    kind: str,
    message: str,
    *,
    symbol: str = "",
    pnl_usdt: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT,
                message TEXT NOT NULL,
                pnl_usdt REAL,
                meta TEXT
            )"""
        )
        c.execute(
            """INSERT INTO activity (ts, kind, symbol, message, pnl_usdt, meta)
               VALUES (?,?,?,?,?,?)""",
            (
                time.time(),
                kind,
                symbol or "",
                message,
                pnl_usdt,
                json.dumps(meta or {}),
            ),
        )


def recent_activity(limit: int = 20) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT ts, kind, symbol, message, pnl_usdt FROM activity
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
