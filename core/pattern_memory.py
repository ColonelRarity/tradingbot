"""
Pattern Memory Module

Stores and retrieves learned market patterns:
- Successful trade patterns
- Failed trade patterns
- Market regime patterns
- Position outcome history

Used for:
- Model training data generation
- Pattern matching during inference
- Strategy adaptation
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

from config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class TradePattern:
    """
    Recorded trade pattern for learning.
    
    Contains:
    - Entry conditions (features at entry)
    - Outcome (profit/loss)
    - Market context
    """
    pattern_id: str
    timestamp: int  # Unix timestamp ms
    symbol: str
    
    # Entry information
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_features: List[float]  # Feature vector at entry
    
    # Outcome
    exit_price: float
    pnl_usdt: float
    pnl_percent: float
    duration_sec: int  # How long position was held
    exit_reason: str  # "TP", "SL", "MANUAL", "HEDGE"
    
    # Was this a winning trade?
    is_profitable: bool
    
    # Market context
    atr_at_entry: float
    volatility_regime: float  # 0-1
    
    # Learning weight (recent patterns weighted higher)
    weight: float = 1.0


@dataclass
class MarketRegime:
    """
    Recorded market regime for context.
    """
    timestamp: int
    symbol: str
    
    # Regime classification
    regime_type: str  # "trending_up", "trending_down", "ranging", "volatile"
    volatility_level: float  # 0-1
    
    # Performance in this regime
    win_rate: float
    avg_pnl: float
    sample_count: int


class PatternMemory:
    """
    Pattern Memory for Self-Learning.
    
    Stores trade patterns and outcomes in SQLite database.
    Used to:
    - Generate training data for ML model
    - Track performance by pattern type
    - Weight recent patterns higher for adaptation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize pattern memory.
        
        Args:
            db_path: Path to SQLite database (uses config default if None)
        """
        settings = get_settings()
        self.db_path = db_path or settings.ml.pattern_db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        # Thread-safe connection
        self._local = threading.local()
        
        # Initialize database
        self._init_db()
        
        logger.info(f"PatternMemory initialized: {self.db_path}")
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        
        conn.executescript("""
            -- Trade patterns table
            CREATE TABLE IF NOT EXISTS trade_patterns (
                pattern_id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_features TEXT NOT NULL,
                exit_price REAL,
                pnl_usdt REAL,
                pnl_percent REAL,
                duration_sec INTEGER,
                exit_reason TEXT,
                is_profitable INTEGER,
                atr_at_entry REAL,
                volatility_regime REAL,
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON trade_patterns(symbol);
            CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON trade_patterns(timestamp);
            CREATE INDEX IF NOT EXISTS idx_patterns_profitable ON trade_patterns(is_profitable);
            
            -- Market regimes table
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                regime_type TEXT NOT NULL,
                volatility_level REAL,
                win_rate REAL,
                avg_pnl REAL,
                sample_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Model performance tracking
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                profit_factor REAL,
                total_trades INTEGER,
                win_rate REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
    
    # ==================== Pattern Storage ====================
    
    def store_pattern(self, pattern: TradePattern) -> bool:
        """
        Store a trade pattern.
        
        Args:
            pattern: TradePattern to store
            
        Returns:
            True if stored successfully
        """
        try:
            conn = self._get_conn()
            
            # Serialize features to JSON
            features_json = json.dumps(pattern.entry_features)
            
            conn.execute("""
                INSERT OR REPLACE INTO trade_patterns
                (pattern_id, timestamp, symbol, side, entry_price, entry_features,
                 exit_price, pnl_usdt, pnl_percent, duration_sec, exit_reason,
                 is_profitable, atr_at_entry, volatility_regime, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.timestamp,
                pattern.symbol,
                pattern.side,
                pattern.entry_price,
                features_json,
                pattern.exit_price,
                pattern.pnl_usdt,
                pattern.pnl_percent,
                pattern.duration_sec,
                pattern.exit_reason,
                1 if pattern.is_profitable else 0,
                pattern.atr_at_entry,
                pattern.volatility_regime,
                pattern.weight
            ))
            
            conn.commit()
            logger.debug(f"Stored pattern {pattern.pattern_id}: PnL={pattern.pnl_usdt:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False
    
    def update_pattern_outcome(
        self,
        pattern_id: str,
        exit_price: float,
        pnl_usdt: float,
        pnl_percent: float,
        duration_sec: int,
        exit_reason: str
    ) -> bool:
        """
        Update pattern with trade outcome (called when position closes).
        
        Args:
            pattern_id: Pattern ID to update
            exit_price: Exit price
            pnl_usdt: Profit/loss in USDT
            pnl_percent: Profit/loss percentage
            duration_sec: Position duration
            exit_reason: Reason for exit
            
        Returns:
            True if updated successfully
        """
        try:
            conn = self._get_conn()
            
            conn.execute("""
                UPDATE trade_patterns
                SET exit_price = ?,
                    pnl_usdt = ?,
                    pnl_percent = ?,
                    duration_sec = ?,
                    exit_reason = ?,
                    is_profitable = ?
                WHERE pattern_id = ?
            """, (
                exit_price,
                pnl_usdt,
                pnl_percent,
                duration_sec,
                exit_reason,
                1 if pnl_usdt > 0 else 0,
                pattern_id
            ))
            
            conn.commit()
            logger.debug(f"Updated pattern {pattern_id} outcome: {exit_reason}, PnL={pnl_usdt:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pattern outcome: {e}")
            return False
    
    # ==================== Pattern Retrieval ====================
    
    def get_training_data(
        self,
        symbol: Optional[str] = None,
        min_samples: int = 100,
        max_samples: int = 10000,
        only_completed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data from stored patterns.
        
        Args:
            symbol: Filter by symbol (None for all)
            min_samples: Minimum samples required
            max_samples: Maximum samples to return
            only_completed: Only include patterns with outcomes
            
        Returns:
            (features, labels, weights) arrays
            - features: (N, feature_dim) array
            - labels: (N,) array with 1=profitable, 0=not
            - weights: (N,) array with sample weights
        """
        conn = self._get_conn()
        
        query = """
            SELECT entry_features, is_profitable, weight, timestamp
            FROM trade_patterns
            WHERE 1=1
        """
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if only_completed:
            query += " AND exit_reason IS NOT NULL"
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max_samples)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        if len(rows) < min_samples:
            logger.warning(f"Insufficient training data: {len(rows)} < {min_samples}")
            return np.array([]), np.array([]), np.array([])
        
        features = []
        labels = []
        weights = []
        
        settings = get_settings()
        decay = settings.ml.decay_factor
        recent_weight = settings.ml.weight_recent_data
        
        for i, row in enumerate(rows):
            # Parse features
            feat = json.loads(row["entry_features"])
            features.append(feat)
            labels.append(row["is_profitable"])
            
            # Apply decay weight (more recent = higher weight)
            base_weight = row["weight"]
            time_weight = recent_weight * (decay ** i)  # Exponential decay
            weights.append(base_weight * time_weight)
        
        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(weights, dtype=np.float32)
        )
    
    def get_recent_patterns(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[TradePattern]:
        """
        Get recent patterns for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum patterns to return
            
        Returns:
            List of TradePattern objects
        """
        conn = self._get_conn()
        
        cursor = conn.execute("""
            SELECT * FROM trade_patterns
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, limit))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append(TradePattern(
                pattern_id=row["pattern_id"],
                timestamp=row["timestamp"],
                symbol=row["symbol"],
                side=row["side"],
                entry_price=row["entry_price"],
                entry_features=json.loads(row["entry_features"]),
                exit_price=row["exit_price"] or 0,
                pnl_usdt=row["pnl_usdt"] or 0,
                pnl_percent=row["pnl_percent"] or 0,
                duration_sec=row["duration_sec"] or 0,
                exit_reason=row["exit_reason"] or "",
                is_profitable=bool(row["is_profitable"]),
                atr_at_entry=row["atr_at_entry"] or 0,
                volatility_regime=row["volatility_regime"] or 0.5,
                weight=row["weight"]
            ))
        
        return patterns
    
    def get_performance_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            symbol: Filter by symbol (None for all)
            days: Number of days to analyze
            
        Returns:
            Dict with performance metrics
        """
        conn = self._get_conn()
        
        # Calculate timestamp threshold
        threshold = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_profitable = 1 THEN 1 ELSE 0 END) as winning_trades,
                SUM(pnl_usdt) as total_pnl,
                AVG(pnl_usdt) as avg_pnl,
                AVG(duration_sec) as avg_duration,
                AVG(CASE WHEN is_profitable = 1 THEN pnl_usdt ELSE NULL END) as avg_win,
                AVG(CASE WHEN is_profitable = 0 THEN pnl_usdt ELSE NULL END) as avg_loss
            FROM trade_patterns
            WHERE timestamp >= ? AND exit_reason IS NOT NULL
        """
        params = [threshold]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        
        total = row["total_trades"] or 0
        winning = row["winning_trades"] or 0
        avg_win = row["avg_win"] or 0
        avg_loss = abs(row["avg_loss"] or 0)
        
        # Calculate profit factor
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": total - winning,
            "win_rate": winning / total if total > 0 else 0,
            "total_pnl": row["total_pnl"] or 0,
            "avg_pnl": row["avg_pnl"] or 0,
            "avg_duration_sec": row["avg_duration"] or 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "days_analyzed": days,
            "symbol": symbol or "ALL"
        }
    
    # ==================== Market Regime ====================
    
    def store_regime(self, regime: MarketRegime) -> bool:
        """Store market regime classification."""
        try:
            conn = self._get_conn()
            
            conn.execute("""
                INSERT INTO market_regimes
                (timestamp, symbol, regime_type, volatility_level, win_rate, avg_pnl, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                regime.timestamp,
                regime.symbol,
                regime.regime_type,
                regime.volatility_level,
                regime.win_rate,
                regime.avg_pnl,
                regime.sample_count
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store regime: {e}")
            return False
    
    def get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get most recent market regime for symbol."""
        conn = self._get_conn()
        
        cursor = conn.execute("""
            SELECT * FROM market_regimes
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return MarketRegime(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            regime_type=row["regime_type"],
            volatility_level=row["volatility_level"],
            win_rate=row["win_rate"],
            avg_pnl=row["avg_pnl"],
            sample_count=row["sample_count"]
        )
    
    # ==================== Model Performance ====================
    
    def log_model_performance(
        self,
        model_version: str,
        accuracy: float,
        precision: float,
        recall: float,
        profit_factor: float,
        total_trades: int,
        win_rate: float
    ) -> None:
        """Log model performance metrics."""
        try:
            conn = self._get_conn()
            
            conn.execute("""
                INSERT INTO model_performance
                (timestamp, model_version, accuracy, precision_score, recall,
                 profit_factor, total_trades, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now(timezone.utc).timestamp() * 1000),
                model_version,
                accuracy,
                precision,
                recall,
                profit_factor,
                total_trades,
                win_rate
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
    
    # ==================== Maintenance ====================
    
    def decay_weights(self, decay_factor: float = 0.99) -> int:
        """
        Apply decay to all pattern weights.
        
        Called periodically to gradually reduce importance of old patterns.
        
        Args:
            decay_factor: Multiply all weights by this factor
            
        Returns:
            Number of patterns updated
        """
        try:
            conn = self._get_conn()
            
            cursor = conn.execute("""
                UPDATE trade_patterns
                SET weight = weight * ?
            """, (decay_factor,))
            
            conn.commit()
            count = cursor.rowcount
            logger.debug(f"Decayed weights for {count} patterns")
            return count
            
        except Exception as e:
            logger.error(f"Failed to decay weights: {e}")
            return 0
    
    def cleanup_old_patterns(self, days: int = 90) -> int:
        """
        Remove patterns older than specified days.
        
        Args:
            days: Remove patterns older than this
            
        Returns:
            Number of patterns removed
        """
        try:
            conn = self._get_conn()
            
            threshold = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
            
            cursor = conn.execute("""
                DELETE FROM trade_patterns
                WHERE timestamp < ?
            """, (threshold,))
            
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Removed {count} patterns older than {days} days")
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old patterns: {e}")
            return 0
    
    def get_pattern_count(self, symbol: Optional[str] = None) -> int:
        """Get total pattern count."""
        conn = self._get_conn()
        
        if symbol:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trade_patterns WHERE symbol = ?",
                (symbol,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM trade_patterns")
        
        return cursor.fetchone()[0]
