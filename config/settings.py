"""
Trading Bot Configuration - Self-Learning Binance USDT-M Futures System

All configuration parameters for:
- Exchange connectivity (Testnet only)
- ML model parameters
- Risk management
- Trading rules
- Hedge logic
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    """Parse float from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Parse int from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


class ModelType(str, Enum):
    """Supported ML model types."""
    MLP = "mlp"
    LSTM = "lstm"


@dataclass
class ExchangeConfig:
    """Binance Testnet API configuration."""
    
    # API Credentials (from environment or fallback to DEMO keys)
    api_key: str = field(default_factory=lambda: os.getenv(
        "BINANCE_FUTURES_API_KEY",
        os.getenv("BINANCE_DEMO_API_KEY", "")
    ))
    api_secret: str = field(default_factory=lambda: os.getenv(
        "BINANCE_FUTURES_API_SECRET",
        os.getenv("BINANCE_DEMO_API_SECRET", "")
    ))
    
    # Base endpoints (default to Testnet; production requires explicit opt-in)
    base_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_FUTURES_BASE_URL", "https://testnet.binancefuture.com")
    )
    ws_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_FUTURES_WS_URL", "wss://stream.binancefuture.com")
    )
    allow_production: bool = field(
        default_factory=lambda: _env_bool("ALLOW_PRODUCTION_TRADING", False)
    )
    dry_run: bool = field(
        default_factory=lambda: _env_bool("DRY_RUN", False)
    )
    
    # Connection settings
    recv_window: int = 5000
    timeout: int = 30
    
    # Leverage (default 20x, minimum 10x)
    default_leverage: int = field(
        default_factory=lambda: _env_int("DEFAULT_LEVERAGE", 20)
    )
    
    # Rate limiting (conservative)
    max_requests_per_minute: int = 1000
    max_orders_per_10sec: int = 40
    max_orders_per_second: int = 8


@dataclass
class MarketDataConfig:
    """Market data collection settings."""
    
    # Multi-pair scanning
    scan_all_pairs: bool = True  # Scan all USDT pairs
    max_pairs_to_scan: int = field(
        default_factory=lambda: _env_int("MAX_PAIRS_TO_SCAN", 150)  # Top pairs by volume (default: 150)
    )
    min_volume_24h: float = 10_000_000  # Min 24h volume in USDT
    
    # Scan interval (seconds) - 12 hours = 43200
    market_scan_interval: float = 43200.0
    
    # Primary trading pair (fallback if scan_all_pairs=False)
    symbol: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "BTCUSDT"))
    
    # Candle interval for ML inference
    candle_interval: str = "1m"
    
    # Ticker update interval for trade management (seconds)
    ticker_interval: float = 10.0
    
    # Historical data for model training
    lookback_candles: int = 500
    
    # Feature calculation windows
    atr_period: int = 14
    momentum_period: int = 10
    volatility_period: int = 20
    
    # Multi-timeframe for confirmation
    mtf_intervals: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])


@dataclass
class MLConfig:
    """Machine Learning model configuration."""
    
    # Model type: MLP or LSTM
    model_type: ModelType = field(
        default_factory=lambda: ModelType(os.getenv("ML_MODEL_TYPE", "mlp").lower())
    )
    
    # Input features
    feature_window: int = 20  # Rolling window size for features
    
    # MLP architecture
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    mlp_dropout: float = 0.2
    
    # LSTM architecture
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs_per_retrain: int = 5
    min_samples_for_training: int = 100
    
    # Incremental learning
    retrain_on_closed_positions: bool = True
    weight_recent_data: float = 2.0  # Weight multiplier for recent samples
    decay_factor: float = 0.95  # Exponential decay for older samples
    
    # Model persistence
    model_save_path: str = "data/models"
    pattern_db_path: str = "data/patterns.db"
    
    # Confidence thresholds (near zero for testnet learning)
    min_confidence_for_entry: float = field(
        default_factory=lambda: _env_float("ML_MIN_CONFIDENCE", 0.001)  # Almost zero for testnet
    )
    high_confidence_threshold: float = 0.10


@dataclass
class RiskConfig:
    """Risk management parameters."""
    
    # Position sizing
    max_position_percent: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_PERCENT", 0.25)
    )
    max_loss_usdt_per_trade: float = field(
        default_factory=lambda: _env_float("MAX_LOSS_PER_TRADE", 2.0)
    )
    
    # Volatility bands for entry (lowered for testnet testing)
    min_volatility_percent: float = 0.03  # Allow low volatility on testnet
    max_volatility_percent: float = 5.0
    
    # ATR-based risk
    atr_sl_multiplier: float = 1.2
    atr_tp_multiplier: float = 1.8
    
    # Hard caps (relaxed for testnet low volatility)
    max_sl_percent: float = 1.0
    min_tp_percent: float = 0.05  # Very low for testnet
    max_tp_percent: float = 5.0
    
    # Daily limits
    daily_max_loss_usdt: float = field(
        default_factory=lambda: _env_float("DAILY_MAX_LOSS", 50.0)
    )
    daily_profit_target_usdt: float = field(
        default_factory=lambda: _env_float("DAILY_PROFIT_TARGET", 20.0)
    )
    max_trades_per_day: int = field(
        default_factory=lambda: _env_int("MAX_TRADES_PER_DAY", 50)
    )
    
    # Concurrent positions
    max_concurrent_positions: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_POSITIONS", 10)
    )
    
    # Maximum unrealized loss before stopping new positions
    max_unrealized_loss_usdt: float = field(
        default_factory=lambda: _env_float("MAX_UNREALIZED_LOSS", 50.0)
    )


@dataclass
class StopLossConfig:
    """
    Stop Loss Configuration
    
    PRIMARY GOAL: Move SL to breakeven as fast as possible
    WORST-CASE OUTCOME: Small profit most of the time
    """
    
    # SL recalculation interval (seconds)
    recalc_interval_sec: float = 10.0
    
    # Breakeven trigger - when position reaches this profit, move SL to entry
    breakeven_trigger_usdt: float = field(
        default_factory=lambda: _env_float("SL_BREAKEVEN_TRIGGER_USDT", 10.0)
    )
    breakeven_trigger_percent: float = field(
        default_factory=lambda: _env_float("SL_BREAKEVEN_TRIGGER_PERCENT", 0.10)
    )
    
    # After breakeven, trail SL to lock in profit
    trail_activation_profit_usdt: float = field(
        default_factory=lambda: _env_float("SL_TRAIL_ACTIVATION_PROFIT_USDT", 1.0)
    )
    trail_distance_atr_mult: float = field(
        default_factory=lambda: _env_float("SL_TRAIL_DISTANCE_ATR_MULT", 0.8)
    )
    
    # Minimum profit buffer above entry for SL
    min_profit_buffer_usdt: float = field(
        default_factory=lambda: _env_float("SL_MIN_PROFIT_BUFFER_USDT", 0.10)
    )
    
    # Emergency SL - never let loss exceed this
    emergency_sl_percent: float = 1.0


@dataclass
class TakeProfitConfig:
    """
    Take Profit Configuration
    
    Uses Fibonacci logic - second step from entry
    Adapts to volatility regime
    """
    
    # TP recalculation interval (seconds)
    recalc_interval_sec: float = 60.0
    
    # Fibonacci levels for TP calculation
    fib_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.500, 0.618, 0.786, 1.0])
    
    # Use second Fibonacci step by default
    default_fib_step: int = 1  # 0-indexed, so 1 = second level (0.382)
    
    # TP distance constraints
    min_distance_percent: float = 0.30
    max_distance_percent: float = 2.5
    
    # TP must not be "near" current price
    min_distance_from_current_price_percent: float = 0.15
    
    # Volatility-based adjustment
    low_volatility_fib_step: int = 0  # Use closer TP in low vol
    high_volatility_fib_step: int = 2  # Use further TP in high vol
    volatility_threshold_low: float = 0.3
    volatility_threshold_high: float = 1.5


@dataclass
class HedgeConfig:
    """
    Hedge (Mirror) Position Configuration
    
    Opens opposite position when main position is losing
    Goal: Reduce net drawdown, never pyramid uncontrollably
    """
    
    # Trigger: Open hedge when main position PnL <= this value
    trigger_pnl_usdt: float = -7.0  # As requested by user
    
    # Hedge position size (adaptive: 30-50% of parent position size)
    fixed_size_usdt: float = 50.0  # Fallback if adaptive calculation fails
    adaptive_size_percent: float = 0.40  # 40% of parent position size
    
    # Hedge close target: Close hedge at this profit
    close_profit_usdt: float = 3.0
    
    # NO TP extension for hedge - strictly closes at target
    allow_tp_extension: bool = False
    
    # Maximum hedges per parent position
    max_hedges_per_position: int = 1
    
    # Cooldown between hedge attempts (seconds)
    hedge_cooldown_sec: float = 60.0
    
    # Hedge must be strictly linked to parent
    require_parent_link: bool = True


@dataclass
class OrderConfig:
    """Order management settings."""
    
    # Order types (MANDATORY)
    entry_order_type: str = "MARKET"
    sl_order_type: str = "STOP_MARKET"
    tp_order_type: str = "TAKE_PROFIT_MARKET"
    
    # Idempotency
    cancel_before_replace: bool = True
    max_cancel_retries: int = 3
    
    # Orphan order cleanup
    cleanup_orphan_orders: bool = True
    orphan_check_interval_sec: float = 30.0
    
    # Order timeout
    order_timeout_sec: float = 30.0


@dataclass
class TelegramConfig:
    """Telegram notifications configuration."""
    
    enabled: bool = field(default_factory=lambda: _env_bool("TELEGRAM_ENABLED", False))
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    
    # Notification settings
    notify_on_entry: bool = True
    notify_on_exit: bool = True
    notify_on_hedge: bool = True
    notify_on_error: bool = True
    notify_on_daily_summary: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = "trading_bot.log"
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "text"))
    
    # Decision logging
    log_all_decisions: bool = True
    log_reason_codes: bool = True


@dataclass
class Settings:
    """
    Master configuration container.
    
    All settings for the self-learning trading bot.
    """
    
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Exchange validation
        if not self.exchange.api_key:
            errors.append("BINANCE_FUTURES_API_KEY not set")
        if not self.exchange.api_secret:
            errors.append("BINANCE_FUTURES_API_SECRET not set")
        if "testnet" not in self.exchange.base_url.lower() and not self.exchange.allow_production:
            errors.append("Production URL set but ALLOW_PRODUCTION_TRADING is not enabled")
        
        # Risk validation
        if self.risk.max_position_percent <= 0 or self.risk.max_position_percent > 1:
            errors.append("max_position_percent must be between 0 and 1")
        if self.risk.max_sl_percent <= 0:
            errors.append("max_sl_percent must be positive")
        
        # ML validation
        if self.ml.min_confidence_for_entry < 0 or self.ml.min_confidence_for_entry > 1:
            errors.append("min_confidence_for_entry must be between 0 and 1")
        
        # Hedge validation
        if self.hedge.trigger_pnl_usdt >= 0:
            errors.append("hedge trigger_pnl_usdt must be negative")
        if self.hedge.close_profit_usdt <= 0:
            errors.append("hedge close_profit_usdt must be positive")
        
        return errors


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
