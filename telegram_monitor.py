"""
Telegram Monitor Bot

Monitors trading bot status and sends updates to Telegram.
Uses only requests library (no telegram bot library).

Features:
- Model status
- Training status
- Open orders
- Active positions with detailed info
- Daily statistics
- Real-time updates
"""

from __future__ import annotations

import os
import time
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram configuration."""
    bot_token: str
    chat_id: str
    enabled: bool = True


class TelegramMonitor:
    """
    Telegram Monitor using only requests API.
    
    Sends formatted messages to Telegram via Bot API.
    """
    
    def __init__(self, config: TelegramConfig):
        """
        Initialize Telegram monitor.
        
        Args:
            config: Telegram configuration
        """
        self.config = config
        self.api_url = f"https://api.telegram.org/bot{config.bot_token}"
        self.last_status_time = 0
        self.status_interval = 300  # 5 minutes
        
        if config.enabled:
            self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            response = requests.get(f"{self.api_url}/getMe", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.info(f"Telegram bot connected: @{data['result']['username']}")
                    return True
            logger.warning("Telegram bot connection failed")
            return False
        except Exception as e:
            logger.warning(f"Telegram connection test failed: {e}")
            return False
    
    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send message to Telegram.
        
        Args:
            text: Message text
            parse_mode: Parse mode (HTML or Markdown)
            
        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.config.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("ok", False)
            
            return False
            
        except Exception as e:
            logger.debug(f"Failed to send Telegram message: {e}")
            return False
    
    def send_status(
        self,
        positions: List,
        orders: List,
        daily_stats: Dict,
        model_stats: Dict,
        training_stats: Dict,
        balance: float,
        available_balance: float
    ) -> None:
        """
        Send comprehensive status update.
        
        Args:
            positions: List of open positions
            orders: List of open orders
            daily_stats: Daily statistics
            model_stats: ML model statistics
            training_stats: Training statistics
            balance: Total balance
            available_balance: Available balance
        """
        now = time.time()
        if now - self.last_status_time < self.status_interval:
            return
        
        self.last_status_time = now
        
        msg = self._format_status(
            positions, orders, daily_stats, model_stats,
            training_stats, balance, available_balance
        )
        
        self._send_message(msg)
    
    def send_position_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        size_usdt: float,
        sl_price: float,
        tp_price: float,
        confidence: float
    ) -> None:
        """Send notification when position is opened."""
        msg = (
            f"<b>✅ POSITION OPENED</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side}</b>\n"
            f"Entry: {entry_price:.6f}\n"
            f"Size: {quantity:.6f} ({size_usdt:.2f} USDT)\n"
            f"SL: {sl_price:.6f}\n"
            f"TP: {tp_price:.6f}\n"
            f"Confidence: {confidence:.3f}"
        )
        self._send_message(msg)
    
    def send_position_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        reason: str
    ) -> None:
        """Send notification when position is closed."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"<b>{emoji} POSITION CLOSED</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: {side}\n"
            f"Entry: {entry_price:.6f}\n"
            f"Exit: {exit_price:.6f}\n"
            f"PnL: <b>{pnl:.2f} USDT</b> ({pnl_percent:.2f}%)\n"
            f"Reason: {reason}"
        )
        self._send_message(msg)
    
    def send_alert(self, title: str, message: str) -> None:
        """Send alert message."""
        msg = f"<b>⚠️ {title}</b>\n\n{message}"
        self._send_message(msg)
    
    def _format_status(
        self,
        positions: List,
        orders: List,
        daily_stats: Dict,
        model_stats: Dict,
        training_stats: Dict,
        balance: float,
        available_balance: float
    ) -> str:
        """Format status message."""
        lines = []
        lines.append("<b>📊 TRADING BOT STATUS</b>\n")
        lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        
        # Balance
        lines.append(f"\n<b>💰 Balance:</b>")
        lines.append(f"Total: {balance:.2f} USDT")
        lines.append(f"Available: {available_balance:.2f} USDT")
        
        # Daily Stats
        lines.append(f"\n<b>📈 Daily Statistics:</b>")
        lines.append(f"Realized PnL: {daily_stats.get('realized_pnl', 0):.2f} USDT")
        lines.append(f"Total PnL: {daily_stats.get('total_pnl', 0):.2f} USDT")
        lines.append(f"Trades: {daily_stats.get('trade_count', 0)}")
        lines.append(f"Win Rate: {daily_stats.get('win_rate', 0):.1%}")
        lines.append(f"Remaining Trades: {daily_stats.get('remaining_trades', 0)}")
        
        # Positions
        lines.append(f"\n<b>📌 Open Positions ({len(positions)}):</b>")
        if positions:
            total_unrealized = 0
            for pos in positions:
                pnl = getattr(pos, 'unrealized_pnl', 0)
                total_unrealized += pnl
                side_emoji = "🟢" if pos.side == "LONG" else "🔴"
                lines.append(
                    f"{side_emoji} <b>{pos.symbol}</b> {pos.side}\n"
                    f"  Entry: {pos.entry_price:.6f}\n"
                    f"  Size: {abs(pos.quantity):.6f}\n"
                    f"  PnL: {pnl:.2f} USDT\n"
                    f"  SL: {pos.sl_price:.6f} | TP: {pos.tp_price:.6f}"
                )
            lines.append(f"\n<b>Total Unrealized PnL: {total_unrealized:.2f} USDT</b>")
        else:
            lines.append("No open positions")
        
        # Orders
        lines.append(f"\n<b>📋 Open Orders ({len(orders)}):</b>")
        if orders:
            for order in orders[:10]:  # Limit to 10
                lines.append(
                    f"{order.order_type} {order.side} {order.symbol}\n"
                    f"  Qty: {order.quantity:.6f} | Price: {order.stop_price:.6f}"
                )
            if len(orders) > 10:
                lines.append(f"... and {len(orders) - 10} more")
        else:
            lines.append("No open orders")
        
        # Model Stats
        lines.append(f"\n<b>🤖 ML Model:</b>")
        lines.append(f"Predictions: {model_stats.get('prediction_count', 0)}")
        lines.append(f"Accuracy: {model_stats.get('accuracy', 0):.1%}")
        lines.append(f"Cache Size: {model_stats.get('cache_size', 0)}")
        lines.append(f"Loaded: {'✅' if model_stats.get('model_loaded', False) else '❌'}")
        
        # Training Stats
        if training_stats:
            lines.append(f"\n<b>🎓 Training:</b>")
            lines.append(f"Epochs: {training_stats.get('epochs_trained', 0)}")
            lines.append(f"Best Val Loss: {training_stats.get('best_val_loss', 0):.4f}")
            lines.append(f"Latest Accuracy: {training_stats.get('latest_accuracy', 0):.1%}")
            lines.append(f"Latest F1: {training_stats.get('latest_f1', 0):.3f}")
        
        return "\n".join(lines)


def create_telegram_monitor() -> Optional[TelegramMonitor]:
    """Create Telegram monitor from environment/config."""
    from config.settings import get_settings
    
    settings = get_settings()
    config = settings.telegram
    
    if not config.enabled or not config.bot_token:
        return None
    
    # Get chat_id from config, env, or file
    chat_id = config.chat_id
    if not chat_id:
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    if not chat_id:
        # Try reading from file
        try:
            chat_id_path = "telegram_chat_id.txt"
            if os.path.exists(chat_id_path):
                with open(chat_id_path, 'r') as f:
                    chat_id = f.read().strip()
        except Exception:
            pass
    
    if not chat_id:
        logger.warning("Telegram chat_id not found. Telegram monitoring disabled.")
        return None
    
    return TelegramMonitor(TelegramConfig(
        bot_token=config.bot_token,
        chat_id=chat_id,
        enabled=True
    ))
