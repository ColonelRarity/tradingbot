"""Telegram command bot (polling) + trade push notifications."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import requests

from bot.config import Config, load_config
from bot.learning.journal import TradeJournal
from bot.notify import send_message
from bot.telegram.activity_log import log_activity
from bot.telegram.formatters import (
    BotSnapshot,
    _e,
    format_balance_short,
    format_help,
    format_history,
    format_positions,
    format_snapshot,
    format_stats,
)

logger = logging.getLogger(__name__)

SnapshotFn = Callable[[], BotSnapshot]


class TelegramMonitor:
    """Long-poll Telegram commands in a background thread."""

    def __init__(
        self,
        config: Config,
        snapshot_fn: SnapshotFn,
        journal: Optional[TradeJournal] = None,
    ):
        self.cfg = config
        self.snapshot_fn = snapshot_fn
        self.journal = journal
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._digest_thread: Optional[threading.Thread] = None
        self._offset = 0

    def start(self) -> None:
        if not self.cfg.telegram_enabled:
            return
        if not self.cfg.telegram_token or not self.cfg.telegram_chat_id:
            logger.warning("Telegram enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
            return
        if self._running:
            return
        self._running = True
        if self.cfg.telegram_commands:
            self._thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="telegram-poll"
            )
            self._thread.start()
            logger.info("Telegram commands: ON (chat %s)", self.cfg.telegram_chat_id)
        if self.cfg.telegram_digest_min > 0:
            self._digest_thread = threading.Thread(
                target=self._digest_loop, daemon=True, name="telegram-digest"
            )
            self._digest_thread.start()
            logger.info(
                "Telegram digest: every %d min", self.cfg.telegram_digest_min
            )

    def stop(self) -> None:
        self._running = False

    def _api(self, method: str, **params) -> dict:
        url = f"https://api.telegram.org/bot{self.cfg.telegram_token}/{method}"
        r = requests.get(url, params=params, timeout=35)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("description", "telegram api error"))
        return data

    def _poll_loop(self) -> None:
        while self._running:
            try:
                data = self._api(
                    "getUpdates",
                    offset=self._offset,
                    timeout=25,
                    allowed_updates='["message"]',
                )
                for upd in data.get("result", []):
                    self._offset = max(self._offset, upd["update_id"] + 1)
                    self._handle_update(upd)
            except Exception as e:
                logger.debug("Telegram poll: %s", e)
                time.sleep(3)

    def _digest_loop(self) -> None:
        interval = max(self.cfg.telegram_digest_min, 5) * 60
        time.sleep(interval)  # first digest after one interval
        while self._running:
            try:
                snap = self.snapshot_fn()
                send_message(
                    "📊 <b>Дайджест</b>\n" + format_balance_short(snap),
                    self.cfg,
                )
            except Exception as e:
                logger.debug("Telegram digest: %s", e)
            for _ in range(int(interval)):
                if not self._running:
                    return
                time.sleep(1)

    def _allowed(self, chat_id: int | str) -> bool:
        return str(chat_id) == str(self.cfg.telegram_chat_id).strip()

    def _handle_update(self, upd: dict) -> None:
        msg = upd.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None or not self._allowed(chat_id):
            return
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        cmd = text.split()[0].split("@")[0].lower()
        try:
            snap = self.snapshot_fn()
            if cmd in ("/start", "/help"):
                reply = format_help()
            elif cmd == "/status":
                reply = format_snapshot(snap)
            elif cmd in ("/balance", "/bal"):
                reply = format_balance_short(snap)
            elif cmd in ("/positions", "/pos"):
                reply = format_positions(snap)
            elif cmd in ("/history", "/hist"):
                reply = format_history(self.journal, limit=12)
            elif cmd == "/stats":
                reply = format_stats(self.journal, snap)
            else:
                reply = "Невідома команда. /help"
            send_message(reply, self.cfg)
        except Exception as e:
            logger.warning("Telegram command %s: %s", cmd, e)
            send_message(f"Помилка: {e}", self.cfg)


def push_trade(
    kind: str,
    symbol: str,
    message: str,
    *,
    pnl_usdt: Optional[float] = None,
    config: Optional[Config] = None,
    notify: bool = True,
) -> None:
    """Log activity + optional Telegram push."""
    cfg = config or load_config()
    log_activity(kind, message, symbol=symbol, pnl_usdt=pnl_usdt)
    if not notify or not cfg.telegram_enabled:
        return
    if kind == "OPEN" and not cfg.telegram_notify_opens:
        return
    if kind in ("CLOSE", "EXIT") and not cfg.telegram_notify_closes:
        return
    icons = {
        "OPEN": "🟢",
        "CLOSE": "🔴",
        "EXIT": "🔴",
        "HEDGE": "🟠",
        "HEDGE_OFF": "⚪",
        "START": "▶️",
        "STOP": "⏹",
    }
    icon = icons.get(kind, "•")
    pnl_part = f"\nPnL: <b>{pnl_usdt:+.2f}</b> USDT" if pnl_usdt is not None else ""
    send_message(
        f"{icon} <b>{_e(kind)}</b> {_e(symbol)}\n{_e(message)}{pnl_part}",
        cfg,
    )
