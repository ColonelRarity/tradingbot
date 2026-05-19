"""Optional Telegram notifications."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from bot.config import Config, load_config

logger = logging.getLogger(__name__)


def send_message(text: str, config: Optional[Config] = None) -> None:
    cfg = config or load_config()
    if not cfg.telegram_enabled or not cfg.telegram_token or not cfg.telegram_chat_id:
        return
    url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.debug("Telegram send failed: %s", e)
