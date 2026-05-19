"""
WebSocket kline stream for low-latency scalping updates.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Dict, List, Optional

import websocket

from bot.config import load_config

logger = logging.getLogger(__name__)

KlineHandler = Callable[[str, dict], None]


class KlineStream:
    """Subscribe to @kline_<interval> for multiple symbols on one connection."""

    def __init__(self, symbols: List[str], interval: str, on_kline: KlineHandler):
        self.config = load_config()
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.on_kline = on_kline
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _streams_path(self) -> str:
        streams = "/".join(f"{s}@kline_{self.interval}" for s in self.symbols)
        if len(self.symbols) == 1:
            return f"{self.config.ws_base}/{streams}"
        return f"wss://fstream.binance.com/stream?streams={streams}"

    def _on_message(self, _ws, message: str) -> None:
        try:
            data = json.loads(message)
            if "stream" in data:
                payload = data.get("data", {})
            else:
                payload = data
            k = payload.get("k") or payload
            symbol = str(k.get("s", "")).upper()
            if symbol:
                self.on_kline(symbol, k)
        except Exception as e:
            logger.debug("WS parse error: %s", e)

    def _on_error(self, _ws, error) -> None:
        logger.warning("WebSocket error: %s", error)

    def _on_close(self, _ws, *args) -> None:
        logger.warning("WebSocket closed")
        if self._running:
            self._connect()

    def _on_open(self, _ws) -> None:
        logger.info("WebSocket connected (%d symbols, %s)", len(self.symbols), self.interval)

    def _connect(self) -> None:
        url = self._streams_path()
        logger.info("WS → %s", url[:120])
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    def set_symbols(self, symbols: List[str]) -> bool:
        """Replace symbol list; reconnect if already running. Returns True if changed."""
        new_syms = sorted(s.lower() for s in symbols if s)
        if new_syms == sorted(self.symbols):
            return False
        self.symbols = new_syms
        if self._running and self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        return True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._connect, daemon=True, name="kline-ws")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()
