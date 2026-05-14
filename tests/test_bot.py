"""
Інтеграційні тести MultiPairTradingBot без реальних викликів Binance / Telegram.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reset_settings_singleton() -> None:
    import config.settings as settings_mod

    settings_mod._settings = None  # type: ignore[attr-defined]


class TestMultiPairTradingBot(unittest.TestCase):
    """Конструктор бота та базова поведінка з замоканим клієнтом біржі."""

    def setUp(self) -> None:
        os.environ["TELEGRAM_ENABLED"] = "false"
        _reset_settings_singleton()
        from exchange.binance_client import reset_binance_client

        reset_binance_client()

    def tearDown(self) -> None:
        _reset_settings_singleton()
        from exchange.binance_client import reset_binance_client

        reset_binance_client()

    def test_bot_initializes_with_mock_client(self) -> None:
        mock_client = MagicMock(name="BinanceClient")
        mock_client.test_connection.return_value = True

        import main

        with patch.object(main, "get_binance_client", return_value=mock_client):
            with patch.object(main, "create_telegram_monitor", return_value=None):
                bot = main.MultiPairTradingBot()

        self.assertIs(bot.client, mock_client)
        self.assertIsNotNone(bot.signal_engine)
        self.assertIsNotNone(bot.position_tracker)
        self.assertIsNotNone(bot.order_manager)
        self.assertEqual(bot.active_symbols, [])

    def test_stop_marks_not_running(self) -> None:
        mock_client = MagicMock(name="BinanceClient")
        mock_client.test_connection.return_value = True

        import main

        with patch.object(main, "get_binance_client", return_value=mock_client):
            with patch.object(main, "create_telegram_monitor", return_value=None):
                bot = main.MultiPairTradingBot()

        bot._running = True
        bot.stop()
        self.assertFalse(bot._running)


if __name__ == "__main__":
    unittest.main()
