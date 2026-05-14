"""
Тести «сигнал → ризик → ордер» і перевірка працездатності ланцюга без реальної біржі.

Використовуються синтетичні свічки та MagicMock для REST-клієнта.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange.binance_client import Candle, Order, OrderSide
from config.settings import MarketDataConfig, reload_settings
from core.market_data import MarketData
from core.signal_engine import SignalEngine, SignalDirection
from core.feature_engineering import FeatureEngine
from core.risk_engine import RiskEngine
from core.order_manager import OrderManager


def _synthetic_volatile_candles(count: int, base_price: float = 50_000.0) -> list[Candle]:
    """OHLCV з помітною волатильністю, щоб atr_percent пройшов пороги SignalEngine."""
    random.seed(42)
    candles: list[Candle] = []
    price = base_price
    start_ms = 1_700_000_000_000
    for i in range(count):
        t = start_ms + i * 60_000
        wobble = random.uniform(-0.015, 0.015) * price
        o = price
        c = max(1.0, price + wobble)
        band = random.uniform(0.004, 0.012) * price
        hi = max(o, c) + band
        lo = min(o, c) - band
        vol = 500_000.0 + random.random() * 200_000.0
        candles.append(
            Candle(
                open_time=t,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=vol,
                close_time=t + 59_999,
                quote_volume=vol * c,
                trades=500,
            )
        )
        price = c
    return candles


def _make_exchange_mock(candles: list[Candle]) -> MagicMock:
    client = MagicMock(name="BinanceClient")

    def _klines(symbol: str, interval: str = "1m", limit: int = 100):
        k = min(limit, len(candles))
        return candles[-k:]

    client.get_candles.side_effect = _klines
    client.get_ticker_price.return_value = float(candles[-1].close)
    client.get_positions.return_value = []
    client.load_exchange_info.return_value = None
    client.is_dual_side_position = MagicMock(return_value=False)
    client.get_symbol_info.return_value = {
        "price_precision": 2,
        "quantity_precision": 3,
        "min_qty": 0.001,
        "step_size": 0.001,
        "max_qty": 1_000_000.0,
        "tick_size": 0.01,
        "percent_price_multiplier_up": 5.0,
        "percent_price_multiplier_down": 0.2,
    }
    client.place_market_order.return_value = Order(
        order_id=987654321,
        client_order_id="test-client-id",
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        status="FILLED",
        price=float(candles[-1].close),
        stop_price=0.0,
        quantity=0.01,
        executed_qty=0.01,
        time=1_700_000_000_000,
    )
    return client


class TestSignalsAndTrading(unittest.TestCase):
    """Сигнали, розмір позиції, розміщення market entry (мок)."""

    def setUp(self) -> None:
        os.environ.setdefault("TELEGRAM_ENABLED", "false")
        reload_settings()

    def test_signal_engine_long_with_mocked_ml(self) -> None:
        candles = _synthetic_volatile_candles(160)
        client = _make_exchange_mock(candles)
        md_cfg = MarketDataConfig(lookback_candles=120, mtf_intervals=[])
        md = MarketData("BTCUSDT", client, config=md_cfg)
        self.assertTrue(md.initialize())
        md.update()

        fe = FeatureEngine()
        from ml.inference import ModelInference

        inf = ModelInference()
        with patch.object(
            inf,
            "predict",
            return_value={
                "direction_prob": 0.92,
                "confidence": 0.45,
                "expected_move": 0.012,
            },
        ):
            engine = SignalEngine(feature_engine=fe, ml_inference=inf)
            sig = engine.generate_signal(md, has_open_position=False, has_hedge=False)

        self.assertTrue(sig.is_valid, msg=f"reasons={sig.reason_codes}")
        self.assertEqual(sig.direction, SignalDirection.LONG)
        self.assertGreater(sig.confidence, 0.1)

    def test_risk_engine_sizes_long_position(self) -> None:
        re = RiskEngine()
        snap_price = 50_000.0
        atr = 250.0
        out = re.calculate_position_size(
            available_balance=10_000.0,
            current_price=snap_price,
            atr=atr,
            side="LONG",
            leverage=5,
            symbol=None,
        )
        self.assertTrue(out.is_valid, msg=out.reason)
        self.assertGreater(out.position_size_usdt, 0)
        self.assertGreater(out.position_size_qty, 0)
        self.assertLess(out.stop_loss_price, snap_price)
        self.assertGreater(out.take_profit_price, snap_price)

    def test_order_manager_entry_market_success(self) -> None:
        candles = _synthetic_volatile_candles(50)
        client = _make_exchange_mock(candles)
        om = OrderManager(client=client)
        res = om.place_entry_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            position_id="test-pos-1",
        )
        self.assertTrue(res.success, msg=res.error_message)
        self.assertIsNotNone(res.order)
        self.assertEqual(res.order.order_id, 987654321)
        client.place_market_order.assert_called_once()

    def test_smoke_signal_risk_order_chain(self) -> None:
        """Повний ланцюг: дані ринку → сигнал LONG → розмір → market entry (усі залежності замокані)."""
        candles = _synthetic_volatile_candles(160)
        client = _make_exchange_mock(candles)
        md_cfg = MarketDataConfig(lookback_candles=120, mtf_intervals=[])
        md = MarketData("BTCUSDT", client, config=md_cfg)
        self.assertTrue(md.initialize())
        md.update()
        snap = md.get_snapshot()
        self.assertIsNotNone(snap)

        from ml.inference import ModelInference

        inf = ModelInference()
        with patch.object(
            inf,
            "predict",
            return_value={
                "direction_prob": 0.88,
                "confidence": 0.42,
                "expected_move": 0.01,
            },
        ):
            engine = SignalEngine(ml_inference=inf)
            sig = engine.generate_signal(md, has_open_position=False, has_hedge=False)

        self.assertTrue(sig.is_valid)
        self.assertEqual(sig.direction, SignalDirection.LONG)

        re = RiskEngine()
        risk = re.calculate_position_size(
            available_balance=10_000.0,
            current_price=snap.current_price,
            atr=snap.atr,
            side="LONG",
            leverage=5,
            symbol=None,
        )
        self.assertTrue(risk.is_valid)

        om = OrderManager(client=client)
        entry = om.place_entry_order(
            symbol="BTCUSDT",
            side="LONG",
            quantity=risk.position_size_qty,
            position_id="smoke-chain-1",
        )
        self.assertTrue(entry.success)


if __name__ == "__main__":
    unittest.main()
