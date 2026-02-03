"""
Тести для кастомних exceptions.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.exceptions import (
    TradingBotError, APIError, RateLimitError, TimestampError,
    InsufficientBalanceError, OrderError, MarketDataError
)


class TestExceptions(unittest.TestCase):
    """Тести для кастомних exceptions"""
    
    def test_trading_bot_error(self):
        """Тест базового exception"""
        error = TradingBotError("Test error", context={'key': 'value'})
        self.assertEqual(str(error), "Test error (key=value)")
        self.assertEqual(error.context, {'key': 'value'})
    
    def test_api_error(self):
        """Тест APIError"""
        error = APIError("API error", error_code=429, symbol="BTCUSDT")
        self.assertEqual(error.error_code, 429)
        self.assertEqual(error.symbol, "BTCUSDT")
        self.assertIn('error_code', error.context)
        self.assertIn('symbol', error.context)
    
    def test_rate_limit_error(self):
        """Тест RateLimitError"""
        error = RateLimitError("Rate limit", retry_after=60)
        self.assertEqual(error.error_code, 429)
        self.assertEqual(error.retry_after, 60)
        self.assertIn('retry_after', error.context)
    
    def test_timestamp_error(self):
        """Тест TimestampError"""
        error = TimestampError("Timestamp error")
        self.assertEqual(error.error_code, -1021)
    
    def test_insufficient_balance_error(self):
        """Тест InsufficientBalanceError"""
        error = InsufficientBalanceError(
            "Insufficient balance",
            required=100.0,
            available=50.0
        )
        self.assertEqual(error.required, 100.0)
        self.assertEqual(error.available, 50.0)
        self.assertIn('required', error.context)
        self.assertIn('available', error.context)
    
    def test_order_error(self):
        """Тест OrderError"""
        error = OrderError("Order error", symbol="BTCUSDT", order_id="12345")
        self.assertEqual(error.symbol, "BTCUSDT")
        self.assertEqual(error.order_id, "12345")
        self.assertIn('symbol', error.context)
        self.assertIn('order_id', error.context)
    
    def test_market_data_error(self):
        """Тест MarketDataError"""
        error = MarketDataError("Market data error", symbol="BTCUSDT")
        self.assertEqual(error.symbol, "BTCUSDT")
        self.assertIn('symbol', error.context)


if __name__ == '__main__':
    unittest.main()

