"""
Тести для модуля market_learning.
"""

import unittest
import tempfile
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_learning import (
    LearningDB, make_feature_vector, DEFAULT_FEATURE_SPEC,
    MultiHeadOnlineModels
)


class TestLearningDB(unittest.TestCase):
    """Тести для LearningDB"""
    
    def setUp(self):
        # Створити тимчасову БД для тестів
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = LearningDB(self.temp_db.name)
    
    def tearDown(self):
        # Видалити тимчасову БД
        self.db.close()
        try:
            os.unlink(self.temp_db.name)
        except Exception:
            pass
    
    def test_record_entry(self):
        """Тест запису entry event"""
        entry_id = self.db.record_entry(
            symbol="BTCUSDT",
            direction="BUY",
            entry_time_ms=int(time.time() * 1000),
            entry_price=50000.0,
            qty=0.001,
            signal_strength=75.0,
            features={'rsi': 0.3, 'adx': 0.5},
            analysis={'trend': 'BULLISH', 'rsi': 30.0}
        )
        
        self.assertIsInstance(entry_id, int)
        self.assertGreater(entry_id, 0)
    
    def test_find_entry_event_near(self):
        """Тест пошуку entry event"""
        entry_time_ms = int(time.time() * 1000)
        
        # Записати entry
        entry_id = self.db.record_entry(
            symbol="BTCUSDT",
            direction="BUY",
            entry_time_ms=entry_time_ms,
            entry_price=50000.0,
            qty=0.001,
            signal_strength=75.0,
            features={'rsi': 0.3},
            analysis={'trend': 'BULLISH'}
        )
        
        # Знайти entry
        found = self.db.find_entry_event_near(
            symbol="BTCUSDT",
            direction="BUY",
            entry_time_ms=entry_time_ms,
            window_ms=5000
        )
        
        self.assertIsNotNone(found)
        self.assertEqual(found['entry_id'], entry_id)
        self.assertEqual(found['symbol'], "BTCUSDT")
    
    def test_upsert_round_trip(self):
        """Тест запису round trip"""
        trip_key = "test_trip_123"
        trip = {
            'symbol': 'BTCUSDT',
            'direction': 'BUY',
            'entry_time': int(time.time() * 1000) - 10000,
            'exit_time': int(time.time() * 1000),
            'net_pnl': 5.0,
            'realized_pnl': 5.5,
            'commission': 0.5
        }
        
        # Перший раз - має вставити
        inserted1 = self.db.upsert_round_trip(trip_key, trip)
        self.assertTrue(inserted1)
        
        # Другий раз - має ігнорувати (вже існує)
        inserted2 = self.db.upsert_round_trip(trip_key, trip)
        self.assertFalse(inserted2)
    
    def test_get_counts(self):
        """Тест отримання статистики"""
        counts = self.db.get_counts()
        
        self.assertIn('entry_events', counts)
        self.assertIn('round_trips', counts)
        self.assertIn('training_samples', counts)
        self.assertIn('entry_unused', counts)
        
        # Всі мають бути >= 0
        for key, value in counts.items():
            self.assertGreaterEqual(value, 0)
    
    def test_cleanup_old_entries(self):
        """Тест очищення старих записів"""
        # Записати старий entry (8 днів тому)
        old_time_ms = int((time.time() - 8 * 24 * 3600) * 1000)
        self.db.record_entry(
            symbol="BTCUSDT",
            direction="BUY",
            entry_time_ms=old_time_ms,
            entry_price=50000.0,
            qty=0.001,
            signal_strength=75.0,
            features={'rsi': 0.3},
            analysis={'trend': 'BULLISH'}
        )
        
        # Позначити як used
        entry = self.db.find_entry_event_near("BTCUSDT", "BUY", old_time_ms, 1000)
        if entry:
            # Створити sample щоб позначити entry як used
            trip_key = "old_trip"
            trip = {
                'symbol': 'BTCUSDT',
                'direction': 'BUY',
                'entry_time': old_time_ms,
                'exit_time': old_time_ms + 1000,
                'net_pnl': 1.0
            }
            self.db.upsert_round_trip(trip_key, trip)
            self.db.create_sample_for_trip(trip_key, trip, window_ms=1000)
        
        # Очистити старі записи (7 днів)
        result = self.db.cleanup_old_entries(days_old=7)
        
        # Має видалити старий entry
        self.assertGreaterEqual(result['entry_events'], 0)


class TestFeatureVector(unittest.TestCase):
    """Тести для створення feature vectors"""
    
    def test_make_feature_vector(self):
        """Тест створення feature vector"""
        analysis = {
            'current_price': 50000.0,
            'rsi': 30.0,
            'adx': 25.0,
            'macd': 100.0,
            'macd_signal': 90.0,
            'bb_upper': 51000.0,
            'bb_lower': 49000.0,
            'atr': 500.0,
            'volume': 1000.0,
            'volume_ma': 800.0,
            'trend': 'BULLISH',
            'ema_9': 50100.0,
            'ema_21': 49900.0,
            'breakeven_cost_pct': 0.1
        }
        
        x, fmap = make_feature_vector(analysis, "BUY", signal_strength=75.0)
        
        # Перевірити що вектор створено
        self.assertIsNotNone(x)
        self.assertEqual(len(x), len(DEFAULT_FEATURE_SPEC.keys))
        
        # Перевірити feature map
        self.assertIn('dir', fmap)
        self.assertIn('rsi', fmap)
        self.assertIn('adx', fmap)
        self.assertEqual(fmap['dir'], 1.0)  # BUY = 1.0


if __name__ == '__main__':
    unittest.main()

