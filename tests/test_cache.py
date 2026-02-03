"""
Тести для модуля кешування.
"""

import time
import unittest
from unittest.mock import patch
import sys
import os

# Додати корінь проєкту до шляху
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cache import TTLCache, cached, cleanup_all_caches


class TestTTLCache(unittest.TestCase):
    """Тести для TTLCache"""
    
    def setUp(self):
        self.cache = TTLCache(maxsize=3, ttl=1.0)
    
    def test_set_and_get(self):
        """Тест базового set/get"""
        self.cache.set(('test',), 'value')
        self.assertEqual(self.cache.get(('test',)), 'value')
    
    def test_expiration(self):
        """Тест застаріння записів"""
        self.cache.set(('test',), 'value')
        time.sleep(1.1)  # Більше ніж TTL
        self.assertIsNone(self.cache.get(('test',)))
    
    def test_maxsize(self):
        """Тест обмеження розміру"""
        self.cache.set(('1',), 'value1')
        self.cache.set(('2',), 'value2')
        self.cache.set(('3',), 'value3')
        self.cache.set(('4',), 'value4')  # Має видалити ('1',)
        
        self.assertIsNone(self.cache.get(('1',)))  # Видалено через maxsize
        self.assertEqual(self.cache.get(('4',)), 'value4')
    
    def test_lru_eviction(self):
        """Тест LRU eviction"""
        self.cache.set(('1',), 'value1')
        self.cache.set(('2',), 'value2')
        self.cache.set(('3',), 'value3')
        
        # Доступ до ('1',) - має стати найновішим
        self.cache.get(('1',))
        
        # Додати новий - має видалити ('2',) а не ('1',)
        self.cache.set(('4',), 'value4')
        
        self.assertIsNone(self.cache.get(('2',)))  # Видалено
        self.assertEqual(self.cache.get(('1',)), 'value1')  # Залишилось
    
    def test_cleanup_expired(self):
        """Тест очищення застарілих записів"""
        self.cache.set(('1',), 'value1')
        self.cache.set(('2',), 'value2')
        time.sleep(1.1)
        
        deleted = self.cache.cleanup_expired()
        self.assertEqual(deleted, 2)
        self.assertEqual(self.cache.size(), 0)


class TestCachedDecorator(unittest.TestCase):
    """Тести для декоратора @cached"""
    
    def setUp(self):
        # Очистити кеші перед кожним тестом
        cleanup_all_caches()
    
    def test_caching(self):
        """Тест що функція кешується"""
        call_count = [0]
        
        @cached(ttl=5.0, maxsize=10)
        def test_func(x):
            call_count[0] += 1
            return x * 2
        
        result1 = test_func(5)
        result2 = test_func(5)  # Має взяти з кешу
        
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # Функція викликалась тільки раз
    
    def test_no_cache_on_error(self):
        """Тест що помилки не кешуються"""
        call_count = [0]
        
        @cached(ttl=5.0, maxsize=10)
        def test_func(x):
            call_count[0] += 1
            return {'error': 'test error'}
        
        result1 = test_func(5)
        result2 = test_func(5)
        
        # Обидва рази має викликатись, бо результат містить 'error'
        self.assertEqual(call_count[0], 2)
    
    def test_expiration(self):
        """Тест застаріння кешу"""
        call_count = [0]
        
        @cached(ttl=0.1, maxsize=10)  # Дуже короткий TTL
        def test_func(x):
            call_count[0] += 1
            return x * 2
        
        test_func(5)
        time.sleep(0.15)
        test_func(5)  # Має викликатись знову
        
        self.assertEqual(call_count[0], 2)


if __name__ == '__main__':
    unittest.main()

