"""
Rate Limiter для Binance API
Обмежує кількість запитів до API для уникнення блокування IP
"""

import time
import threading
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter для обмеження кількості API запитів.
    
    Binance має обмеження:
    - 1200 запитів на хвилину (weight-based limits)
    - 50 orders на 10 секунд
    - 10 orders на секунду
    
    При перевищенні → 429 Too Many Requests або -1003 IP ban
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 1000,  # Консервативний ліміт (залишаємо запас)
        max_orders_per_10sec: int = 40,  # Консервативний ліміт
        max_orders_per_second: int = 8,  # Консервативний ліміт
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_orders_per_10sec = max_orders_per_10sec
        self.max_orders_per_second = max_orders_per_second
        
        # Черги для відстеження запитів
        self._request_times = deque()  # Час кожного запиту
        self._order_times_10s = deque()  # Час кожного ордеру (10s вікно)
        self._order_times_1s = deque()  # Час кожного ордеру (1s вікно)
        
        # Lock для thread-safety
        self._lock = threading.Lock()
        
        # Статистика
        self._total_requests = 0
        self._total_waited = 0.0
        
    def wait_if_needed(self, is_order: bool = False) -> float:
        """
        Чекати якщо потрібно для дотримання rate limits.
        
        Args:
            is_order: Чи це запит на створення/скасування ордеру
            
        Returns:
            Час очікування в секундах (якщо було очікування)
        """
        with self._lock:
            now = time.time()
            wait_time = 0.0
            
            # Очистити старі запити (старше 1 хвилини)
            while self._request_times and (now - self._request_times[0]) > 60:
                self._request_times.popleft()
            
            # Перевірити ліміт запитів на хвилину
            if len(self._request_times) >= self.max_requests_per_minute:
                # Потрібно дочекатися поки найстаріший запит не стане старше 60 секунд
                oldest_time = self._request_times[0]
                wait_time = max(wait_time, 60 - (now - oldest_time) + 0.1)  # +0.1 для запасу
            
            # Для ордерів перевіряємо додаткові ліміти
            if is_order:
                # Очистити старі ордери (старше 10 секунд)
                while self._order_times_10s and (now - self._order_times_10s[0]) > 10:
                    self._order_times_10s.popleft()
                
                # Очистити старі ордери (старше 1 секунди)
                while self._order_times_1s and (now - self._order_times_1s[0]) > 1:
                    self._order_times_1s.popleft()
                
                # Перевірити ліміт ордерів на 10 секунд
                if len(self._order_times_10s) >= self.max_orders_per_10sec:
                    oldest_time = self._order_times_10s[0]
                    wait_time = max(wait_time, 10 - (now - oldest_time) + 0.1)
                
                # Перевірити ліміт ордерів на секунду
                if len(self._order_times_1s) >= self.max_orders_per_second:
                    oldest_time = self._order_times_1s[0]
                    wait_time = max(wait_time, 1 - (now - oldest_time) + 0.1)
            
            # Якщо потрібно чекати
            if wait_time > 0:
                self._total_waited += wait_time
                logger.debug(f"⏳ Rate limiter: очікування {wait_time:.2f}s (запитів: {len(self._request_times)}/{self.max_requests_per_minute}, "
                           f"ордерів 10s: {len(self._order_times_10s)}/{self.max_orders_per_10sec}, "
                           f"ордерів 1s: {len(self._order_times_1s)}/{self.max_orders_per_second})")
                time.sleep(wait_time)
                now = time.time()  # Оновити час після очікування
            
            # Додати поточний запит до черги
            self._request_times.append(now)
            self._total_requests += 1
            
            if is_order:
                self._order_times_10s.append(now)
                self._order_times_1s.append(now)
            
            return wait_time
    
    def get_stats(self) -> dict:
        """Отримати статистику rate limiter"""
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'total_waited_seconds': self._total_waited,
                'current_requests_per_minute': len(self._request_times),
                'current_orders_per_10sec': len(self._order_times_10s),
                'current_orders_per_second': len(self._order_times_1s),
                'max_requests_per_minute': self.max_requests_per_minute,
                'max_orders_per_10sec': self.max_orders_per_10sec,
                'max_orders_per_second': self.max_orders_per_second,
            }
    
    def reset_stats(self):
        """Скинути статистику"""
        with self._lock:
            self._total_requests = 0
            self._total_waited = 0.0


# Глобальний rate limiter (singleton)
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Отримати глобальний rate limiter"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def reset_rate_limiter():
    """Скинути глобальний rate limiter (для тестування)"""
    global _global_rate_limiter
    _global_rate_limiter = None
