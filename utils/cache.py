"""
LRU Cache з TTL (Time To Live) для кешування результатів API викликів.

Використання:
    from utils.cache import cached
    
    @cached(ttl=5, maxsize=100)
    def expensive_function(symbol: str):
        # ... важкі обчислення
        return result
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from collections import OrderedDict
import threading


class TTLCache:
    """
    Thread-safe LRU cache з TTL (Time To Live).
    
    Args:
        maxsize: Максимальна кількість елементів у кеші
        ttl: Час життя запису в секундах
    """
    
    def __init__(self, maxsize: int = 128, ttl: float = 5.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[Tuple, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: Tuple) -> Optional[Any]:
        """
        Отримати значення з кешу.
        
        Returns:
            Значення або None якщо не знайдено або застаріло
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            timestamp, value = self._cache[key]
            
            # Перевірити чи не застаріло
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            
            # Перемістити в кінець (LRU)
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: Tuple, value: Any) -> None:
        """Додати значення до кешу."""
        with self._lock:
            # Якщо ключ вже є - оновити
            if key in self._cache:
                self._cache.move_to_end(key)
            
            # Додати новий запис
            self._cache[key] = (time.time(), value)
            
            # Якщо перевищено maxsize - видалити найстаріший
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
    
    def clear(self) -> None:
        """Очистити весь кеш."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Видалити застарілі записи.
        
        Returns:
            Кількість видалених записів
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (timestamp, _) in self._cache.items()
                if now - timestamp > self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def size(self) -> int:
        """Повернути поточний розмір кешу."""
        with self._lock:
            return len(self._cache)


# Глобальні кеші для різних типів даних
_analysis_cache: Optional[TTLCache] = None
_price_cache: Optional[TTLCache] = None


def get_analysis_cache(ttl: float = 5.0, maxsize: int = 200) -> TTLCache:
    """Отримати глобальний кеш для аналізу ринку."""
    global _analysis_cache
    if _analysis_cache is None:
        _analysis_cache = TTLCache(maxsize=maxsize, ttl=ttl)
    return _analysis_cache


def get_price_cache(ttl: float = 2.0, maxsize: int = 500) -> TTLCache:
    """Отримати глобальний кеш для цін."""
    global _price_cache
    if _price_cache is None:
        _price_cache = TTLCache(maxsize=maxsize, ttl=ttl)
    return _price_cache


def cached(ttl: float = 5.0, maxsize: int = 128, cache_type: str = "analysis"):
    """
    Декоратор для кешування результатів функції з TTL.
    
    Args:
        ttl: Час життя кешу в секундах
        maxsize: Максимальна кількість записів у кеші
        cache_type: Тип кешу ("analysis" або "price")
    
    Приклад:
        @cached(ttl=5.0, maxsize=100)
        def analyze_market(symbol: str):
            # ... важкі обчислення
            return result
    """
    def decorator(func: Callable) -> Callable:
        # Вибрати кеш залежно від типу
        if cache_type == "price":
            cache = get_price_cache(ttl=ttl, maxsize=maxsize)
        else:
            cache = get_analysis_cache(ttl=ttl, maxsize=maxsize)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Створити ключ кешу з аргументів
            # Для методів класу перший аргумент (self) ігноруємо
            if args and hasattr(args[0], '__class__'):
                # Метод класу - ігноруємо self
                cache_key = (func.__name__,) + tuple(args[1:]) + tuple(sorted(kwargs.items()))
            else:
                # Звичайна функція
                cache_key = (func.__name__,) + tuple(args) + tuple(sorted(kwargs.items()))
            
            # Спробувати отримати з кешу
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Виконати функцію
            result = func(*args, **kwargs)
            
            # Зберегти результат у кеш (тільки якщо не помилка)
            if isinstance(result, dict) and 'error' not in result:
                cache.set(cache_key, result)
            
            return result
        
        # Додати методи для управління кешем
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.cleanup_expired = cache.cleanup_expired
        
        return wrapper
    
    return decorator


def clear_all_caches() -> None:
    """Очистити всі кеші."""
    global _analysis_cache, _price_cache
    if _analysis_cache:
        _analysis_cache.clear()
    if _price_cache:
        _price_cache.clear()


def cleanup_all_caches() -> int:
    """
    Очистити застарілі записи з усіх кешів.
    
    Returns:
        Загальна кількість видалених записів
    """
    total = 0
    global _analysis_cache, _price_cache
    if _analysis_cache:
        total += _analysis_cache.cleanup_expired()
    if _price_cache:
        total += _price_cache.cleanup_expired()
    return total

