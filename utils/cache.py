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
import inspect
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


# Глобальні кеші: ключ (тип, ttl, maxsize) — щоб декоратори з різним TTL не змішували записи
_analysis_caches: Dict[Tuple[str, float, int], TTLCache] = {}
_price_caches: Dict[Tuple[str, float, int], TTLCache] = {}


def get_analysis_cache(ttl: float = 5.0, maxsize: int = 200) -> TTLCache:
    """Отримати кеш для аналізу ринку (окремий екземпляр на кожну пару ttl/maxsize)."""
    key = ("analysis", ttl, maxsize)
    if key not in _analysis_caches:
        _analysis_caches[key] = TTLCache(maxsize=maxsize, ttl=ttl)
    return _analysis_caches[key]


def get_price_cache(ttl: float = 2.0, maxsize: int = 500) -> TTLCache:
    """Отримати кеш для цін."""
    key = ("price", ttl, maxsize)
    if key not in _price_caches:
        _price_caches[key] = TTLCache(maxsize=maxsize, ttl=ttl)
    return _price_caches[key]


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
            # Ключ кешу: для методів класу відкидаємо лише реальний self (інстанс), не примітиви
            try:
                params = list(inspect.signature(func).parameters.keys())
            except (ValueError, TypeError, OSError):
                params = []

            if (
                args
                and params
                and params[0] in ("self", "cls")
                and not isinstance(args[0], (int, float, str, bool, bytes, type(None)))
            ):
                cache_key = (id(func), func.__name__) + tuple(args[1:]) + tuple(sorted(kwargs.items()))
            else:
                cache_key = (id(func), func.__name__) + tuple(args) + tuple(sorted(kwargs.items()))

            # Спробувати отримати з кешу
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Виконати функцію
            result = func(*args, **kwargs)

            # Не кешувати відповіді з явною помилкою в dict
            if not (isinstance(result, dict) and "error" in result):
                cache.set(cache_key, result)

            return result
        
        # Додати методи для управління кешем
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.cleanup_expired = cache.cleanup_expired
        
        return wrapper
    
    return decorator


def clear_all_caches() -> None:
    """Очистити всі кеші та скинути глобальні екземпляри."""
    global _analysis_caches, _price_caches
    for c in _analysis_caches.values():
        c.clear()
    for c in _price_caches.values():
        c.clear()
    _analysis_caches = {}
    _price_caches = {}


def cleanup_all_caches() -> int:
    """
    Очистити застарілі записи з усіх кешів.
    
    Returns:
        Загальна кількість видалених записів
    """
    total = 0
    for c in list(_analysis_caches.values()):
        total += c.cleanup_expired()
    for c in list(_price_caches.values()):
        total += c.cleanup_expired()
    return total

